// Copyright 2024 Erik Garrison. Apache 2.0 License.
// Elman RNN variants with different activation combinations.
//
// All gated variants follow the same architecture:
//   raw = R @ h + Wx[t] + b          -- [B, 2D] single matmul
//   [h_cand_raw, gate_raw] = split(raw)
//   h_candidate = ACT1(h_cand_raw)   -- [B, D]
//   gate = ACT2(gate_raw)            -- [B, D]
//   h_new = h_candidate * gate       -- [B, D] elementwise
//
// NoGate variant:
//   raw = R @ h + Wx[t] + b          -- [B, D] single matmul
//   h_new = tanh(raw)                -- [B, D]
//
// Variants:
//   ElmanTanh:    ACT1=tanh, ACT2=tanh
//   ElmanSigmoid: ACT1=tanh, ACT2=sigmoid
//   ElmanSwish:   ACT1=silu, ACT2=silu
//   ElmanGelu:    ACT1=tanh, ACT2=gelu
//   ElmanNoGate:  ACT1=tanh, no gate (ablation baseline)

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// ============================================================================
// Activation functions
// ============================================================================

__device__ __forceinline__
float device_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__
float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__
float device_silu(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return x * sig;
}

__device__ __forceinline__
float device_gelu(float x) {
    // Approximate GELU using tanh: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float c = 0.7978845608f;  // sqrt(2/π)
    const float k = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + k * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ============================================================================
// Derivative functions for backward pass
// ============================================================================

__device__ __forceinline__
float d_tanh(float tanh_x) {
    // d/dx tanh(x) = 1 - tanh(x)^2, given tanh(x) already computed
    return 1.0f - tanh_x * tanh_x;
}

__device__ __forceinline__
float d_sigmoid(float sig_x) {
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    return sig_x * (1.0f - sig_x);
}

__device__ __forceinline__
float d_silu(float x, float sig) {
    // d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    return sig * (1.0f + x * (1.0f - sig));
}

__device__ __forceinline__
float d_gelu(float x) {
    // Derivative of approximate GELU
    const float c = 0.7978845608f;
    const float k = 0.044715f;
    float x2 = x * x;
    float inner = c * (x + k * x * x2);
    float tanh_inner = tanhf(inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    float d_inner = c * (1.0f + 3.0f * k * x2);
    return 0.5f * (1.0f + tanh_inner + x * sech2 * d_inner);
}

// ============================================================================
// ElmanTanh: tanh + tanh gate
// ============================================================================

template<typename T>
__global__
void ElmanTanhPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,
    const T* __restrict__ Wx,
    const T* __restrict__ b,
    T* __restrict__ h_next,
    T* __restrict__ v
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    float h_raw = static_cast<float>(Rh[h_cand_idx]) +
                  static_cast<float>(Wx[h_cand_idx]) +
                  static_cast<float>(b[d_idx]);
    float g_raw = static_cast<float>(Rh[gate_idx]) +
                  static_cast<float>(Wx[gate_idx]) +
                  static_cast<float>(b[D + d_idx]);

    float h_candidate = device_tanh(h_raw);
    float gate = device_tanh(g_raw);  // tanh gate
    float h_new = h_candidate * gate;

    h_next[idx] = static_cast<T>(h_new);
    v[h_cand_idx] = static_cast<T>(h_raw);
    v[gate_idx] = static_cast<T>(g_raw);
}

template<typename T>
__global__
void ElmanTanhBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,
    const T* __restrict__ dh_recurrent,
    const T* __restrict__ v,
    T* __restrict__ d_raw
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    const float h_cand_raw = static_cast<float>(v[h_cand_idx]);
    const float gate_raw = static_cast<float>(v[gate_idx]);

    const float h_candidate = device_tanh(h_cand_raw);
    const float gate = device_tanh(gate_raw);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    const float d_h_candidate = dh * gate;
    const float d_gate = dh * h_candidate;

    const float d_gate_raw = d_gate * d_tanh(gate);
    const float d_h_cand_raw = d_h_candidate * d_tanh(h_candidate);

    d_raw[h_cand_idx] = static_cast<T>(d_h_cand_raw);
    d_raw[gate_idx] = static_cast<T>(d_gate_raw);
}

// ============================================================================
// ElmanSigmoid: tanh + sigmoid gate
// ============================================================================

template<typename T>
__global__
void ElmanSigmoidPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,
    const T* __restrict__ Wx,
    const T* __restrict__ b,
    T* __restrict__ h_next,
    T* __restrict__ v
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    float h_raw = static_cast<float>(Rh[h_cand_idx]) +
                  static_cast<float>(Wx[h_cand_idx]) +
                  static_cast<float>(b[d_idx]);
    float g_raw = static_cast<float>(Rh[gate_idx]) +
                  static_cast<float>(Wx[gate_idx]) +
                  static_cast<float>(b[D + d_idx]);

    float h_candidate = device_tanh(h_raw);
    float gate = device_sigmoid(g_raw);  // sigmoid gate
    float h_new = h_candidate * gate;

    h_next[idx] = static_cast<T>(h_new);
    v[h_cand_idx] = static_cast<T>(h_raw);
    v[gate_idx] = static_cast<T>(g_raw);
}

template<typename T>
__global__
void ElmanSigmoidBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,
    const T* __restrict__ dh_recurrent,
    const T* __restrict__ v,
    T* __restrict__ d_raw
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    const float h_cand_raw = static_cast<float>(v[h_cand_idx]);
    const float gate_raw = static_cast<float>(v[gate_idx]);

    const float h_candidate = device_tanh(h_cand_raw);
    const float gate = device_sigmoid(gate_raw);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    const float d_h_candidate = dh * gate;
    const float d_gate = dh * h_candidate;

    const float d_gate_raw = d_gate * d_sigmoid(gate);
    const float d_h_cand_raw = d_h_candidate * d_tanh(h_candidate);

    d_raw[h_cand_idx] = static_cast<T>(d_h_cand_raw);
    d_raw[gate_idx] = static_cast<T>(d_gate_raw);
}

// ============================================================================
// ElmanSwish: silu + silu gate
// ============================================================================

template<typename T>
__global__
void ElmanSwishPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,
    const T* __restrict__ Wx,
    const T* __restrict__ b,
    T* __restrict__ h_next,
    T* __restrict__ v
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    float h_raw = static_cast<float>(Rh[h_cand_idx]) +
                  static_cast<float>(Wx[h_cand_idx]) +
                  static_cast<float>(b[d_idx]);
    float g_raw = static_cast<float>(Rh[gate_idx]) +
                  static_cast<float>(Wx[gate_idx]) +
                  static_cast<float>(b[D + d_idx]);

    float sig_h = device_sigmoid(h_raw);
    float sig_g = device_sigmoid(g_raw);
    float h_candidate = device_silu(h_raw);  // silu h_candidate
    float gate = device_silu(g_raw);         // silu gate
    float h_new = h_candidate * gate;

    h_next[idx] = static_cast<T>(h_new);
    v[h_cand_idx] = static_cast<T>(h_raw);
    v[gate_idx] = static_cast<T>(g_raw);
}

template<typename T>
__global__
void ElmanSwishBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,
    const T* __restrict__ dh_recurrent,
    const T* __restrict__ v,
    T* __restrict__ d_raw
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    const float h_cand_raw = static_cast<float>(v[h_cand_idx]);
    const float gate_raw = static_cast<float>(v[gate_idx]);

    const float sig_h = device_sigmoid(h_cand_raw);
    const float sig_g = device_sigmoid(gate_raw);
    const float h_candidate = device_silu(h_cand_raw);
    const float gate = device_silu(gate_raw);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    const float d_h_candidate = dh * gate;
    const float d_gate = dh * h_candidate;

    const float d_gate_raw = d_gate * d_silu(gate_raw, sig_g);
    const float d_h_cand_raw = d_h_candidate * d_silu(h_cand_raw, sig_h);

    d_raw[h_cand_idx] = static_cast<T>(d_h_cand_raw);
    d_raw[gate_idx] = static_cast<T>(d_gate_raw);
}

// ============================================================================
// ElmanGelu: tanh + gelu gate
// ============================================================================

template<typename T>
__global__
void ElmanGeluPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,
    const T* __restrict__ Wx,
    const T* __restrict__ b,
    T* __restrict__ h_next,
    T* __restrict__ v
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    float h_raw = static_cast<float>(Rh[h_cand_idx]) +
                  static_cast<float>(Wx[h_cand_idx]) +
                  static_cast<float>(b[d_idx]);
    float g_raw = static_cast<float>(Rh[gate_idx]) +
                  static_cast<float>(Wx[gate_idx]) +
                  static_cast<float>(b[D + d_idx]);

    float h_candidate = device_tanh(h_raw);
    float gate = device_gelu(g_raw);  // gelu gate
    float h_new = h_candidate * gate;

    h_next[idx] = static_cast<T>(h_new);
    v[h_cand_idx] = static_cast<T>(h_raw);
    v[gate_idx] = static_cast<T>(g_raw);
}

template<typename T>
__global__
void ElmanGeluBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,
    const T* __restrict__ dh_recurrent,
    const T* __restrict__ v,
    T* __restrict__ d_raw
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    const float h_cand_raw = static_cast<float>(v[h_cand_idx]);
    const float gate_raw = static_cast<float>(v[gate_idx]);

    const float h_candidate = device_tanh(h_cand_raw);
    const float gate = device_gelu(gate_raw);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    const float d_h_candidate = dh * gate;
    const float d_gate = dh * h_candidate;

    const float d_gate_raw = d_gate * d_gelu(gate_raw);
    const float d_h_cand_raw = d_h_candidate * d_tanh(h_candidate);

    d_raw[h_cand_idx] = static_cast<T>(d_h_cand_raw);
    d_raw[gate_idx] = static_cast<T>(d_gate_raw);
}

// ============================================================================
// ElmanNoGate: tanh only, no gating (ablation baseline)
// ============================================================================

template<typename T>
__global__
void ElmanNoGatePointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,
    const T* __restrict__ Wx,
    const T* __restrict__ b,
    T* __restrict__ h_next,
    T* __restrict__ v
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(b[idx % D]);

    float h_new = device_tanh(raw);

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);  // Save pre-activation for backward
}

template<typename T>
__global__
void ElmanNoGateBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,
    const T* __restrict__ dh_recurrent,
    const T* __restrict__ v,
    T* __restrict__ d_raw
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const float raw = static_cast<float>(v[idx]);
    const float h_new = device_tanh(raw);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_raw_val = dh * d_tanh(h_new);

    d_raw[idx] = static_cast<T>(d_raw_val);
}

// ============================================================================
// Shared bias accumulation kernel
// ============================================================================

template<typename T>
__global__
void AccumulateBiasGradientKernel(
    const int batch_size,
    const int gate_dim,
    const T* __restrict__ d_raw,
    T* __restrict__ db
) {
    const int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (g_idx >= gate_dim) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        sum += static_cast<float>(d_raw[b * gate_dim + g_idx]);
    }
    db[g_idx] = static_cast<T>(static_cast<float>(db[g_idx]) + sum);
}

}  // anonymous namespace

// ============================================================================
// Forward/Backward Pass implementations for each variant
// ============================================================================

namespace haste {
namespace v0 {

// ============================================================================
// ElmanTanh Implementation
// ============================================================================
namespace elman_tanh {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,
    const T* b,
    const T* Wx,
    T* h,
    T* v,
    T* tmp_Rh
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * batch_size * gate_dim;
        const T* Wx_t = Wx + t * batch_size * gate_dim;

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            gate_dim, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, gate_dim);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanTanhPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,
    const T* h,
    const T* v,
    const T* dh_new,
    T* dWx,
    T* dR,
    T* db,
    T* dh,
    T* tmp_dRh
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * batch_size * gate_dim;
        T* dWx_t = dWx + t * batch_size * gate_dim;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanTanhBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, tmp_dRh);

        cudaMemcpyAsync(dWx_t, tmp_dRh, batch_size * gate_dim * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        const int bias_threads = 256;
        const int bias_blocks = (gate_dim + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, gate_dim, tmp_dRh, db);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, gate_dim, batch_size,
            &alpha,
            h_t, D,
            tmp_dRh, gate_dim,
            &beta_one,
            dR, D);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, gate_dim,
            &alpha,
            R, D,
            tmp_dRh, gate_dim,
            &beta,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_tanh

// ============================================================================
// ElmanSigmoid Implementation
// ============================================================================
namespace elman_sigmoid {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,
    const T* b,
    const T* Wx,
    T* h,
    T* v,
    T* tmp_Rh
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * batch_size * gate_dim;
        const T* Wx_t = Wx + t * batch_size * gate_dim;

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            gate_dim, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, gate_dim);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanSigmoidPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,
    const T* h,
    const T* v,
    const T* dh_new,
    T* dWx,
    T* dR,
    T* db,
    T* dh,
    T* tmp_dRh
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * batch_size * gate_dim;
        T* dWx_t = dWx + t * batch_size * gate_dim;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanSigmoidBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, tmp_dRh);

        cudaMemcpyAsync(dWx_t, tmp_dRh, batch_size * gate_dim * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        const int bias_threads = 256;
        const int bias_blocks = (gate_dim + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, gate_dim, tmp_dRh, db);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, gate_dim, batch_size,
            &alpha,
            h_t, D,
            tmp_dRh, gate_dim,
            &beta_one,
            dR, D);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, gate_dim,
            &alpha,
            R, D,
            tmp_dRh, gate_dim,
            &beta,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_sigmoid

// ============================================================================
// ElmanSwish Implementation
// ============================================================================
namespace elman_swish {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,
    const T* b,
    const T* Wx,
    T* h,
    T* v,
    T* tmp_Rh
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * batch_size * gate_dim;
        const T* Wx_t = Wx + t * batch_size * gate_dim;

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            gate_dim, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, gate_dim);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanSwishPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,
    const T* h,
    const T* v,
    const T* dh_new,
    T* dWx,
    T* dR,
    T* db,
    T* dh,
    T* tmp_dRh
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * batch_size * gate_dim;
        T* dWx_t = dWx + t * batch_size * gate_dim;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanSwishBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, tmp_dRh);

        cudaMemcpyAsync(dWx_t, tmp_dRh, batch_size * gate_dim * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        const int bias_threads = 256;
        const int bias_blocks = (gate_dim + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, gate_dim, tmp_dRh, db);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, gate_dim, batch_size,
            &alpha,
            h_t, D,
            tmp_dRh, gate_dim,
            &beta_one,
            dR, D);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, gate_dim,
            &alpha,
            R, D,
            tmp_dRh, gate_dim,
            &beta,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_swish

// ============================================================================
// ElmanGelu Implementation
// ============================================================================
namespace elman_gelu {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,
    const T* b,
    const T* Wx,
    T* h,
    T* v,
    T* tmp_Rh
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * batch_size * gate_dim;
        const T* Wx_t = Wx + t * batch_size * gate_dim;

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            gate_dim, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, gate_dim);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanGeluPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,
    const T* h,
    const T* v,
    const T* dh_new,
    T* dWx,
    T* dR,
    T* db,
    T* dh,
    T* tmp_dRh
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * batch_size * gate_dim;
        T* dWx_t = dWx + t * batch_size * gate_dim;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanGeluBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, tmp_dRh);

        cudaMemcpyAsync(dWx_t, tmp_dRh, batch_size * gate_dim * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        const int bias_threads = 256;
        const int bias_blocks = (gate_dim + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, gate_dim, tmp_dRh, db);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, gate_dim, batch_size,
            &alpha,
            h_t, D,
            tmp_dRh, gate_dim,
            &beta_one,
            dR, D);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, gate_dim,
            &alpha,
            R, D,
            tmp_dRh, gate_dim,
            &beta,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_gelu

// ============================================================================
// ElmanNoGate Implementation (ablation baseline)
// ============================================================================
namespace elman_nogate {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,       // [D, D] - no gate, so half the size
    const T* b,       // [D]
    const T* Wx,      // [T*B, D]
    T* h,             // [(T+1)*B, D]
    T* v,             // [T*B, D] - saves pre-activation
    T* tmp_Rh         // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD;
        const T* Wx_t = Wx + t * BD;

        // GEMM: tmp_Rh = R @ h_t
        // R is [D, D] (not 2D), so different dimensions
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, D);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanNoGatePointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,        // [D, D]
    const T* h,        // [(T+1)*B, D]
    const T* v,        // [T*B, D]
    const T* dh_new,   // [(T+1)*B, D]
    T* dWx,            // [T*B, D]
    T* dR,             // [D, D]
    T* db,             // [D]
    T* dh,             // [B, D]
    T* tmp_dRh         // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * BD;
        T* dWx_t = dWx + t * BD;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanNoGateBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, tmp_dRh);

        cudaMemcpyAsync(dWx_t, tmp_dRh, BD * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_dRh, db);

        // dR += tmp_dRh^T @ h_t = [D, B] @ [B, D] = [D, D]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha,
            h_t, D,
            tmp_dRh, D,
            &beta_one,
            dR, D);

        // dh = R @ tmp_dRh = [D, D] @ [D, B] = [D, B]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            tmp_dRh, D,
            &beta,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_nogate

// ============================================================================
// ElmanLeaky Implementation (input-dependent leaky integration)
// ============================================================================
namespace elman_leaky {

namespace {

template<typename T>
__global__
void ElmanLeakyPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,      // [B, D] - R @ h_prev
    const T* __restrict__ Wx,      // [B, D]
    const T* __restrict__ b,       // [D]
    const T* __restrict__ delta,   // [B, D] - precomputed delta
    const T* __restrict__ h_prev,  // [B, D] - previous hidden state
    T* __restrict__ h_next,        // [B, D] - output
    T* __restrict__ v              // [B, D] - save pre-activation
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(b[idx % D]);

    float candidate = tanhf(raw);
    float d = static_cast<float>(delta[idx]);
    float h_p = static_cast<float>(h_prev[idx]);

    // Leaky integration: h_new = (1 - delta) * h_prev + delta * candidate
    float h_new = (1.0f - d) * h_p + d * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);  // Save pre-activation for backward
}

template<typename T>
__global__
void ElmanLeakyBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,       // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent, // [B, D] - gradient from next timestep (or nullptr)
    const T* __restrict__ v,            // [B, D] - pre-activation (raw)
    const T* __restrict__ delta,        // [B, D] - delta used in forward
    const T* __restrict__ h_prev,       // [B, D] - previous hidden state
    T* __restrict__ d_raw,              // [B, D] - gradient w.r.t. pre-activation
    T* __restrict__ d_delta,            // [B, D] - gradient w.r.t. delta
    T* __restrict__ dh_prev_out         // [B, D] - gradient w.r.t. h_prev (from leaky)
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const float raw = static_cast<float>(v[idx]);
    const float candidate = tanhf(raw);
    const float d = static_cast<float>(delta[idx]);
    const float h_p = static_cast<float>(h_prev[idx]);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // Backward through: h_new = (1 - d) * h_prev + d * candidate
    // d_candidate = dh * d
    // d_h_prev = dh * (1 - d)
    // d_delta = dh * (candidate - h_prev)

    const float d_candidate = dh * d;
    const float d_h_prev = dh * (1.0f - d);
    const float d_d = dh * (candidate - h_p);

    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_raw_val = d_candidate * (1.0f - candidate * candidate);

    d_raw[idx] = static_cast<T>(d_raw_val);
    d_delta[idx] = static_cast<T>(d_d);
    dh_prev_out[idx] = static_cast<T>(d_h_prev);
}

}  // anonymous namespace

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,        // [D, D]
    const T* b,        // [D]
    const T* Wx,       // [T*B, D]
    const T* delta,    // [T*B, D]
    T* h,              // [(T+1)*B, D]
    T* v,              // [T*B, D]
    T* tmp_Rh          // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD;
        const T* Wx_t = Wx + t * BD;
        const T* delta_t = delta + t * BD;

        // GEMM: tmp_Rh = R @ h_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, D);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanLeakyPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, delta_t, h_t, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,        // [D, D]
    const T* h,        // [(T+1)*B, D]
    const T* v,        // [T*B, D]
    const T* delta,    // [T*B, D]
    const T* dh_new,   // [(T+1)*B, D]
    T* dWx,            // [T*B, D]
    T* dR,             // [D, D]
    T* db,             // [D]
    T* d_delta,        // [T*B, D]
    T* dh,             // [B, D] - recurrent gradient
    T* tmp_dRh         // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    // Temporary buffer for dh_prev from leaky integration
    // We'll reuse tmp_dRh for the d_raw, need separate space for dh_prev
    // Actually, we need to be careful here. Let me use dh for both purposes.

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta + t * BD;
        T* dWx_t = dWx + t * BD;
        T* d_delta_t = d_delta + t * BD;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        // This kernel computes:
        // - d_raw (gradient w.r.t. pre-activation) -> tmp_dRh
        // - d_delta (gradient w.r.t. delta) -> d_delta_t
        // - dh_prev (gradient from leaky blending) -> dh (will be combined with matmul gradient)
        ElmanLeakyBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, delta_t, h_t,
            tmp_dRh, d_delta_t, dh);

        // Copy d_raw to dWx_t
        cudaMemcpyAsync(dWx_t, tmp_dRh, BD * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // Accumulate bias gradient
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_dRh, db);

        // dR += h_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha,
            h_t, D,
            tmp_dRh, D,
            &beta_one,
            dR, D);

        // dh (from matmul) += R @ d_raw
        // But we already have dh from the leaky backward, so we need to add to it
        // Use a temporary or do in-place addition
        // Actually, the leaky backward already wrote to dh, so we add the matmul part
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            tmp_dRh, D,
            &beta_one,  // Add to existing dh from leaky backward
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_leaky

// ============================================================================
// ElmanLeakyMamba2Delta Implementation (Mamba2-style softplus/exp delta)
// ============================================================================
// Like ElmanLeaky but with Mamba2's log-space delta parameterization:
// - delta_raw = W_delta @ x + b_delta        -- raw (can be any real)
// - delta = softplus(delta_raw)              -- always positive
// - decay = exp(-delta)                      -- between 0 and 1
// - candidate = tanh(R @ h + Wx @ x + b)     -- same as ElmanLeaky
// - h_new = decay * h + (1 - decay) * candidate
//
// Advantages over sigmoid:
// - Wider dynamic range (delta can be 0.001 to 100+)
// - Better gradients (softplus linear for large values)
// - More numerically stable (log-space parameterization)
namespace elman_leaky_mamba2_delta {

namespace {

__device__ __forceinline__ float device_softplus(float x) {
    // Numerically stable softplus: log(1 + exp(x))
    if (x > 20.0f) return x;  // For large x, softplus(x) ≈ x
    return logf(1.0f + expf(x));
}

__device__ __forceinline__ float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template<typename T>
__global__
void ElmanLeakyMamba2DeltaPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,           // [B, D] - R @ h_prev
    const T* __restrict__ Wx,           // [B, D]
    const T* __restrict__ b,            // [D]
    const T* __restrict__ delta_raw,    // [B, D] - RAW delta (before softplus!)
    const T* __restrict__ h_prev,       // [B, D] - previous hidden state
    T* __restrict__ h_next,             // [B, D] - output
    T* __restrict__ v,                  // [B, D] - save pre-activation
    T* __restrict__ decay_cache         // [B, D] - save decay for backward
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(b[idx % D]);

    float candidate = tanhf(raw);
    float d_raw = static_cast<float>(delta_raw[idx]);
    float h_p = static_cast<float>(h_prev[idx]);

    // Mamba2-style: softplus then exp for decay
    float delta = device_softplus(d_raw);
    float decay = expf(-delta);

    // h_new = decay * h + (1 - decay) * candidate
    float h_new = decay * h_p + (1.0f - decay) * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);
    decay_cache[idx] = static_cast<T>(decay);  // Save for backward
}

template<typename T>
__global__
void ElmanLeakyMamba2DeltaBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,         // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent,   // [B, D] - gradient from next timestep
    const T* __restrict__ v,              // [B, D] - pre-activation (raw)
    const T* __restrict__ delta_raw,      // [B, D] - raw delta
    const T* __restrict__ decay_cache,    // [B, D] - cached decay
    const T* __restrict__ h_prev,         // [B, D] - previous hidden state
    T* __restrict__ d_raw,                // [B, D] - gradient w.r.t. pre-activation
    T* __restrict__ d_delta_raw,          // [B, D] - gradient w.r.t. raw delta
    T* __restrict__ dh_prev_out           // [B, D] - gradient w.r.t. h_prev
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const float raw = static_cast<float>(v[idx]);
    const float candidate = tanhf(raw);
    const float d_raw_val = static_cast<float>(delta_raw[idx]);
    const float decay = static_cast<float>(decay_cache[idx]);
    const float h_p = static_cast<float>(h_prev[idx]);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // h_new = decay * h_prev + (1 - decay) * candidate
    // d_candidate = dh * (1 - decay)
    // d_h_prev = dh * decay
    // d_decay = dh * (h_prev - candidate)

    const float d_candidate = dh * (1.0f - decay);
    const float d_h_prev = dh * decay;
    const float d_decay = dh * (h_p - candidate);

    // decay = exp(-softplus(delta_raw))
    // d_decay/d_delta_raw = -decay * sigmoid(delta_raw)
    const float sigmoid_d_raw = device_sigmoid(d_raw_val);
    const float d_delta_raw_val = d_decay * (-decay) * sigmoid_d_raw;

    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_raw_out = d_candidate * (1.0f - candidate * candidate);

    d_raw[idx] = static_cast<T>(d_raw_out);
    d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);
    dh_prev_out[idx] = static_cast<T>(d_h_prev);
}

}  // anonymous namespace

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,            // [D, D]
    const T* b,            // [D]
    const T* Wx,           // [T*B, D]
    const T* delta_raw,    // [T*B, D] - RAW delta (before softplus!)
    T* h,                  // [(T+1)*B, D]
    T* v,                  // [T*B, D]
    T* decay_cache,        // [T*B, D] - cache decay for backward
    T* tmp_Rh              // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD;
        T* decay_t = decay_cache + t * BD;
        const T* Wx_t = Wx + t * BD;
        const T* delta_raw_t = delta_raw + t * BD;

        // Rh = R @ h_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, D);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanLeakyMamba2DeltaPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, delta_raw_t, h_t, h_next, v_t, decay_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,              // [D, D]
    const T* h,              // [(T+1)*B, D]
    const T* v,              // [T*B, D]
    const T* delta_raw,      // [T*B, D]
    const T* decay_cache,    // [T*B, D]
    const T* dh_new,         // [(T+1)*B, D]
    T* dx,                   // [T*B, D]
    T* dh,                   // [B, D] - gradient to pass back
    T* dR,                   // [D, D]
    T* db,                   // [D]
    T* d_delta_raw,          // [T*B, D]
    T* tmp_dRh               // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    // Initialize dh to zero
    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    const int threads = 256;
    const int blocks = (BD + threads - 1) / threads;

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * BD;
        const T* delta_raw_t = delta_raw + t * BD;
        const T* decay_t = decay_cache + t * BD;
        const T* dh_out_t = dh_new + (t + 1) * BD;
        T* dx_t = dx + t * BD;
        T* d_delta_raw_t = d_delta_raw + t * BD;

        // Pointwise backward
        ElmanLeakyMamba2DeltaBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out_t, dh, v_t, delta_raw_t, decay_t, h_t,
            tmp_dRh, d_delta_raw_t, dh);

        // dx = Wx^T @ d_raw (but Wx is identity in our case, so dx = d_raw)
        // Actually for this kernel, we need to accumulate into dx
        cudaMemcpyAsync(dx_t, tmp_dRh, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);

        // db += sum(d_raw, dim=0)
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_dRh, db);

        // dR += h_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha,
            tmp_dRh, D,
            h_t, D,
            &beta_one,
            dR, D);

        // dh += R^T @ d_raw (from matmul R @ h)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            tmp_dRh, D,
            &beta_one,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_leaky_mamba2_delta

// ============================================================================
// ElmanNoDelta Implementation (FIXED decay, no input-dependent delta)
// ============================================================================
// Simplified version of ElmanLeaky with constant alpha:
// - candidate = tanh(R @ h + Wx @ x + b)
// - h_new = alpha * h + (1 - alpha) * candidate  -- FIXED alpha!
// This tests whether input-dependent delta is actually needed.
namespace elman_no_delta {

namespace {

template<typename T>
__global__
void ElmanNoDeltaPointwiseKernel(
    const int batch_size,
    const int D,
    const float alpha,              // SCALAR fixed decay factor
    const T* __restrict__ Rh,       // [B, D] - R @ h_prev
    const T* __restrict__ Wx,       // [B, D]
    const T* __restrict__ b,        // [D]
    const T* __restrict__ h_prev,   // [B, D] - previous hidden state
    T* __restrict__ h_next,         // [B, D] - output
    T* __restrict__ v               // [B, D] - save pre-activation
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(b[idx % D]);

    float candidate = tanhf(raw);
    float h_p = static_cast<float>(h_prev[idx]);

    // Fixed alpha leaky integration: h_new = alpha * h_prev + (1 - alpha) * candidate
    float h_new = alpha * h_p + (1.0f - alpha) * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);  // Save pre-activation for backward
}

template<typename T>
__global__
void ElmanNoDeltaBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const float alpha,                    // SCALAR fixed decay factor
    const T* __restrict__ dh_out,         // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent,   // [B, D] - gradient from next timestep (or nullptr)
    const T* __restrict__ v,              // [B, D] - pre-activation (raw)
    T* __restrict__ d_raw,                // [B, D] - gradient w.r.t. pre-activation
    T* __restrict__ dh_prev_out           // [B, D] - gradient w.r.t. h_prev (from leaky)
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const float raw = static_cast<float>(v[idx]);
    const float candidate = tanhf(raw);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // Backward through: h_new = alpha * h_prev + (1 - alpha) * candidate
    // d_candidate = dh * (1 - alpha)
    // d_h_prev = dh * alpha
    // NO d_delta since alpha is fixed!

    const float d_candidate = dh * (1.0f - alpha);
    const float d_h_prev = dh * alpha;

    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_raw_val = d_candidate * (1.0f - candidate * candidate);

    d_raw[idx] = static_cast<T>(d_raw_val);
    dh_prev_out[idx] = static_cast<T>(d_h_prev);
}

}  // anonymous namespace

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    float alpha;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float alpha,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->alpha = alpha;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,        // [D, D]
    const T* b,        // [D]
    const T* Wx,       // [T*B, D]
    T* h,              // [(T+1)*B, D]
    T* v,              // [T*B, D]
    T* tmp_Rh          // [B, D]
) {
    static const T cublas_alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;
    const float alpha = data_->alpha;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD;
        const T* Wx_t = Wx + t * BD;

        // GEMM: tmp_Rh = R @ h_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &cublas_alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, D);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanNoDeltaPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, alpha, tmp_Rh, Wx_t, b, h_t, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    float alpha;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const float alpha,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->alpha = alpha;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,        // [D, D]
    const T* h,        // [(T+1)*B, D]
    const T* v,        // [T*B, D]
    const T* dh_new,   // [(T+1)*B, D]
    T* dWx,            // [T*B, D]
    T* dR,             // [D, D]
    T* db,             // [D]
    T* dh,             // [B, D] - recurrent gradient
    T* tmp_dRh         // [B, D]
) {
    static const T cublas_alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;
    const float alpha = data_->alpha;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    const int threads = 256;
    const int blocks = (BD + threads - 1) / threads;

    // Backward through time
    for (int t = steps - 1; t >= 0; --t) {
        const T* dh_out = dh_new + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* h_t = h + t * BD;
        T* dWx_t = dWx + t * BD;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanNoDeltaBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, alpha, dh_out, dh_recurrent, v_t,
            tmp_dRh, dh);

        // dWx_t = d_raw (gradient w.r.t. Wx @ x contribution)
        cudaMemcpyAsync(dWx_t, tmp_dRh, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);

        // Accumulate db: db += sum(d_raw, dim=0) using kernel
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_dRh, db);

        // Accumulate dR: dR += h_t^T @ d_raw = [D, B] @ [B, D] = [D, D]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &cublas_alpha,
            h_t, D,
            tmp_dRh, D,
            &beta_one,
            dR, D);

        // dh = R @ d_raw (gradient from R @ h term)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &cublas_alpha,
            R, D,
            tmp_dRh, D,
            &beta_zero,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_no_delta

// ============================================================================
// ElmanMamba2 Implementation (Mamba2-style linear recurrence)
// ============================================================================
// Mamba2-style discretization with NO R matrix and NO tanh:
// - delta = softplus(W_delta @ x + b_delta)      -- always positive
// - decay = exp(-delta)                          -- between 0 and 1
// - Bx = B @ x + b_B                            -- input projection
// - h_new = decay * h + (1 - decay) * Bx        -- linear combination
// Key: No R matrix means no recurrent GEMM! Much faster than standard Elman.
namespace elman_mamba2 {

namespace {

__device__ __forceinline__
float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__
float device_softplus(float x) {
    // softplus(x) = log(1 + exp(x))
    // Numerically stable version
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

// Broadcast bias [D] to [B, D]
template<typename T>
__global__
void BroadcastBiasKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ bias,  // [D]
    T* __restrict__ out          // [B, D]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;
    if (idx >= total) return;
    out[idx] = bias[idx % D];
}

template<typename T>
__global__
void ElmanMamba2PointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Wx_delta,  // [B, D] - W_delta @ x + b_delta
    const T* __restrict__ Bx,         // [B, D] - B @ x + b_B
    const T* __restrict__ h_prev,     // [B, D] - previous hidden state
    T* __restrict__ h_next,           // [B, D] - output
    T* __restrict__ v                 // [B, D] - save Wx_delta for backward
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    float wx_d = static_cast<float>(Wx_delta[idx]);
    float bx = static_cast<float>(Bx[idx]);
    float h_p = static_cast<float>(h_prev[idx]);

    // Mamba2 formula: delta = softplus(Wx_delta), decay = exp(-delta)
    float delta = device_softplus(wx_d);
    float decay = expf(-delta);

    // h_new = decay * h + (1 - decay) * Bx
    float h_new = decay * h_p + (1.0f - decay) * bx;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(wx_d);  // Save for backward
}

template<typename T>
__global__
void ElmanMamba2BackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,       // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent, // [B, D] - gradient from next timestep (or nullptr)
    const T* __restrict__ v,            // [B, D] - saved Wx_delta
    const T* __restrict__ Bx,           // [B, D] - saved Bx
    const T* __restrict__ h_prev,       // [B, D] - previous hidden state
    T* __restrict__ d_Wx_delta,         // [B, D] - gradient w.r.t. W_delta projection
    T* __restrict__ d_Bx,               // [B, D] - gradient w.r.t. B projection
    T* __restrict__ dh_prev_out         // [B, D] - gradient w.r.t. h_prev
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const float wx_d = static_cast<float>(v[idx]);
    const float bx = static_cast<float>(Bx[idx]);
    const float h_p = static_cast<float>(h_prev[idx]);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // Recompute forward values
    float delta = device_softplus(wx_d);
    float decay = expf(-delta);

    // Backward through: h_new = decay * h_prev + (1 - decay) * Bx
    float d_decay = dh * (h_p - bx);       // d/d(decay)
    float d_bx = dh * (1.0f - decay);      // d/d(Bx)
    float d_h_prev = dh * decay;           // d/d(h_prev)

    // d/dx exp(-softplus(x)) = exp(-softplus(x)) * (-sigmoid(x))
    //                        = decay * (-sigmoid(x))
    float d_delta = d_decay * (-decay);
    float d_wx_delta = d_delta * device_sigmoid(wx_d);  // d/dx softplus(x) = sigmoid(x)

    d_Wx_delta[idx] = static_cast<T>(d_wx_delta);
    d_Bx[idx] = static_cast<T>(d_bx);
    dh_prev_out[idx] = static_cast<T>(d_h_prev);
}

}  // anonymous namespace

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* W_delta,    // [D, D] - delta projection weights
    const T* b_delta,    // [D] - delta projection bias
    const T* W_B,        // [D, D] - B projection weights
    const T* b_B,        // [D] - B projection bias
    const T* x,          // [T, B, D] - input sequence
    T* h,                // [T+1, B, D] - hidden states (h[0] = h0)
    T* Wx_delta_cache,   // [T, B, D] - cached Wx_delta for backward
    T* Bx_cache,         // [T, B, D] - cached Bx for backward
    T* tmp_Wx_delta,     // [B, D] - workspace
    T* tmp_Bx) {         // [B, D] - workspace

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    const int threads = 256;
    const int blocks = (BD + threads - 1) / threads;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* Wx_delta_t = Wx_delta_cache + t * BD;
        T* Bx_t = Bx_cache + t * BD;

        // Compute Wx_delta = W_delta @ x + b_delta
        // First broadcast bias to tmp_Wx_delta
        BroadcastBiasKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, b_delta, tmp_Wx_delta);

        // Then GEMM: tmp_Wx_delta += W_delta @ x
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            W_delta, D,
            x_t, D,
            &beta_one,
            tmp_Wx_delta, D);

        // Compute Bx = W_B @ x + b_B
        BroadcastBiasKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, b_B, tmp_Bx);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            W_B, D,
            x_t, D,
            &beta_one,
            tmp_Bx, D);

        // Cache for backward
        cudaMemcpyAsync(Wx_delta_t, tmp_Wx_delta, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(Bx_t, tmp_Bx, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);

        // Pointwise kernel: h_next = decay * h_t + (1 - decay) * Bx
        ElmanMamba2PointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Wx_delta, tmp_Bx, h_t,
            h_next, Wx_delta_t);  // v = Wx_delta for backward (overwrites cache with same value)
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* W_delta,      // [D, D]
    const T* W_B,          // [D, D]
    const T* x,            // [T, B, D]
    const T* h,            // [T+1, B, D]
    const T* Wx_delta_cache, // [T, B, D] - saved Wx_delta
    const T* Bx_cache,     // [T, B, D] - saved Bx
    const T* dh_new,       // [T+1, B, D] - gradient w.r.t. output
    T* dx,                 // [T, B, D] - gradient w.r.t. input
    T* dW_delta,           // [D, D] - gradient w.r.t. W_delta
    T* db_delta,           // [D] - gradient w.r.t. b_delta
    T* dW_B,               // [D, D] - gradient w.r.t. W_B
    T* db_B,               // [D] - gradient w.r.t. b_B
    T* dh,                 // [B, D] - gradient w.r.t. h (carried through time)
    T* tmp_d_Wx_delta,     // [B, D] - workspace
    T* tmp_d_Bx) {         // [B, D] - workspace

    static const T cublas_alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    const int threads = 256;
    const int blocks = (BD + threads - 1) / threads;

    // Backward through time
    for (int t = steps - 1; t >= 0; --t) {
        const T* dh_out = dh_new + (t + 1) * BD;
        const T* Wx_delta_t = Wx_delta_cache + t * BD;
        const T* Bx_t = Bx_cache + t * BD;
        const T* h_t = h + t * BD;
        const T* x_t = x + t * BD;
        T* dx_t = dx + t * BD;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        // Pointwise backward
        ElmanMamba2BackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, Wx_delta_t, Bx_t, h_t,
            tmp_d_Wx_delta, tmp_d_Bx, dh);

        // Accumulate db_delta += sum(d_Wx_delta, dim=0)
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_d_Wx_delta, db_delta);

        // Accumulate db_B += sum(d_Bx, dim=0)
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_d_Bx, db_B);

        // Accumulate dW_delta += x_t^T @ d_Wx_delta = [D, B] @ [B, D] = [D, D]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &cublas_alpha,
            x_t, D,
            tmp_d_Wx_delta, D,
            &beta_one,
            dW_delta, D);

        // Accumulate dW_B += x_t^T @ d_Bx
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &cublas_alpha,
            x_t, D,
            tmp_d_Bx, D,
            &beta_one,
            dW_B, D);

        // dx = W_delta^T @ d_Wx_delta + W_B^T @ d_Bx
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &cublas_alpha,
            W_delta, D,
            tmp_d_Wx_delta, D,
            &beta_zero,
            dx_t, D);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &cublas_alpha,
            W_B, D,
            tmp_d_Bx, D,
            &beta_one,
            dx_t, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_mamba2

// ============================================================================
// ElmanMamba2Silu Implementation (Mamba2 structure + silu nonlinearity)
// ============================================================================
// Like elman_mamba2 but with silu(Bx) instead of just Bx:
// - delta = softplus(W_delta @ x + b_delta)  -- input-dependent discretization
// - decay = exp(-delta)                      -- between 0 and 1
// - candidate = silu(B @ x + b_B)            -- silu nonlinearity (unbounded for x>0)
// - h_new = decay * h + (1 - decay) * candidate
//
// Testing if silu (unbounded) works better than tanh (bounded) in Mamba2 structure.
namespace elman_mamba2_silu {

namespace {

__device__ __forceinline__
float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__
float device_softplus(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

__device__ __forceinline__
float device_silu(float x) {
    return x * device_sigmoid(x);
}

__device__ __forceinline__
float d_silu(float x, float sig) {
    // d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    return sig * (1.0f + x * (1.0f - sig));
}

template<typename T>
__global__
void BroadcastBiasKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ bias,
    T* __restrict__ out
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;
    if (idx >= total) return;
    out[idx] = bias[idx % D];
}

template<typename T>
__global__
void ElmanMamba2SiluPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Wx_delta,  // [B, D] - W_delta @ x + b_delta
    const T* __restrict__ Bx,         // [B, D] - B @ x + b_B
    const T* __restrict__ h_prev,     // [B, D] - previous hidden state
    T* __restrict__ h_next,           // [B, D] - output
    T* __restrict__ v                 // [B, D] - save Wx_delta for backward
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    float wx_d = static_cast<float>(Wx_delta[idx]);
    float bx = static_cast<float>(Bx[idx]);
    float h_p = static_cast<float>(h_prev[idx]);

    // Mamba2 formula with silu: delta = softplus(Wx_delta), decay = exp(-delta)
    float delta = device_softplus(wx_d);
    float decay = expf(-delta);

    // candidate = silu(Bx) instead of just Bx
    float candidate = device_silu(bx);

    // h_new = decay * h + (1 - decay) * candidate
    float h_new = decay * h_p + (1.0f - decay) * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(wx_d);  // Save for backward
}

template<typename T>
__global__
void ElmanMamba2SiluBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,       // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent, // [B, D] - gradient from next timestep (or nullptr)
    const T* __restrict__ v,            // [B, D] - saved Wx_delta
    const T* __restrict__ Bx,           // [B, D] - saved Bx
    const T* __restrict__ h_prev,       // [B, D] - previous hidden state
    T* __restrict__ d_Wx_delta,         // [B, D] - gradient w.r.t. W_delta projection
    T* __restrict__ d_Bx,               // [B, D] - gradient w.r.t. B projection
    T* __restrict__ dh_prev_out         // [B, D] - gradient w.r.t. h_prev
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    float wx_d = static_cast<float>(v[idx]);
    float bx = static_cast<float>(Bx[idx]);
    float h_p = static_cast<float>(h_prev[idx]);

    // Recompute forward values
    float delta = device_softplus(wx_d);
    float decay = expf(-delta);
    float sig_bx = device_sigmoid(bx);
    float candidate = bx * sig_bx;  // silu(bx)

    // Backward through: h_new = decay * h_p + (1 - decay) * candidate
    float d_decay = dh * (h_p - candidate);
    float d_candidate = dh * (1.0f - decay);
    float d_h_prev = dh * decay;

    // Backward through silu: d_bx = d_candidate * d_silu(bx)
    float d_bx = d_candidate * d_silu(bx, sig_bx);

    // Backward through decay = exp(-delta): d_delta = d_decay * (-decay)
    float d_delta = d_decay * (-decay);

    // Backward through delta = softplus(wx_d): d_wx_delta = d_delta * sigmoid(wx_d)
    float d_wx_delta = d_delta * device_sigmoid(wx_d);

    d_Wx_delta[idx] = static_cast<T>(d_wx_delta);
    d_Bx[idx] = static_cast<T>(d_bx);
    dh_prev_out[idx] = static_cast<T>(d_h_prev);
}

}  // anonymous namespace

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* W_delta,    // [D, D] - delta projection weights
    const T* b_delta,    // [D] - delta projection bias
    const T* W_B,        // [D, D] - B projection weights
    const T* b_B,        // [D] - B projection bias
    const T* x,          // [T, B, D] - input sequence
    T* h,                // [T+1, B, D] - hidden states (h[0] = h0)
    T* Wx_delta_cache,   // [T, B, D] - cached Wx_delta for backward
    T* Bx_cache,         // [T, B, D] - cached Bx for backward
    T* tmp_Wx_delta,     // [B, D] - workspace
    T* tmp_Bx) {         // [B, D] - workspace

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    const int threads = 256;
    const int blocks = (BD + threads - 1) / threads;

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* Wx_delta_t = Wx_delta_cache + t * BD;
        T* Bx_t = Bx_cache + t * BD;

        // Compute Wx_delta = W_delta @ x + b_delta
        BroadcastBiasKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, b_delta, tmp_Wx_delta);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            W_delta, D,
            x_t, D,
            &beta_one,
            tmp_Wx_delta, D);

        // Compute Bx = W_B @ x + b_B
        BroadcastBiasKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, b_B, tmp_Bx);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            W_B, D,
            x_t, D,
            &beta_one,
            tmp_Bx, D);

        // Cache for backward
        cudaMemcpyAsync(Wx_delta_t, tmp_Wx_delta, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(Bx_t, tmp_Bx, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream);

        // Pointwise kernel: h_next = decay * h_t + (1 - decay) * silu(Bx)
        ElmanMamba2SiluPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Wx_delta, tmp_Bx, h_t,
            h_next, Wx_delta_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* W_delta,      // [D, D]
    const T* W_B,          // [D, D]
    const T* x,            // [T, B, D]
    const T* h,            // [T+1, B, D]
    const T* Wx_delta_cache, // [T, B, D] - saved Wx_delta
    const T* Bx_cache,     // [T, B, D] - saved Bx
    const T* dh_new,       // [T+1, B, D] - gradient w.r.t. output
    T* dx,                 // [T, B, D] - gradient w.r.t. input
    T* dW_delta,           // [D, D] - gradient w.r.t. W_delta
    T* db_delta,           // [D] - gradient w.r.t. b_delta
    T* dW_B,               // [D, D] - gradient w.r.t. W_B
    T* db_B,               // [D] - gradient w.r.t. b_B
    T* dh,                 // [B, D] - gradient w.r.t. h (carried through time)
    T* tmp_d_Wx_delta,     // [B, D] - workspace
    T* tmp_d_Bx) {         // [B, D] - workspace

    static const T cublas_alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    const int threads = 256;
    const int blocks = (BD + threads - 1) / threads;

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* h_prev_t = h + t * BD;
        const T* v_t = Wx_delta_cache + t * BD;
        const T* Bx_t = Bx_cache + t * BD;
        const T* dh_t = dh_new + (t + 1) * BD;
        T* dx_t = dx + t * BD;

        // Pointwise backward
        ElmanMamba2SiluBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D,
            dh_t,
            (t == steps - 1) ? nullptr : dh,
            v_t,
            Bx_t,
            h_prev_t,
            tmp_d_Wx_delta,
            tmp_d_Bx,
            dh);

        // Accumulate gradients for W_delta: dW_delta += d_Wx_delta @ x^T
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &cublas_alpha,
            tmp_d_Wx_delta, D,
            x_t, D,
            &beta_one,
            dW_delta, D);

        // Accumulate db_delta += sum(d_Wx_delta, dim=0)
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_d_Wx_delta, db_delta);

        // Accumulate gradients for W_B: dW_B += d_Bx @ x^T
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &cublas_alpha,
            tmp_d_Bx, D,
            x_t, D,
            &beta_one,
            dW_B, D);

        // Accumulate db_B += sum(d_Bx, dim=0)
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_d_Bx, db_B);

        // dx = W_delta^T @ d_Wx_delta + W_B^T @ d_Bx
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &cublas_alpha,
            W_delta, D,
            tmp_d_Wx_delta, D,
            &beta_zero,
            dx_t, D);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &cublas_alpha,
            W_B, D,
            tmp_d_Bx, D,
            &beta_one,
            dx_t, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_mamba2_silu

// ============================================================================
// ElmanLeakySilu Implementation (silu activation + leaky integration)
// ============================================================================
// Same as ElmanLeaky but with silu instead of tanh:
// - candidate = silu(R @ h + Wx @ x + b)         -- silu often beats tanh
// - delta = sigmoid(W_delta @ x + b_delta)       -- input-dependent blend
// - h_new = (1 - delta) * h + delta * candidate  -- leaky integration
// - output = h_new                               -- NO output gate
namespace elman_leaky_silu {

namespace {

__device__ __forceinline__
float device_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__
float device_silu(float x) {
    return x * device_sigmoid(x);
}

template<typename T>
__global__
void ElmanLeakySiluPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,      // [B, D] - R @ h_prev
    const T* __restrict__ Wx,      // [B, D]
    const T* __restrict__ b,       // [D]
    const T* __restrict__ delta,   // [B, D] - precomputed delta (sigmoid output)
    const T* __restrict__ h_prev,  // [B, D] - previous hidden state
    T* __restrict__ h_next,        // [B, D] - output
    T* __restrict__ v              // [B, D] - save pre-activation
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(b[idx % D]);

    float candidate = device_silu(raw);  // silu instead of tanh!
    float d = static_cast<float>(delta[idx]);
    float h_p = static_cast<float>(h_prev[idx]);

    // Leaky integration: h_new = (1 - delta) * h_prev + delta * candidate
    float h_new = (1.0f - d) * h_p + d * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);  // Save pre-activation for backward
}

template<typename T>
__global__
void ElmanLeakySiluBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,       // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent, // [B, D] - gradient from next timestep (or nullptr)
    const T* __restrict__ v,            // [B, D] - pre-activation (raw)
    const T* __restrict__ delta,        // [B, D] - delta used in forward
    const T* __restrict__ h_prev,       // [B, D] - previous hidden state
    T* __restrict__ d_raw,              // [B, D] - gradient w.r.t. pre-activation
    T* __restrict__ d_delta,            // [B, D] - gradient w.r.t. delta
    T* __restrict__ dh_prev_out         // [B, D] - gradient w.r.t. h_prev (from leaky)
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const float raw = static_cast<float>(v[idx]);
    const float sig = device_sigmoid(raw);
    const float candidate = raw * sig;  // silu(raw) = raw * sigmoid(raw)
    const float d = static_cast<float>(delta[idx]);
    const float h_p = static_cast<float>(h_prev[idx]);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // Backward through: h_new = (1 - d) * h_prev + d * candidate
    const float d_candidate = dh * d;
    const float d_h_prev = dh * (1.0f - d);
    const float d_d = dh * (candidate - h_p);

    // d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    const float d_silu = sig * (1.0f + raw * (1.0f - sig));
    const float d_raw_val = d_candidate * d_silu;

    d_raw[idx] = static_cast<T>(d_raw_val);
    d_delta[idx] = static_cast<T>(d_d);
    dh_prev_out[idx] = static_cast<T>(d_h_prev);
}

}  // anonymous namespace

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,        // [D, D]
    const T* b,        // [D]
    const T* Wx,       // [T*B, D]
    const T* delta,    // [T*B, D]
    T* h,              // [(T+1)*B, D]
    T* v,              // [T*B, D]
    T* tmp_Rh          // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD;
        const T* Wx_t = Wx + t * BD;
        const T* delta_t = delta + t * BD;

        // GEMM: tmp_Rh = R @ h_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, D);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanLeakySiluPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, delta_t, h_t, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,        // [D, D]
    const T* h,        // [(T+1)*B, D]
    const T* v,        // [T*B, D]
    const T* delta,    // [T*B, D]
    const T* dh_new,   // [(T+1)*B, D]
    T* dWx,            // [T*B, D]
    T* dR,             // [D, D]
    T* db,             // [D]
    T* d_delta,        // [T*B, D]
    T* dh,             // [B, D] - recurrent gradient
    T* tmp_dRh         // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta + t * BD;
        T* dWx_t = dWx + t * BD;
        T* d_delta_t = d_delta + t * BD;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanLeakySiluBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, delta_t, h_t,
            tmp_dRh, d_delta_t, dh);

        // Copy d_raw to dWx_t
        cudaMemcpyAsync(dWx_t, tmp_dRh, BD * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // Accumulate bias gradient
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_dRh, db);

        // dR += h_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha,
            h_t, D,
            tmp_dRh, D,
            &beta_one,
            dR, D);

        // dh += R @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            tmp_dRh, D,
            &beta_one,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_leaky_silu

// ============================================================================
// ElmanLeakyDiag Implementation (diagonal R - like Mamba's diagonal A!)
// ============================================================================
// Same as ElmanLeaky but with DIAGONAL R instead of full matrix:
// - candidate = tanh(r ⊙ h + Wx @ x + b)           -- r is [D] not [D,D]!
// - delta = sigmoid(W_delta @ x + b_delta)         -- input-dependent blend
// - h_new = (1 - delta) * h + delta * candidate    -- leaky integration
//
// Benefits:
// - D params instead of D² for recurrence (massive reduction!)
// - No GEMM needed - pure elementwise ops (faster!)
// - Cross-channel mixing happens via W @ x (input projection)
// - Matches Mamba's diagonal A architecture
namespace elman_leaky_diag {

namespace {

template<typename T>
__global__
void ElmanLeakyDiagPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ r,       // [D] - diagonal recurrence weights
    const T* __restrict__ h_prev,  // [B, D] - previous hidden state
    const T* __restrict__ Wx,      // [B, D]
    const T* __restrict__ b,       // [D]
    const T* __restrict__ delta,   // [B, D] - precomputed delta
    T* __restrict__ h_next,        // [B, D] - output
    T* __restrict__ v              // [B, D] - save pre-activation
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int d = idx % D;  // which dimension

    // raw = r[d] * h[idx] + Wx[idx] + b[d]  -- diagonal R!
    float raw = static_cast<float>(r[d]) * static_cast<float>(h_prev[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(b[d]);

    float candidate = tanhf(raw);
    float d_val = static_cast<float>(delta[idx]);
    float h_p = static_cast<float>(h_prev[idx]);

    // Leaky integration: h_new = (1 - delta) * h_prev + delta * candidate
    float h_new = (1.0f - d_val) * h_p + d_val * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);  // Save pre-activation for backward
}

template<typename T>
__global__
void ElmanLeakyDiagBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ r,            // [D] - diagonal recurrence weights
    const T* __restrict__ dh_out,       // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent, // [B, D] - gradient from next timestep (or nullptr)
    const T* __restrict__ v,            // [B, D] - pre-activation (raw)
    const T* __restrict__ delta,        // [B, D] - delta used in forward
    const T* __restrict__ h_prev,       // [B, D] - previous hidden state
    T* __restrict__ d_raw,              // [B, D] - gradient w.r.t. pre-activation
    T* __restrict__ d_delta,            // [B, D] - gradient w.r.t. delta
    T* __restrict__ dh_prev_out         // [B, D] - gradient w.r.t. h_prev
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int d = idx % D;

    const float raw = static_cast<float>(v[idx]);
    const float candidate = tanhf(raw);
    const float d_val = static_cast<float>(delta[idx]);
    const float h_p = static_cast<float>(h_prev[idx]);
    const float r_d = static_cast<float>(r[d]);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // Backward through: h_new = (1 - d) * h_prev + d * candidate
    const float d_candidate = dh * d_val;
    const float d_h_prev_from_leaky = dh * (1.0f - d_val);
    const float d_d = dh * (candidate - h_p);

    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_raw_val = d_candidate * (1.0f - candidate * candidate);

    // Backward through: raw = r[d] * h_prev + Wx + b
    // dh_prev += r[d] * d_raw (contribution from matmul)
    const float d_h_prev_from_raw = r_d * d_raw_val;

    d_raw[idx] = static_cast<T>(d_raw_val);
    d_delta[idx] = static_cast<T>(d_d);
    dh_prev_out[idx] = static_cast<T>(d_h_prev_from_leaky + d_h_prev_from_raw);
}

// Reduction kernel for dr gradient
template<typename T>
__global__
void AccumulateDiagGradientKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ h,       // [B, D] - hidden state
    const T* __restrict__ d_raw,   // [B, D] - gradient w.r.t. pre-activation
    T* __restrict__ dr             // [D] - accumulated gradient for r
) {
    const int d = blockDim.x * blockIdx.x + threadIdx.x;
    if (d >= D) return;

    // dr[d] += sum over batch of h[b, d] * d_raw[b, d]
    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        const int idx = b * D + d;
        sum += static_cast<float>(h[idx]) * static_cast<float>(d_raw[idx]);
    }
    dr[d] = static_cast<T>(static_cast<float>(dr[d]) + sum);
}

// Bias gradient accumulation (reuse from elman_leaky)
template<typename T>
__global__
void AccumulateBiasGradientKernelDiag(
    const int batch_size,
    const int D,
    const T* __restrict__ d_raw,
    T* __restrict__ db
) {
    const int d = blockDim.x * blockIdx.x + threadIdx.x;
    if (d >= D) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        sum += static_cast<float>(d_raw[b * D + d]);
    }
    db[d] = static_cast<T>(static_cast<float>(db[d]) + sum);
}

}  // anonymous namespace

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,  // unused but kept for API compatibility
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* r,        // [D] - diagonal recurrence weights
    const T* b,        // [D]
    const T* Wx,       // [T*B, D]
    const T* delta,    // [T*B, D]
    T* h,              // [(T+1)*B, D]
    T* v               // [T*B, D]
    // Note: no tmp_Rh needed! Pure elementwise.
) {
    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;
    cudaStream_t stream = data_->stream;

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD;
        const T* Wx_t = Wx + t * BD;
        const T* delta_t = delta + t * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanLeakyDiagPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, r, h_t, Wx_t, b, delta_t, h_next, v_t);
    }
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,  // unused but kept for API compatibility
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* r,        // [D] - diagonal recurrence weights
    const T* h,        // [(T+1)*B, D]
    const T* v,        // [T*B, D]
    const T* delta,    // [T*B, D]
    const T* dh_new,   // [(T+1)*B, D]
    T* dWx,            // [T*B, D]
    T* dr,             // [D] - gradient for diagonal r
    T* db,             // [D]
    T* d_delta,        // [T*B, D]
    T* dh              // [B, D] - recurrent gradient
) {
    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;
    cudaStream_t stream = data_->stream;

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta + t * BD;
        T* dWx_t = dWx + t * BD;
        T* d_delta_t = d_delta + t * BD;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        // Compute d_raw, d_delta, and full dh_prev (including r*d_raw term)
        ElmanLeakyDiagBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, r, dh_out, dh_recurrent, v_t, delta_t, h_t,
            dWx_t, d_delta_t, dh);  // dWx_t doubles as d_raw storage

        // Accumulate bias gradient
        const int d_threads = 256;
        const int d_blocks = (D + d_threads - 1) / d_threads;
        AccumulateBiasGradientKernelDiag<T><<<d_blocks, d_threads, 0, stream>>>(
            batch_size, D, dWx_t, db);

        // Accumulate dr gradient: dr[d] += sum_b h[b,d] * d_raw[b,d]
        AccumulateDiagGradientKernel<T><<<d_blocks, d_threads, 0, stream>>>(
            batch_size, D, h_t, dWx_t, dr);
    }
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_leaky_diag

// ============================================================================
// ElmanLeakySelective Implementation (Mamba2-style discretization + nonlinearity)
// ============================================================================
// This is the key architecture for nonlinear RNNs:
// - candidate = tanh(R @ h + Wx @ x + b)       -- NONLINEAR (our innovation!)
// - delta_raw = W_delta @ x + b_delta          -- input-dependent (precomputed)
// - alpha = exp(-softplus(delta_raw) * exp(A)) -- Mamba2-style per-channel decay
// - h_new = alpha * h + (1 - alpha) * candidate
//
// Per-channel A gives each dimension its own decay rate, like Mamba2's diagonal A.
// The exp(-softplus(...) * exp(A)) parameterization ensures alpha ∈ (0, 1) always.
namespace elman_leaky_selective {

namespace {

__device__ __forceinline__
float device_softplus(float x) {
    // softplus(x) = log(1 + exp(x))
    // Numerically stable version
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

template<typename T>
__global__
void ElmanLeakySelectivePointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,        // [B, D] - R @ h_prev
    const T* __restrict__ Wx,        // [B, D]
    const T* __restrict__ b,         // [D]
    const T* __restrict__ delta_raw, // [B, D] - precomputed W_delta @ x + b_delta
    const T* __restrict__ A,         // [D] - per-channel decay (log-space, typically negative)
    const T* __restrict__ h_prev,    // [B, D] - previous hidden state
    T* __restrict__ h_next,          // [B, D] - output
    T* __restrict__ v                // [B, 2*D] - save [raw, delta_raw] for backward
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int d_idx = idx % D;
    const int batch_idx = idx / D;

    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(b[d_idx]);

    float candidate = tanhf(raw);
    float d_raw = static_cast<float>(delta_raw[idx]);
    float a_raw = static_cast<float>(A[d_idx]);  // A is raw learnable param
    float h_p = static_cast<float>(h_prev[idx]);

    // LOG-SPACE Mamba-style parameterization (STABLE!):
    // dt = softplus(delta_raw)      -- input-dependent timestep (always positive)
    // decay_rate = exp(-exp(A_log)) -- ALWAYS in (0, 1) no matter what A_log is!
    // alpha = exp(-dt * decay_rate) -- blend factor
    //
    // Why log-space? exp(-exp(x)) maps ANY x → (0, 1):
    //   A_log = -2 → decay_rate = exp(-0.14) = 0.87
    //   A_log = -1 → decay_rate = exp(-0.37) = 0.69
    //   A_log =  0 → decay_rate = exp(-1.00) = 0.37
    //   A_log =  1 → decay_rate = exp(-2.72) = 0.07
    //
    // This prevents alpha from going to extremes during training.
    // The double exponential constrains the learning dynamics naturally.

    float dt = device_softplus(d_raw);
    float decay_rate = expf(-expf(a_raw));  // Log-space: ALWAYS in (0, 1)

    // alpha = exp(-dt * decay_rate)
    // With dt=3, decay_rate=0.37: alpha = exp(-1.1) = 0.33 → 67% candidate
    float alpha = expf(-dt * decay_rate);

    // Small epsilon for numerical stability in backward
    alpha = fmaxf(alpha, 1e-6f);
    alpha = fminf(alpha, 1.0f - 1e-6f);  // Also clamp upper bound

    // Leaky integration with Mamba2-style blend
    float h_new = alpha * h_p + (1.0f - alpha) * candidate;

    h_next[idx] = static_cast<T>(h_new);

    // Save for backward: raw and delta_raw
    const int v_idx = batch_idx * 2 * D + d_idx;
    v[v_idx] = static_cast<T>(raw);
    v[v_idx + D] = static_cast<T>(d_raw);
}

template<typename T>
__global__
void ElmanLeakySelectiveBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,       // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent, // [B, D] - gradient from next timestep (or nullptr)
    const T* __restrict__ v,            // [B, 2*D] - [raw, delta_raw]
    const T* __restrict__ A,            // [D] - per-channel decay
    const T* __restrict__ h_prev,       // [B, D] - previous hidden state
    T* __restrict__ d_raw,              // [B, D] - gradient w.r.t. pre-activation
    T* __restrict__ d_delta_raw,        // [B, D] - gradient w.r.t. delta_raw (for W_delta, b_delta)
    T* __restrict__ dA,                 // [D] - gradient w.r.t. A (accumulated)
    T* __restrict__ dh_prev_out         // [B, D] - gradient w.r.t. h_prev
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int d_idx = idx % D;
    const int batch_idx = idx / D;
    const int v_idx = batch_idx * 2 * D + d_idx;

    const float raw = static_cast<float>(v[v_idx]);
    const float d_raw_val = static_cast<float>(v[v_idx + D]);
    const float a_raw = static_cast<float>(A[d_idx]);  // A is raw learnable param
    const float h_p = static_cast<float>(h_prev[idx]);

    const float candidate = tanhf(raw);
    const float dt = device_softplus(d_raw_val);
    const float decay_rate = expf(-expf(a_raw));  // Log-space: ALWAYS in (0, 1)

    // LOG-SPACE: alpha = exp(-dt * decay_rate)
    float alpha = expf(-dt * decay_rate);
    alpha = fmaxf(alpha, 1e-6f);
    alpha = fminf(alpha, 1.0f - 1e-6f);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // Backward through: h_new = alpha * h_prev + (1 - alpha) * candidate
    const float d_candidate = dh * (1.0f - alpha);
    const float d_h_prev = dh * alpha;
    const float d_alpha = dh * (h_p - candidate);

    // Backward through: alpha = exp(-dt * decay_rate)
    // d/d(decay_rate) alpha = -dt * alpha
    const float d_decay_rate = d_alpha * (-dt * alpha);
    // d/d(dt) alpha = -decay_rate * alpha
    const float d_dt = d_alpha * (-decay_rate * alpha);

    // Backward through: decay_rate = exp(-exp(A_log))
    // Let u = exp(A_log), decay_rate = exp(-u)
    // d(decay_rate)/d(A_log) = d(exp(-u))/du * du/d(A_log)
    //                        = -exp(-u) * exp(A_log)
    //                        = -decay_rate * exp(A_log)
    const float exp_a_raw = expf(a_raw);
    const float d_a_raw = d_decay_rate * (-decay_rate * exp_a_raw);

    // Backward through: dt = softplus(d_raw_val)
    // d/dx softplus(x) = sigmoid(x)
    const float sigmoid_d_raw = 1.0f / (1.0f + expf(-d_raw_val));
    const float d_delta_raw_val = d_dt * sigmoid_d_raw;

    // Backward through candidate = tanh(raw)
    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_raw_from_candidate = d_candidate * (1.0f - candidate * candidate);

    d_raw[idx] = static_cast<T>(d_raw_from_candidate);
    d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);
    dh_prev_out[idx] = static_cast<T>(d_h_prev);

    // Atomic add for dA since multiple batch elements contribute
    atomicAdd(&dA[d_idx], d_a_raw);
}

}  // anonymous namespace

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* R,          // [D, D]
    const T* b,          // [D]
    const T* Wx,         // [T*B, D]
    const T* delta_raw,  // [T*B, D] - precomputed W_delta @ x + b_delta
    const T* A,          // [D] - per-channel decay rates
    T* h,                // [(T+1)*B, D]
    T* v,                // [T*B, 2*D] - saves [raw, delta_raw]
    T* tmp_Rh            // [B, D]
) {
    static const T alpha_blas = static_cast<T>(1.0);
    static const T beta_blas = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * batch_size * 2 * D;
        const T* Wx_t = Wx + t * BD;
        const T* delta_raw_t = delta_raw + t * BD;

        // GEMM: tmp_Rh = R @ h_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha_blas,
            R, D,
            h_t, D,
            &beta_blas,
            tmp_Rh, D);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        ElmanLeakySelectivePointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, Wx_t, b, delta_raw_t, A, h_t, h_next, v_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    data_->stream = stream;
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* R,          // [D, D]
    const T* h,          // [(T+1)*B, D]
    const T* v,          // [T*B, 2*D]
    const T* A,          // [D]
    const T* dh_new,     // [(T+1)*B, D]
    T* dWx,              // [T*B, D]
    T* dR,               // [D, D]
    T* db,               // [D]
    T* d_delta_raw,      // [T*B, D] - gradient for W_delta, b_delta
    T* dA,               // [D] - gradient for A
    T* dh,               // [B, D] - recurrent gradient
    T* tmp_dRh           // [B, D]
) {
    static const T alpha_blas = static_cast<T>(1.0);
    static const T beta_blas = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);
    cudaMemsetAsync(dA, 0, D * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * batch_size * 2 * D;
        T* dWx_t = dWx + t * BD;
        T* d_delta_raw_t = d_delta_raw + t * BD;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanLeakySelectiveBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, A, h_t,
            tmp_dRh, d_delta_raw_t, dA, dh);

        // Copy d_raw to dWx_t
        cudaMemcpyAsync(dWx_t, tmp_dRh, BD * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // Accumulate bias gradient
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_dRh, db);

        // dR += h_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha_blas,
            h_t, D,
            tmp_dRh, D,
            &beta_one,
            dR, D);

        // dh += R @ d_raw (add to existing dh from leaky backward)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha_blas,
            R, D,
            tmp_dRh, D,
            &beta_one,
            dh, D);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace elman_leaky_selective

}  // namespace v0
}  // namespace haste
