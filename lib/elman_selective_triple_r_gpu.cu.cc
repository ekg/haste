// Copyright 2024 Erik Garrison. Apache 2.0 License.
// ElmanSelectiveTripleR: Triple R with input-dependent B gate (like Mamba2)
//
// Architecture:
//   B_gate = sigmoid(W_B @ x + b_B)                      -- input-dependent write gate
//   candidate = tanh(R_h @ h + B_gate * (R_x @ x) + b)   -- B_gate modulates input
//   delta = sigmoid(R_delta @ h + W_delta @ x + b_delta) -- h-dependent delta
//   h_new = (1 - delta) * h + delta * candidate

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// ============================================================================
// Pointwise kernels
// ============================================================================

template<typename T>
__global__
void SelectiveTripleRPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,          // [B, D] - R_h @ h_prev
    const T* __restrict__ Rx,          // [B, D] - R_x @ x
    const T* __restrict__ b,           // [D]
    const T* __restrict__ Rdelta,      // [B, D] - R_delta @ h_prev
    const T* __restrict__ Wdelta_x,    // [B, D] - W_delta @ x
    const T* __restrict__ b_delta,     // [D]
    const T* __restrict__ WB_x,        // [B, D] - W_B @ x (NEW)
    const T* __restrict__ b_B,         // [D] (NEW)
    const T* __restrict__ h_prev,      // [B, D]
    T* __restrict__ h_next,            // [B, D]
    T* __restrict__ v,                 // [B, D] - pre-activation cache
    T* __restrict__ delta_cache,       // [B, D] - cached delta
    T* __restrict__ B_gate_cache       // [B, D] - cached B_gate (NEW)
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int d_idx = idx % D;

    // B_gate: sigmoid(W_B @ x + b_B) - input-dependent write selectivity
    float B_raw = static_cast<float>(WB_x[idx]) + static_cast<float>(b_B[d_idx]);
    float B_gate = 1.0f / (1.0f + expf(-B_raw));

    // Candidate: tanh(R_h @ h + B_gate * (R_x @ x) + b)
    // B_gate modulates the input contribution element-wise
    float raw = static_cast<float>(Rh[idx]) +
                B_gate * static_cast<float>(Rx[idx]) +
                static_cast<float>(b[d_idx]);
    float candidate = tanhf(raw);

    // Delta: sigmoid(R_delta @ h + W_delta @ x + b_delta)
    float delta_raw = static_cast<float>(Rdelta[idx]) +
                      static_cast<float>(Wdelta_x[idx]) +
                      static_cast<float>(b_delta[d_idx]);
    float delta = 1.0f / (1.0f + expf(-delta_raw));

    float h_p = static_cast<float>(h_prev[idx]);

    // Leaky integration
    float h_new = (1.0f - delta) * h_p + delta * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);
    delta_cache[idx] = static_cast<T>(delta);
    B_gate_cache[idx] = static_cast<T>(B_gate);
}

template<typename T>
__global__
void SelectiveTripleRBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,         // [B, D]
    const T* __restrict__ dh_recurrent,   // [B, D] or nullptr
    const T* __restrict__ v,              // [B, D] - pre-activation (with B_gate applied)
    const T* __restrict__ delta_cache,    // [B, D]
    const T* __restrict__ B_gate_cache,   // [B, D]
    const T* __restrict__ Rx,             // [B, D] - R_x @ x (needed for B_gate grad)
    const T* __restrict__ h_prev,         // [B, D]
    T* __restrict__ d_raw,                // [B, D] - grad for candidate pathway
    T* __restrict__ d_delta_raw,          // [B, D] - grad for delta pathway
    T* __restrict__ d_B_raw,              // [B, D] - grad for B_gate pathway (NEW)
    T* __restrict__ dh_prev_out           // [B, D] - grad w.r.t. h_prev
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const float raw = static_cast<float>(v[idx]);
    const float candidate = tanhf(raw);
    const float delta = static_cast<float>(delta_cache[idx]);
    const float B_gate = static_cast<float>(B_gate_cache[idx]);
    const float Rx_val = static_cast<float>(Rx[idx]);
    const float h_p = static_cast<float>(h_prev[idx]);

    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // Backward through: h_new = (1 - delta) * h_prev + delta * candidate
    const float d_candidate = dh * delta;
    const float d_h_prev = dh * (1.0f - delta);
    const float d_delta = dh * (candidate - h_p);

    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_raw_val = d_candidate * (1.0f - candidate * candidate);

    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    const float d_delta_raw_val = d_delta * delta * (1.0f - delta);

    // Backward through B_gate * Rx contribution to raw
    // raw contains R_h @ h + B_gate * Rx + b, so d_raw flows to B_gate
    // d(B_gate * Rx)/d(B_gate) = Rx
    // d(B_gate)/d(B_raw) = B_gate * (1 - B_gate)
    const float d_B_gate = d_raw_val * Rx_val;
    const float d_B_raw_val = d_B_gate * B_gate * (1.0f - B_gate);

    d_raw[idx] = static_cast<T>(d_raw_val);
    d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);
    d_B_raw[idx] = static_cast<T>(d_B_raw_val);
    dh_prev_out[idx] = static_cast<T>(d_h_prev);
}

template<typename T>
__global__
void AccumulateBiasGradientKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ d_raw,
    T* __restrict__ db
) {
    const int d_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (d_idx >= D) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        sum += static_cast<float>(d_raw[b * D + d_idx]);
    }
    db[d_idx] = static_cast<T>(static_cast<float>(db[d_idx]) + sum);
}

// Kernel to scale d_raw by B_gate for dR_x computation
template<typename T>
__global__
void ScaleByBGateKernel(
    const int total,
    const T* __restrict__ d_raw,
    const T* __restrict__ B_gate_cache,
    T* __restrict__ d_raw_scaled
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;

    float d = static_cast<float>(d_raw[idx]);
    float B = static_cast<float>(B_gate_cache[idx]);
    d_raw_scaled[idx] = static_cast<T>(d * B);
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace elman_selective_triple_r {

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
    const T* R_h,           // [D, D]
    const T* R_x,           // [D, input_size]
    const T* R_delta,       // [D, D]
    const T* W_delta,       // [D, input_size]
    const T* W_B,           // [D, input_size]
    const T* b,             // [D]
    const T* b_delta,       // [D]
    const T* b_B,           // [D]
    const T* x,             // [T, B, input_size]
    T* h,                   // [T+1, B, D]
    T* v,                   // [T, B, D]
    T* delta_cache,         // [T, B, D]
    T* B_gate_cache,        // [T, B, D]
    T* tmp_Rh,              // [B, D]
    T* tmp_Rx,              // [B, D]
    T* tmp_Rdelta,          // [B, D]
    T* tmp_B                // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;
    const int BI = batch_size * input_size;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD;
        T* delta_t = delta_cache + t * BD;
        T* B_gate_t = B_gate_cache + t * BD;
        const T* x_t = x + t * BI;

        // GEMM 1: tmp_Rh = R_h @ h_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R_h, D,
            h_t, D,
            &beta,
            tmp_Rh, D);

        // GEMM 2: tmp_Rx = R_x @ x_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            R_x, input_size,
            x_t, input_size,
            &beta,
            tmp_Rx, D);

        // GEMM 3: tmp_Rdelta = R_delta @ h_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R_delta, D,
            h_t, D,
            &beta,
            tmp_Rdelta, D);

        // GEMM 4: v_t (temp) = W_delta @ x_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            W_delta, input_size,
            x_t, input_size,
            &beta,
            v_t, D);  // Temporary: v_t holds W_delta @ x

        // GEMM 5: tmp_B = W_B @ x_t (for B_gate)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            W_B, input_size,
            x_t, input_size,
            &beta,
            tmp_B, D);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        SelectiveTripleRPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, tmp_Rx, b, tmp_Rdelta, v_t, b_delta,
            tmp_B, b_B, h_t, h_next, v_t, delta_t, B_gate_t);
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
    const T* R_h,
    const T* R_x,
    const T* R_delta,
    const T* W_delta,
    const T* W_B,
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* B_gate_cache,
    const T* dh_new,
    T* dx,
    T* dR_h,
    T* dR_x,
    T* dR_delta,
    T* dW_delta,
    T* dW_B,
    T* db,
    T* db_delta,
    T* db_B,
    T* dh,
    T* tmp_Rh,
    T* tmp_Rx,
    T* tmp_Rdelta,
    T* tmp_B
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int D = data_->hidden_size;
    const int BD = batch_size * D;
    const int BI = batch_size * input_size;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* B_gate_t = B_gate_cache + t * BD;
        const T* x_t = x + t * BI;
        T* dx_t = dx + t * BI;
        const T* dh_out = dh_new + (t + 1) * BD;

        // First, recompute R_x @ x for this timestep (needed for B_gate grad)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            R_x, input_size,
            x_t, input_size,
            &beta,
            tmp_Rx, D);

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        // Compute pointwise gradients
        // d_raw -> tmp_Rh
        // d_delta_raw -> tmp_Rdelta
        // d_B_raw -> tmp_B
        // dh_prev -> dh
        SelectiveTripleRBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, delta_t, B_gate_t, tmp_Rx, h_t,
            tmp_Rh, tmp_Rdelta, tmp_B, dh);

        // Accumulate bias gradients
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_Rh, db);
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_Rdelta, db_delta);
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_B, db_B);

        // dR_h += h_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha,
            h_t, D,
            tmp_Rh, D,
            &beta_one,
            dR_h, D);

        // dR_delta += h_t^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha,
            h_t, D,
            tmp_Rdelta, D,
            &beta_one,
            dR_delta, D);

        // For dR_x: need d_raw * B_gate (elementwise), then x_t^T @ (d_raw * B_gate)
        // Use tmp_Rx as scratch for scaled gradient
        ScaleByBGateKernel<T><<<blocks, threads, 0, stream>>>(
            BD, tmp_Rh, B_gate_t, tmp_Rx);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_Rx, D,
            &beta_one,
            dR_x, input_size);

        // dW_delta += x_t^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_Rdelta, D,
            &beta_one,
            dW_delta, input_size);

        // dW_B += x_t^T @ d_B_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_B, D,
            &beta_one,
            dW_B, input_size);

        // dx_t = R_x^T @ (d_raw * B_gate) + W_delta^T @ d_delta_raw + W_B^T @ d_B_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            R_x, input_size,
            tmp_Rx, D,  // tmp_Rx now holds d_raw * B_gate
            &beta,
            dx_t, input_size);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            W_delta, input_size,
            tmp_Rdelta, D,
            &beta_one,
            dx_t, input_size);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            W_B, input_size,
            tmp_B, D,
            &beta_one,
            dx_t, input_size);

        // dh (from matmul) += R_h @ d_raw + R_delta @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R_h, D,
            tmp_Rh, D,
            &beta_one,
            dh, D);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R_delta, D,
            tmp_Rdelta, D,
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

}  // namespace elman_selective_triple_r
}  // namespace v0
}  // namespace haste
