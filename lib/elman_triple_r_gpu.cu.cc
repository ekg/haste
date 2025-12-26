// Copyright 2024 Erik Garrison. Apache 2.0 License.
// ElmanTripleR: Three separate R matrices for different signal pathways
//
// Architecture:
//   candidate = tanh(R_h @ h + R_x @ x + b)       -- separate R for hidden vs input
//   delta = sigmoid(R_delta @ h + W_delta @ x + b_delta)  -- h-dependent delta!
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
void TripleRPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,          // [B, D] - R_h @ h_prev
    const T* __restrict__ Rx,          // [B, D] - R_x @ x
    const T* __restrict__ b,           // [D]
    const T* __restrict__ Rdelta,      // [B, D] - R_delta @ h_prev
    const T* __restrict__ Wdelta_x,    // [B, D] - W_delta @ x
    const T* __restrict__ b_delta,     // [D]
    const T* __restrict__ h_prev,      // [B, D]
    T* __restrict__ h_next,            // [B, D]
    T* __restrict__ v,                 // [B, D] - pre-activation cache
    T* __restrict__ delta_cache        // [B, D] - cached delta
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int d_idx = idx % D;

    // Candidate: tanh(R_h @ h + R_x @ x + b)
    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Rx[idx]) +
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
}

template<typename T>
__global__
void TripleRBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,         // [B, D]
    const T* __restrict__ dh_recurrent,   // [B, D] or nullptr
    const T* __restrict__ v,              // [B, D] - pre-activation
    const T* __restrict__ delta_cache,    // [B, D]
    const T* __restrict__ h_prev,         // [B, D]
    T* __restrict__ d_raw,                // [B, D] - grad for candidate pathway
    T* __restrict__ d_delta_raw,          // [B, D] - grad for delta pathway
    T* __restrict__ dh_prev_out           // [B, D] - grad w.r.t. h_prev
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const float raw = static_cast<float>(v[idx]);
    const float candidate = tanhf(raw);
    const float delta = static_cast<float>(delta_cache[idx]);
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

    d_raw[idx] = static_cast<T>(d_raw_val);
    d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);
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

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace elman_triple_r {

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
    const T* b,             // [D]
    const T* b_delta,       // [D]
    const T* x,             // [T, B, input_size]
    T* h,                   // [T+1, B, D]
    T* v,                   // [T, B, D]
    T* delta_cache,         // [T, B, D]
    T* tmp_Rh,              // [B, D]
    T* tmp_Rx,              // [B, D]
    T* tmp_Rdelta           // [B, D]
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

    // Workspace for W_delta @ x
    // We'll reuse tmp_Rdelta for this since they're same size [B, D]

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD;
        T* delta_t = delta_cache + t * BD;
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

        // GEMM 3: tmp_Rdelta = R_delta @ h_t (h-dependent delta!)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R_delta, D,
            h_t, D,
            &beta,
            tmp_Rdelta, D);

        // We need W_delta @ x_t as well. Use a temporary.
        // Actually, let's compute it into delta_t first, then add in kernel
        // No wait, we need both R_delta@h and W_delta@x separately for backward
        // Let's use v_t temporarily since it's overwritten by kernel anyway
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            W_delta, input_size,
            x_t, input_size,
            &beta,
            v_t, D);  // Temporary: v_t holds W_delta @ x

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        // Kernel uses v_t as temporary for W_delta@x, then overwrites with pre-activation
        TripleRPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, tmp_Rx, b, tmp_Rdelta, v_t, b_delta, h_t, h_next, v_t, delta_t);
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
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* dh_new,
    T* dx,
    T* dR_h,
    T* dR_x,
    T* dR_delta,
    T* dW_delta,
    T* db,
    T* db_delta,
    T* dh,
    T* tmp_Rh,
    T* tmp_Rx,
    T* tmp_Rdelta
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
        const T* x_t = x + t * BI;
        T* dx_t = dx + t * BI;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        // Compute pointwise gradients
        // d_raw -> tmp_Rh (reuse)
        // d_delta_raw -> tmp_Rx (reuse)
        // dh_prev -> dh
        TripleRBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, delta_t, h_t,
            tmp_Rh, tmp_Rx, dh);

        // Accumulate bias gradients
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_Rh, db);
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_Rx, db_delta);

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
            tmp_Rx, D,
            &beta_one,
            dR_delta, D);

        // dR_x += x_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_Rh, D,
            &beta_one,
            dR_x, input_size);

        // dW_delta += x_t^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_Rx, D,
            &beta_one,
            dW_delta, input_size);

        // dx_t = R_x^T @ d_raw + W_delta^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            R_x, input_size,
            tmp_Rh, D,
            &beta,
            dx_t, input_size);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            W_delta, input_size,
            tmp_Rx, D,
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
            tmp_Rx, D,
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

}  // namespace elman_triple_r
}  // namespace v0
}  // namespace haste
