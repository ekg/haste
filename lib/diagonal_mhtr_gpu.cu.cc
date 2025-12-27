// Copyright 2024 Erik Garrison. Apache 2.0 License.
// DiagonalMHTR: Diagonal Multi-Head Triple R (depth-stable variant)
//
// KEY DIFFERENCE FROM FULL MHTR:
//   Full MHTR: R_h @ h (matrix-vector multiply, O(d^2))
//   Diagonal:  R_h * h (element-wise multiply, O(d))
//
// This eliminates strided batched GEMM from the recurrent loop!
// Only W_delta @ x needs GEMM, and it can be pre-computed once.
//
// Architecture (per head):
//   candidate = tanh(R_h * h + R_x * x + b)      // element-wise
//   delta = sigmoid(R_delta * h + W_delta @ x + b_delta)
//   h_new = (1 - delta) * h + delta * candidate

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// ============================================================================
// Diagonal MHTR forward kernel - ALL operations element-wise in recurrent loop
// ============================================================================

template<typename T>
__global__
void DiagonalMHTRForwardKernel(
    const int batch_size,
    const int nheads,
    const int headdim,
    const T* __restrict__ R_h,          // [nheads, headdim] - DIAGONAL
    const T* __restrict__ R_x,          // [nheads, headdim] - DIAGONAL
    const T* __restrict__ R_delta,      // [nheads, headdim] - DIAGONAL
    const T* __restrict__ b,            // [nheads, headdim]
    const T* __restrict__ b_delta,      // [nheads, headdim]
    const T* __restrict__ h_prev,       // [B, nheads, headdim]
    const T* __restrict__ x_t,          // [B, nheads, headdim]
    const T* __restrict__ Wdelta_x,     // [B, nheads, headdim] - pre-computed W_delta @ x
    T* __restrict__ h_next,             // [B, nheads, headdim]
    T* __restrict__ v,                  // [B, nheads, headdim] - pre-activation cache
    T* __restrict__ delta_cache         // [B, nheads, headdim] - cached delta
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * nheads * headdim;

    if (idx >= total) return;

    // Compute head and dim indices for parameter lookup
    const int rem = idx % (nheads * headdim);
    const int h_idx = rem / headdim;
    const int d_idx = rem % headdim;
    const int param_idx = h_idx * headdim + d_idx;

    // Load diagonal R values (shared across batch)
    const float r_h = static_cast<float>(R_h[param_idx]);
    const float r_x = static_cast<float>(R_x[param_idx]);
    const float r_delta = static_cast<float>(R_delta[param_idx]);

    // Load h_prev and x_t
    const float h_p = static_cast<float>(h_prev[idx]);
    const float x_val = static_cast<float>(x_t[idx]);

    // DIAGONAL: element-wise multiply instead of matrix-vector
    // Candidate: tanh(R_h * h + R_x * x + b)
    float raw = r_h * h_p + r_x * x_val + static_cast<float>(b[param_idx]);
    float candidate = tanhf(raw);

    // Delta: sigmoid(R_delta * h + W_delta @ x + b_delta)
    // W_delta @ x is pre-computed (only matrix op needed)
    float delta_raw = r_delta * h_p +
                      static_cast<float>(Wdelta_x[idx]) +
                      static_cast<float>(b_delta[param_idx]);
    float delta = 1.0f / (1.0f + expf(-delta_raw));

    // Leaky integration
    float h_new = (1.0f - delta) * h_p + delta * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);
    delta_cache[idx] = static_cast<T>(delta);
}

template<typename T>
__global__
void DiagonalMHTRBackwardKernel(
    const int batch_size,
    const int nheads,
    const int headdim,
    const T* __restrict__ R_h,          // [nheads, headdim] - DIAGONAL
    const T* __restrict__ R_delta,      // [nheads, headdim] - DIAGONAL
    const T* __restrict__ dh_out,       // [B, nheads, headdim]
    const T* __restrict__ dh_recurrent, // [B, nheads, headdim] or nullptr
    const T* __restrict__ v,            // [B, nheads, headdim] - pre-activation
    const T* __restrict__ delta_cache,  // [B, nheads, headdim]
    const T* __restrict__ h_prev,       // [B, nheads, headdim]
    const T* __restrict__ x_t,          // [B, nheads, headdim]
    T* __restrict__ dh_prev_out,        // [B, nheads, headdim]
    T* __restrict__ d_raw,              // [B, nheads, headdim] - for gradient accumulation
    T* __restrict__ d_delta_raw,        // [B, nheads, headdim]
    T* __restrict__ dx_out              // [B, nheads, headdim]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * nheads * headdim;

    if (idx >= total) return;

    // Compute head and dim indices
    const int rem = idx % (nheads * headdim);
    const int h_idx = rem / headdim;
    const int d_idx = rem % headdim;
    const int param_idx = h_idx * headdim + d_idx;

    // Load diagonal R values
    const float r_h = static_cast<float>(R_h[param_idx]);
    const float r_delta = static_cast<float>(R_delta[param_idx]);

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
    const float d_h_prev_direct = dh * (1.0f - delta);
    const float d_delta = dh * (candidate - h_p);

    // d/dx tanh(x) = 1 - tanh(x)^2
    const float d_raw_val = d_candidate * (1.0f - candidate * candidate);

    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    const float d_delta_raw_val = d_delta * delta * (1.0f - delta);

    // Gradient through DIAGONAL R matrices
    // raw = r_h * h_prev + r_x * x + b
    // d_h_prev from raw: d_raw * r_h
    // delta_raw = r_delta * h_prev + ...
    // d_h_prev from delta_raw: d_delta_raw * r_delta
    const float dh_prev = d_h_prev_direct + d_raw_val * r_h + d_delta_raw_val * r_delta;

    d_raw[idx] = static_cast<T>(d_raw_val);
    d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);
    dh_prev_out[idx] = static_cast<T>(dh_prev);

    // dx gets gradient from R_x path (accumulated separately)
    // Will be computed with d_raw * R_x in a separate pass or here
}

// Kernel to accumulate gradients for diagonal R vectors
template<typename T>
__global__
void AccumulateDiagonalRGradientKernel(
    const int batch_size,
    const int seq_len,
    const int nheads,
    const int headdim,
    const T* __restrict__ d_raw,        // [T, B, nheads, headdim]
    const T* __restrict__ h_prev,       // [T, B, nheads, headdim]
    const T* __restrict__ x,            // [T, B, nheads, headdim]
    const T* __restrict__ d_delta_raw,  // [T, B, nheads, headdim]
    T* __restrict__ dR_h,               // [nheads, headdim]
    T* __restrict__ dR_x,               // [nheads, headdim]
    T* __restrict__ dR_delta            // [nheads, headdim]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = nheads * headdim;
    if (idx >= total) return;

    const int h_idx = idx / headdim;
    const int d_idx = idx % headdim;

    float sum_h = 0.0f;
    float sum_x = 0.0f;
    float sum_delta = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        for (int b = 0; b < batch_size; ++b) {
            const int src_idx = t * batch_size * nheads * headdim +
                               b * nheads * headdim +
                               h_idx * headdim + d_idx;

            const float d_raw_val = static_cast<float>(d_raw[src_idx]);
            const float d_delta_raw_val = static_cast<float>(d_delta_raw[src_idx]);
            const float h_val = static_cast<float>(h_prev[src_idx]);
            const float x_val = static_cast<float>(x[src_idx]);

            // dR_h += d_raw * h_prev (element-wise)
            sum_h += d_raw_val * h_val;
            // dR_x += d_raw * x (element-wise)
            sum_x += d_raw_val * x_val;
            // dR_delta += d_delta_raw * h_prev (element-wise)
            sum_delta += d_delta_raw_val * h_val;
        }
    }

    dR_h[idx] = static_cast<T>(static_cast<float>(dR_h[idx]) + sum_h);
    dR_x[idx] = static_cast<T>(static_cast<float>(dR_x[idx]) + sum_x);
    dR_delta[idx] = static_cast<T>(static_cast<float>(dR_delta[idx]) + sum_delta);
}

template<typename T>
__global__
void AccumulateBiasGradientKernel(
    const int batch_size,
    const int seq_len,
    const int nheads,
    const int headdim,
    const T* __restrict__ d_raw,
    const T* __restrict__ d_delta_raw,
    T* __restrict__ db,
    T* __restrict__ db_delta
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = nheads * headdim;
    if (idx >= total) return;

    const int h_idx = idx / headdim;
    const int d_idx = idx % headdim;

    float sum_b = 0.0f;
    float sum_b_delta = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        for (int b = 0; b < batch_size; ++b) {
            const int src_idx = t * batch_size * nheads * headdim +
                               b * nheads * headdim +
                               h_idx * headdim + d_idx;
            sum_b += static_cast<float>(d_raw[src_idx]);
            sum_b_delta += static_cast<float>(d_delta_raw[src_idx]);
        }
    }

    db[idx] = static_cast<T>(static_cast<float>(db[idx]) + sum_b);
    db_delta[idx] = static_cast<T>(static_cast<float>(db_delta[idx]) + sum_b_delta);
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace diagonal_mhtr {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int nheads;
    int headdim;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int nheads,
    const int headdim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->nheads = nheads;
    data_->headdim = headdim;
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
    const T* R_h,           // [nheads, headdim] - DIAGONAL
    const T* R_x,           // [nheads, headdim] - DIAGONAL
    const T* R_delta,       // [nheads, headdim] - DIAGONAL
    const T* W_delta,       // [nheads, headdim, headdim] - still full matrix
    const T* b,             // [nheads, headdim]
    const T* b_delta,       // [nheads, headdim]
    const T* x,             // [T, B, nheads, headdim]
    T* h,                   // [T+1, B, nheads, headdim]
    T* v,                   // [T, B, nheads, headdim]
    T* delta_cache,         // [T, B, nheads, headdim]
    T* tmp_Wdelta           // [T, B, nheads, headdim] - pre-computed W_delta @ x for ALL timesteps
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int nheads = data_->nheads;
    const int headdim = data_->headdim;
    const int BNH = batch_size * nheads * headdim;
    const long long strideR = headdim * headdim;  // Stride between heads in W_delta matrix
    const long long strideX = headdim;            // Stride between heads in x
    const long long strideOut = headdim;          // Stride between heads in output

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    // PRE-COMPUTE W_delta @ x for ALL timesteps at once!
    // This is the ONLY GEMM needed for diagonal MHTR
    // Shape: [T, B, nheads, headdim] = W_delta @ x
    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BNH;
        T* Wdelta_t = tmp_Wdelta + t * BNH;

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            W_delta, headdim, strideR,
            x_t, nheads * headdim, strideX,
            &beta,
            Wdelta_t, nheads * headdim, strideOut,
            nheads);
    }

    // Now run the recurrence with purely element-wise operations!
    // NO GEMM in the loop - just the diagonal kernel
    const int threads = 256;
    const int blocks = (BNH + threads - 1) / threads;

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BNH;
        T* h_next = h + (t + 1) * BNH;
        T* v_t = v + t * BNH;
        T* delta_t = delta_cache + t * BNH;
        const T* x_t = x + t * BNH;
        const T* Wdelta_t = tmp_Wdelta + t * BNH;

        DiagonalMHTRForwardKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, nheads, headdim,
            R_h, R_x, R_delta, b, b_delta,
            h_t, x_t, Wdelta_t,
            h_next, v_t, delta_t);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int nheads;
    int headdim;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int nheads,
    const int headdim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->nheads = nheads;
    data_->headdim = headdim;
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
    const T* R_h,           // [nheads, headdim] - DIAGONAL
    const T* R_x,           // [nheads, headdim] - DIAGONAL
    const T* R_delta,       // [nheads, headdim] - DIAGONAL
    const T* W_delta,       // [nheads, headdim, headdim]
    const T* x,             // [T, B, nheads, headdim]
    const T* h,             // [T+1, B, nheads, headdim]
    const T* v,             // [T, B, nheads, headdim]
    const T* delta_cache,   // [T, B, nheads, headdim]
    const T* dh_out,        // [T, B, nheads, headdim]
    T* dx,                  // [T, B, nheads, headdim]
    T* dR_h,                // [nheads, headdim]
    T* dR_x,                // [nheads, headdim]
    T* dR_delta,            // [nheads, headdim]
    T* dW_delta,            // [nheads, headdim, headdim]
    T* db,                  // [nheads, headdim]
    T* db_delta,            // [nheads, headdim]
    T* d_raw_all,           // [T, B, nheads, headdim] - workspace
    T* d_delta_raw_all,     // [T, B, nheads, headdim] - workspace
    T* dh_prev              // [B, nheads, headdim] - workspace for recurrent gradient
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int nheads = data_->nheads;
    const int headdim = data_->headdim;
    const int BNH = batch_size * nheads * headdim;
    const long long strideR = headdim * headdim;
    const long long strideX = headdim;
    const long long strideOut = headdim;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    const int threads = 256;
    const int blocks = (BNH + threads - 1) / threads;

    // Zero out gradients
    cudaMemsetAsync(dR_h, 0, nheads * headdim * sizeof(T), stream);
    cudaMemsetAsync(dR_x, 0, nheads * headdim * sizeof(T), stream);
    cudaMemsetAsync(dR_delta, 0, nheads * headdim * sizeof(T), stream);
    cudaMemsetAsync(db, 0, nheads * headdim * sizeof(T), stream);
    cudaMemsetAsync(db_delta, 0, nheads * headdim * sizeof(T), stream);
    cudaMemsetAsync(dW_delta, 0, nheads * headdim * headdim * sizeof(T), stream);
    cudaMemsetAsync(dh_prev, 0, BNH * sizeof(T), stream);

    // Backward pass: iterate from t=steps-1 to t=0
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BNH;
        const T* x_t = x + t * BNH;
        const T* v_t = v + t * BNH;
        const T* delta_t = delta_cache + t * BNH;
        const T* dh_t = dh_out + t * BNH;
        T* d_raw_t = d_raw_all + t * BNH;
        T* d_delta_raw_t = d_delta_raw_all + t * BNH;
        T* dx_t = dx + t * BNH;

        DiagonalMHTRBackwardKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, nheads, headdim,
            R_h, R_delta,
            dh_t, (t == steps - 1) ? nullptr : dh_prev,
            v_t, delta_t, h_t, x_t,
            dh_prev, d_raw_t, d_delta_raw_t, dx_t);

        // dx += d_raw * R_x (element-wise broadcast)
        // This needs a separate kernel or can be merged above
        // For now, we'll compute dx from d_delta_raw through W_delta^T

        // dW_delta += outer_product(d_delta_raw, x) - need batched gemm
        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            d_delta_raw_t, nheads * headdim, strideX,
            x_t, nheads * headdim, strideX,
            &beta_one,
            dW_delta, headdim, strideR,
            nheads);
    }

    // Accumulate gradients for diagonal R vectors and biases
    const int param_blocks = (nheads * headdim + threads - 1) / threads;

    AccumulateDiagonalRGradientKernel<T><<<param_blocks, threads, 0, stream>>>(
        batch_size, steps, nheads, headdim,
        d_raw_all, h, x, d_delta_raw_all,
        dR_h, dR_x, dR_delta);

    AccumulateBiasGradientKernel<T><<<param_blocks, threads, 0, stream>>>(
        batch_size, steps, nheads, headdim,
        d_raw_all, d_delta_raw_all,
        db, db_delta);

    cublasSetStream(blas_handle, save_stream);
}

// Explicit template instantiations
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;

}  // namespace diagonal_mhtr
}  // namespace v0
}  // namespace haste
