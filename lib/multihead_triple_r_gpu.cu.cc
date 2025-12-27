// Copyright 2024 Erik Garrison. Apache 2.0 License.
// MultiHeadTripleR: Multi-head Triple R with 32× state expansion (Mamba2-style)
//
// Architecture (per head):
//   candidate = tanh(R_h @ h + R_x @ x + b)
//   delta = sigmoid(R_delta @ h + W_delta @ x + b_delta)
//   h_new = (1 - delta) * h + delta * candidate
//
// Uses strided batched GEMM for efficient multi-head processing.
// Total state: nheads × headdim × d_state_r = 32× model dim (like Mamba2)

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// ============================================================================
// Multi-head pointwise kernel - processes all heads in parallel
// ============================================================================

template<typename T>
__global__
void MultiHeadTripleRPointwiseKernel(
    const int batch_size,
    const int nheads,
    const int headdim,
    const T* __restrict__ Rh,          // [B, nheads, headdim] - R_h @ h_prev per head
    const T* __restrict__ Rx,          // [B, nheads, headdim] - R_x @ x per head
    const T* __restrict__ b,           // [nheads, headdim]
    const T* __restrict__ Rdelta,      // [B, nheads, headdim] - R_delta @ h_prev
    const T* __restrict__ Wdelta_x,    // [B, nheads, headdim] - W_delta @ x
    const T* __restrict__ b_delta,     // [nheads, headdim]
    const T* __restrict__ h_prev,      // [B, nheads, headdim]
    T* __restrict__ h_next,            // [B, nheads, headdim]
    T* __restrict__ v,                 // [B, nheads, headdim] - pre-activation cache
    T* __restrict__ delta_cache        // [B, nheads, headdim] - cached delta
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * nheads * headdim;

    if (idx >= total) return;

    // Compute head and dim indices for bias lookup
    const int rem = idx % (nheads * headdim);
    const int h_idx = rem / headdim;
    const int d_idx = rem % headdim;
    const int bias_idx = h_idx * headdim + d_idx;

    // Candidate: tanh(R_h @ h + R_x @ x + b)
    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Rx[idx]) +
                static_cast<float>(b[bias_idx]);
    float candidate = tanhf(raw);

    // Delta: sigmoid(R_delta @ h + W_delta @ x + b_delta)
    float delta_raw = static_cast<float>(Rdelta[idx]) +
                      static_cast<float>(Wdelta_x[idx]) +
                      static_cast<float>(b_delta[bias_idx]);
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
void MultiHeadTripleRBackwardPointwiseKernel(
    const int batch_size,
    const int nheads,
    const int headdim,
    const T* __restrict__ dh_out,         // [B, nheads, headdim]
    const T* __restrict__ dh_recurrent,   // [B, nheads, headdim] or nullptr
    const T* __restrict__ v,              // [B, nheads, headdim] - pre-activation
    const T* __restrict__ delta_cache,    // [B, nheads, headdim]
    const T* __restrict__ h_prev,         // [B, nheads, headdim]
    T* __restrict__ d_raw,                // [B, nheads, headdim]
    T* __restrict__ d_delta_raw,          // [B, nheads, headdim]
    T* __restrict__ dh_prev_out           // [B, nheads, headdim]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * nheads * headdim;

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
void AccumulateMultiHeadBiasGradientKernel(
    const int batch_size,
    const int nheads,
    const int headdim,
    const T* __restrict__ d_raw,
    T* __restrict__ db
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = nheads * headdim;
    if (idx >= total) return;

    const int h_idx = idx / headdim;
    const int d_idx = idx % headdim;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        const int src_idx = b * nheads * headdim + h_idx * headdim + d_idx;
        sum += static_cast<float>(d_raw[src_idx]);
    }
    db[idx] = static_cast<T>(static_cast<float>(db[idx]) + sum);
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace multihead_triple_r {

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
    const T* R_h,           // [nheads, headdim, headdim]
    const T* R_x,           // [nheads, headdim, headdim]
    const T* R_delta,       // [nheads, headdim, headdim]
    const T* W_delta,       // [nheads, headdim, headdim]
    const T* b,             // [nheads, headdim]
    const T* b_delta,       // [nheads, headdim]
    const T* x,             // [T, B, nheads, headdim]
    T* h,                   // [T+1, B, nheads, headdim]
    T* v,                   // [T, B, nheads, headdim]
    T* delta_cache,         // [T, B, nheads, headdim]
    T* tmp_Rh,              // [B, nheads, headdim]
    T* tmp_Rx,              // [B, nheads, headdim]
    T* tmp_Rdelta,          // [B, nheads, headdim]
    T* tmp_Wdelta           // [B, nheads, headdim]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int nheads = data_->nheads;
    const int headdim = data_->headdim;
    const int BNH = batch_size * nheads * headdim;
    const long long strideR = headdim * headdim;  // Stride between heads in R matrices
    const long long strideH = headdim;            // Stride between heads in h/x (within batch)
    const long long strideOut = headdim;          // Stride between heads in output

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BNH;
        T* h_next = h + (t + 1) * BNH;
        T* v_t = v + t * BNH;
        T* delta_t = delta_cache + t * BNH;
        const T* x_t = x + t * BNH;

        // Use strided batched GEMM for all heads in parallel
        // For each head h: tmp_Rh[:, h, :] = h_t[:, h, :] @ R_h[h]^T
        //
        // Memory layout (row-major/C order):
        //   R_h[nheads, headdim, headdim]: R_h[h] at offset h * headdim * headdim
        //   h_t[B, nheads, headdim]: h_t[:, h, :] at offset h * headdim (strided by nheads*headdim)
        //   tmp_Rh[B, nheads, headdim]: tmp_Rh[:, h, :] at offset h * headdim (strided by nheads*headdim)
        //
        // cuBLAS interprets as column-major, so:
        //   R_h[h] is headdim x headdim with lda = headdim
        //   h_t[:, h, :] is headdim x B with ldb = nheads * headdim (stride between batches)
        //   output is headdim x B with ldc = nheads * headdim
        //
        // GEMM: C = alpha * op(A) * op(B) + beta * C
        // We want: output = R_h @ h_t^T for each head
        // Using CUBLAS_OP_T on A: output = R_h^T @ h_t where matrices are in column-major
        // Since our R_h is row-major, cuBLAS sees it transposed, so CUBLAS_OP_T gives us R_h @ h_t

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            R_h, headdim, strideR,
            h_t, nheads * headdim, strideH,
            &beta,
            tmp_Rh, nheads * headdim, strideOut,
            nheads);

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            R_x, headdim, strideR,
            x_t, nheads * headdim, strideH,
            &beta,
            tmp_Rx, nheads * headdim, strideOut,
            nheads);

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            R_delta, headdim, strideR,
            h_t, nheads * headdim, strideH,
            &beta,
            tmp_Rdelta, nheads * headdim, strideOut,
            nheads);

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            W_delta, headdim, strideR,
            x_t, nheads * headdim, strideH,
            &beta,
            tmp_Wdelta, nheads * headdim, strideOut,
            nheads);

        const int threads = 256;
        const int blocks = (BNH + threads - 1) / threads;

        MultiHeadTripleRPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, nheads, headdim,
            tmp_Rh, tmp_Rx, b, tmp_Rdelta, tmp_Wdelta, b_delta,
            h_t, h_next, v_t, delta_t);
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
    T* tmp_Rdelta,
    T* tmp_Wdelta
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int nheads = data_->nheads;
    const int headdim = data_->headdim;
    const int BNH = batch_size * nheads * headdim;
    const int NH = nheads * headdim;
    const long long strideR = headdim * headdim;
    const long long strideH = headdim;
    const long long strideOut = headdim;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    cudaMemsetAsync(dh, 0, BNH * sizeof(T), stream);

    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BNH;
        const T* v_t = v + t * BNH;
        const T* delta_t = delta_cache + t * BNH;
        const T* x_t = x + t * BNH;
        T* dx_t = dx + t * BNH;
        const T* dh_out = dh_new + (t + 1) * BNH;

        const int threads = 256;
        const int blocks = (BNH + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        // Compute pointwise gradients
        // d_raw -> tmp_Rh, d_delta_raw -> tmp_Rx
        MultiHeadTripleRBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, nheads, headdim,
            dh_out, dh_recurrent, v_t, delta_t, h_t,
            tmp_Rh, tmp_Rx, dh);

        // Accumulate bias gradients
        const int bias_threads = 256;
        const int bias_blocks = (NH + bias_threads - 1) / bias_threads;
        AccumulateMultiHeadBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, nheads, headdim, tmp_Rh, db);
        AccumulateMultiHeadBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, nheads, headdim, tmp_Rx, db_delta);

        // Gradient accumulation for R matrices using strided batched GEMM
        // dR_h[h] += h_t[:, h, :]^T @ d_raw[:, h, :] (outer product sum over batch)
        // This is: [headdim, B] @ [B, headdim] = [headdim, headdim]
        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            h_t, nheads * headdim, strideH,
            tmp_Rh, nheads * headdim, strideH,
            &beta_one,
            dR_h, headdim, strideR,
            nheads);

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            h_t, nheads * headdim, strideH,
            tmp_Rx, nheads * headdim, strideH,
            &beta_one,
            dR_delta, headdim, strideR,
            nheads);

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            x_t, nheads * headdim, strideH,
            tmp_Rh, nheads * headdim, strideH,
            &beta_one,
            dR_x, headdim, strideR,
            nheads);

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            x_t, nheads * headdim, strideH,
            tmp_Rx, nheads * headdim, strideH,
            &beta_one,
            dW_delta, headdim, strideR,
            nheads);

        // dx = R_x^T @ d_raw + W_delta^T @ d_delta_raw
        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            R_x, headdim, strideR,
            tmp_Rh, nheads * headdim, strideH,
            &beta,
            dx_t, nheads * headdim, strideOut,
            nheads);

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            W_delta, headdim, strideR,
            tmp_Rx, nheads * headdim, strideH,
            &beta_one,
            dx_t, nheads * headdim, strideOut,
            nheads);

        // dh += R_h @ d_raw + R_delta @ d_delta_raw
        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            R_h, headdim, strideR,
            tmp_Rh, nheads * headdim, strideH,
            &beta_one,
            dh, nheads * headdim, strideOut,
            nheads);

        blas<T>::gemmStridedBatched(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            R_delta, headdim, strideR,
            tmp_Rx, nheads * headdim, strideH,
            &beta_one,
            dh, nheads * headdim, strideOut,
            nheads);
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

}  // namespace multihead_triple_r
}  // namespace v0
}  // namespace haste
