// Copyright 2024 Erik Garrison. Apache 2.0 License.
// ElmanLowRankR: R = U @ V^T + S decomposition
//
// Architecture:
//   Rh = U @ (V^T @ h) + S @ h    -- two matmuls instead of one D×D
//   candidate = tanh(Rh + W_x @ x + b)
//   delta = sigmoid(W_delta @ x + b_delta)
//   h_new = (1 - delta) * h + delta * candidate

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cstdio>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// ============================================================================
// Pointwise kernel
// ============================================================================

template<typename T>
__global__
void LowRankRPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,           // [B, D] - (U @ V^T + S) @ h
    const T* __restrict__ Wx,           // [B, D]
    const T* __restrict__ b,            // [D]
    const T* __restrict__ delta_Wx,     // [B, D] - W_delta @ x (pre-sigmoid, WITHOUT b_delta)
    const T* __restrict__ b_delta,      // [D] - delta bias
    const T* __restrict__ h_prev,       // [B, D]
    T* __restrict__ h_next,             // [B, D]
    T* __restrict__ v,                  // [B, D] - pre-activation cache
    T* __restrict__ delta_cache         // [B, D]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int d_idx = idx % D;

    // Candidate: tanh(Rh + Wx + b)
    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(b[d_idx]);
    float candidate = tanhf(raw);

    // Delta: sigmoid(W_delta @ x + b_delta)
    float d_raw = static_cast<float>(delta_Wx[idx]) + static_cast<float>(b_delta[d_idx]);
    float delta = 1.0f / (1.0f + expf(-d_raw));

    float h_p = static_cast<float>(h_prev[idx]);
    float h_new = (1.0f - delta) * h_p + delta * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);
    delta_cache[idx] = static_cast<T>(delta);
}

template<typename T>
__global__
void LowRankRBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,
    const T* __restrict__ dh_recurrent,
    const T* __restrict__ v,
    const T* __restrict__ delta_cache,
    const T* __restrict__ h_prev,
    T* __restrict__ d_raw,              // [B, D]
    T* __restrict__ d_delta_raw,        // [B, D]
    T* __restrict__ dh_prev_out         // [B, D]
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

    const float d_candidate = dh * delta;
    const float d_h_prev = dh * (1.0f - delta);
    const float d_delta = dh * (candidate - h_p);

    const float d_raw_val = d_candidate * (1.0f - candidate * candidate);
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

// Add two tensors element-wise
template<typename T>
__global__
void AddTensorsKernel(
    const int size,
    const T* __restrict__ a,
    T* __restrict__ b   // b += a
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    b[idx] = static_cast<T>(static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace elman_lowrank_r {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    int rank;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int rank,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->rank = rank;
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
    const T* U,             // [D, rank]
    const T* V,             // [D, rank]
    const T* S,             // [D, D]
    const T* W_x,           // [D, input_size]
    const T* b,             // [D]
    const T* W_delta,       // [D, input_size]
    const T* b_delta,       // [D]
    const T* x,             // [T, B, input_size]
    T* h,                   // [T+1, B, D]
    T* v,                   // [T, B, D]
    T* delta_cache,         // [T, B, D]
    T* tmp_Vh,              // [B, rank]
    T* tmp_UVh,             // [B, D]
    T* tmp_Sh,              // [B, D]
    T* tmp_Wx               // [B, D]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int D = data_->hidden_size;
    const int rank = data_->rank;
    const int BD = batch_size * D;
    const int BI = batch_size * input_size;
    const int BR = batch_size * rank;

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
        const T* x_t = x + t * BI;

        // Step 1: Compute low-rank part: U @ (V^T @ h)
        // h @ V = [B, D] @ [D, rank] = [B, rank]
        // In col-major: V=[rank,D] @ h=[D,B] doesn't work dimensionally
        // So compute V @ h^T... no, let's use: V (col-maj [rank,D]) @ h (col-maj [D,B])
        // Actually: result [rank,B] = V[rank,D] @ h[D,B] - this is op(V)=V @ op(h)=h
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank, batch_size, D,
            &alpha,
            V, rank,       // V is [D,rank] row-maj = [rank,D] col-maj, lda=rank
            h_t, D,        // h is [B,D] row-maj = [D,B] col-maj, ldb=D
            &beta,
            tmp_Vh, rank);

        // U @ (V^T @ h): [D, rank] @ [rank, B]^T -> [B, D]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, rank,
            &alpha,
            U, rank,
            tmp_Vh, rank,
            &beta,
            tmp_UVh, D);

        // Step 2: Compute sparse/residual part: S @ h
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            S, D,
            h_t, D,
            &beta,
            tmp_Sh, D);

        // Step 3: Combine: Rh = UVh + Sh
        const int add_threads = 256;
        const int add_blocks = (BD + add_threads - 1) / add_threads;
        AddTensorsKernel<T><<<add_blocks, add_threads, 0, stream>>>(BD, tmp_Sh, tmp_UVh);
        // Now tmp_UVh = (U @ V^T + S) @ h

        // Step 4: W_x @ x
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            W_x, input_size,
            x_t, input_size,
            &beta,
            tmp_Wx, D);

        // Step 5: W_delta @ x + b_delta -> delta_t (pre-sigmoid)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            W_delta, input_size,
            x_t, input_size,
            &beta,
            delta_t, D);

        // Step 6: Pointwise kernel (b_delta is added inside the kernel)
        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        LowRankRPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_UVh, tmp_Wx, b, delta_t, b_delta, h_t, h_next, v_t, delta_t);
        // delta_t is input (W_delta @ x) and output (final delta after sigmoid)
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    int rank;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int rank,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->rank = rank;
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
    const T* U,
    const T* V,
    const T* S,
    const T* W_x,
    const T* W_delta,
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* dh_new,
    T* dx,
    T* dU,
    T* dV,
    T* dS,
    T* dW_x,
    T* db,
    T* dW_delta,
    T* db_delta,
    T* dh,
    T* tmp_Vh,
    T* tmp_d_raw,
    T* tmp_d_delta_raw
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int D = data_->hidden_size;
    const int rank = data_->rank;
    const int BD = batch_size * D;
    const int BI = batch_size * input_size;
    const int BR = batch_size * rank;

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

        // Pointwise backward
        LowRankRBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, delta_t, h_t,
            tmp_d_raw, tmp_d_delta_raw, dh);

        // Accumulate bias gradients
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_d_raw, db);
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_d_delta_raw, db_delta);

        // dS += h_t^T @ d_raw (gradient for S component)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha,
            h_t, D,
            tmp_d_raw, D,
            &beta_one,
            dS, D);

        // For low-rank: dU and dV
        // d_raw flows back through U @ V^T @ h
        // d(U @ V^T @ h) = d_raw
        // dU += d_raw @ (V^T @ h)^T = d_raw @ h^T @ V
        // dV += h @ (U^T @ d_raw)^T = h @ d_raw^T @ U

        // Compute h @ V -> tmp_Vh [B, rank]
        // V is [D,rank] row-maj = [rank,D] col-maj, h is [B,D] row-maj = [D,B] col-maj
        // Result: [rank,D] @ [D,B] = [rank,B] col-maj = [B,rank] row-maj
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank, batch_size, D,
            &alpha,
            V, rank,
            h_t, D,
            &beta,
            tmp_Vh, rank);

        // dU += tmp_Vh @ d_raw^T: [rank, B] @ [B, D] -> [rank, D] (col-major view)
        // Row-major dU is [D, rank], col-major is [rank, D]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rank, D, batch_size,
            &alpha,
            tmp_Vh, rank,
            tmp_d_raw, D,
            &beta_one,
            dU, rank);

        // For dV: need U^T @ d_raw first
        // d_z = U^T @ d_raw: [rank, D] @ [D, B]^T -> [B, rank]
        T* tmp_dz = tmp_Vh;  // Reuse
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank, batch_size, D,
            &alpha,
            U, rank,
            tmp_d_raw, D,
            &beta,
            tmp_dz, rank);

        // dV = H^T @ D_Z = [D,B]@[B,rank] = [D,rank] row-maj = [rank,D] col-maj
        // In CUBLAS: D_Z @ H^T = [rank,B] @ [B,D] = [rank,D]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            rank, D, batch_size,
            &alpha,
            tmp_dz, rank,
            h_t, D,
            &beta_one,
            dV, rank);

        // dW_x += x_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_d_raw, D,
            &beta_one,
            dW_x, input_size);

        // dW_delta += x_t^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_d_delta_raw, D,
            &beta_one,
            dW_delta, input_size);

        // dx = W_x^T @ d_raw + W_delta^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            W_x, input_size,
            tmp_d_raw, D,
            &beta,
            dx_t, input_size);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            W_delta, input_size,
            tmp_d_delta_raw, D,
            &beta_one,
            dx_t, input_size);

        // dh += R^T @ d_raw where R = U @ V^T + S
        // R^T = (U @ V^T + S)^T = V @ U^T + S^T
        // First: S^T @ d_raw (S stored row-major = S^T col-major, so OP_N gives S^T)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            S, D,
            tmp_d_raw, D,
            &beta_one,
            dh, D);

        // Second: V @ U^T @ d_raw (NOT U^T @ V^T!)
        // Step 1: z = U^T @ d_raw = [rank, batch]
        // U stored [D,rank] row-maj = [rank,D] col-maj, with OP_N gives [rank,D]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rank, batch_size, D,
            &alpha,
            U, rank,
            tmp_d_raw, D,
            &beta,
            tmp_Vh, rank);

        // Step 2: dh += V @ z where V is [D,rank] row-maj = [rank,D] col-maj
        // Use OP_T to get [D,rank] @ [rank,batch] = [D,batch]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, rank,
            &alpha,
            V, rank,
            tmp_Vh, rank,
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

}  // namespace elman_lowrank_r
}  // namespace v0
}  // namespace haste
