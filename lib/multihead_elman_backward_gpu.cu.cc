// Copyright 2024 Erik Garrison. Apache 2.0 License.
// Multi-head Elman RNN backward pass with RESET GATE.
//
// Forward architecture:
//   r[i] = sigmoid(Wr[i] @ x[i])           // Reset gate (input-only)
//   h_gated[i] = r[i] * h[i]               // Gate the history
//   h_new[i] = activation(R[i] @ h_gated[i] + Wx[i] @ x[i] + b[i])
//
// Backward path:
//   dpre = (dy + dh_next) * d_activation(pre)
//   dR += dpre @ h_gated.T
//   dWx += dpre @ x.T
//   db += sum(dpre)
//   dh_gated = R.T @ dpre
//   dh = dh_gated * r  (gradient through h_gated = r * h)
//   dr = dh_gated * h
//   dWrx = dr * r * (1-r)  (sigmoid gradient)
//   dWr += dWrx @ x.T
//   dx += Wx.T @ dpre + Wr.T @ dWrx
//
// Uses cublasSgemmStridedBatched to process all heads in single kernel calls.

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// Softsign gradient: d/dx [x / (1 + |x|)] = 1 / (1 + |x|)^2
template<typename T>
__device__ __forceinline__
T d_softsign(const T pre) {
    const T denom = static_cast<T>(1.0) + fabs(pre);
    return static_cast<T>(1.0) / (denom * denom);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
template<>
__device__ __forceinline__
__half d_softsign(const __half pre) {
    const __half one = __float2half(1.0f);
    const __half denom = __hadd(one, __habs(pre));
    return __hdiv(one, __hmul(denom, denom));
}
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template<>
__device__ __forceinline__
__nv_bfloat16 d_softsign(const __nv_bfloat16 pre) {
    float pf = __bfloat162float(pre);
    float denom = 1.0f + fabsf(pf);
    return __float2bfloat16(1.0f / (denom * denom));
}
#endif

// Tanh residual gradient: d/dx [x + tanh(x)] = 1 + (1 - tanh(x)^2)
template<typename T>
__device__ __forceinline__
T d_tanh_residual(const T pre) {
    const T tanh_pre = tanh(pre);
    return static_cast<T>(1.0) + (static_cast<T>(1.0) - tanh_pre * tanh_pre);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template<>
__device__ __forceinline__
__nv_bfloat16 d_tanh_residual(const __nv_bfloat16 pre) {
    float pf = __bfloat162float(pre);
    float tanh_pre = tanhf(pf);
    return __float2bfloat16(1.0f + (1.0f - tanh_pre * tanh_pre));
}
#endif

// Fused kernel: dpre = (dy + dh) * d_activation(pre)
template<typename T, int Activation>
__global__
void FusedActivationBackwardKernel(
    const int size,
    const T* __restrict__ dy,
    const T* __restrict__ dh,
    const T* __restrict__ pre_act,
    T* __restrict__ dpre
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    const T pre = pre_act[idx];
    const T dh_total = dy[idx] + dh[idx];
    T d_act;

    if (Activation == 0) {
        d_act = d_softsign(pre);
    } else if (Activation == 1) {
        d_act = d_tanh_residual(pre);
    } else {
        const T tanh_pre = tanh(pre);
        d_act = static_cast<T>(1.0) - tanh_pre * tanh_pre;
    }

    dpre[idx] = dh_total * d_act;
}

// Reset gate backward kernel
// Computes:
//   dh = dh_gated * r  (gradient through h_gated = r * h)
//   dr = dh_gated * h
//   dWrx = dr * r * (1-r)  (sigmoid derivative: r*(1-r))
template<typename T>
__global__
void ResetGateBackwardKernel(
    const int size,
    const T* __restrict__ dh_gated,   // [B, nheads, headdim] - gradient from R.T @ dpre
    const T* __restrict__ r,          // [B, nheads, headdim] - saved reset gate values
    const T* __restrict__ h,          // [B, nheads, headdim] - hidden state
    T* __restrict__ dh,               // [B, nheads, headdim] - gradient to previous h
    T* __restrict__ dWrx              // [B, nheads, headdim] - gradient for Wr @ x
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    const T r_val = r[idx];
    const T h_val = h[idx];
    const T dh_gated_val = dh_gated[idx];

    // dh = dh_gated * r
    dh[idx] = dh_gated_val * r_val;

    // dr = dh_gated * h
    // dWrx = dr * sigmoid'(Wrx) = dr * r * (1-r)
    const T dr = dh_gated_val * h_val;
    dWrx[idx] = dr * r_val * (static_cast<T>(1.0) - r_val);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template<>
__global__
void ResetGateBackwardKernel(
    const int size,
    const __nv_bfloat16* __restrict__ dh_gated,
    const __nv_bfloat16* __restrict__ r,
    const __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ dh,
    __nv_bfloat16* __restrict__ dWrx
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    // Compute in float for numerical stability
    float r_val = __bfloat162float(r[idx]);
    float h_val = __bfloat162float(h[idx]);
    float dh_gated_val = __bfloat162float(dh_gated[idx]);

    // dh = dh_gated * r
    dh[idx] = __float2bfloat16(dh_gated_val * r_val);

    // dr = dh_gated * h
    // dWrx = dr * sigmoid'(Wrx) = dr * r * (1-r)
    float dr = dh_gated_val * h_val;
    dWrx[idx] = __float2bfloat16(dr * r_val * (1.0f - r_val));
}
#endif

// Simple elementwise multiply: out = a * b
template<typename T>
__global__
void ElementwiseMultiplyKernel(
    const int size,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = a[idx] * b[idx];
}

// Kernel: Accumulate bias gradient
template<typename T>
__global__
void AccumulateBiasGradKernel(
    const int batch_size,
    const int nheads,
    const int headdim,
    const T* __restrict__ dpre,
    T* __restrict__ db
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = nheads * headdim;

    if (idx >= total) return;

    T sum = static_cast<T>(0.0);
    for (int b = 0; b < batch_size; ++b) {
        sum += dpre[b * total + idx];
    }

    atomicAdd(&db[idx], sum);
}

// Strided batched GEMM wrappers
inline cublasStatus_t stridedBatchedGemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    const float* beta,
    float* C, int ldc, long long strideC,
    int batchCount) {
    return cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
        alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline cublasStatus_t stridedBatchedGemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda, long long strideA,
    const double* B, int ldb, long long strideB,
    const double* beta,
    double* C, int ldc, long long strideC,
    int batchCount) {
    return cublasDgemmStridedBatched(handle, transa, transb, m, n, k,
        alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline cublasStatus_t stridedBatchedGemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const __half* alpha,
    const __half* A, int lda, long long strideA,
    const __half* B, int ldb, long long strideB,
    const __half* beta,
    __half* C, int ldc, long long strideC,
    int batchCount) {
    return cublasHgemmStridedBatched(handle, transa, transb, m, n, k,
        alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

inline cublasStatus_t stridedBatchedGemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const __nv_bfloat16* alpha,
    const __nv_bfloat16* A, int lda, long long strideA,
    const __nv_bfloat16* B, int ldb, long long strideB,
    const __nv_bfloat16* beta,
    __nv_bfloat16* C, int ldc, long long strideC,
    int batchCount) {
    float alpha_f = __bfloat162float(*alpha);
    float beta_f = __bfloat162float(*beta);
    return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k,
        &alpha_f, A, CUDA_R_16BF, lda, strideA,
        B, CUDA_R_16BF, ldb, strideB,
        &beta_f, C, CUDA_R_16BF, ldc, strideC,
        batchCount, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace multihead_elman {

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int nheads;
    int headdim;
    int activation;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int nheads,
    const int headdim,
    const int activation,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->nheads = nheads;
    data_->headdim = headdim;
    data_->activation = activation;
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
    const T* R,         // [nheads, headdim, headdim] - recurrent weights
    const T* Wr,        // [nheads, headdim, headdim] - reset gate weights
    const T* Wx,        // [nheads, headdim, headdim] - input weights
    const T* x,         // [T, B, nheads, headdim] - input sequence
    const T* h,         // [T+1, B, nheads, headdim] - all hidden states
    const T* r_gate,    // [T, B, nheads, headdim] - saved reset gate values
    const T* pre_act,   // [T, B, nheads, headdim] - saved pre-activations
    const T* dy,        // [T, B, nheads, headdim] - output gradients
    T* dx,              // [T, B, nheads, headdim] - input gradients
    T* dR,              // [nheads, headdim, headdim] - R gradients
    T* dWr,             // [nheads, headdim, headdim] - Wr gradients
    T* dWx,             // [nheads, headdim, headdim] - Wx gradients
    T* db,              // [nheads, headdim] - bias gradients
    T* dh0,             // [B, nheads, headdim] - initial hidden state gradient
    T* tmp_dpre,        // [B, nheads, headdim] - workspace
    T* tmp_dh,          // [B, nheads, headdim] - workspace for dh from next step
    T* tmp_dh_gated,    // [B, nheads, headdim] - workspace for dh_gated
    T* tmp_dWrx,        // [B, nheads, headdim] - workspace for dWrx
    T* tmp_h_gated      // [B, nheads, headdim] - workspace to recompute h_gated
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int nheads = data_->nheads;
    const int headdim = data_->headdim;
    const int activation = data_->activation;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream = data_->stream;

    cublasSetStream(blas_handle, stream);

    const int BNH = batch_size * nheads * headdim;
    const int NH = nheads * headdim;
    const long long strideW = headdim * headdim;
    const long long strideH = headdim;

    // Initialize dh to zeros
    cudaMemsetAsync(tmp_dh, 0, BNH * sizeof(T), stream);

    // Process timesteps in reverse
    for (int t = steps - 1; t >= 0; --t) {
        const T* dy_t = dy + t * BNH;
        const T* pre_act_t = pre_act + t * BNH;
        const T* x_t = x + t * BNH;
        const T* h_t = h + t * BNH;       // h[t], the input hidden state for this step
        const T* r_t = r_gate + t * BNH;  // saved reset gate values
        T* dx_t = dx + t * BNH;

        const int threads = 256;
        const int blocks = (BNH + threads - 1) / threads;

        // Step 1: dpre = (dy + dh_from_next) * d_activation(pre)
        if (activation == 0) {
            FusedActivationBackwardKernel<T, 0><<<blocks, threads, 0, stream>>>(
                BNH, dy_t, tmp_dh, pre_act_t, tmp_dpre);
        } else if (activation == 1) {
            FusedActivationBackwardKernel<T, 1><<<blocks, threads, 0, stream>>>(
                BNH, dy_t, tmp_dh, pre_act_t, tmp_dpre);
        } else {
            FusedActivationBackwardKernel<T, 2><<<blocks, threads, 0, stream>>>(
                BNH, dy_t, tmp_dh, pre_act_t, tmp_dpre);
        }

        // Step 2: Recompute h_gated = r * h for dR gradient
        ElementwiseMultiplyKernel<T><<<blocks, threads, 0, stream>>>(
            BNH, r_t, h_t, tmp_h_gated);

        // Step 3: Accumulate dR using h_gated
        // dR[n] += dpre[n] @ h_gated[n].T
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            tmp_dpre, NH, strideH,
            tmp_h_gated, NH, strideH,
            &beta_one,
            dR, headdim, strideW,
            nheads);

        // Step 4: Accumulate dWx
        // dWx[n] += dpre[n] @ x[n].T
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            tmp_dpre, NH, strideH,
            x_t, NH, strideH,
            &beta_one,
            dWx, headdim, strideW,
            nheads);

        // Step 5: Accumulate db
        AccumulateBiasGradKernel<T><<<(NH + 255) / 256, 256, 0, stream>>>(
            batch_size, nheads, headdim, tmp_dpre, db);

        // Step 6: Compute dh_gated = R.T @ dpre
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            R, headdim, strideW,
            tmp_dpre, NH, strideH,
            &beta_zero,
            tmp_dh_gated, NH, strideH,
            nheads);

        // Step 7: Backprop through reset gate
        // dh = dh_gated * r  (gradient through h_gated = r * h)
        // dr = dh_gated * h
        // dWrx = dr * r * (1-r)  (sigmoid derivative)
        ResetGateBackwardKernel<T><<<blocks, threads, 0, stream>>>(
            BNH, tmp_dh_gated, r_t, h_t, tmp_dh, tmp_dWrx);

        // Step 8: Compute dx
        // dx[t] = Wx.T @ dpre + Wr.T @ dWrx
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            Wx, headdim, strideW,
            tmp_dpre, NH, strideH,
            &beta_zero,
            dx_t, NH, strideH,
            nheads);

        // dx += Wr.T @ dWrx
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            Wr, headdim, strideW,
            tmp_dWrx, NH, strideH,
            &beta_one,
            dx_t, NH, strideH,
            nheads);

        // Step 9: Accumulate dWr
        // dWr += dWrx @ x.T
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            tmp_dWrx, NH, strideH,
            x_t, NH, strideH,
            &beta_one,
            dWr, headdim, strideW,
            nheads);
    }

    // Copy final dh to dh0
    cudaMemcpyAsync(dh0, tmp_dh, BNH * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace multihead_elman
}  // namespace v0
}  // namespace haste
