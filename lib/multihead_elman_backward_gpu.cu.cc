// Copyright 2024 Erik Garrison. Apache 2.0 License.
// Multi-head Elman RNN backward pass.
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
    const T* R,
    const T* Wx,
    const T* x,
    const T* h,
    const T* pre_act,
    const T* dy,
    T* dx,
    T* dR,
    T* dWx,
    T* db,
    T* dh0,
    T* tmp_dpre,
    T* tmp_dh
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
        const T* h_t = h + t * BNH;
        T* dx_t = dx + t * BNH;

        // Step 1: dpre = (dy + dh_from_next) * d_activation(pre)
        const int threads = 256;
        const int blocks = (BNH + threads - 1) / threads;

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

        // Step 2: Accumulate dR and dWx using batched GEMM
        // dR[n] += dpre[n].T @ h[n] for each head n
        // Shape: [headdim, B] @ [B, headdim] = [headdim, headdim]
        // In cuBLAS: [headdim, headdim] = [headdim, B] @ [B, headdim].T
        //         = dpre[n] @ h[n].T with CUBLAS_OP_N, CUBLAS_OP_T

        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,  // M, N, K
            &alpha,
            tmp_dpre, NH, strideH,         // dpre, lda, strideA
            h_t, NH, strideH,              // h, ldb, strideB
            &beta_one,                     // Accumulate!
            dR, headdim, strideW,          // dR, ldc, strideC
            nheads);

        // dWx[n] += dpre[n].T @ x[n]
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            headdim, headdim, batch_size,
            &alpha,
            tmp_dpre, NH, strideH,
            x_t, NH, strideH,
            &beta_one,
            dWx, headdim, strideW,
            nheads);

        // Accumulate db
        AccumulateBiasGradKernel<T><<<(NH + 255) / 256, 256, 0, stream>>>(
            batch_size, nheads, headdim, tmp_dpre, db);

        // Step 3: Compute dx and dh for previous timestep
        // dx[t] = Wx.T @ dpre (batched)
        // dh = R.T @ dpre (batched)

        cudaMemsetAsync(tmp_dh, 0, BNH * sizeof(T), stream);

        // dx = Wx.T @ dpre
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,  // M, N, K
            &alpha,
            Wx, headdim, strideW,          // Wx.T
            tmp_dpre, NH, strideH,
            &beta_zero,
            dx_t, NH, strideH,
            nheads);

        // dh = R.T @ dpre
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            headdim, batch_size, headdim,
            &alpha,
            R, headdim, strideW,           // R.T
            tmp_dpre, NH, strideH,
            &beta_zero,
            tmp_dh, NH, strideH,
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
