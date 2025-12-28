// Copyright 2024 Erik Garrison. Apache 2.0 License.
// Multi-head Elman RNN forward pass with per-head R matrices and RESET GATE.
//
// Architecture per timestep per head:
//   r[i] = sigmoid(Wr[i] @ x[i])           // Reset gate (input-only)
//   h_gated[i] = r[i] * h[i]               // Gate the history
//   h_new[i] = activation(R[i] @ h_gated[i] + Wx[i] @ x[i] + b[i])
//
// Key optimization: Use cublasSgemmStridedBatched to process ALL heads
// in a SINGLE kernel call, eliminating per-head loop overhead.
//
// Shapes:
//   x:  [T, B, nheads, headdim] - input
//   h:  [B, nheads, headdim] - hidden state
//   R:  [nheads, headdim, headdim] - recurrent weights
//   Wr: [nheads, headdim, headdim] - reset gate weights
//   Wx: [nheads, headdim, headdim] - input weights
//   b:  [nheads, headdim] - bias

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// Softsign activation: x / (1 + |x|)
template<typename T>
__device__ __forceinline__
T softsign(const T x) {
    return x / (static_cast<T>(1.0) + fabs(x));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
template<>
__device__ __forceinline__
__half softsign(const __half x) {
    const __half one = __float2half(1.0f);
    return __hdiv(x, __hadd(one, __habs(x)));
}
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template<>
__device__ __forceinline__
__nv_bfloat16 softsign(const __nv_bfloat16 x) {
    float xf = __bfloat162float(x);
    return __float2bfloat16(xf / (1.0f + fabsf(xf)));
}
#endif

// Sigmoid activation for reset gate
template<typename T>
__device__ __forceinline__
T reset_gate_sigmoid(const T x) {
    return static_cast<T>(1.0) / (static_cast<T>(1.0) + expf(static_cast<float>(-x)));
}

// Specialization for float
template<>
__device__ __forceinline__
float reset_gate_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Specialization for double
template<>
__device__ __forceinline__
double reset_gate_sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
template<>
__device__ __forceinline__
__half reset_gate_sigmoid(const __half x) {
    float xf = __half2float(x);
    return __float2half(1.0f / (1.0f + expf(-xf)));
}
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template<>
__device__ __forceinline__
__nv_bfloat16 reset_gate_sigmoid(const __nv_bfloat16 x) {
    float xf = __bfloat162float(x);
    return __float2bfloat16(1.0f / (1.0f + expf(-xf)));
}
#endif

// Reset gate kernel: r = sigmoid(Wrx), h_gated = r * h
template<typename T>
__global__
void ResetGateKernel(
    const int size,
    const T* __restrict__ Wrx,     // [B, nheads, headdim] - Wr @ x
    const T* __restrict__ h,       // [B, nheads, headdim] - hidden state
    T* __restrict__ r,             // [B, nheads, headdim] - reset gate values (saved for backward)
    T* __restrict__ h_gated        // [B, nheads, headdim] - gated hidden state
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    const T r_val = reset_gate_sigmoid(Wrx[idx]);
    r[idx] = r_val;
    h_gated[idx] = r_val * h[idx];
}

// Fused kernel: h_new = activation(Rh_gated + Wxx + b)
template<typename T, int Activation>
__global__
void FusedActivationKernel(
    const int batch_size,
    const int nheads,
    const int headdim,
    const T* __restrict__ Rh,      // [B, nheads, headdim]
    const T* __restrict__ Wxx,     // [B, nheads, headdim]
    const T* __restrict__ b,       // [nheads, headdim]
    T* __restrict__ out,           // [B, nheads, headdim]
    T* __restrict__ pre_act        // [B, nheads, headdim] - saved pre-activation
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * nheads * headdim;

    if (idx >= total) return;

    // Compute indices for bias lookup
    const int remainder = idx % (nheads * headdim);

    // Compute pre-activation
    const T pre = Rh[idx] + Wxx[idx] + b[remainder];

    // Save pre-activation for backward if needed
    if (pre_act != nullptr) {
        pre_act[idx] = pre;
    }

    // Apply activation
    if (Activation == 0) {
        // Softsign
        out[idx] = softsign(pre);
    } else if (Activation == 1) {
        // Tanh residual: x + tanh(x)
        out[idx] = pre + tanh(pre);
    } else {
        // Tanh
        out[idx] = tanh(pre);
    }
}

// Strided batched GEMM wrappers for different types
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
    // BF16 uses cublasGemmStridedBatchedEx
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
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int nheads;
    int headdim;
    int activation;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int nheads,
    const int headdim,
    const int activation,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->nheads = nheads;
    data_->headdim = headdim;
    data_->activation = activation;
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
    const T* R,         // [nheads, headdim, headdim] - recurrent weights
    const T* Wr,        // [nheads, headdim, headdim] - reset gate weights
    const T* Wx,        // [nheads, headdim, headdim] - input weights
    const T* b,         // [nheads, headdim] - bias
    const T* x,         // [T, B, nheads, headdim] - input sequence
    T* h,               // [T+1, B, nheads, headdim] - all hidden states (h[0]=h0 on input)
    T* y,               // [T, B, nheads, headdim] - output sequence
    T* r_gate,          // [T, B, nheads, headdim] - saved reset gate values for backward
    T* pre_act,         // [T, B, nheads, headdim] - saved pre-activations
    T* tmp_Rh,          // [B, nheads, headdim] - workspace for R @ h_gated
    T* tmp_Wrx,         // [T, B, nheads, headdim] - pre-computed Wr @ x
    T* tmp_Wxx,         // [T, B, nheads, headdim] - pre-computed Wx @ x
    T* tmp_h_gated      // [B, nheads, headdim] - workspace for gated hidden state
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int nheads = data_->nheads;
    const int headdim = data_->headdim;
    const int activation = data_->activation;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream = data_->stream;

    cublasSetStream(blas_handle, stream);

    const int BNH = batch_size * nheads * headdim;
    const int NH = nheads * headdim;
    const long long strideW = headdim * headdim;  // Stride between weight matrices
    const long long strideH = headdim;            // Stride between head slices in h

    // =========================================================================
    // Pre-compute Wr @ x and Wx @ x for ALL timesteps
    // =========================================================================
    for (int head = 0; head < nheads; ++head) {
        const T* Wr_head = Wr + head * headdim * headdim;
        const T* Wx_head = Wx + head * headdim * headdim;

        // Wr @ x for reset gate
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            headdim, steps * batch_size, headdim,
            &alpha,
            Wr_head, headdim,
            x + head * headdim, NH,
            &beta_zero,
            tmp_Wrx + head * headdim, NH);

        // Wx @ x for input
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            headdim, steps * batch_size, headdim,
            &alpha,
            Wx_head, headdim,
            x + head * headdim, NH,
            &beta_zero,
            tmp_Wxx + head * headdim, NH);
    }

    // =========================================================================
    // Main recurrence loop with reset gate
    // =========================================================================
    // h[0] = h0 (already initialized by caller)
    // For each timestep t:
    //   r = sigmoid(Wr @ x[t])         // Reset gate
    //   h_gated = r * h[t]             // Gate the history
    //   tmp_Rh = R @ h_gated           // Recurrence on gated history
    //   h[t+1] = y[t] = activation(tmp_Rh + Wx @ x[t] + b)

    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BNH;           // Current hidden state
        T* h_next = h + (t + 1) * BNH;        // Next hidden state
        const T* Wrx_t = tmp_Wrx + t * BNH;
        const T* Wxx_t = tmp_Wxx + t * BNH;
        T* y_t = y + t * BNH;
        T* r_t = training ? (r_gate + t * BNH) : nullptr;
        T* pre_act_t = training ? (pre_act + t * BNH) : nullptr;

        // Step 1: Apply reset gate
        // r = sigmoid(Wrx), h_gated = r * h
        const int threads = 256;
        const int blocks = (BNH + threads - 1) / threads;

        if (training) {
            ResetGateKernel<T><<<blocks, threads, 0, stream>>>(
                BNH, Wrx_t, h_t, r_t, tmp_h_gated);
        } else {
            // Non-training: don't save r, use tmp buffer for r
            ResetGateKernel<T><<<blocks, threads, 0, stream>>>(
                BNH, Wrx_t, h_t, tmp_h_gated, tmp_h_gated);  // r written to tmp, then overwritten
            // Actually we need a separate tmp for this case, let's just always save r
            // For simplicity, always compute r even if not training
            ResetGateKernel<T><<<blocks, threads, 0, stream>>>(
                BNH, Wrx_t, h_t, tmp_Rh, tmp_h_gated);  // Use tmp_Rh as temp r storage
        }

        // Step 2: Compute R @ h_gated for ALL heads in ONE batched GEMM call
        stridedBatchedGemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            headdim, batch_size, headdim,  // M, N, K
            &alpha,
            R, headdim, strideW,           // A=R, lda, strideA
            tmp_h_gated, NH, strideH,      // B=h_gated, ldb=NH, strideB=headdim
            &beta_zero,
            tmp_Rh, NH, strideH,           // C, ldc=NH, strideC=headdim
            nheads);

        // Step 3: Fused activation kernel
        if (activation == 0) {
            FusedActivationKernel<T, 0><<<blocks, threads, 0, stream>>>(
                batch_size, nheads, headdim, tmp_Rh, Wxx_t, b, y_t, pre_act_t);
        } else if (activation == 1) {
            FusedActivationKernel<T, 1><<<blocks, threads, 0, stream>>>(
                batch_size, nheads, headdim, tmp_Rh, Wxx_t, b, y_t, pre_act_t);
        } else {
            FusedActivationKernel<T, 2><<<blocks, threads, 0, stream>>>(
                batch_size, nheads, headdim, tmp_Rh, Wxx_t, b, y_t, pre_act_t);
        }

        // Step 4: Copy y[t] to h[t+1] for next timestep
        cudaMemcpyAsync(h_next, y_t, BNH * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;

}  // namespace multihead_elman
}  // namespace v0
}  // namespace haste
