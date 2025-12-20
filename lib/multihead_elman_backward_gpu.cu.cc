// Copyright 2024 Erik Garrison. Apache 2.0 License.
// Multi-head Elman RNN backward pass.
//
// Backward through: h_new = activation(R @ h + Wx @ x + b)
//
// Gradient computation:
//   dpre = dh_new * d_activation(pre)
//   dR += dpre @ h.T  (per head, accumulated)
//   dWx += dpre @ x.T  (per head, accumulated)
//   db += dpre  (per head, accumulated)
//   dh = R.T @ dpre  (for previous timestep)
//   dx = Wx.T @ dpre

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

// Fused kernel: Compute dpre = (dy + dh) * d_activation(pre)
// This combines the gradient from output (dy) and hidden state (dh_from_next)
template<typename T, int Activation>
__global__
void FusedActivationBackwardKernel(
    const int size,
    const T* __restrict__ dy,       // [B, nheads, headdim]
    const T* __restrict__ dh,       // [B, nheads, headdim] - gradient from next timestep
    const T* __restrict__ pre_act,  // [B, nheads, headdim]
    T* __restrict__ dpre            // [B, nheads, headdim]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    const T pre = pre_act[idx];
    const T dh_total = dy[idx] + dh[idx];  // Combine gradients
    T d_act;

    if (Activation == 0) {
        // Softsign
        d_act = d_softsign(pre);
    } else if (Activation == 1) {
        // Tanh residual
        d_act = d_tanh_residual(pre);
    } else {
        // Tanh
        const T tanh_pre = tanh(pre);
        d_act = static_cast<T>(1.0) - tanh_pre * tanh_pre;
    }

    dpre[idx] = dh_total * d_act;
}

// Kernel: Accumulate bias gradient (sum over batch dimension)
template<typename T>
__global__
void AccumulateBiasGradKernel(
    const int batch_size,
    const int nheads,
    const int headdim,
    const T* __restrict__ dpre,  // [B, nheads, headdim]
    T* __restrict__ db           // [nheads, headdim]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = nheads * headdim;

    if (idx >= total) return;

    T sum = static_cast<T>(0.0);
    for (int b = 0; b < batch_size; ++b) {
        sum += dpre[b * total + idx];
    }

    // Atomic add for accumulation across timesteps
    atomicAdd(&db[idx], sum);
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
    const T* R,         // [nheads, headdim, headdim]
    const T* Wx,        // [nheads, headdim, headdim]
    const T* x,         // [T, B, nheads, headdim]
    const T* h,         // [T+1, B, nheads, headdim] - h[0] is initial state
    const T* pre_act,   // [T, B, nheads, headdim]
    const T* dy,        // [T, B, nheads, headdim]
    T* dx,              // [T, B, nheads, headdim]
    T* dR,              // [nheads, headdim, headdim]
    T* dWx,             // [nheads, headdim, headdim]
    T* db,              // [nheads, headdim]
    T* dh0,             // [B, nheads, headdim]
    T* tmp_dpre,        // [B, nheads, headdim]
    T* tmp_dh           // [B, nheads, headdim]
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
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

    // Initialize dh to zeros (will accumulate gradient from future timesteps)
    cudaMemsetAsync(tmp_dh, 0, BNH * sizeof(T), stream);

    // Process timesteps in reverse order
    for (int t = steps - 1; t >= 0; --t) {
        const T* dy_t = dy + t * BNH;
        const T* pre_act_t = pre_act + t * BNH;
        const T* x_t = x + t * BNH;
        const T* h_t = h + t * BNH;  // h[t] is the state BEFORE timestep t
        T* dx_t = dx + t * BNH;

        // Step 1: Compute dpre = (dy + dh_from_next) * d_activation(pre)
        const int threads = 256;
        const int blocks = (BNH + threads - 1) / threads;

        // Fused kernel: dpre = (dy + dh_from_next) * d_activation(pre)
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

        // Step 2: Accumulate gradients for R, Wx, b

        // For each head: dR[head] += dpre[head] @ h[head].T
        // dWx[head] += dpre[head] @ x[head].T
        for (int head = 0; head < nheads; ++head) {
            const T* dpre_head = tmp_dpre + head * headdim;
            const T* h_head = h_t + head * headdim;
            const T* x_head = x_t + head * headdim;
            T* dR_head = dR + head * headdim * headdim;
            T* dWx_head = dWx + head * headdim * headdim;

            // dR[head] += dpre[batch, head, :].T @ h[batch, head, :]
            // This is: headdim x B @ B x headdim = headdim x headdim
            // Using: dpre as [B, headdim] with stride NH, h as [B, headdim] with stride NH
            blas<T>::gemm(blas_handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                headdim, headdim, batch_size,
                &alpha,
                dpre_head, NH,  // [B, headdim] with stride NH
                h_head, NH,
                &beta_one,  // Accumulate
                dR_head, headdim);

            // dWx[head] += dpre.T @ x
            blas<T>::gemm(blas_handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                headdim, headdim, batch_size,
                &alpha,
                dpre_head, NH,
                x_head, NH,
                &beta_one,
                dWx_head, headdim);
        }

        // Accumulate bias gradient
        AccumulateBiasGradKernel<T><<<(NH + 255) / 256, 256, 0, stream>>>(
            batch_size, nheads, headdim, tmp_dpre, db);

        // Step 3: Compute gradients for previous timestep
        // dx[t] = Wx.T @ dpre
        // dh[t-1] = R.T @ dpre

        // Clear tmp_dh for accumulating dh
        cudaMemsetAsync(tmp_dh, 0, BNH * sizeof(T), stream);

        for (int head = 0; head < nheads; ++head) {
            const T* dpre_head = tmp_dpre + head * headdim;
            const T* R_head = R + head * headdim * headdim;
            const T* Wx_head = Wx + head * headdim * headdim;
            T* dx_head = dx_t + head * headdim;
            T* dh_head = tmp_dh + head * headdim;

            // dx[t, head] = Wx.T @ dpre
            blas<T>::gemm(blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                headdim, batch_size, headdim,
                &alpha,
                Wx_head, headdim,
                dpre_head, NH,
                &beta,
                dx_head, NH);

            // dh = R.T @ dpre (for next iteration, going backwards)
            blas<T>::gemm(blas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                headdim, batch_size, headdim,
                &alpha,
                R_head, headdim,
                dpre_head, NH,
                &beta_one,  // Accumulate across heads (though they're independent)
                dh_head, NH);
        }
    }

    // Copy final dh to dh0 (gradient w.r.t. initial hidden state)
    cudaMemcpyAsync(dh0, tmp_dh, BNH * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace multihead_elman
}  // namespace v0
}  // namespace haste
