// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 6 FAST: Log-Space Triple R with cuBLAS matmuls
//
// Key optimization: Use cuBLAS gemm for R @ h instead of custom log-space kernel.
// Only store hidden state in log-space, compute everything in linear.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cfloat>

#include "haste/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

constexpr float LOG_ZERO = -1e10f;
constexpr float LOG_EPS = 1e-10f;

__device__ __forceinline__ void to_log_space(float x, float& log_x, float& sign_x) {
    sign_x = (x >= 0) ? 1.0f : -1.0f;
    float abs_x = fabsf(x);
    log_x = (abs_x > LOG_EPS) ? logf(abs_x) : LOG_ZERO;
}

__device__ __forceinline__ float from_log_space(float log_x, float sign_x) {
    if (log_x <= LOG_ZERO + 1.0f) return 0.0f;
    return sign_x * expf(log_x);
}

// Convert log/sign to linear
template<typename T>
__global__ void LogToLinearKernel(
    const int n,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    T* __restrict__ h_linear) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float log_val = static_cast<float>(log_h[idx]);
        float sign_val = static_cast<float>(sign_h[idx]);
        h_linear[idx] = static_cast<T>(from_log_space(log_val, sign_val));
    }
}

// Convert linear to log/sign
template<typename T>
__global__ void LinearToLogKernel(
    const int n,
    const T* __restrict__ x_linear,
    T* __restrict__ log_x,
    T* __restrict__ sign_x) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(x_linear[idx]);
        float log_val, sign_val;
        to_log_space(x, log_val, sign_val);
        log_x[idx] = static_cast<T>(log_val);
        sign_x[idx] = static_cast<T>(sign_val);
    }
}

// Fused gated update: h_new = (1-delta) * h_prev + delta * tanh(v)
// All inputs in linear, output converted to log
template<typename T>
__global__ void FastTripleRGatedUpdateKernel(
    const int batch_size,
    const int dim,
    const T* __restrict__ h_prev_linear,  // h_{t-1} in linear
    const T* __restrict__ Rx_x,           // R_x @ x (linear)
    const T* __restrict__ Rh_h,           // R_h @ h_{t-1} (linear)
    const T* __restrict__ Wdelta_x,       // W_delta @ x (linear)
    const T* __restrict__ Rdelta_h,       // R_delta @ h_{t-1} (linear)
    const T* __restrict__ b,
    const T* __restrict__ b_delta,
    T* __restrict__ log_h_out,
    T* __restrict__ sign_h_out,
    T* __restrict__ h_out_linear,         // Also output linear for W_out
    T* __restrict__ v_cache,
    T* __restrict__ delta_cache) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // v_t = R_x @ x + R_h @ h_{t-1} + b
        float v_f = static_cast<float>(Rx_x[idx]) +
                    static_cast<float>(Rh_h[idx]) +
                    static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v_f);
        float candidate = tanhf(v_f);

        // delta = sigmoid(W_delta @ x + R_delta @ h_{t-1} + b_delta)
        float delta_in = static_cast<float>(Wdelta_x[idx]) +
                         static_cast<float>(Rdelta_h[idx]) +
                         static_cast<float>(b_delta[d]);
        float delta_f = 1.0f / (1.0f + expf(-delta_in));
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta_f);

        // Gated update
        float h_prev = static_cast<float>(h_prev_linear[idx]);
        float h_new = (1.0f - delta_f) * h_prev + delta_f * candidate;

        // Store in log space
        float log_h_new, sign_h_new;
        to_log_space(h_new, log_h_new, sign_h_new);
        log_h_out[idx] = static_cast<T>(log_h_new);
        sign_h_out[idx] = static_cast<T>(sign_h_new);

        // Also store linear for W_out
        h_out_linear[idx] = static_cast<T>(h_new);
    }
}

// Selective output: compete(h) * silu(W_out @ h)
template<typename T>
__global__ void FastSelectiveOutputKernel(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h_linear,       // h in linear
    const T* __restrict__ w_out_h,        // W_out @ h
    T* __restrict__ output,
    T* __restrict__ compete_cache) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // Find max for softmax stability
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float h_val = static_cast<float>(h_linear[base + i]);
        max_val = fmaxf(max_val, h_val);
    }
    smem[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Compute exp sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float h_val = static_cast<float>(h_linear[base + i]);
        sum += expf(h_val - max_val);
    }
    float* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = sum_smem[0];

    // Compute output
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float h_val = static_cast<float>(h_linear[base + i]);
        float compete = expf(h_val - max_val) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w_out = static_cast<float>(w_out_h[base + i]);
        float silu = w_out / (1.0f + expf(-w_out));
        output[base + i] = static_cast<T>(compete * silu);
    }
}

}  // anonymous namespace


namespace haste {
namespace v0 {
namespace elman_ladder {

// Fast Triple R Forward - uses cuBLAS for all matmuls
template<typename T>
class LogSpaceTripleRFastForward {
public:
    LogSpaceTripleRFastForward(
        bool training,
        int batch_size,
        int dim,
        int n_groups,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream)
        : training_(training),
          batch_size_(batch_size),
          dim_(dim),
          n_groups_(n_groups),
          blas_handle_(blas_handle),
          stream_(stream) {}

    void Run(
        int steps,
        const T* R_h,
        const T* R_x,
        const T* R_delta,
        const T* W_delta,
        const T* W_out,
        const T* b,
        const T* b_delta,
        const T* x,
        T* log_h,
        T* sign_h,
        T* output,
        T* v,
        T* delta_cache,
        T* compete_cache) {

        static const T alpha = static_cast<T>(1.0);
        static const T beta_zero = static_cast<T>(0.0);

        const int BD = batch_size_ * dim_;
        const int block_size = 256;
        const int num_blocks = (BD + block_size - 1) / block_size;
        const int group_size = dim_ / n_groups_;

        // Workspace - all linear
        T *h_linear, *h_linear_out;
        T *Rx_x, *Rh_h, *Wdelta_x, *Rdelta_h, *w_out_h;

        cudaMalloc(&h_linear, BD * sizeof(T));
        cudaMalloc(&h_linear_out, BD * sizeof(T));
        cudaMalloc(&Rx_x, BD * sizeof(T));
        cudaMalloc(&Rh_h, BD * sizeof(T));
        cudaMalloc(&Wdelta_x, BD * sizeof(T));
        cudaMalloc(&Rdelta_h, BD * sizeof(T));
        cudaMalloc(&w_out_h, BD * sizeof(T));

        for (int t = 0; t < steps; ++t) {
            const T* x_t = x + t * BD;
            const T* log_h_prev = log_h + t * BD;
            const T* sign_h_prev = sign_h + t * BD;
            T* log_h_t = log_h + (t + 1) * BD;
            T* sign_h_t = sign_h + (t + 1) * BD;
            T* out_t = output + t * BD;
            T* v_t = training_ ? (v + t * BD) : nullptr;
            T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
            T* compete_t = training_ ? (compete_cache + t * BD) : nullptr;

            // 1. Convert log_h_prev to linear
            LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                BD, log_h_prev, sign_h_prev, h_linear);

            // 2. cuBLAS gemm for all R @ vectors (FAST!)
            // R_x @ x_t
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha, R_x, dim_, x_t, dim_, &beta_zero, Rx_x, dim_);

            // R_h @ h_linear
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha, R_h, dim_, h_linear, dim_, &beta_zero, Rh_h, dim_);

            // W_delta @ x_t
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha, W_delta, dim_, x_t, dim_, &beta_zero, Wdelta_x, dim_);

            // R_delta @ h_linear
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha, R_delta, dim_, h_linear, dim_, &beta_zero, Rdelta_h, dim_);

            // 3. Fused gated update (outputs both log and linear)
            FastTripleRGatedUpdateKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                batch_size_, dim_, h_linear,
                Rx_x, Rh_h, Wdelta_x, Rdelta_h,
                b, b_delta,
                log_h_t, sign_h_t, h_linear_out, v_t, delta_t);

            // 4. W_out @ h_linear_out
            blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                dim_, batch_size_, dim_, &alpha, W_out, dim_, h_linear_out, dim_, &beta_zero, w_out_h, dim_);

            // 5. Selective output
            dim3 out_grid(batch_size_, n_groups_);
            int out_smem_size = 2 * block_size * sizeof(float);
            FastSelectiveOutputKernel<T><<<out_grid, block_size, out_smem_size, stream_>>>(
                batch_size_, dim_, n_groups_, group_size,
                h_linear_out, w_out_h, out_t, compete_t);

            // Copy h_linear_out to h_linear for next iteration
            cudaMemcpyAsync(h_linear, h_linear_out, BD * sizeof(T), cudaMemcpyDeviceToDevice, stream_);
        }

        // Free workspace
        cudaFree(h_linear);
        cudaFree(h_linear_out);
        cudaFree(Rx_x);
        cudaFree(Rh_h);
        cudaFree(Wdelta_x);
        cudaFree(Rdelta_h);
        cudaFree(w_out_h);
    }

private:
    bool training_;
    int batch_size_;
    int dim_;
    int n_groups_;
    cublasHandle_t blas_handle_;
    cudaStream_t stream_;
};

// Explicit instantiations
template class LogSpaceTripleRFastForward<float>;
template class LogSpaceTripleRFastForward<__half>;
template class LogSpaceTripleRFastForward<__nv_bfloat16>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace haste
