// Copyright 2024 Erik Garrison. Apache 2.0 License.
// ElmanNeuralMemory: NTM-inspired external memory bank
//
// Architecture:
//   read_key = W_read @ h
//   read_weights = softmax(M @ read_key / sqrt(memory_dim))
//   memory_read = read_weights @ M
//   candidate = tanh(R @ h + W_x @ x + W_mem @ memory_read + b)
//   delta = sigmoid(W_delta @ x + W_delta_mem @ memory_read + b_delta)
//   h_new = (1 - delta) * h + delta * candidate

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cmath>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// ============================================================================
// Softmax kernel for attention weights
// ============================================================================

template<typename T>
__global__
void SoftmaxKernel(
    const int batch_size,
    const int num_slots,
    const float scale,
    T* __restrict__ weights    // [B, num_slots] - in-place
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    T* row = weights + batch_idx * num_slots;

    // Find max for numerical stability (sequential per thread)
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < num_slots; i += blockDim.x) {
        float val = static_cast<float>(row[i]) * scale;
        if (val > max_val) max_val = val;
    }

    // Warp reduction for max using shuffle (works for all float values)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset);
        max_val = fmaxf(max_val, other);
    }

    // Broadcast max to all threads
    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_slots; i += blockDim.x) {
        float val = expf(static_cast<float>(row[i]) * scale - max_val);
        row[i] = static_cast<T>(val);
        sum += val;
    }

    // Warp reduction for sum using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Broadcast sum to all threads
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    sum = shared_sum;

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int i = threadIdx.x; i < num_slots; i += blockDim.x) {
        row[i] = static_cast<T>(static_cast<float>(row[i]) * inv_sum);
    }
}

// Simpler softmax for small num_slots (single thread per batch)
template<typename T>
__global__
void SoftmaxKernelSimple(
    const int batch_size,
    const int num_slots,
    const float scale,
    T* __restrict__ weights
) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    T* row = weights + batch_idx * num_slots;

    // Find max
    float max_val = -1e30f;
    for (int i = 0; i < num_slots; ++i) {
        float val = static_cast<float>(row[i]) * scale;
        if (val > max_val) max_val = val;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < num_slots; ++i) {
        float val = expf(static_cast<float>(row[i]) * scale - max_val);
        row[i] = static_cast<T>(val);
        sum += val;
    }

    // Normalize
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int i = 0; i < num_slots; ++i) {
        row[i] = static_cast<T>(static_cast<float>(row[i]) * inv_sum);
    }
}

// ============================================================================
// Pointwise kernel for RNN with memory
// ============================================================================

template<typename T>
__global__
void NeuralMemoryPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ Rh,           // [B, D]
    const T* __restrict__ Wx,           // [B, D]
    const T* __restrict__ Wmem,         // [B, D] - W_mem @ memory_read
    const T* __restrict__ b,            // [D]
    const T* __restrict__ Wdelta_x,     // [B, D] - W_delta @ x
    const T* __restrict__ Wdelta_mem,   // [B, D] - W_delta_mem @ memory_read
    const T* __restrict__ b_delta,      // [D]
    const T* __restrict__ h_prev,       // [B, D]
    T* __restrict__ h_next,             // [B, D]
    T* __restrict__ v,                  // [B, D] - pre-activation
    T* __restrict__ delta_cache         // [B, D]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int d_idx = idx % D;

    // Candidate: tanh(R @ h + W_x @ x + W_mem @ memory_read + b)
    float raw = static_cast<float>(Rh[idx]) +
                static_cast<float>(Wx[idx]) +
                static_cast<float>(Wmem[idx]) +
                static_cast<float>(b[d_idx]);
    float candidate = tanhf(raw);

    // Delta: sigmoid(W_delta @ x + W_delta_mem @ memory_read + b_delta)
    float delta_raw = static_cast<float>(Wdelta_x[idx]) +
                      static_cast<float>(Wdelta_mem[idx]) +
                      static_cast<float>(b_delta[d_idx]);
    float delta = 1.0f / (1.0f + expf(-delta_raw));

    float h_p = static_cast<float>(h_prev[idx]);
    float h_new = (1.0f - delta) * h_p + delta * candidate;

    h_next[idx] = static_cast<T>(h_new);
    v[idx] = static_cast<T>(raw);
    delta_cache[idx] = static_cast<T>(delta);
}

template<typename T>
__global__
void NeuralMemoryBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,
    const T* __restrict__ dh_recurrent,
    const T* __restrict__ v,
    const T* __restrict__ delta_cache,
    const T* __restrict__ h_prev,
    T* __restrict__ d_raw,              // [B, D] - for candidate pathway
    T* __restrict__ d_delta_raw,        // [B, D] - for delta pathway
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

// Softmax backward
template<typename T>
__global__
void SoftmaxBackwardKernel(
    const int batch_size,
    const int num_slots,
    const float scale,
    const T* __restrict__ softmax_out,  // [B, num_slots]
    const T* __restrict__ grad_out,     // [B, num_slots]
    T* __restrict__ grad_in             // [B, num_slots]
) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const T* s = softmax_out + batch_idx * num_slots;
    const T* dout = grad_out + batch_idx * num_slots;
    T* din = grad_in + batch_idx * num_slots;

    // Compute dot product s . dout
    float dot = 0.0f;
    for (int i = 0; i < num_slots; ++i) {
        dot += static_cast<float>(s[i]) * static_cast<float>(dout[i]);
    }

    // d/d(input) = s * (dout - dot) * scale
    for (int i = 0; i < num_slots; ++i) {
        float grad = static_cast<float>(s[i]) * (static_cast<float>(dout[i]) - dot) * scale;
        din[i] = static_cast<T>(grad);
    }
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
namespace elman_neural_memory {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    int num_memory_slots;
    int memory_dim;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int num_memory_slots,
    const int memory_dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->num_memory_slots = num_memory_slots;
    data_->memory_dim = memory_dim;
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
    const T* R,
    const T* W_x,
    const T* b,
    const T* W_delta,
    const T* b_delta,
    const T* M,
    const T* W_read,
    const T* W_mem,
    const T* W_delta_mem,
    const T* x,
    T* h,
    T* v,
    T* delta_cache,
    T* read_weights_cache,
    T* memory_read_cache,
    T* tmp_Rh,
    T* tmp_Wx,
    T* tmp_read_key,
    T* tmp_read_weights,
    T* tmp_memory_read
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int D = data_->hidden_size;
    const int num_slots = data_->num_memory_slots;
    const int mem_dim = data_->memory_dim;
    const int BD = batch_size * D;
    const int BI = batch_size * input_size;
    const int BS = batch_size * num_slots;
    const int BM = batch_size * mem_dim;

    const float scale = 1.0f / sqrtf(static_cast<float>(mem_dim));

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
        T* rw_t = read_weights_cache + t * BS;
        T* mr_t = memory_read_cache + t * BM;
        const T* x_t = x + t * BI;

        // Step 1: Compute read key from h
        // read_key = W_read @ h_t  [mem_dim, D] @ [D, B]^T -> [B, mem_dim]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            mem_dim, batch_size, D,
            &alpha,
            W_read, D,
            h_t, D,
            &beta,
            tmp_read_key, mem_dim);

        // Step 2: Compute attention weights
        // weights = M @ read_key  [num_slots, mem_dim] @ [mem_dim, B]^T -> [B, num_slots]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            num_slots, batch_size, mem_dim,
            &alpha,
            M, mem_dim,
            tmp_read_key, mem_dim,
            &beta,
            tmp_read_weights, num_slots);

        // Step 3: Softmax
        const int softmax_threads = 256;
        const int softmax_blocks = (batch_size + softmax_threads - 1) / softmax_threads;
        SoftmaxKernelSimple<T><<<softmax_blocks, softmax_threads, 0, stream>>>(
            batch_size, num_slots, scale, tmp_read_weights);

        // Save read weights for backward
        cudaMemcpyAsync(rw_t, tmp_read_weights, BS * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // Step 4: Read from memory
        // memory_read = read_weights @ M  [B, num_slots] @ [num_slots, mem_dim] -> [B, mem_dim]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            mem_dim, batch_size, num_slots,
            &alpha,
            M, mem_dim,
            tmp_read_weights, num_slots,
            &beta,
            tmp_memory_read, mem_dim);

        // Save memory read for backward
        cudaMemcpyAsync(mr_t, tmp_memory_read, BM * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // Step 5: RNN update
        // R @ h
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            h_t, D,
            &beta,
            tmp_Rh, D);

        // W_x @ x
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            W_x, input_size,
            x_t, input_size,
            &beta,
            tmp_Wx, D);

        // W_mem @ memory_read -> v_t (temporary)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, mem_dim,
            &alpha,
            W_mem, mem_dim,
            tmp_memory_read, mem_dim,
            &beta,
            v_t, D);  // Using v_t as temp

        // W_delta @ x -> delta_t (temporary)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, input_size,
            &alpha,
            W_delta, input_size,
            x_t, input_size,
            &beta,
            delta_t, D);  // Using delta_t as temp for W_delta @ x

        // W_delta_mem @ memory_read - need another temp, use tmp_read_key
        // Actually we need [B, D] but tmp_read_key is [B, mem_dim]
        // Let's allocate this in-kernel or reuse carefully
        // For now, accumulate in delta_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, batch_size, mem_dim,
            &alpha,
            W_delta_mem, mem_dim,
            tmp_memory_read, mem_dim,
            &beta_one,  // Add to existing (beta_one = 1.0)
            delta_t, D);

        // Now run pointwise kernel
        // v_t has W_mem @ memory_read
        // delta_t has W_delta @ x + W_delta_mem @ memory_read
        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        NeuralMemoryPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, tmp_Rh, tmp_Wx, v_t, b, delta_t, delta_t, b_delta,
            h_t, h_next, v_t, delta_t);
        // Note: delta_t input is reused for both Wdelta_x and Wdelta_mem (summed)
        // and v_t input is W_mem @ memory_read
        // Kernel overwrites v_t with pre-activation and delta_t with delta
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    int num_memory_slots;
    int memory_dim;
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int num_memory_slots,
    const int memory_dim,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->num_memory_slots = num_memory_slots;
    data_->memory_dim = memory_dim;
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
    const T* W_x,
    const T* W_delta,
    const T* M,
    const T* W_read,
    const T* W_mem,
    const T* W_delta_mem,
    const T* x,
    const T* h,
    const T* v,
    const T* delta_cache,
    const T* read_weights_cache,
    const T* memory_read_cache,
    const T* dh_new,
    T* dx,
    T* dR,
    T* dW_x,
    T* db,
    T* dW_delta,
    T* db_delta,
    T* dM,
    T* dW_read,
    T* dW_mem,
    T* dW_delta_mem,
    T* dh,
    T* tmp_Rh,
    T* tmp_Wx,
    T* tmp_d_memory_read,
    T* tmp_d_read_weights
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int D = data_->hidden_size;
    const int num_slots = data_->num_memory_slots;
    const int mem_dim = data_->memory_dim;
    const int BD = batch_size * D;
    const int BI = batch_size * input_size;
    const int BS = batch_size * num_slots;
    const int BM = batch_size * mem_dim;

    const float scale = 1.0f / sqrtf(static_cast<float>(mem_dim));

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
        const T* rw_t = read_weights_cache + t * BS;
        const T* mr_t = memory_read_cache + t * BM;
        const T* x_t = x + t * BI;
        T* dx_t = dx + t * BI;
        const T* dh_out = dh_new + (t + 1) * BD;

        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        // Pointwise backward - d_raw and d_delta_raw
        NeuralMemoryBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D, dh_out, dh_recurrent, v_t, delta_t, h_t,
            tmp_Rh, tmp_Wx, dh);  // tmp_Rh = d_raw, tmp_Wx = d_delta_raw

        // Accumulate bias gradients
        const int bias_threads = 256;
        const int bias_blocks = (D + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_Rh, db);
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, D, tmp_Wx, db_delta);

        // dR += h_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, D, batch_size,
            &alpha,
            h_t, D,
            tmp_Rh, D,
            &beta_one,
            dR, D);

        // dW_x += x_t^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_Rh, D,
            &beta_one,
            dW_x, input_size);

        // dW_mem += memory_read^T @ d_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            mem_dim, D, batch_size,
            &alpha,
            mr_t, mem_dim,
            tmp_Rh, D,
            &beta_one,
            dW_mem, mem_dim);

        // dW_delta += x_t^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, D, batch_size,
            &alpha,
            x_t, input_size,
            tmp_Wx, D,
            &beta_one,
            dW_delta, input_size);

        // dW_delta_mem += memory_read^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            mem_dim, D, batch_size,
            &alpha,
            mr_t, mem_dim,
            tmp_Wx, D,
            &beta_one,
            dW_delta_mem, mem_dim);

        // d_memory_read = W_mem^T @ d_raw + W_delta_mem^T @ d_delta_raw
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            mem_dim, batch_size, D,
            &alpha,
            W_mem, mem_dim,
            tmp_Rh, D,
            &beta,
            tmp_d_memory_read, mem_dim);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            mem_dim, batch_size, D,
            &alpha,
            W_delta_mem, mem_dim,
            tmp_Wx, D,
            &beta_one,
            tmp_d_memory_read, mem_dim);

        // Backward through memory read: d_read_weights = d_memory_read @ M^T
        // d_memory_read: [B, mem_dim], M: [num_slots, mem_dim]
        // d_read_weights = d_memory_read @ M^T -> [B, num_slots]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            num_slots, batch_size, mem_dim,
            &alpha,
            M, mem_dim,
            tmp_d_memory_read, mem_dim,
            &beta,
            tmp_d_read_weights, num_slots);

        // dM += read_weights^T @ d_memory_read (for each batch, then sum)
        // This is tricky - need to do batched outer product and sum
        // For now, approximate: dM += sum over batch of outer(rw, d_mr)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            mem_dim, num_slots, batch_size,
            &alpha,
            tmp_d_memory_read, mem_dim,
            rw_t, num_slots,
            &beta_one,
            dM, mem_dim);

        // Backward through softmax
        const int sm_threads = 256;
        const int sm_blocks = (batch_size + sm_threads - 1) / sm_threads;
        SoftmaxBackwardKernel<T><<<sm_blocks, sm_threads, 0, stream>>>(
            batch_size, num_slots, scale, rw_t, tmp_d_read_weights, tmp_d_read_weights);

        // d_read_key = M^T @ d_softmax_in
        // Reuse tmp_d_memory_read for d_read_key since it's [B, mem_dim]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            mem_dim, batch_size, num_slots,
            &alpha,
            M, mem_dim,
            tmp_d_read_weights, num_slots,
            &beta,
            tmp_d_memory_read, mem_dim);  // Now holds d_read_key

        // dM += read_key @ d_softmax_in^T (contribution from attention)
        // We need to save read_key... for now skip this gradient path
        // TODO: properly compute this

        // dW_read += h_t^T @ d_read_key
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, mem_dim, batch_size,
            &alpha,
            h_t, D,
            tmp_d_memory_read, mem_dim,
            &beta_one,
            dW_read, D);

        // dx contribution from input
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            W_x, input_size,
            tmp_Rh, D,
            &beta,
            dx_t, input_size);

        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, D,
            &alpha,
            W_delta, input_size,
            tmp_Wx, D,
            &beta_one,
            dx_t, input_size);

        // dh += R @ d_raw (plus W_read contribution)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, D,
            &alpha,
            R, D,
            tmp_Rh, D,
            &beta_one,
            dh, D);

        // dh += W_read^T @ d_read_key
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, mem_dim,
            &alpha,
            W_read, D,
            tmp_d_memory_read, mem_dim,
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

}  // namespace elman_neural_memory
}  // namespace v0
}  // namespace haste
