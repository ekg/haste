// Copyright 2024 Erik Garrison. Apache 2.0 License.
// Fused Elman RNN forward pass.
//
// Architecture per timestep:
//   hidden = tanh(W1 @ [x_t, h])    -- [B, H] where H = expansion * D
//   h_new = W2 @ hidden             -- [B, D]
//   gate = silu(Wgx @ x_t + Wgh @ h_new + bias)  -- [B, D]
//   out = h_new * gate              -- [B, D]
//
// Strategy:
//   1. Pre-compute Wgx @ x for ALL timesteps (one big GEMM)
//   2. For each timestep:
//      a. Two GEMMs: W1_x @ x_t + W1_h @ h -> tmp_combined
//      b. Fused kernel: hidden = tanh(tmp_combined), save for backward
//      c. GEMM: W2 @ hidden -> h_new
//      d. GEMM: Wgh @ h_new -> tmp_gate
//      e. Fused kernel: out = h_new * silu(gate_x[t] + tmp_gate + bias)

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// Device helper for exp that handles __half
template<typename T>
__device__ __forceinline__
T device_exp(const T x) {
    return exp(x);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
template<>
__device__ __forceinline__
__half device_exp(const __half x) {
    return hexp(x);
}
#endif

// Kernel 1: Fused add + tanh
// Computes: out = tanh(a + b)
template<typename T>
__global__
void AddTanhKernel(
    const int size,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = tanh(a[idx] + b[idx]);
}

// Kernel 1b: Compute tanh activation (standalone)
// Input: pre [B, H] - pre-activation values
// Output: out [B, H] = tanh(pre)
template<typename T>
__global__
void TanhKernel(
    const int size,
    const T* __restrict__ pre,
    T* __restrict__ out
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = tanh(pre[idx]);
}

// Kernel 2: Fused SiLU gate and output multiplication
// Computes: out = h_new * silu(gate_x + gate_h + bias)
// Also saves gate_logits and gate for backward pass
template<typename T, bool Training>
__global__
void SiluGateOutputKernel(
    const int batch_size,
    const int output_dim,
    const T* __restrict__ h_new,       // [B, D]
    const T* __restrict__ gate_x,      // [B, D] - pre-computed Wgx @ x
    const T* __restrict__ gate_h,      // [B, D] - Wgh @ h_new
    const T* __restrict__ bias,        // [D]
    T* __restrict__ out,               // [B, D] - output (also next h)
    T* __restrict__ v                  // [B, D*3] - saved activations for backward
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;  // output dim
    const int col = blockDim.y * blockIdx.y + threadIdx.y;  // batch

    if (row >= output_dim || col >= batch_size)
        return;

    const int idx = col * output_dim + row;

    // Compute gate_logits = gate_x + gate_h + bias
    const T h_new_val = h_new[idx];
    const T gate_logit = gate_x[idx] + gate_h[idx] + bias[row];

    // SiLU: gate = gate_logit * sigmoid(gate_logit)
    const T sig = static_cast<T>(1.0) / (static_cast<T>(1.0) + device_exp(-gate_logit));
    const T gate = gate_logit * sig;

    // Output: out = h_new * gate
    out[idx] = h_new_val * gate;

    // Save activations for backward pass
    if (Training) {
        v[col * output_dim * 3 + row] = h_new_val;                  // h_new
        v[col * output_dim * 3 + output_dim + row] = gate_logit;    // gate_logit (for d_silu)
        v[col * output_dim * 3 + output_dim * 2 + row] = gate;      // gate
    }
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace elman {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;    // H = expansion * D
    int output_size;    // D
    cublasHandle_t blas_handle;
    cudaStream_t stream[3];
    cudaEvent_t event[2];
    cudaStream_t sync_stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int output_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->output_size = output_size;
    data_->blas_handle = blas_handle;
    data_->sync_stream = stream;
    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaStreamCreate(&data_->stream[2]);
    cudaEventCreateWithFlags(&data_->event[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&data_->event[1], cudaEventDisableTiming);
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    if (data_->sync_stream) {
        for (int i = 0; i < 3; ++i) {
            cudaEvent_t event;
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
            cudaEventRecord(event, data_->stream[i]);
            cudaStreamWaitEvent(data_->sync_stream, event, 0);
            cudaEventDestroy(event);
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            cudaStreamSynchronize(data_->stream[i]);
        }
    }
    cudaEventDestroy(data_->event[0]);
    cudaEventDestroy(data_->event[1]);
    cudaStreamDestroy(data_->stream[0]);
    cudaStreamDestroy(data_->stream[1]);
    cudaStreamDestroy(data_->stream[2]);
    delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* W1,        // [H, D+D] - first D columns for input, second D for recurrent
    const T* W2,        // [D, H] - output projection
    const T* Wgx,       // [D, D] - gate input weights
    const T* Wgh,       // [D, D] - gate recurrent weights
    const T* bias,      // [D] - gate bias
    const T* x,         // [T*B, D] - input sequence (time-major, flattened)
    T* h,               // [(T+1)*B, D] - hidden states (includes initial h0)
    T* v,               // [T*B, D*3] - saved activations for backward
    T* hidden,          // [T*B, H] - tanh outputs (saved for backward)
    T* tmp_Wx1,         // [T*B, H] - W1_x @ x pre-computed for all steps
    T* tmp_gx,          // [T*B, D] - Wgx @ x (pre-computed for all steps)
    T* tmp_gh,          // [B, D] - workspace for gate_h
    T* tmp_Wg2          // [D, H] - pre-computed Wgh @ W2 (fused gate weight)
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const int output_size = data_->output_size;
    const cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    const int BD = batch_size * output_size;
    const int BH = batch_size * hidden_size;

    // Pre-compute all input projections in parallel on different streams

    // Stream 0: Wg2 = Wgh @ W2 (fused gate weight for faster computation)
    // This allows computing gate_h = Wg2 @ hidden instead of Wgh @ (W2 @ hidden)
    cublasSetStream(blas_handle, data_->stream[0]);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        output_size, hidden_size, output_size,
        &alpha,
        Wgh, output_size,
        W2, output_size,
        &beta,
        tmp_Wg2, output_size);

    // Stream 1: W1_x @ x for ALL timesteps (large matrix)
    cublasSetStream(blas_handle, data_->stream[1]);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, steps * batch_size, input_size,
        &alpha,
        W1, hidden_size,  // W1_x is first input_size columns
        x, input_size,
        &beta,
        tmp_Wx1, hidden_size);
    cudaEventRecord(data_->event[0], data_->stream[1]);

    // Stream 2: Wgx @ x for all timesteps (large matrix)
    cublasSetStream(blas_handle, data_->stream[2]);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        output_size, steps * batch_size, input_size,
        &alpha,
        Wgx, output_size,
        x, input_size,
        &beta,
        tmp_gx, output_size);
    cudaEventRecord(data_->event[1], data_->stream[2]);

    // Wait for pre-computations on stream[0] (main loop stream)
    cublasSetStream(blas_handle, data_->stream[0]);
    cudaStreamWaitEvent(data_->stream[0], data_->event[0], 0);  // Wait for W1_x @ x
    cudaStreamWaitEvent(data_->stream[0], data_->event[1], 0);  // Wait for Wgx @ x

    // Process each timestep - single stream for simplicity
    for (int t = 0; t < steps; ++t) {
        const T* h_t = h + t * BD;
        T* h_next = h + (t + 1) * BD;
        T* v_t = v + t * BD * 3;
        T* hidden_t = hidden + t * BH;
        const T* tmp_Wx1_t = tmp_Wx1 + t * BH;  // Pre-computed W1_x @ x_t
        T* tmp_gx_t = tmp_gx + t * BD;

        // Step 1: W1_h @ h_t -> hidden_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_size, batch_size, output_size,
            &alpha,
            W1 + hidden_size * input_size, hidden_size,  // W1_h is after W1_x
            h_t, output_size,
            &beta,  // beta=0, overwrite
            hidden_t, hidden_size);

        // Step 2: Fused: hidden = tanh(W1_h @ h + W1_x @ x)
        const int tanh_threads = 256;
        const int tanh_blocks = (BH + tanh_threads - 1) / tanh_threads;
        AddTanhKernel<T><<<tanh_blocks, tanh_threads, 0, data_->stream[0]>>>(
            BH, hidden_t, tmp_Wx1_t, hidden_t);

        // Step 3: h_new = W2 @ hidden
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            output_size, batch_size, hidden_size,
            &alpha,
            W2, output_size,
            hidden_t, hidden_size,
            &beta,
            h_next, output_size);

        // Step 4: gate_h = Wg2 @ hidden (using pre-computed Wg2 = Wgh @ W2)
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            output_size, batch_size, hidden_size,
            &alpha,
            tmp_Wg2, output_size,
            hidden_t, hidden_size,
            &beta,
            tmp_gh, output_size);

        // Step 5: Fused SiLU gate and output

        const dim3 blockDim(32, 16);
        const dim3 gridDim(
            (output_size + blockDim.x - 1) / blockDim.x,
            (batch_size + blockDim.y - 1) / blockDim.y);

        if (training) {
            SiluGateOutputKernel<T, true><<<gridDim, blockDim, 0, data_->stream[0]>>>(
                batch_size,
                output_size,
                h_next,      // h_new (input)
                tmp_gx_t,    // gate_x
                tmp_gh,      // gate_h
                bias,
                h_next,      // output (overwrite h_new with final output)
                v_t);
        } else {
            SiluGateOutputKernel<T, false><<<gridDim, blockDim, 0, data_->stream[0]>>>(
                batch_size,
                output_size,
                h_next,
                tmp_gx_t,
                tmp_gh,
                bias,
                h_next,
                nullptr);
        }
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;

}  // namespace elman
}  // namespace v0
}  // namespace haste
