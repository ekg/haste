// Copyright 2024 Erik Garrison. Apache 2.0 License.
// Fused Elman RNN backward pass.
//
// Forward architecture (for reference):
//   hidden = tanh(W1 @ [x_t, h])    -- [B, H] where H = expansion * D
//   h_new = W2 @ hidden             -- [B, D]
//   gate = silu(Wgx @ x_t + Wgh @ h_new + bias)  -- [B, D]
//   out = h_new * gate              -- [B, D]
//
// Backward pass computes gradients w.r.t. all inputs and weights.
// The recurrent gradient is properly accumulated through dh_accum.

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

// Derivative of SiLU: d/dx[x * sigmoid(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
template<typename T>
__device__ __forceinline__
T d_silu(const T x) {
    const T sig = static_cast<T>(1.0) / (static_cast<T>(1.0) + device_exp(-x));
    return sig * (static_cast<T>(1.0) + x * (static_cast<T>(1.0) - sig));
}

// Kernel: Backward through output gate and SiLU
// Adds recurrent gradient (dh_recurrent) to external gradient (dout) to get dh_total
// Computes:
//   dh_new = dh_total * gate (partial, before adding Wgh contribution)
//   dgate_logits = dh_total * h_new * d_silu(gate_logits)
//   dbias += dgate_logits
template<typename T>
__global__
void GateBackwardKernel(
    const int batch_size,
    const int output_dim,
    const T* __restrict__ dout,           // [B, D] - gradient from loss
    const T* __restrict__ dh_recurrent,   // [B, D] - recurrent gradient from later timestep
    const T* __restrict__ v,              // [B, D*3] - saved: h_new, gate_logit, gate
    T* __restrict__ dh_new,               // [B, D] - gradient w.r.t. h_new (partial)
    T* __restrict__ dgate_logits,         // [B, D] - gradient w.r.t. gate_logits
    T* __restrict__ dbias                 // [D] - gradient w.r.t. bias (atomicAdd)
) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;  // output dim
    const int col = blockDim.y * blockIdx.y + threadIdx.y;  // batch

    if (row >= output_dim || col >= batch_size)
        return;

    const int idx = col * output_dim + row;
    const int v_base = col * output_dim * 3;

    // Total gradient = external + recurrent
    const T dh_total = dout[idx] + dh_recurrent[idx];

    const T h_new_val = v[v_base + row];                      // h_new
    const T gate_logit = v[v_base + output_dim + row];        // gate_logit
    const T gate = v[v_base + output_dim * 2 + row];          // gate

    // Backward through: out = h_new * gate
    // dh_new (partial) = dh_total * gate
    // dgate = dh_total * h_new
    const T dgate = dh_total * h_new_val;
    dh_new[idx] = dh_total * gate;

    // Backward through: gate = silu(gate_logit)
    // dgate_logits = dgate * d_silu(gate_logit)
    const T dgate_logit = dgate * d_silu(gate_logit);
    dgate_logits[idx] = dgate_logit;

    // Accumulate bias gradient (atomic for thread safety)
    atomicAdd(&dbias[row], dgate_logit);
}

// Kernel: Backward through tanh activation
// Computes: dhidden_pre = dhidden * (1 - hidden^2)
template<typename T>
__global__
void TanhBackwardKernel(
    const int size,
    const T* __restrict__ dhidden,        // [B, H]
    const T* __restrict__ hidden,         // [B, H] - tanh output from forward
    T* __restrict__ dhidden_pre           // [B, H]
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return;

    const T h = hidden[idx];
    dhidden_pre[idx] = dhidden[idx] * (static_cast<T>(1.0) - h * h);
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace elman {

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    int output_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[2];
    cudaEvent_t event;
    cudaStream_t sync_stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const int output_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->output_size = output_size;
    data_->blas_handle = blas_handle;
    data_->sync_stream = stream;
    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    if (data_->sync_stream) {
        cudaEventRecord(data_->event, data_->stream[1]);
        cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
        cudaEventRecord(data_->event, data_->stream[0]);
        cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    } else {
        cudaStreamSynchronize(data_->stream[1]);
        cudaStreamSynchronize(data_->stream[0]);
    }
    cudaEventDestroy(data_->event);
    cudaStreamDestroy(data_->stream[1]);
    cudaStreamDestroy(data_->stream[0]);
    delete data_;
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* W1_t,        // [D+D, H] - transpose of W1
    const T* W2_t,        // [H, D] - transpose of W2
    const T* Wgx_t,       // [D, D] - transpose of Wgx
    const T* Wgh_t,       // [D, D] - transpose of Wgh
    const T* x_t,         // [D, T*B] - transpose of input sequence
    const T* h,           // [(T+1)*B, D] - hidden states from forward
    const T* hidden,      // [T*B, H] - tanh outputs from forward
    const T* v,           // [T*B, D*3] - saved activations
    const T* dh_new,      // [(T+1)*B, D] - gradient of output (from loss)
    T* dx,                // [T*B, D] - gradient w.r.t. input
    T* dW1,               // [H, D+D] - gradient w.r.t. W1
    T* dW2,               // [D, H] - gradient w.r.t. W2
    T* dWgx,              // [D, D] - gradient w.r.t. Wgx
    T* dWgh,              // [D, D] - gradient w.r.t. Wgh
    T* dbias,             // [D] - gradient w.r.t. bias
    T* dh,                // [B, D] - gradient of initial hidden state (INPUT/OUTPUT - used as recurrent accumulator)
    T* tmp_dgate,         // [B, D] - workspace for dgate_logits
    T* tmp_dh_new_local,  // [B, D] - workspace for local dh_new
    T* tmp_dhidden,       // [B, H] - workspace for dhidden
    T* tmp_dcombined      // [B, D+D] - workspace for dcombined (unused, kept for API)
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const int output_size = data_->output_size;
    const cublasHandle_t blas_handle = data_->blas_handle;

    const int BD = batch_size * output_size;
    const int BH = batch_size * hidden_size;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, data_->stream[0]);

    const dim3 blockDim(32, 16);
    const dim3 gridDim_D(
        (output_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    const int tanh_threads = 256;
    const int tanh_blocks = (BH + tanh_threads - 1) / tanh_threads;

    // dh is used as recurrent gradient accumulator
    // Initialize to zeros (caller should provide zeroed buffer)
    // NOTE: Caller must zero dh before calling Run()

    // Process timesteps in reverse order
    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t_step = x_t + t * BD;  // [D, B] for this timestep
        const T* h_t = h + t * BD;         // h at timestep t
        const T* hidden_t = hidden + t * BH;
        const T* v_t = v + t * BD * 3;
        const T* dout_t = dh_new + (t + 1) * BD;  // gradient from loss at t+1

        T* dx_t = dx + t * BD;

        // Step 1: Backward through output gate
        // Combines external gradient (dout_t) with recurrent gradient (dh)
        // dh_new_local = dh_total * gate (partial)
        // dgate_logits = dh_total * h_new * d_silu(gate_logit)
        // dbias += dgate_logits
        GateBackwardKernel<T><<<gridDim_D, blockDim, 0, data_->stream[0]>>>(
            batch_size,
            output_size,
            dout_t,
            dh,  // recurrent gradient from later timestep (zeros for t = steps-1)
            v_t,
            tmp_dh_new_local,
            tmp_dgate,
            dbias);

        // Step 2: Add Wgh contribution to dh_new
        // dh_new += Wgh^T @ dgate_logits
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            output_size, batch_size, output_size,
            &alpha,
            Wgh_t, output_size,
            tmp_dgate, output_size,
            &beta_one,
            tmp_dh_new_local, output_size);

        // Step 3: dWgh += dgate_logits @ h_new^T
        // h_new is stored in v_t[0:BD]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            output_size, output_size, batch_size,
            &alpha,
            tmp_dgate, output_size,
            v_t, output_size,
            &beta_one,
            dWgh, output_size);

        // Step 4: dWgx += dgate_logits @ x_t^T
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            output_size, input_size, batch_size,
            &alpha,
            tmp_dgate, output_size,
            x_t_step, input_size,
            &beta_one,
            dWgx, output_size);

        // Step 5: dx (partial) = Wgx^T @ dgate_logits
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, output_size,
            &alpha,
            Wgx_t, input_size,
            tmp_dgate, output_size,
            &beta_zero,
            dx_t, input_size);

        // Step 6: Backward through h_new = W2 @ hidden
        // dhidden = W2^T @ dh_new_local
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_size, batch_size, output_size,
            &alpha,
            W2_t, hidden_size,
            tmp_dh_new_local, output_size,
            &beta_zero,
            tmp_dhidden, hidden_size);

        // Step 7: dW2 += dh_new_local @ hidden^T
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            output_size, hidden_size, batch_size,
            &alpha,
            tmp_dh_new_local, output_size,
            hidden_t, hidden_size,
            &beta_one,
            dW2, output_size);

        // Step 8: Backward through hidden = tanh(W1 @ combined)
        // dcombined_pre = dhidden * (1 - hidden^2)
        TanhBackwardKernel<T><<<tanh_blocks, tanh_threads, 0, data_->stream[0]>>>(
            BH, tmp_dhidden, hidden_t, tmp_dhidden);  // In-place

        // Step 9: dW1_x += dcombined_pre @ x_t^T
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            hidden_size, input_size, batch_size,
            &alpha,
            tmp_dhidden, hidden_size,
            x_t_step, input_size,
            &beta_one,
            dW1, hidden_size);

        // Step 10: dW1_h += dcombined_pre @ h_t^T
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            hidden_size, output_size, batch_size,
            &alpha,
            tmp_dhidden, hidden_size,
            h_t, output_size,
            &beta_one,
            dW1 + hidden_size * input_size, hidden_size);

        // Step 11: dx += W1_x^T @ dcombined_pre
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch_size, hidden_size,
            &alpha,
            W1_t, input_size,
            tmp_dhidden, hidden_size,
            &beta_one,
            dx_t, input_size);

        // Step 12: Update dh (recurrent gradient for next iteration)
        // dh = W1_h^T @ dcombined_pre
        // This becomes the recurrent gradient for timestep t-1
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            output_size, batch_size, hidden_size,
            &alpha,
            W1_t + input_size, output_size,
            tmp_dhidden, hidden_size,
            &beta_zero,  // Overwrite - this is the new recurrent gradient
            dh, output_size);
    }

    // After the loop, dh contains the gradient of the initial hidden state
    cublasSetStream(blas_handle, save_stream);
}

template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;

}  // namespace elman
}  // namespace v0
}  // namespace haste
