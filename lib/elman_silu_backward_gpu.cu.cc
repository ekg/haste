// Copyright 2024 Erik Garrison. Apache 2.0 License.
// SiLU-gated Elman RNN backward pass.
//
// Architecture per timestep (forward):
//   raw = R @ h + Wx[t] + b          -- [B, 2D]
//   [h_cand_raw, gate_raw] = split(raw)
//   h_candidate = tanh(h_cand_raw)
//   gate = silu(gate_raw)
//   h_new = h_candidate * gate
//
// Backward pass computes:
//   dR  - gradient w.r.t. recurrent weights [2D, D]
//   db  - gradient w.r.t. bias [2D]
//   dWx - gradient w.r.t. input projections [T*B, 2D]
//   dh  - gradient w.r.t. initial hidden state [B, D]

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// Kernel: Compute gradients through SiLU gate and tanh, produce d_raw
// Input:
//   dh_out: [B, D] - gradient from output at this timestep
//   dh_recurrent: [B, D] - gradient from recurrence (from t+1)
//   v: [B, 2D] - saved activations (h_cand_raw, gate_raw)
// Output:
//   d_raw: [B, 2D] - gradient w.r.t. pre-activation
template<typename T>
__global__
void ElmanSiluBackwardPointwiseKernel(
    const int batch_size,
    const int D,
    const T* __restrict__ dh_out,       // [B, D] - gradient from output
    const T* __restrict__ dh_recurrent, // [B, D] - gradient from recurrence (can be null)
    const T* __restrict__ v,            // [B, 2D] - saved (h_cand_raw, gate_raw)
    T* __restrict__ d_raw               // [B, 2D] - gradient w.r.t. raw
) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_size * D;

    if (idx >= total) return;

    const int batch_idx = idx / D;
    const int d_idx = idx % D;

    // Indices into 2D tensors
    const int h_cand_idx = batch_idx * 2 * D + d_idx;
    const int gate_idx = batch_idx * 2 * D + D + d_idx;

    // Load saved pre-activations
    const float h_cand_raw = static_cast<float>(v[h_cand_idx]);
    const float gate_raw = static_cast<float>(v[gate_idx]);

    // Recompute activations
    const float h_candidate = tanhf(h_cand_raw);
    const float sig = 1.0f / (1.0f + expf(-gate_raw));
    const float gate = gate_raw * sig;  // SiLU(x) = x * sigmoid(x)

    // Total gradient from output + recurrence
    float dh = static_cast<float>(dh_out[idx]);
    if (dh_recurrent != nullptr) {
        dh += static_cast<float>(dh_recurrent[idx]);
    }

    // Backprop through elementwise multiply: h_new = h_candidate * gate
    const float d_h_candidate = dh * gate;
    const float d_gate = dh * h_candidate;

    // Backprop through SiLU: gate = gate_raw * sigmoid(gate_raw)
    // d_silu/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    //           = sig * (1 + gate_raw * (1 - sig))
    const float d_silu = sig * (1.0f + gate_raw * (1.0f - sig));
    const float d_gate_raw = d_gate * d_silu;

    // Backprop through tanh: h_candidate = tanh(h_cand_raw)
    // d_tanh/dx = 1 - tanh(x)^2
    const float d_tanh = 1.0f - h_candidate * h_candidate;
    const float d_h_cand_raw = d_h_candidate * d_tanh;

    // Write gradients
    d_raw[h_cand_idx] = static_cast<T>(d_h_cand_raw);
    d_raw[gate_idx] = static_cast<T>(d_gate_raw);
}

// Kernel: Accumulate bias gradient
// db += sum(d_raw, dim=batch)
template<typename T>
__global__
void AccumulateBiasGradientKernel(
    const int batch_size,
    const int gate_dim,  // 2D
    const T* __restrict__ d_raw,  // [B, 2D]
    T* __restrict__ db            // [2D]
) {
    const int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (g_idx >= gate_dim) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        sum += static_cast<float>(d_raw[b * gate_dim + g_idx]);
    }
    db[g_idx] = static_cast<T>(static_cast<float>(db[g_idx]) + sum);
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace elman_silu {

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;  // D
    int hidden_size; // D
    cublasHandle_t blas_handle;
    cudaStream_t stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
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
    const T* R,        // [2D, D] - recurrent weights (same as forward)
    const T* h,        // [(T+1)*B, D] - hidden states from forward
    const T* v,        // [T*B, 2D] - saved activations
    const T* dh_new,   // [(T+1)*B, D] - gradient of output
    T* dWx,            // [T*B, 2D] - gradient w.r.t. input projections
    T* dR,             // [2D, D] - gradient w.r.t. R (accumulated)
    T* db,             // [2D] - gradient w.r.t. bias (accumulated)
    T* dh,             // [B, D] - gradient of initial hidden state
    T* tmp_dRh         // [B, 2D] - workspace
) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int D = data_->hidden_size;
    const int gate_dim = 2 * D;

    const int BD = batch_size * D;

    cudaStream_t stream = data_->stream;
    cublasHandle_t blas_handle = data_->blas_handle;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);
    cublasSetStream(blas_handle, stream);

    // Initialize dh to zero
    cudaMemsetAsync(dh, 0, BD * sizeof(T), stream);

    // Process timesteps in reverse
    for (int t = steps - 1; t >= 0; --t) {
        const T* h_t = h + t * BD;
        const T* v_t = v + t * batch_size * gate_dim;
        T* dWx_t = dWx + t * batch_size * gate_dim;

        // Gradient from this timestep's output
        const T* dh_out = dh_new + (t + 1) * BD;

        // Step 1: Compute d_raw from dh_out + dh (recurrent gradient)
        const int threads = 256;
        const int blocks = (BD + threads - 1) / threads;

        // For t = steps-1, dh is zero (no recurrent gradient from future)
        // For t < steps-1, dh contains gradient through recurrence from t+1
        const T* dh_recurrent = (t == steps - 1) ? nullptr : dh;

        ElmanSiluBackwardPointwiseKernel<T><<<blocks, threads, 0, stream>>>(
            batch_size, D,
            dh_out,       // Gradient from output at this timestep
            dh_recurrent, // Gradient from recurrence (from t+1)
            v_t,
            tmp_dRh);     // d_raw [B, 2D]

        // Step 2: dWx = d_raw (gradient flows through directly)
        cudaMemcpyAsync(dWx_t, tmp_dRh, batch_size * gate_dim * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);

        // Step 3: Accumulate db += sum(d_raw, dim=batch)
        const int bias_threads = 256;
        const int bias_blocks = (gate_dim + bias_threads - 1) / bias_threads;
        AccumulateBiasGradientKernel<T><<<bias_blocks, bias_threads, 0, stream>>>(
            batch_size, gate_dim, tmp_dRh, db);

        // Step 4: Accumulate dR += d_raw^T @ h_t
        // d_raw is [B, 2D], h_t is [B, D]
        // dR is [2D, D]
        // dR += d_raw^T @ h_t = [2D, B] @ [B, D] = [2D, D]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            D, gate_dim, batch_size,
            &alpha,
            h_t, D,
            tmp_dRh, gate_dim,
            &beta_one,
            dR, D);

        // Step 5: Compute recurrent gradient dh = d_raw @ R (row-major)
        // d_raw is [B, 2D] row-major = [2D, B] col-major
        // R is [2D, D] row-major = [D, 2D] col-major
        // dh is [B, D] row-major = [D, B] col-major
        // In cuBLAS: dh = R @ d_raw = [D, 2D] @ [2D, B] = [D, B]
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            D, batch_size, gate_dim,
            &alpha,
            R, D,         // R is [D, 2D] col-major, lda = D
            tmp_dRh, gate_dim,  // d_raw is [2D, B] col-major, ldb = 2D
            &beta,
            dh, D);       // dh is [D, B] col-major = [B, D] row-major
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;

}  // namespace elman_silu
}  // namespace v0
}  // namespace haste
