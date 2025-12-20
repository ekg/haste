// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// SiLU-gated Elman RNN - FlashRNN-compatible architecture.
//
// Architecture per timestep (2 gates, 1 matmul):
//   raw = R @ h + Wx[t] + b          -- [B, 2D] single matmul
//   [h_candidate, gate_logit] = split(raw)
//   h_candidate = tanh(h_candidate)   -- [B, D]
//   gate = silu(gate_logit)           -- [B, D]
//   h_new = h_candidate * gate        -- [B, D] elementwise
//
// This matches FlashRNN's structure exactly, enabling cuDNN-level performance.
// Key insight: With 2 gates computed together, we get 1 matmul per timestep.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace elman_silu {

// Number of gates: 2 (state candidate + output gate)
constexpr int NUM_GATES = 2;

template<typename T>
class ForwardPass {
  public:
    // training: `true` if the caller intends to perform a backward pass.
    // batch_size: the number of training/inference inputs.
    // input_size: the dimension of each input vector (D).
    // hidden_size: the hidden dimension (D) - same as input for this cell.
    // blas_handle: an initialized cuBLAS handle.
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    ~ForwardPass();

    // Performs the forward pass.
    //
    // steps: number of timesteps T.
    // R: [2D, D] recurrent weight matrix (combines W_h and Wg_h).
    // b: [2D] bias vector.
    // Wx: [T*B, 2D] pre-computed input projections for all timesteps.
    // h: [(T+1)*B, D] hidden states. h[0:B*D] should be initialized.
    // v: [T*B, 2D] saved activations for backward (h_candidate, gate_logit).
    void Run(
        const int steps,
        const T* R,
        const T* b,
        const T* Wx,
        T* h,
        T* v,
        T* tmp_Rh);  // [B, 2D] workspace

  private:
    struct private_data;
    private_data* data_;
};

template<typename T>
class BackwardPass {
  public:
    BackwardPass(
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    ~BackwardPass();

    void Run(
        const int steps,
        const T* R,        // [2D, D] recurrent weights (same as forward)
        const T* h,        // [(T+1)*B, D] hidden states from forward
        const T* v,        // [T*B, 2D] saved activations
        const T* dh_new,   // [(T+1)*B, D] gradient of output
        T* dWx,            // [T*B, 2D] gradient w.r.t. input projections
        T* dR,             // [2D, D] gradient w.r.t. R (accumulated)
        T* db,             // [2D] gradient w.r.t. bias (accumulated)
        T* dh,             // [B, D] gradient of initial hidden state
        T* tmp_dRh);       // [B, 2D] workspace

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_silu
}  // namespace v0
}  // namespace haste
