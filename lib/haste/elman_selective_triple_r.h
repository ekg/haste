// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// ElmanSelectiveTripleR: Triple R with input-dependent B gate (like Mamba2)
//
// Architecture:
//   B_gate = sigmoid(W_B @ x + b_B)                      -- input-dependent write gate
//   candidate = tanh(R_h @ h + B_gate * (R_x @ x) + b)   -- B_gate modulates input contribution
//   delta = sigmoid(R_delta @ h + W_delta @ x + b_delta) -- context-aware delta
//   h_new = (1 - delta) * h + delta * candidate          -- leaky integration
//
// Key difference from Triple R:
// - B_gate adds input-dependent selectivity: different inputs write to different
//   parts of the hidden state, inspired by Mamba2's selective state space model.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace elman_selective_triple_r {

template<typename T>
class ForwardPass {
  public:
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);
    ~ForwardPass();

    // Three R matrices + delta projection + B gate
    void Run(
        const int steps,
        const T* R_h,           // [D, D] - hidden-to-hidden
        const T* R_x,           // [D, input_size] - input projection
        const T* R_delta,       // [D, D] - hidden-to-delta
        const T* W_delta,       // [D, input_size] - input-to-delta
        const T* W_B,           // [D, input_size] - input-to-B_gate (NEW)
        const T* b,             // [D] - candidate bias
        const T* b_delta,       // [D] - delta bias
        const T* b_B,           // [D] - B_gate bias (NEW)
        const T* x,             // [T, B, input_size] - input sequence
        T* h,                   // [T+1, B, D] - hidden states
        T* v,                   // [T, B, D] - pre-activation cache
        T* delta_cache,         // [T, B, D] - cached delta for backward
        T* B_gate_cache,        // [T, B, D] - cached B_gate for backward (NEW)
        T* tmp_Rh,              // [B, D] - workspace
        T* tmp_Rx,              // [B, D] - workspace
        T* tmp_Rdelta,          // [B, D] - workspace
        T* tmp_B);              // [B, D] - workspace (NEW)

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
        const T* R_h,
        const T* R_x,
        const T* R_delta,
        const T* W_delta,
        const T* W_B,
        const T* x,
        const T* h,
        const T* v,
        const T* delta_cache,
        const T* B_gate_cache,
        const T* dh_new,
        T* dx,                  // [T, B, input_size]
        T* dR_h,                // [D, D]
        T* dR_x,                // [D, input_size]
        T* dR_delta,            // [D, D]
        T* dW_delta,            // [D, input_size]
        T* dW_B,                // [D, input_size] (NEW)
        T* db,                  // [D]
        T* db_delta,            // [D]
        T* db_B,                // [D] (NEW)
        T* dh,                  // [B, D] - recurrent gradient
        T* tmp_Rh,
        T* tmp_Rx,
        T* tmp_Rdelta,
        T* tmp_B);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_selective_triple_r
}  // namespace v0
}  // namespace haste
