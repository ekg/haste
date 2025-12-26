// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// ElmanTripleR: Three separate R matrices for different signal pathways
//
// Architecture:
//   candidate = tanh(R_h @ h + R_x @ x + b)       -- R_h for hidden, R_x for input
//   delta = sigmoid(R_delta @ h + W_delta @ x + b_delta)  -- R_delta for context-aware delta
//   h_new = (1 - delta) * h + delta * candidate   -- leaky integration
//
// This separates:
// - R_h: hidden-to-hidden temporal patterns
// - R_x: input transformation (richer than simple Wx)
// - R_delta: context-aware forget decisions (h-dependent delta!)

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace elman_triple_r {

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

    // Three R matrices + delta projection
    void Run(
        const int steps,
        const T* R_h,           // [D, D] - hidden-to-hidden
        const T* R_x,           // [D, input_size] - input projection (replaces Wx)
        const T* R_delta,       // [D, D] - hidden-to-delta (context-aware!)
        const T* W_delta,       // [D, input_size] - input-to-delta
        const T* b,             // [D] - candidate bias
        const T* b_delta,       // [D] - delta bias
        const T* x,             // [T, B, input_size] - input sequence
        T* h,                   // [T+1, B, D] - hidden states
        T* v,                   // [T, B, D] - pre-activation cache
        T* delta_cache,         // [T, B, D] - cached delta for backward
        T* tmp_Rh,              // [B, D] - workspace
        T* tmp_Rx,              // [B, D] - workspace
        T* tmp_Rdelta);         // [B, D] - workspace

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
        const T* x,
        const T* h,
        const T* v,
        const T* delta_cache,
        const T* dh_new,
        T* dx,                  // [T, B, input_size]
        T* dR_h,                // [D, D]
        T* dR_x,                // [D, input_size]
        T* dR_delta,            // [D, D]
        T* dW_delta,            // [D, input_size]
        T* db,                  // [D]
        T* db_delta,            // [D]
        T* dh,                  // [B, D] - recurrent gradient
        T* tmp_Rh,
        T* tmp_Rx,
        T* tmp_Rdelta);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_triple_r
}  // namespace v0
}  // namespace haste
