// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// MultiHeadTripleR: Multi-head Triple R with 32× state expansion (Mamba2-style)
//
// Architecture (per head):
//   candidate = tanh(R_h @ h + R_x @ x + b)
//   delta = sigmoid(R_delta @ h + W_delta @ x + b_delta)
//   h_new = (1 - delta) * h + delta * candidate
//
// State expansion: nheads × headdim × d_state = 32× model dim

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace multihead_triple_r {

template<typename T>
class ForwardPass {
  public:
    ForwardPass(
        const bool training,
        const int batch_size,
        const int nheads,
        const int headdim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);
    ~ForwardPass();

    // Multi-head Triple R forward pass
    void Run(
        const int steps,
        const T* R_h,           // [nheads, headdim, headdim]
        const T* R_x,           // [nheads, headdim, headdim]
        const T* R_delta,       // [nheads, headdim, headdim]
        const T* W_delta,       // [nheads, headdim, headdim]
        const T* b,             // [nheads, headdim]
        const T* b_delta,       // [nheads, headdim]
        const T* x,             // [T, B, nheads, headdim]
        T* h,                   // [T+1, B, nheads, headdim]
        T* v,                   // [T, B, nheads, headdim]
        T* delta_cache,         // [T, B, nheads, headdim]
        T* tmp_Rh,              // [B, nheads, headdim]
        T* tmp_Rx,              // [B, nheads, headdim]
        T* tmp_Rdelta,          // [B, nheads, headdim]
        T* tmp_Wdelta);         // [B, nheads, headdim]

  private:
    struct private_data;
    private_data* data_;
};

template<typename T>
class BackwardPass {
  public:
    BackwardPass(
        const int batch_size,
        const int nheads,
        const int headdim,
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
        T* dx,                  // [T, B, nheads, headdim]
        T* dR_h,                // [nheads, headdim, headdim]
        T* dR_x,                // [nheads, headdim, headdim]
        T* dR_delta,            // [nheads, headdim, headdim]
        T* dW_delta,            // [nheads, headdim, headdim]
        T* db,                  // [nheads, headdim]
        T* db_delta,            // [nheads, headdim]
        T* dh,                  // [B, nheads, headdim]
        T* tmp_Rh,
        T* tmp_Rx,
        T* tmp_Rdelta,
        T* tmp_Wdelta);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace multihead_triple_r
}  // namespace v0
}  // namespace haste
