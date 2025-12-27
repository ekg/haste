// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// DiagonalMHTR: Diagonal Multi-Head Triple R (depth-stable variant)
//
// KEY DIFFERENCE: R matrices are DIAGONAL (vectors) instead of full matrices.
// This makes the recurrence element-wise, stable at depth=48+.
//
// Architecture (per head):
//   candidate = tanh(R_h * h + R_x * x + b)      // element-wise
//   delta = sigmoid(R_delta * h + W_delta @ x + b_delta)
//   h_new = (1 - delta) * h + delta * candidate

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace diagonal_mhtr {

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

    // Diagonal MHTR forward pass
    void Run(
        const int steps,
        const T* R_h,           // [nheads, headdim] - DIAGONAL
        const T* R_x,           // [nheads, headdim] - DIAGONAL
        const T* R_delta,       // [nheads, headdim] - DIAGONAL
        const T* W_delta,       // [nheads, headdim, headdim] - still full
        const T* b,             // [nheads, headdim]
        const T* b_delta,       // [nheads, headdim]
        const T* x,             // [T, B, nheads, headdim]
        T* h,                   // [T+1, B, nheads, headdim]
        T* v,                   // [T, B, nheads, headdim]
        T* delta_cache,         // [T, B, nheads, headdim]
        T* tmp_Wdelta);         // [T, B, nheads, headdim] - pre-computed W_delta @ x

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
        const T* R_h,           // [nheads, headdim] - DIAGONAL
        const T* R_x,           // [nheads, headdim] - DIAGONAL
        const T* R_delta,       // [nheads, headdim] - DIAGONAL
        const T* W_delta,       // [nheads, headdim, headdim]
        const T* x,
        const T* h,
        const T* v,
        const T* delta_cache,
        const T* dh_new,
        T* dx,                  // [T, B, nheads, headdim]
        T* dR_h,                // [nheads, headdim] - DIAGONAL
        T* dR_x,                // [nheads, headdim] - DIAGONAL
        T* dR_delta,            // [nheads, headdim] - DIAGONAL
        T* dW_delta,            // [nheads, headdim, headdim]
        T* db,                  // [nheads, headdim]
        T* db_delta,            // [nheads, headdim]
        T* d_raw_all,           // [T, B, nheads, headdim] - workspace
        T* d_delta_raw_all,     // [T, B, nheads, headdim] - workspace
        T* dh_prev);            // [B, nheads, headdim] - workspace

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace diagonal_mhtr
}  // namespace v0
}  // namespace haste
