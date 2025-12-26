// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// ElmanLowRankR: R matrix decomposed as R = U @ V^T + S
//
// Architecture:
//   R_effective = U @ V^T + S              -- [D, D] = [D, rank] @ [rank, D] + sparse
//   candidate = tanh(R_effective @ h + W_x @ x + b)
//   delta = sigmoid(W_delta @ x + b_delta)
//   h_new = (1 - delta) * h + delta * candidate
//
// Benefits:
// - Parameter efficiency: D*rank*2 + sparse_nnz instead of D*D
// - Low-rank captures global patterns
// - Sparse captures local/specialized patterns
// - For rank=256, D=2048: 1M params vs 4M params (75% reduction!)
//
// For simplicity, S is dense but can be masked/pruned during training.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace elman_lowrank_r {

template<typename T>
class ForwardPass {
  public:
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);
    ~ForwardPass();

    void Run(
        const int steps,
        const T* U,             // [D, rank] - left factor
        const T* V,             // [D, rank] - right factor (R = U @ V^T)
        const T* S,             // [D, D] - sparse/residual term
        const T* W_x,           // [D, input_size]
        const T* b,             // [D]
        const T* W_delta,       // [D, input_size]
        const T* b_delta,       // [D]
        const T* x,             // [T, B, input_size]
        T* h,                   // [T+1, B, D]
        T* v,                   // [T, B, D] - pre-activation cache
        T* delta_cache,         // [T, B, D]
        T* tmp_Vh,              // [B, rank] - V^T @ h
        T* tmp_UVh,             // [B, D] - U @ V^T @ h
        T* tmp_Sh,              // [B, D] - S @ h (can add to UVh)
        T* tmp_Wx);             // [B, D]

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
        const int rank,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);
    ~BackwardPass();

    void Run(
        const int steps,
        const T* U,
        const T* V,
        const T* S,
        const T* W_x,
        const T* W_delta,
        const T* x,
        const T* h,
        const T* v,
        const T* delta_cache,
        const T* dh_new,
        T* dx,
        T* dU,                  // [D, rank]
        T* dV,                  // [D, rank]
        T* dS,                  // [D, D]
        T* dW_x,
        T* db,
        T* dW_delta,
        T* db_delta,
        T* dh,                  // [B, D]
        T* tmp_Vh,              // [B, rank]
        T* tmp_d_raw,           // [B, D]
        T* tmp_d_delta_raw);    // [B, D]

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_lowrank_r
}  // namespace v0
}  // namespace haste
