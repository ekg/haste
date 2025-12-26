// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// ElmanNeuralMemory: NTM-inspired external memory bank with read/write heads
//
// Architecture:
//   # Read from memory
//   read_key = W_read @ h                         -- [B, memory_dim]
//   read_weights = softmax(M @ read_key / sqrt(memory_dim))  -- [B, num_slots]
//   memory_read = read_weights @ M               -- [B, memory_dim]
//
//   # RNN update uses memory
//   candidate = tanh(R @ h + W_x @ x + W_mem @ memory_read + b)
//   delta = sigmoid(W_delta @ x + W_delta_mem @ memory_read + b_delta)
//   h_new = (1 - delta) * h + delta * candidate
//
//   # Write to memory (optional, controlled by write gate)
//   write_key = W_write @ h_new
//   write_weights = softmax(M @ write_key / sqrt(memory_dim))
//   M_new = M + write_gate * outer(write_weights, W_content @ h_new)
//
// For simplicity, memory is NOT updated (read-only for this version).
// A trainable memory bank is learned during training.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace elman_neural_memory {

template<typename T>
class ForwardPass {
  public:
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const int num_memory_slots,
        const int memory_dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);
    ~ForwardPass();

    void Run(
        const int steps,
        // RNN weights
        const T* R,             // [D, D] - hidden-to-hidden
        const T* W_x,           // [D, input_size] - input projection
        const T* b,             // [D] - candidate bias
        const T* W_delta,       // [D, input_size]
        const T* b_delta,       // [D]
        // Memory weights
        const T* M,             // [num_slots, memory_dim] - memory bank
        const T* W_read,        // [memory_dim, D] - read key projection
        const T* W_mem,         // [D, memory_dim] - memory-to-candidate
        const T* W_delta_mem,   // [D, memory_dim] - memory-to-delta
        // Input/output
        const T* x,             // [T, B, input_size]
        T* h,                   // [T+1, B, D]
        // Cache for backward
        T* v,                   // [T, B, D] - pre-activation
        T* delta_cache,         // [T, B, D]
        T* read_weights_cache,  // [T, B, num_slots] - attention weights
        T* memory_read_cache,   // [T, B, memory_dim] - read values
        // Workspace
        T* tmp_Rh,              // [B, D]
        T* tmp_Wx,              // [B, D]
        T* tmp_read_key,        // [B, memory_dim]
        T* tmp_read_weights,    // [B, num_slots]
        T* tmp_memory_read);    // [B, memory_dim]

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
        const int num_memory_slots,
        const int memory_dim,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);
    ~BackwardPass();

    void Run(
        const int steps,
        // Weights
        const T* R,
        const T* W_x,
        const T* W_delta,
        const T* M,
        const T* W_read,
        const T* W_mem,
        const T* W_delta_mem,
        // Forward cache
        const T* x,
        const T* h,
        const T* v,
        const T* delta_cache,
        const T* read_weights_cache,
        const T* memory_read_cache,
        // Incoming gradient
        const T* dh_new,
        // Output gradients
        T* dx,
        T* dR,
        T* dW_x,
        T* db,
        T* dW_delta,
        T* db_delta,
        T* dM,                  // [num_slots, memory_dim]
        T* dW_read,             // [memory_dim, D]
        T* dW_mem,              // [D, memory_dim]
        T* dW_delta_mem,        // [D, memory_dim]
        T* dh,                  // [B, D] - recurrent gradient
        // Workspace
        T* tmp_Rh,
        T* tmp_Wx,
        T* tmp_d_memory_read,   // [B, memory_dim]
        T* tmp_d_read_weights); // [B, num_slots]

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_neural_memory
}  // namespace v0
}  // namespace haste
