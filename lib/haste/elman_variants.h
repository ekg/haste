// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// Elman RNN variants with different activation combinations.
//
// All variants follow the same architecture:
//   raw = R @ h + Wx[t] + b          -- [B, 2D] single matmul
//   [h_cand_raw, gate_raw] = split(raw)
//   h_candidate = ACT1(h_cand_raw)   -- [B, D]
//   gate = ACT2(gate_raw)            -- [B, D]
//   h_new = h_candidate * gate       -- [B, D] elementwise
//
// Variants:
//   ElmanTanh:    ACT1=tanh, ACT2=tanh
//   ElmanSigmoid: ACT1=tanh, ACT2=sigmoid
//   ElmanSwish:   ACT1=silu, ACT2=silu
//   ElmanGelu:    ACT1=tanh, ACT2=gelu

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {

// ============================================================================
// ElmanTanh: tanh + tanh gate
// ============================================================================
namespace elman_tanh {

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

    void Run(
        const int steps,
        const T* R,        // [2D, D]
        const T* b,        // [2D]
        const T* Wx,       // [T*B, 2D]
        T* h,              // [(T+1)*B, D]
        T* v,              // [T*B, 2D]
        T* tmp_Rh);        // [B, 2D]

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
        const T* R,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_tanh

// ============================================================================
// ElmanSigmoid: tanh + sigmoid gate
// ============================================================================
namespace elman_sigmoid {

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

    void Run(
        const int steps,
        const T* R,
        const T* b,
        const T* Wx,
        T* h,
        T* v,
        T* tmp_Rh);

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
        const T* R,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_sigmoid

// ============================================================================
// ElmanSwish: silu + silu gate (both activations are SiLU/Swish)
// ============================================================================
namespace elman_swish {

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

    void Run(
        const int steps,
        const T* R,
        const T* b,
        const T* Wx,
        T* h,
        T* v,
        T* tmp_Rh);

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
        const T* R,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_swish

// ============================================================================
// ElmanGelu: tanh + gelu gate
// ============================================================================
namespace elman_gelu {

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

    void Run(
        const int steps,
        const T* R,
        const T* b,
        const T* Wx,
        T* h,
        T* v,
        T* tmp_Rh);

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
        const T* R,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_gelu

// ============================================================================
// ElmanNoGate: tanh only, no gating (ablation baseline)
// Architecture:
//   h_new = tanh(R @ h + Wx[t] + b)   -- simple Elman RNN
// ============================================================================
namespace elman_nogate {

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

    // Note: For NoGate, R is [D, D] and b is [D] (no gate dimension)
    void Run(
        const int steps,
        const T* R,        // [D, D]
        const T* b,        // [D]
        const T* Wx,       // [T*B, D]
        T* h,              // [(T+1)*B, D]
        T* v,              // [T*B, D] - saves pre-activation for backward
        T* tmp_Rh);        // [B, D]

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
        const T* R,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_nogate

}  // namespace v0
}  // namespace haste
