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

// ============================================================================
// ElmanLeaky: tanh + input-dependent leaky integration (Mamba2-style Δ)
// Architecture:
//   candidate = tanh(R @ h + Wx[t] + b)       -- [B, D] candidate state
//   h_new = (1 - delta[t]) * h + delta[t] * candidate  -- leaky integration
//
// Key difference from NoGate: delta[t] is precomputed per-timestep and
// passed in, allowing input-dependent control over state retention.
// This is the mathematically correct discretized Elman with W_h seeing
// the properly blended state at each timestep.
// ============================================================================
namespace elman_leaky {

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

    // Note: R is [D, D], b is [D], delta is [T*B, D] (precomputed)
    void Run(
        const int steps,
        const T* R,        // [D, D]
        const T* b,        // [D]
        const T* Wx,       // [T*B, D]
        const T* delta,    // [T*B, D] - precomputed delta values (sigmoid output)
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
        const T* delta,    // [T*B, D] - same delta used in forward
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* d_delta,        // [T*B, D] - gradient w.r.t. delta
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_leaky

// ============================================================================
// ElmanLeakyMamba2Delta: Mamba2-style softplus/exp delta parameterization
// ============================================================================
// Like ElmanLeaky but with log-space delta:
// - delta = softplus(raw_delta)
// - decay = exp(-delta)
// - h_new = decay * h + (1 - decay) * tanh(R @ h + Wx @ x + b)
namespace elman_leaky_mamba2_delta {

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

    // Note: delta_raw is RAW (before softplus!), decay_cache is for backward
    void Run(
        const int steps,
        const T* R,            // [D, D]
        const T* b,            // [D]
        const T* Wx,           // [T*B, D]
        const T* delta_raw,    // [T*B, D] - raw delta (before softplus)
        T* h,                  // [(T+1)*B, D]
        T* v,                  // [T*B, D]
        T* decay_cache,        // [T*B, D] - cached decay for backward
        T* tmp_Rh);            // [B, D]

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
        const T* delta_raw,    // [T*B, D] - raw delta
        const T* decay_cache,  // [T*B, D] - cached decay
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* d_delta_raw,        // [T*B, D] - gradient w.r.t. raw delta
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_leaky_mamba2_delta

// ============================================================================
// ElmanNoDelta: FIXED decay, no input-dependent delta (ablation baseline)
// Architecture:
//   candidate = tanh(R @ h + Wx[t] + b)       -- [B, D] candidate state
//   h_new = alpha * h + (1 - alpha) * candidate  -- FIXED alpha leaky integration
//
// Same as ElmanLeaky but with constant alpha instead of input-dependent delta.
// This tests whether input-dependent delta is actually needed.
// ============================================================================
namespace elman_no_delta {

template<typename T>
class ForwardPass {
  public:
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const float alpha,  // SCALAR fixed decay factor
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);
    ~ForwardPass();

    // Note: R is [D, D], b is [D], NO delta input!
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
        const float alpha,  // SCALAR fixed decay factor
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);
    ~BackwardPass();

    // Note: NO d_delta output since alpha is fixed!
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

}  // namespace elman_no_delta

// ============================================================================
// ElmanMamba2: Mamba2-style linear recurrence (NO R matrix, NO tanh!)
// Architecture:
//   delta = softplus(W_delta @ x + b_delta)      -- always positive
//   decay = exp(-delta)                          -- between 0 and 1
//   Bx = B @ x + b_B                             -- input projection
//   h_new = decay * h + (1 - decay) * Bx         -- linear combination
//
// Key difference from Elman: NO R matrix (no recurrent GEMM!), NO tanh.
// This is the simplified Mamba2 architecture for ablation testing.
// ============================================================================
namespace elman_mamba2 {

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

    // Note: W_delta, W_B are [D, D], biases are [D]
    void Run(
        const int steps,
        const T* W_delta,       // [D, D] - delta projection weights
        const T* b_delta,       // [D] - delta projection bias
        const T* W_B,           // [D, D] - B projection weights
        const T* b_B,           // [D] - B projection bias
        const T* x,             // [T, B, D] - input sequence
        T* h,                   // [T+1, B, D] - hidden states (h[0] = h0)
        T* Wx_delta_cache,      // [T, B, D] - cached Wx_delta for backward
        T* Bx_cache,            // [T, B, D] - cached Bx for backward
        T* tmp_Wx_delta,        // [B, D] - workspace
        T* tmp_Bx);             // [B, D] - workspace

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
        const T* W_delta,         // [D, D]
        const T* W_B,             // [D, D]
        const T* x,               // [T, B, D]
        const T* h,               // [T+1, B, D]
        const T* Wx_delta_cache,  // [T, B, D]
        const T* Bx_cache,        // [T, B, D]
        const T* dh_new,          // [T+1, B, D]
        T* dx,                    // [T, B, D]
        T* dW_delta,              // [D, D]
        T* db_delta,              // [D]
        T* dW_B,                  // [D, D]
        T* db_B,                  // [D]
        T* dh,                    // [B, D] - carried through time
        T* tmp_d_Wx_delta,        // [B, D] - workspace
        T* tmp_d_Bx);             // [B, D] - workspace

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_mamba2

// ============================================================================
// ElmanMamba2Silu: Mamba2 structure + silu nonlinearity (no R matrix)
// Architecture:
//   delta = softplus(W_delta @ x + b_delta)  -- input-dependent discretization
//   decay = exp(-delta)                      -- between 0 and 1
//   candidate = silu(B @ x + b_B)            -- silu nonlinearity (unbounded for x>0)
//   h_new = decay * h + (1 - decay) * candidate
//
// Testing if silu (unbounded) works better than tanh (bounded) in Mamba2 structure.
// ============================================================================
namespace elman_mamba2_silu {

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
        const T* W_delta,       // [D, D] - delta projection weights
        const T* b_delta,       // [D] - delta projection bias
        const T* W_B,           // [D, D] - B projection weights
        const T* b_B,           // [D] - B projection bias
        const T* x,             // [T, B, D] - input sequence
        T* h,                   // [T+1, B, D] - hidden states (h[0] = h0)
        T* Wx_delta_cache,      // [T, B, D] - cached Wx_delta for backward
        T* Bx_cache,            // [T, B, D] - cached Bx for backward
        T* tmp_Wx_delta,        // [B, D] - workspace
        T* tmp_Bx);             // [B, D] - workspace

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
        const T* W_delta,         // [D, D]
        const T* W_B,             // [D, D]
        const T* x,               // [T, B, D]
        const T* h,               // [T+1, B, D]
        const T* Wx_delta_cache,  // [T, B, D]
        const T* Bx_cache,        // [T, B, D]
        const T* dh_new,          // [T+1, B, D]
        T* dx,                    // [T, B, D]
        T* dW_delta,              // [D, D]
        T* db_delta,              // [D]
        T* dW_B,                  // [D, D]
        T* db_B,                  // [D]
        T* dh,                    // [B, D] - workspace for recurrent gradient
        T* tmp_d_Wx_delta,        // [B, D]
        T* tmp_d_Bx);             // [B, D]

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_mamba2_silu

// ============================================================================
// ElmanLeakySilu: silu activation + leaky integration (NO output gate)
// Architecture:
//   candidate = silu(R @ h + Wx[t] + b)            -- silu often beats tanh
//   delta = sigmoid(W_delta @ x + b_delta)         -- input-dependent blend
//   h_new = (1 - delta) * h + delta * candidate    -- leaky integration
//   output = h_new                                 -- NO output gate!
//
// Same as ElmanLeaky but with silu instead of tanh.
// ============================================================================
namespace elman_leaky_silu {

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
        const T* R,        // [D, D]
        const T* b,        // [D]
        const T* Wx,       // [T*B, D]
        const T* delta,    // [T*B, D] - precomputed delta values (sigmoid output)
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
        const T* delta,
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* d_delta,
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_leaky_silu

// ============================================================================
// ElmanLeakyDiag: tanh + DIAGONAL R + leaky integration (like Mamba's diagonal A!)
// Architecture:
//   candidate = tanh(r ⊙ h + Wx[t] + b)           -- r is [D] not [D,D]!
//   delta = sigmoid(W_delta @ x + b_delta)        -- input-dependent blend
//   h_new = (1 - delta) * h + delta * candidate   -- leaky integration
//
// Benefits:
// - D params instead of D² for recurrence (massive reduction!)
// - No GEMM needed - pure elementwise ops (faster!)
// - Cross-channel mixing happens via W @ x (input projection)
// - Matches Mamba's diagonal A architecture
// ============================================================================
namespace elman_leaky_diag {

template<typename T>
class ForwardPass {
  public:
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,  // unused but kept for API compat
        const cudaStream_t& stream = 0);
    ~ForwardPass();

    // Note: r is [D] (diagonal), no tmp_Rh needed!
    void Run(
        const int steps,
        const T* r,        // [D] - diagonal recurrence weights
        const T* b,        // [D]
        const T* Wx,       // [T*B, D]
        const T* delta,    // [T*B, D] - precomputed delta values
        T* h,              // [(T+1)*B, D]
        T* v);             // [T*B, D] - saves pre-activation

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
        const cublasHandle_t& blas_handle,  // unused but kept for API compat
        const cudaStream_t& stream = 0);
    ~BackwardPass();

    void Run(
        const int steps,
        const T* r,        // [D] - diagonal recurrence weights
        const T* h,
        const T* v,
        const T* delta,
        const T* dh_new,
        T* dWx,
        T* dr,             // [D] - gradient for diagonal r
        T* db,
        T* d_delta,
        T* dh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_leaky_diag

// ============================================================================
// ElmanLeakySelective: tanh + Mamba2-style discretization + per-channel decay
// Architecture:
//   candidate = tanh(R @ h + Wx[t] + b)           -- [B, D] nonlinear candidate
//   dt = softplus(delta_raw[t])                   -- input-dependent timestep
//   alpha = exp(-dt * exp(A))                     -- Mamba2-style per-channel decay
//   h_new = alpha * h + (1 - alpha) * candidate   -- exponential blend
//
// Key features:
// - Per-channel decay rates A (like Mamba2's diagonal A matrix)
// - Input-dependent timestep via softplus (like Mamba2's Δ)
// - NONLINEAR candidate (tanh) - our innovation beyond Mamba2!
// - Output gate computed outside kernel for h+x selectivity
//
// This is the target architecture for achieving Mamba2 parity with nonlinearity.
// ============================================================================
namespace elman_leaky_selective {

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
        const T* R,          // [D, D]
        const T* b,          // [D]
        const T* Wx,         // [T*B, D]
        const T* delta_raw,  // [T*B, D] - precomputed W_delta @ x + b_delta
        const T* A,          // [D] - per-channel decay rates (log-space)
        T* h,                // [(T+1)*B, D]
        T* v,                // [T*B, 2*D] - saves [raw, delta_raw] for backward
        T* tmp_Rh);          // [B, D]

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
        const T* A,
        const T* dh_new,
        T* dWx,
        T* dR,
        T* db,
        T* d_delta_raw,  // [T*B, D] - gradient for W_delta, b_delta
        T* dA,           // [D] - gradient for A
        T* dh,
        T* tmp_dRh);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman_leaky_selective

}  // namespace v0
}  // namespace haste
