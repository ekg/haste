// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// Multi-head Elman RNN with per-head recurrence matrices and RESET GATE.
//
// Architecture per timestep per head:
//   r[i] = sigmoid(Wr[i] @ x[i])           // Reset gate (input-only)
//   h_gated[i] = r[i] * h[i]               // Gate the history
//   h_new[i] = activation(R[i] @ h_gated[i] + Wx[i] @ x[i] + b[i])
//
// Key insight: Each head has its own headdim×headdim R matrix,
// giving 2048x more expressive recurrence than Mamba2's scalar decays.
// The reset gate enables SELECTIVE recurrence like GRU.
//
// Activations:
//   0 = softsign: x / (1 + |x|) - gradient-friendly, non-saturating
//   1 = tanh_residual: x + tanh(x) - preserves gradients via skip connection
//   2 = tanh: standard tanh

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace multihead_elman {

template<typename T>
class ForwardPass {
  public:
    // training: `true` if the caller intends to perform a backward pass.
    // batch_size: the number of training/inference inputs per tensor.
    // nheads: number of independent heads.
    // headdim: dimension per head.
    // activation: 0=softsign, 1=tanh_residual, 2=tanh.
    // blas_handle: an initialized cuBLAS handle.
    ForwardPass(
        const bool training,
        const int batch_size,
        const int nheads,
        const int headdim,
        const int activation,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    ~ForwardPass();

    // Performs the full forward pass of the multi-head Elman RNN with reset gate.
    //
    // steps: number of timesteps T.
    // R: [nheads, headdim, headdim] - recurrent weights per head.
    // Wr: [nheads, headdim, headdim] - reset gate weights per head.
    // Wx: [nheads, headdim, headdim] - input weights per head.
    // b: [nheads, headdim] - bias per head.
    // x: [T, B, nheads, headdim] - input sequence.
    // h: [T+1, B, nheads, headdim] - all hidden states (h[0]=h0 on input).
    // y: [T, B, nheads, headdim] - output sequence.
    // r_gate: [T, B, nheads, headdim] - saved reset gate values for backward.
    //         Can be nullptr if training is false.
    // pre_act: [T, B, nheads, headdim] - saved pre-activations for backward.
    //          Can be nullptr if training is false.
    // tmp_Rh: [B, nheads, headdim] - workspace for R @ h_gated.
    // tmp_Wrx: [T, B, nheads, headdim] - workspace for pre-computed Wr @ x.
    // tmp_Wxx: [T, B, nheads, headdim] - workspace for pre-computed Wx @ x.
    // tmp_h_gated: [B, nheads, headdim] - workspace for gated hidden state.
    void Run(
        const int steps,
        const T* R,
        const T* Wr,
        const T* Wx,
        const T* b,
        const T* x,
        T* h,
        T* y,
        T* r_gate,
        T* pre_act,
        T* tmp_Rh,
        T* tmp_Wrx,
        T* tmp_Wxx,
        T* tmp_h_gated);

  private:
    struct private_data;
    private_data* data_;
};

template<typename T>
class BackwardPass {
  public:
    // batch_size: the number of training inputs per tensor.
    // nheads: number of independent heads.
    // headdim: dimension per head.
    // activation: 0=softsign, 1=tanh_residual, 2=tanh.
    // blas_handle: an initialized cuBLAS handle.
    BackwardPass(
        const int batch_size,
        const int nheads,
        const int headdim,
        const int activation,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    ~BackwardPass();

    // Performs the full backward pass of the multi-head Elman RNN with reset gate.
    //
    // steps: number of timesteps T.
    // R: [nheads, headdim, headdim] - recurrent weights.
    // Wr: [nheads, headdim, headdim] - reset gate weights.
    // Wx: [nheads, headdim, headdim] - input weights.
    // x: [T, B, nheads, headdim] - input sequence.
    // h: [T+1, B, nheads, headdim] - hidden states from forward (h[0]=initial).
    // r_gate: [T, B, nheads, headdim] - saved reset gate values from forward.
    // pre_act: [T, B, nheads, headdim] - saved pre-activations.
    // dy: [T, B, nheads, headdim] - gradient of loss w.r.t. output.
    // dx: [T, B, nheads, headdim] - output: gradient w.r.t. input.
    // dR: [nheads, headdim, headdim] - output: gradient w.r.t. R (accumulated).
    // dWr: [nheads, headdim, headdim] - output: gradient w.r.t. Wr (accumulated).
    // dWx: [nheads, headdim, headdim] - output: gradient w.r.t. Wx (accumulated).
    // db: [nheads, headdim] - output: gradient w.r.t. bias (accumulated).
    // dh0: [B, nheads, headdim] - output: gradient w.r.t. initial hidden state.
    // tmp_dpre: [B, nheads, headdim] - workspace.
    // tmp_dh: [B, nheads, headdim] - workspace.
    // tmp_dh_gated: [B, nheads, headdim] - workspace.
    // tmp_dWrx: [B, nheads, headdim] - workspace.
    // tmp_h_gated: [B, nheads, headdim] - workspace to recompute h_gated.
    void Run(
        const int steps,
        const T* R,
        const T* Wr,
        const T* Wx,
        const T* x,
        const T* h,
        const T* r_gate,
        const T* pre_act,
        const T* dy,
        T* dx,
        T* dR,
        T* dWr,
        T* dWx,
        T* db,
        T* dh0,
        T* tmp_dpre,
        T* tmp_dh,
        T* tmp_dh_gated,
        T* tmp_dWrx,
        T* tmp_h_gated);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace multihead_elman
}  // namespace v0
}  // namespace haste
