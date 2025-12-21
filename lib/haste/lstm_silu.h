// Copyright 2020 LMNT, Inc. All Rights Reserved.
// LSTM + SiLU Selectivity Gate extension by Erik Gaasedelen, 2024.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace lstm_silu {

// LSTM + SiLU Selectivity Gate
//
// This module combines a standard LSTM with an input-dependent output gate:
//   1. c, h_lstm = LSTM(x, c_prev, h_prev)  -- Standard LSTM recurrence
//   2. gate = silu(Wg_x @ x + Wg_h @ h_lstm + bg)  -- SiLU selectivity gate
//   3. output = h_lstm * gate  -- Gated output
//
// This matches the selectivity mechanism used in Mamba2 and provides
// significant improvement over vanilla LSTM.

template<typename T>
class ForwardPass {
  public:
    // training: `true` if the caller intends to perform a backward pass.
    // batch_size: the number of training/inference inputs in each tensor.
    // input_size: the dimension of each input vector (C).
    // hidden_size: the dimension of each hidden/output vector (H).
    // blas_handle: an initialized cuBLAS handle.
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    ~ForwardPass();

    // Performs one forward iteration of the LSTM+SiLU cell.
    //
    // W: [C, H*4] the LSTM input weight matrix.
    // R: [H, H*4] the LSTM recurrent weight matrix.
    // b: [H*4] the LSTM bias.
    // Wg_x: [C, H] the gate input weight matrix.
    // Wg_h: [H, H] the gate hidden weight matrix.
    // bg: [H] the gate bias.
    // x: [N, C] the input for this iteration.
    // h: [N, H] the previous hidden state.
    // c: [N, H] the previous cell state.
    // h_out: [N, H] the LSTM hidden output (before gating).
    // c_out: [N, H] the LSTM cell output.
    // y_out: [N, H] the final gated output.
    // v: [N, H*4] LSTM intermediate activations for backward pass.
    // gate_pre: [N, H] pre-activation gate values (before SiLU) for backward.
    // tmp_Rh: [N, H*4] temporary workspace.
    // tmp_gate: [N, H] temporary workspace for gate computation.
    void Iterate(
        const T* W,
        const T* R,
        const T* b,
        const T* Wg_x,
        const T* Wg_h,
        const T* bg,
        const T* x,
        const T* h,
        const T* c,
        T* h_out,
        T* c_out,
        T* y_out,
        T* v,
        T* gate_pre,
        T* tmp_Rh,
        T* tmp_gate);

    // Runs the full sequence.
    //
    // steps: number of time steps.
    // All other parameters follow the same convention as Iterate.
    // h: [T+1, N, H] hidden states (h[0] is initial state, h[1:] are outputs)
    // c: [T+1, N, H] cell states (c[0] is initial state, c[1:] are outputs)
    // y: [T, N, H] gated outputs
    // v: [T, N, H*4] LSTM activations
    // gate_pre: [T, N, H] pre-activation gate values
    void Run(
        const int steps,
        const T* W,
        const T* R,
        const T* b,
        const T* Wg_x,
        const T* Wg_h,
        const T* bg,
        const T* x,
        T* h,
        T* c,
        T* y,
        T* v,
        T* gate_pre,
        T* tmp_Rh,
        T* tmp_gate);

  private:
    void IterateInternal(
        const T* R,
        const T* b,
        const T* Wg_h,
        const T* bg,
        const T* h,
        const T* c,
        T* h_out,
        T* c_out,
        T* y_out,
        T* v,
        T* gate_pre,
        T* tmp_Rh,
        T* tmp_gate_x,
        T* tmp_gate_h);

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

    // Performs one backward iteration.
    //
    // W_t: [H*4, C] transpose of LSTM input weight matrix.
    // R_t: [H*4, H] transpose of LSTM recurrent weight matrix.
    // b: [H*4] LSTM bias.
    // Wg_x_t: [H, C] transpose of gate input weight matrix.
    // Wg_h_t: [H, H] transpose of gate hidden weight matrix.
    // bg: [H] gate bias.
    // x_t: [C, N] transpose of input.
    // h: [N, H] previous hidden state (from forward pass).
    // c: [N, H] previous cell state (from forward pass).
    // c_new: [N, H] current cell state (from forward pass).
    // h_out: [N, H] LSTM output (from forward pass).
    // v: [N, H*4] LSTM activations (from forward pass).
    // gate_pre: [N, H] pre-activation gate (from forward pass).
    // dy: [N, H] gradient of loss w.r.t. gated output.
    // dx: [N, C] gradient of loss w.r.t. input (output).
    // dW: [C, H*4] gradient w.r.t. LSTM input weights (accumulated).
    // dR: [H, H*4] gradient w.r.t. LSTM recurrent weights (accumulated).
    // db: [H*4] gradient w.r.t. LSTM bias (accumulated).
    // dWg_x: [C, H] gradient w.r.t. gate input weights (accumulated).
    // dWg_h: [H, H] gradient w.r.t. gate hidden weights (accumulated).
    // dbg: [H] gradient w.r.t. gate bias (accumulated).
    // dh: [N, H] gradient w.r.t. previous hidden state (input/output).
    // dc: [N, H] gradient w.r.t. previous cell state (input/output).
    // dp: [N, H*4] temporary workspace.
    // tmp_dgate: [N, H] temporary workspace.
    void Iterate(
        const T* W_t,
        const T* R_t,
        const T* b,
        const T* Wg_x_t,
        const T* Wg_h_t,
        const T* bg,
        const T* x_t,
        const T* h,
        const T* c,
        const T* c_new,
        const T* h_out,
        const T* v,
        const T* gate_pre,
        const T* dy,
        T* dx,
        T* dW,
        T* dR,
        T* db,
        T* dWg_x,
        T* dWg_h,
        T* dbg,
        T* dh,
        T* dc,
        T* dp,
        T* dh_lstm,
        T* tmp_dgate);

    void Run(
        const int steps,
        const T* W_t,
        const T* R_t,
        const T* b,
        const T* Wg_x_t,
        const T* Wg_h_t,
        const T* bg,
        const T* x_t,
        const T* h,
        const T* c,
        const T* h_out,
        const T* v,
        const T* gate_pre,
        const T* dy,
        T* dx,
        T* dW,
        T* dR,
        T* db,
        T* dWg_x,
        T* dWg_h,
        T* dbg,
        T* dh,
        T* dc,
        T* dp,
        T* dh_lstm,
        T* tmp_dgate);

  private:
    void IterateInternal(
        const T* R_t,
        const T* Wg_h_t,
        const T* h,
        const T* c,
        const T* c_new,
        const T* h_out,
        const T* v,
        const T* gate_pre,
        const T* dy,
        T* db,
        T* dbg,
        T* dh,
        T* dc,
        T* dh_lstm,
        T* dp,
        T* tmp_dgate);

    struct private_data;
    private_data* data_;
};

}  // namespace lstm_silu
}  // namespace v0
}  // namespace haste
