// Copyright 2020 LMNT, Inc. All Rights Reserved.
// GRU + SiLU Selectivity Gate extension by Erik Gaasedelen, 2024.
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
namespace gru_silu {

// GRU + SiLU Selectivity Gate
//
// This module combines a standard GRU with an input-dependent output gate:
//   1. h_gru = GRU(x, h_prev)     -- Standard GRU recurrence
//   2. gate = silu(Wg_x @ x + Wg_h @ h_gru + bg)  -- SiLU selectivity gate
//   3. output = h_gru * gate      -- Gated output
//
// This matches the selectivity mechanism used in Mamba2 and provides
// ~1 nat improvement over vanilla GRU (from 3.77 to 2.80 loss).

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

    // Performs one forward iteration of the GRU+SiLU cell.
    //
    // W: [C, H*3] the GRU input weight matrix.
    // R: [H, H*3] the GRU recurrent weight matrix.
    // bx: [H*3] the GRU bias for input weights.
    // br: [H*3] the GRU bias for recurrent weights.
    // Wg_x: [C, H] the gate input weight matrix.
    // Wg_h: [H, H] the gate hidden weight matrix.
    // bg: [H] the gate bias.
    // x: [N, C] the input for this iteration.
    // h: [N, H] the previous hidden state.
    // h_out: [N, H] the GRU hidden output (before gating).
    // y_out: [N, H] the final gated output.
    // v: [N, H*4] GRU intermediate activations for backward pass.
    // gate_pre: [N, H] pre-activation gate values (before SiLU) for backward.
    // tmp_Wx: [N, H*3] temporary workspace.
    // tmp_Rh: [N, H*3] temporary workspace.
    // tmp_gate: [N, H] temporary workspace for gate computation.
    void Iterate(
        const T* W,
        const T* R,
        const T* bx,
        const T* br,
        const T* Wg_x,
        const T* Wg_h,
        const T* bg,
        const T* x,
        const T* h,
        T* h_out,
        T* y_out,
        T* v,
        T* gate_pre,
        T* tmp_Wx,
        T* tmp_Rh,
        T* tmp_gate);

    // Runs the full sequence.
    //
    // steps: number of time steps.
    // All other parameters follow the same convention as Iterate.
    // h: [T+1, N, H] hidden states (h[0] is initial state, h[1:] are outputs)
    // y: [T, N, H] gated outputs
    // v: [T, N, H*4] GRU activations
    // gate_pre: [T, N, H] pre-activation gate values
    void Run(
        const int steps,
        const T* W,
        const T* R,
        const T* bx,
        const T* br,
        const T* Wg_x,
        const T* Wg_h,
        const T* bg,
        const T* x,
        T* h,
        T* y,
        T* v,
        T* gate_pre,
        T* tmp_Wx,
        T* tmp_Rh,
        T* tmp_gate);

  private:
    void IterateInternal(
        const T* R,
        const T* bx,
        const T* br,
        const T* Wg_h,
        const T* bg,
        const T* h,
        T* h_out,
        T* y_out,
        T* v,
        T* gate_pre,
        T* tmp_Wx,
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
    // W_t: [H*3, C] transpose of GRU input weight matrix.
    // R_t: [H*3, H] transpose of GRU recurrent weight matrix.
    // bx: [H*3] GRU input bias.
    // br: [H*3] GRU recurrent bias.
    // Wg_x_t: [H, C] transpose of gate input weight matrix.
    // Wg_h_t: [H, H] transpose of gate hidden weight matrix.
    // bg: [H] gate bias.
    // x_t: [C, N] transpose of input.
    // h: [N, H] previous hidden state (from forward pass).
    // h_out: [N, H] GRU output (from forward pass).
    // v: [N, H*4] GRU activations (from forward pass).
    // gate_pre: [N, H] pre-activation gate (from forward pass).
    // dy: [N, H] gradient of loss w.r.t. gated output.
    // dx: [N, C] gradient of loss w.r.t. input (output).
    // dW: [C, H*3] gradient w.r.t. GRU input weights (accumulated).
    // dR: [H, H*3] gradient w.r.t. GRU recurrent weights (accumulated).
    // dbx: [H*3] gradient w.r.t. GRU input bias (accumulated).
    // dbr: [H*3] gradient w.r.t. GRU recurrent bias (accumulated).
    // dWg_x: [C, H] gradient w.r.t. gate input weights (accumulated).
    // dWg_h: [H, H] gradient w.r.t. gate hidden weights (accumulated).
    // dbg: [H] gradient w.r.t. gate bias (accumulated).
    // dh: [N, H] gradient w.r.t. previous hidden state (input/output).
    // dp, dq: [N, H*3] temporary workspaces.
    // tmp_dgate: [N, H] temporary workspace.
    void Iterate(
        const T* W_t,
        const T* R_t,
        const T* bx,
        const T* br,
        const T* Wg_x_t,
        const T* Wg_h_t,
        const T* bg,
        const T* x_t,
        const T* h,
        const T* h_out,
        const T* v,
        const T* gate_pre,
        const T* dy,
        T* dx,
        T* dW,
        T* dR,
        T* dbx,
        T* dbr,
        T* dWg_x,
        T* dWg_h,
        T* dbg,
        T* dh,
        T* dp,
        T* dq,
        T* tmp_dgate);

    void Run(
        const int steps,
        const T* W_t,
        const T* R_t,
        const T* bx,
        const T* br,
        const T* Wg_x_t,
        const T* Wg_h_t,
        const T* bg,
        const T* x_t,
        const T* h,
        const T* h_out,
        const T* v,
        const T* gate_pre,
        const T* dy,
        T* dx,
        T* dW,
        T* dR,
        T* dbx,
        T* dbr,
        T* dWg_x,
        T* dWg_h,
        T* dbg,
        T* dh,
        T* dp,
        T* dq,
        T* tmp_dgate);

  private:
    void IterateInternal(
        const T* R_t,
        const T* Wg_h_t,
        const T* h,
        const T* h_out,
        const T* v,
        const T* gate_pre,
        const T* dy,
        T* dbx,
        T* dbr,
        T* dbg,
        T* dh,
        T* dh_gru,
        T* dp,
        T* dq,
        T* tmp_dgate);

    struct private_data;
    private_data* data_;
};

}  // namespace gru_silu
}  // namespace v0
}  // namespace haste
