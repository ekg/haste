// Copyright 2020 LMNT, Inc. All Rights Reserved.
// Modified 2024 for SkipElman
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
//
// SkipElman: Elman RNN with skip connection and separate input/hidden biases
//   z = sigmoid(Wz @ x + Rz @ h + bx_z + bh_z)   # update gate
//   a = tanh(Wa @ x + Ra @ h + bx_a + bh_a)      # candidate
//   h_new = z * h + (1-z) * a                     # skip connection!
//
// Simpler than GRU (2 components instead of 3, no reset gate).
// Retains gradient highway via z*h skip connection.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace skip_elman {

template<typename T>
class ForwardPass {
  public:
    // training: `true` if the caller intends to perform a backward pass.
    // batch_size: the number of training/inference inputs provided in each tensor.
    // input_size: the dimension of each input vector.
    // hidden_size: the expected dimension of each output vector.
    // blas_handle: an initialized cuBLAS handle.
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    ~ForwardPass();

    // Run full sequence forward pass.
    //
    // W: [C,H*2] input weight matrix (z and a components).
    // R: [H,H*2] recurrent weight matrix.
    // bx: [H*2] input bias (separate from hidden bias).
    // bh: [H*2] hidden bias (separate from input bias).
    // x: [T*N,C] input sequence.
    // h: [(T+1)*N,H] hidden states (h[0] is initial state, typically zeros).
    // v: [T*N,H*2] intermediate activations (for backward pass).
    // tmp_Wx: [T*N,H*2] temporary workspace.
    // tmp_Rh: [N,H*2] temporary workspace (reused each step).
    // zoneout_prob: probability of zoning out hidden activation.
    // zoneout_mask: [T*N,H] binary mask for zoneout (may be null).
    void Run(
        const int steps,
        const T* W,
        const T* R,
        const T* bx,
        const T* bh,
        const T* x,
        T* h,
        T* v,
        T* tmp_Wx,
        T* tmp_Rh,
        const float zoneout_prob,
        const T* zoneout_mask);

  private:
    void IterateInternal(
        const T* R,
        const T* bx,
        const T* bh,
        const T* h,
        T* h_out,
        T* v,
        T* tmp_Wx,
        T* tmp_Rh,
        const float zoneout_prob,
        const T* zoneout_mask);

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

    // Run full sequence backward pass.
    //
    // W_t: [H*2,C] transpose of input weight matrix.
    // R_t: [H*2,H] transpose of recurrent weight matrix.
    // bx: [H*2] input bias.
    // bh: [H*2] hidden bias.
    // x_t: [C,T*N] transpose of input sequence.
    // h: [(T+1)*N,H] hidden states from forward pass.
    // v: [T*N,H*2] intermediate activations from forward pass.
    // dh_new: [(T+1)*N,H] gradient of hidden states.
    // dx: [T*N,C] output: gradient of input.
    // dW: [C,H*2] output: gradient of input weights.
    // dR: [H,H*2] output: gradient of recurrent weights.
    // dbx: [H*2] output: gradient of input bias.
    // dbh: [H*2] output: gradient of hidden bias.
    // dh: [N,H] in/out: gradient of initial hidden state.
    // dp: [T*N,H*2] temporary workspace.
    // dq: [T*N,H*2] temporary workspace.
    // zoneout_mask: [T*N,H] same mask as forward pass (may be null).
    void Run(
        const int steps,
        const T* W_t,
        const T* R_t,
        const T* bx,
        const T* bh,
        const T* x_t,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dx,
        T* dW,
        T* dR,
        T* dbx,
        T* dbh,
        T* dh,
        T* dp,
        T* dq,
        const T* zoneout_mask);

  private:
    void IterateInternal(
        const T* R_t,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dbx,
        T* dbh,
        T* dh,
        T* dp,
        T* dq,
        const T* zoneout_mask);

    struct private_data;
    private_data* data_;
};

}  // namespace skip_elman
}  // namespace v0
}  // namespace haste
