// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// Fused Elman RNN with SiLU gating.
//
// Architecture per timestep:
//   hidden = tanh(W1 @ [x_t, h])    -- [B, H] where H = expansion * D
//   h_new = W2 @ hidden             -- [B, D]
//   gate = silu(Wgx @ x_t + Wgh @ h_new + bias)  -- [B, D]
//   out = h_new * gate              -- [B, D]

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace elman {

template<typename T>
class ForwardPass {
  public:
    // training: `true` if the caller intends to perform a backward pass to compute gradients.
    // batch_size: the number of training/inference inputs provided in each tensor.
    // input_size: the dimension of each input vector (typically D).
    // hidden_size: the expanded hidden dimension (typically expansion * D).
    // output_size: the output dimension (typically D).
    // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const int output_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    // Releases internal resources.
    // Blocks until all iterations have completed executing on the GPU.
    ~ForwardPass();

    // Performs the full forward pass of the Elman RNN.
    //
    // steps: number of timesteps T.
    // W1: [H, D+D] the first layer weights, first D columns for input, second D for recurrent.
    // W2: [D, H] the output projection weights.
    // Wgx: [D, D] the gate input weights.
    // Wgh: [D, D] the gate recurrent weights.
    // bias: [D] the gate bias.
    // x: [T*B, D] the input sequence (time-major, flattened).
    // h: [(T+1)*B, D] the hidden states. h[0:B*D] should be initialized to the initial state (typically zeros).
    //    After the call, h[(t+1)*B*D : (t+2)*B*D] contains the output at timestep t.
    // v: [T*B, D*3] saved activations for backward pass (h_new, gate_logit, gate per timestep).
    //    Can be nullptr if training is false.
    // hidden: [T*B, H] workspace for tanh outputs (also saved for backward).
    // tmp_Wx1: [T*B, H] workspace for pre-computed W1_x @ x.
    // tmp_gx: [T*B, D] workspace for pre-computed Wgx @ x.
    // tmp_gh: [B, D] workspace for gate_h computation.
    // tmp_Wg2: [D, H] workspace for pre-computed Wgh @ W2 (fused gate weight).
    void Run(
        const int steps,
        const T* W1,
        const T* W2,
        const T* Wgx,
        const T* Wgh,
        const T* bias,
        const T* x,
        T* h,
        T* v,
        T* hidden,
        T* tmp_Wx1,
        T* tmp_gx,
        T* tmp_gh,
        T* tmp_Wg2);

  private:
    struct private_data;
    private_data* data_;
};

template<typename T>
class BackwardPass {
  public:
    // batch_size: the number of training inputs provided in each tensor.
    // input_size: the dimension of each input vector (typically D).
    // hidden_size: the expanded hidden dimension (typically expansion * D).
    // output_size: the output dimension (typically D).
    // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
    BackwardPass(
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const int output_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    // Releases internal resources.
    // Blocks until all iterations have completed executing on the GPU.
    ~BackwardPass();

    // Performs the full backward pass of the Elman RNN.
    //
    // steps: number of timesteps T.
    // W1_t: [D+D, H] the transpose of W1.
    // W2_t: [H, D] the transpose of W2.
    // Wgx_t: [D, D] the transpose of Wgx.
    // Wgh_t: [D, D] the transpose of Wgh.
    // x_t: [D, T*B] the transpose of the input sequence.
    // h: [(T+1)*B, D] the hidden states from forward pass.
    // hidden: [T*B, H] the tanh outputs from forward pass.
    // v: [T*B, D*3] the saved activations from forward pass.
    // dh_new: [(T+1)*B, D] the gradient of the output (loss gradient w.r.t. h).
    // dx: [T*B, D] output gradient w.r.t. input.
    // dW1: [H, D+D] output gradient w.r.t. W1 (accumulated).
    // dW2: [D, H] output gradient w.r.t. W2 (accumulated).
    // dWgx: [D, D] output gradient w.r.t. Wgx (accumulated).
    // dWgh: [D, D] output gradient w.r.t. Wgh (accumulated).
    // dbias: [D] output gradient w.r.t. bias (accumulated).
    // dh: [B, D] gradient of initial hidden state (output).
    // tmp workspaces: various temporary buffers.
    void Run(
        const int steps,
        const T* W1_t,
        const T* W2_t,
        const T* Wgx_t,
        const T* Wgh_t,
        const T* x_t,
        const T* h,
        const T* hidden,
        const T* v,
        const T* dh_new,
        T* dx,
        T* dW1,
        T* dW2,
        T* dWgx,
        T* dWgh,
        T* dbias,
        T* dh,
        T* tmp_dgate,
        T* tmp_dh_new,
        T* tmp_dhidden,
        T* tmp_dcombined);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace elman
}  // namespace v0
}  // namespace haste
