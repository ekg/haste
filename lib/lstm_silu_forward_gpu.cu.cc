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

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// SiLU activation function
template<typename T>
__device__ __forceinline__
T silu(T x) {
    return x / (static_cast<T>(1.0) + exp(-x));
}

template<>
__device__ __forceinline__
__half silu(__half x) {
    float xf = __half2float(x);
    return __float2half(xf / (1.0f + expf(-xf)));
}

template<>
__device__ __forceinline__
__nv_bfloat16 silu(__nv_bfloat16 x) {
    float xf = __bfloat162float(x);
    return __float2bfloat16(xf / (1.0f + expf(-xf)));
}

// LSTM pointwise operations with SiLU gating
template<typename T, bool Training>
__global__
void LSTMPointwiseWithSiLU(
    const int batch_dim,
    const int hidden_dim,
    const T* Wx,        // [N, H*4] precomputed W @ x
    const T* Rh,        // [N, H*4] precomputed R @ h
    const T* b,         // [H*4] LSTM bias
    const T* gate_in,   // [N, H] precomputed Wg_x @ x + Wg_h @ h (will be computed later for h_out)
    const T* bg,        // [H] gate bias
    const T* h,         // [N, H] previous hidden state
    const T* c,         // [N, H] previous cell state
    T* h_out,           // [N, H] LSTM hidden output
    T* c_out,           // [N, H] LSTM cell output
    T* y_out,           // [N, H] gated output
    T* v_out,           // [N, H*4] LSTM activations for backward
    T* gate_pre_out) {  // [N, H] gate pre-activation for backward

    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim)
        return;

    const int weight_idx = col * (hidden_dim * 4) + row;
    const int output_idx = col * hidden_dim + row;

    const int i_idx = weight_idx + 0 * hidden_dim;
    const int g_idx = weight_idx + 1 * hidden_dim;
    const int f_idx = weight_idx + 2 * hidden_dim;
    const int o_idx = weight_idx + 3 * hidden_dim;

    // LSTM gates
    const T i = sigmoid(Wx[i_idx] + Rh[i_idx] + b[row + 0 * hidden_dim]);
    const T g = tanh   (Wx[g_idx] + Rh[g_idx] + b[row + 1 * hidden_dim]);
    const T f = sigmoid(Wx[f_idx] + Rh[f_idx] + b[row + 2 * hidden_dim]);
    const T o = sigmoid(Wx[o_idx] + Rh[o_idx] + b[row + 3 * hidden_dim]);

    if (Training) {
        v_out[i_idx] = i;
        v_out[g_idx] = g;
        v_out[f_idx] = f;
        v_out[o_idx] = o;
    }

    // LSTM cell update
    const T c_new = (f * c[output_idx]) + (i * g);
    const T h_new = o * tanh(c_new);

    c_out[output_idx] = c_new;
    h_out[output_idx] = h_new;
}

// Separate kernel for SiLU gating (needs h_out to compute gate)
template<typename T, bool Training>
__global__
void SiLUGatingKernel(
    const int batch_dim,
    const int hidden_dim,
    const T* h_out,         // [N, H] LSTM hidden output
    const T* gate_in,       // [N, H] Wg_x @ x + Wg_h @ h_out
    const T* bg,            // [H] gate bias
    T* y_out,               // [N, H] gated output
    T* gate_pre_out) {      // [N, H] gate pre-activation

    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim)
        return;

    const int idx = col * hidden_dim + row;

    // SiLU gate: silu(Wg_x @ x + Wg_h @ h_out + bg)
    const T pre = gate_in[idx] + bg[row];
    const T gate_val = silu(pre);

    if (Training && gate_pre_out != nullptr) {
        gate_pre_out[idx] = pre;
    }

    y_out[idx] = h_out[idx] * gate_val;
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace lstm_silu {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[2];
    cudaEvent_t event;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
    data_->training = training;
    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;
    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
    cudaStreamSynchronize(data_->stream[1]);
    cudaStreamSynchronize(data_->stream[0]);
    cudaEventDestroy(data_->event);
    cudaStreamDestroy(data_->stream[1]);
    cudaStreamDestroy(data_->stream[0]);
    delete data_;
}

template<typename T>
void ForwardPass<T>::IterateInternal(
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
    T* tmp_gate_h) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);
    static const T beta_one = static_cast<T>(1.0);

    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    // Compute R @ h
    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 4, batch_size, hidden_size,
        &alpha,
        R, hidden_size * 4,
        h, hidden_size,
        &beta,
        tmp_Rh, hidden_size * 4);

    cudaStreamWaitEvent(stream1, event, 0);

    // Compute LSTM gates (without SiLU gating yet)
    const dim3 blockDim(64, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    if (training) {
        LSTMPointwiseWithSiLU<T, true><<<gridDim, blockDim, 0, stream1>>>(
            batch_size,
            hidden_size,
            v,  // Wx is stored in v initially
            tmp_Rh,
            b,
            nullptr,  // gate_in computed later
            bg,
            h,
            c,
            h_out,
            c_out,
            y_out,
            v,
            gate_pre);
    } else {
        LSTMPointwiseWithSiLU<T, false><<<gridDim, blockDim, 0, stream1>>>(
            batch_size,
            hidden_size,
            v,
            tmp_Rh,
            b,
            nullptr,
            bg,
            h,
            c,
            h_out,
            c_out,
            y_out,
            nullptr,
            nullptr);
    }

    // Now compute gate using h_out: Wg_h @ h_out
    // tmp_gate_h already contains Wg_x @ x from the caller
    // We add Wg_h @ h_out to it
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size,
        &alpha,
        Wg_h, hidden_size,
        h_out, hidden_size,
        &beta_one,  // Add to existing Wg_x @ x
        tmp_gate_x, hidden_size);

    // Apply SiLU gating
    if (training) {
        SiLUGatingKernel<T, true><<<gridDim, blockDim, 0, stream1>>>(
            batch_size,
            hidden_size,
            h_out,
            tmp_gate_x,
            bg,
            y_out,
            gate_pre);
    } else {
        SiLUGatingKernel<T, false><<<gridDim, blockDim, 0, stream1>>>(
            batch_size,
            hidden_size,
            h_out,
            tmp_gate_x,
            bg,
            y_out,
            nullptr);
    }
}

template<typename T>
void ForwardPass<T>::Iterate(
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
    T* tmp_gate) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    // Compute W @ x -> v
    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 4, batch_size, input_size,
        &alpha,
        W, hidden_size * 4,
        x, input_size,
        &beta,
        v, hidden_size * 4);

    // Compute Wg_x @ x -> tmp_gate
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, input_size,
        &alpha,
        Wg_x, hidden_size,
        x, input_size,
        &beta,
        tmp_gate, hidden_size);

    cudaEventRecord(event, stream2);

    IterateInternal(
        R, b, Wg_h, bg,
        h, c, h_out, c_out, y_out,
        v, gate_pre, tmp_Rh, tmp_gate, nullptr);

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void ForwardPass<T>::Run(
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
    T* tmp_gate) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    // Compute W @ x for all timesteps -> v
    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 4, steps * batch_size, input_size,
        &alpha,
        W, hidden_size * 4,
        x, input_size,
        &beta,
        v, hidden_size * 4);

    // Compute Wg_x @ x for all timesteps -> tmp_gate (reusing gate_pre as temp)
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, steps * batch_size, input_size,
        &alpha,
        Wg_x, hidden_size,
        x, input_size,
        &beta,
        gate_pre, hidden_size);

    const int NH = batch_size * hidden_size;

    for (int i = 0; i < steps; ++i) {
        // Copy Wg_x @ x for this timestep to tmp_gate
        cudaMemcpyAsync(tmp_gate, gate_pre + i * NH, NH * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream1);

        IterateInternal(
            R, b, Wg_h, bg,
            h + i * NH,
            c + i * NH,
            h + (i + 1) * NH,
            c + (i + 1) * NH,
            y + i * NH,
            v + i * NH * 4,
            gate_pre + i * NH,
            tmp_Rh,
            tmp_gate,
            nullptr);
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;

}  // namespace lstm_silu
}  // namespace v0
}  // namespace haste
