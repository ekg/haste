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

// LSTM + SiLU Selectivity Gate Backward Pass
//
// Gradient flow:
// 1. dy -> dh_lstm, dgate_pre (through SiLU gating)
// 2. dgate_pre -> dWg_x, dWg_h, dbg, additional dh_lstm
// 3. dh_lstm -> standard LSTM backward

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// SiLU derivative: d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//                                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
template<typename T>
__device__ __forceinline__
T d_silu(T x) {
    T sig = sigmoid(x);
    return sig * (static_cast<T>(1.0) + x * (static_cast<T>(1.0) - sig));
}

template<>
__device__ __forceinline__
__half d_silu(__half x) {
    float xf = __half2float(x);
    float sig = 1.0f / (1.0f + expf(-xf));
    return __float2half(sig * (1.0f + xf * (1.0f - sig)));
}

template<>
__device__ __forceinline__
__nv_bfloat16 d_silu(__nv_bfloat16 x) {
    float xf = __bfloat162float(x);
    float sig = 1.0f / (1.0f + expf(-xf));
    return __float2bfloat16(sig * (1.0f + xf * (1.0f - sig)));
}

// SiLU backward kernel
// y = h_lstm * silu(gate_pre)
// dy -> dh_lstm = dy * silu(gate_pre)
// dy -> dgate_pre = dy * h_lstm * d_silu(gate_pre)
template<typename T>
__global__
void SiLUGatingBackwardKernel(
    const int batch_dim,
    const int hidden_dim,
    const T* dy,           // [N, H] gradient of output
    const T* h_lstm,       // [N, H] LSTM output (from forward)
    const T* gate_pre,     // [N, H] pre-activation (from forward)
    T* dh_lstm,            // [N, H] gradient w.r.t. LSTM output
    T* dgate_pre,          // [N, H] gradient w.r.t. gate pre-activation
    T* dbg) {              // [H] gradient w.r.t. gate bias (atomically accumulated)

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_dim * hidden_dim;

    if (idx >= total)
        return;

    const int h_idx = idx % hidden_dim;
    const T dy_val = dy[idx];
    const T h_val = h_lstm[idx];
    const T pre_val = gate_pre[idx];

    // silu(x) = x * sigmoid(x)
    const T sig = sigmoid(pre_val);
    const T silu_val = pre_val * sig;

    // dh_lstm = dy * silu(gate_pre)
    dh_lstm[idx] = dy_val * silu_val;

    // dgate_pre = dy * h_lstm * d_silu(gate_pre)
    const T dsilu = d_silu(pre_val);
    const T dg = dy_val * h_val * dsilu;
    dgate_pre[idx] = dg;

    // Accumulate bias gradient
    atomicAdd(&dbg[h_idx], dg);
}

// LSTM pointwise backward kernel
// Note: For LSTM_SiLU, cell state is not an external output, so dc_new is typically nullptr
template<typename T>
__global__
void LSTMPointwiseBackward(
    const int batch_dim,
    const int hidden_dim,
    const T* c,            // [N, H] previous cell state
    const T* v,            // [N, H*4] LSTM activations (i, g, f, o)
    const T* c_new,        // [N, H] current cell state
    const T* dh_new,       // [N, H] gradient of loss w.r.t. h_out
    const T* dc_new,       // [N, H] gradient of loss w.r.t. c_out (can be nullptr)
    T* db_out,             // [H*4] gradient w.r.t. bias (accumulated)
    T* dh_inout,           // [N, H] gradient w.r.t. previous hidden (input/output)
    T* dc_inout,           // [N, H] gradient w.r.t. previous cell (input/output)
    T* dv_out) {           // [N, H*4] gradient w.r.t. pre-activation

    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim)
        return;

    const int base_idx = col * hidden_dim + row;

    // dc_new is nullptr for LSTM_SiLU since cell state is not an external output
    const T dc_new_val = (dc_new != nullptr) ? dc_new[base_idx] : static_cast<T>(0.0);
          T dc_total = dc_new_val + dc_inout[base_idx];
          T dh_total = dh_new[base_idx] + dh_inout[base_idx];
    const T c_tanh = tanh(c_new[base_idx]);

    const int stride4_base_idx = col * (hidden_dim * 4) + row;
    const int i_idx = stride4_base_idx + 0 * hidden_dim;
    const int g_idx = stride4_base_idx + 1 * hidden_dim;
    const int f_idx = stride4_base_idx + 2 * hidden_dim;
    const int o_idx = stride4_base_idx + 3 * hidden_dim;

    const T i = v[i_idx];
    const T g = v[g_idx];
    const T f = v[f_idx];
    const T o = v[o_idx];

    dh_inout[base_idx] = static_cast<T>(0.0);

    const T do_ = c_tanh * dh_total;
    const T dc_tanh = o * dh_total;
            dc_total += d_tanh(c_tanh) * dc_tanh;
    const T df = c[base_idx] * dc_total;
    const T dc = f * dc_total;
    const T di = g * dc_total;
    const T dg = i * dc_total;
    const T dv_g = d_tanh(g) * dg;
    const T dv_o = d_sigmoid(o) * do_;
    const T dv_i = d_sigmoid(i) * di;
    const T dv_f = d_sigmoid(f) * df;

    atomicAdd(&db_out[row + 0 * hidden_dim], dv_i);
    atomicAdd(&db_out[row + 1 * hidden_dim], dv_g);
    atomicAdd(&db_out[row + 2 * hidden_dim], dv_f);
    atomicAdd(&db_out[row + 3 * hidden_dim], dv_o);

    dc_inout[base_idx] = dc;

    dv_out[i_idx] = dv_i;
    dv_out[g_idx] = dv_g;
    dv_out[f_idx] = dv_f;
    dv_out[o_idx] = dv_o;
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace lstm_silu {

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[3];
    cudaEvent_t event[2];
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {

    data_->batch_size = batch_size;
    data_->input_size = input_size;
    data_->hidden_size = hidden_size;
    data_->blas_handle = blas_handle;

    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaStreamCreate(&data_->stream[2]);
    cudaEventCreateWithFlags(&data_->event[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&data_->event[1], cudaEventDisableTiming);
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    for (int i = 0; i < 3; ++i) {
        cudaStreamSynchronize(data_->stream[i]);
    }
    cudaEventDestroy(data_->event[1]);
    cudaEventDestroy(data_->event[0]);
    cudaStreamDestroy(data_->stream[2]);
    cudaStreamDestroy(data_->stream[1]);
    cudaStreamDestroy(data_->stream[0]);
    delete data_;
}

template<typename T>
void BackwardPass<T>::IterateInternal(
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
    T* tmp_dgate) {

    const T alpha = static_cast<T>(1.0);
    const T beta_sum = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream0 = data_->stream[0];

    // Step 1: SiLU gating backward
    const int total = batch_size * hidden_size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    SiLUGatingBackwardKernel<T><<<blocks, threads, 0, stream0>>>(
        batch_size, hidden_size, dy, h_out, gate_pre,
        dh_lstm, tmp_dgate, dbg);

    // Step 2: Add gradient from Wg_h to dh_lstm
    cublasSetStream(blas_handle, stream0);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size,
        &alpha,
        Wg_h_t, hidden_size,
        tmp_dgate, hidden_size,
        &beta_sum,
        dh_lstm, hidden_size);

    // Step 3: LSTM pointwise backward
    const dim3 blockDim2D(64, 16);
    const dim3 gridDim2D(
        (hidden_size + blockDim2D.x - 1) / blockDim2D.x,
        (batch_size + blockDim2D.y - 1) / blockDim2D.y);

    // dc_new is nullptr because cell state is not an external output in LSTM_SiLU
    // dc_inout accumulates cell gradients from future timesteps
    LSTMPointwiseBackward<T><<<gridDim2D, blockDim2D, 0, stream0>>>(
        batch_size, hidden_size,
        c, v, c_new, dh_lstm, static_cast<T*>(nullptr),  // dc_new = nullptr
        db, dh, dc, dp);

    cudaEventRecord(data_->event[0], stream0);

    // dh += R_t @ dp
    cublasSetStream(blas_handle, stream0);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size * 4,
        &alpha,
        R_t, hidden_size,
        dp, hidden_size * 4,
        &beta_sum,
        dh, hidden_size);
}

template<typename T>
void BackwardPass<T>::Iterate(
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
    T* tmp_dgate) {

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const T alpha = static_cast<T>(1.0);
    const T beta_sum = static_cast<T>(1.0);
    const T beta_assign = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream0 = data_->stream[0];
    const cudaStream_t stream1 = data_->stream[1];
    const cudaStream_t stream2 = data_->stream[2];

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    IterateInternal(
        R_t, Wg_h_t, h, c, c_new, h_out, v, gate_pre, dy,
        db, dbg, dh, dc, dh_lstm, dp, tmp_dgate);

    cudaStreamWaitEvent(stream1, data_->event[0], 0);
    cudaStreamWaitEvent(stream2, data_->event[0], 0);

    // Gate weight gradients
    cublasSetStream(blas_handle, stream1);

    // dWg_x += tmp_dgate @ x_t
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, input_size, batch_size,
        &alpha,
        tmp_dgate, hidden_size,
        x_t, batch_size,
        &beta_sum,
        dWg_x, hidden_size);

    // dWg_h += tmp_dgate @ h_out_t
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size, hidden_size, batch_size,
        &alpha,
        tmp_dgate, hidden_size,
        h_out, hidden_size,
        &beta_sum,
        dWg_h, hidden_size);

    // LSTM weight gradients
    // dR += dp @ h_t
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size * 4, hidden_size, batch_size,
        &alpha,
        dp, hidden_size * 4,
        h, hidden_size,
        &beta_sum,
        dR, hidden_size * 4);

    // dW += dp @ x_t
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 4, input_size, batch_size,
        &alpha,
        dp, hidden_size * 4,
        x_t, batch_size,
        &beta_sum,
        dW, hidden_size * 4);

    // Input gradients
    cublasSetStream(blas_handle, stream2);

    // dx = W_t @ dp
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        input_size, batch_size, hidden_size * 4,
        &alpha,
        W_t, input_size,
        dp, hidden_size * 4,
        &beta_assign,
        dx, input_size);

    // dx += Wg_x_t @ tmp_dgate
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        input_size, batch_size, hidden_size,
        &alpha,
        Wg_x_t, input_size,
        tmp_dgate, hidden_size,
        &beta_sum,
        dx, input_size);

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void BackwardPass<T>::Run(
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
    T* tmp_dgate) {

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const T alpha = static_cast<T>(1.0);
    const T beta_sum = static_cast<T>(1.0);
    const T beta_assign = static_cast<T>(0.0);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream0 = data_->stream[0];
    const cudaStream_t stream1 = data_->stream[1];
    const cudaStream_t stream2 = data_->stream[2];

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    const int NH = batch_size * hidden_size;
    const int NC = batch_size * input_size;
    const int NH4 = batch_size * hidden_size * 4;

    // Process timesteps in reverse order
    for (int i = steps - 1; i >= 0; --i) {
        IterateInternal(
            R_t, Wg_h_t,
            h + i * NH,
            c + i * NH,
            c + (i + 1) * NH,
            h_out + i * NH,
            v + i * NH4,
            gate_pre + i * NH,
            dy + i * NH,
            db, dbg, dh, dc, dh_lstm + i * NH,
            dp + i * NH4, tmp_dgate + i * NH);
    }

    cudaEventRecord(data_->event[0], stream0);
    cudaStreamWaitEvent(stream1, data_->event[0], 0);
    cudaStreamWaitEvent(stream2, data_->event[0], 0);

    // Batch compute weight gradients
    cublasSetStream(blas_handle, stream1);

    // dR += dp @ h_t (batched)
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size * 4, hidden_size, batch_size * steps,
        &alpha,
        dp, hidden_size * 4,
        h, hidden_size,
        &beta_sum,
        dR, hidden_size * 4);

    // dW += dp @ x_t (batched)
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 4, input_size, batch_size * steps,
        &alpha,
        dp, hidden_size * 4,
        x_t, batch_size * steps,
        &beta_sum,
        dW, hidden_size * 4);

    // dWg_x += tmp_dgate @ x_t (batched)
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, input_size, batch_size * steps,
        &alpha,
        tmp_dgate, hidden_size,
        x_t, batch_size * steps,
        &beta_sum,
        dWg_x, hidden_size);

    // dWg_h += tmp_dgate @ h_out_t (batched)
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size, hidden_size, batch_size * steps,
        &alpha,
        tmp_dgate, hidden_size,
        h_out, hidden_size,
        &beta_sum,
        dWg_h, hidden_size);

    // Input gradients
    cublasSetStream(blas_handle, stream2);

    // dx = W_t @ dp
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        input_size, batch_size * steps, hidden_size * 4,
        &alpha,
        W_t, input_size,
        dp, hidden_size * 4,
        &beta_assign,
        dx, input_size);

    // dx += Wg_x_t @ tmp_dgate
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        input_size, batch_size * steps, hidden_size,
        &alpha,
        Wg_x_t, input_size,
        tmp_dgate, hidden_size,
        &beta_sum,
        dx, input_size);

    cublasSetStream(blas_handle, save_stream);
}

template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__half>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace lstm_silu
}  // namespace v0
}  // namespace haste
