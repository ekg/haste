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

// GRU + SiLU Selectivity Gate Backward Pass
//
// Gradient flow:
// 1. dy -> dh_gru, dgate_pre (through SiLU gating)
// 2. dgate_pre -> dWg_x, dWg_h, dbg, additional dh_gru
// 3. dh_gru -> standard GRU backward

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "blas.h"
#include "device_assert.h"
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
// y = h_gru * silu(gate_pre)
// dy -> dh_gru = dy * silu(gate_pre)
// dy -> dgate_pre = dy * h_gru * d_silu(gate_pre)
template<typename T>
__global__
void SiLUGatingBackwardKernel(
    const int batch_dim,
    const int hidden_dim,
    const T* dy,           // [N, H] gradient of output
    const T* h_gru,        // [N, H] GRU output (from forward)
    const T* gate_pre,     // [N, H] pre-activation (from forward)
    T* dh_gru,             // [N, H] gradient w.r.t. GRU output
    T* dgate_pre,          // [N, H] gradient w.r.t. gate pre-activation
    T* dbg) {              // [H] gradient w.r.t. gate bias (atomically accumulated)

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_dim * hidden_dim;

    if (idx >= total)
        return;

    const int h_idx = idx % hidden_dim;
    const T dy_val = dy[idx];
    const T h_val = h_gru[idx];
    const T pre_val = gate_pre[idx];

    // silu(x) = x * sigmoid(x)
    const T sig = sigmoid(pre_val);
    const T silu_val = pre_val * sig;

    // dh_gru = dy * silu(gate_pre)
    dh_gru[idx] = dy_val * silu_val;

    // dgate_pre = dy * h_gru * d_silu(gate_pre)
    const T dsilu = d_silu(pre_val);
    const T dg = dy_val * h_val * dsilu;
    dgate_pre[idx] = dg;

    // Accumulate bias gradient
    atomicAdd(&dbg[h_idx], dg);
}

// GRU pointwise backward kernel (same as original)
template<typename T>
__global__
void GRUPointwiseBackward(
    const int batch_dim,
    const int hidden_dim,
    const T* h,
    const T* v,
    const T* dh_new,
    T* dbx_out,
    T* dbr_out,
    T* dh_inout,
    T* dp_out,
    T* dq_out) {

    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim)
        return;

    const int base_idx = col * hidden_dim + row;

    T dh_total = dh_new[base_idx] + dh_inout[base_idx];

    const int stride4_base_idx = col * (hidden_dim * 4) + row;
    const int z_idx = stride4_base_idx + 0 * hidden_dim;
    const int r_idx = stride4_base_idx + 1 * hidden_dim;
    const int g_idx = stride4_base_idx + 2 * hidden_dim;
    const int q_g_idx = stride4_base_idx + 3 * hidden_dim;

    const T z = v[z_idx];
    const T r = v[r_idx];
    const T g = v[g_idx];
    const T q_g = v[q_g_idx];

    dh_inout[base_idx] = z * dh_total;

    const T dg = (static_cast<T>(1.0) - z) * dh_total;
    const T dz = (h[base_idx] - g) * dh_total;
    const T dp_g = d_tanh(g) * dg;
    const T dq_g = dp_g * r;
    const T dr = dp_g * q_g;
    const T dp_r = d_sigmoid(r) * dr;
    const T dq_r = dp_r;
    const T dp_z = d_sigmoid(z) * dz;
    const T dq_z = dp_z;

    const int idx = col * (hidden_dim * 3) + row;

    dp_out[idx + 0 * hidden_dim] = dp_z;
    dp_out[idx + 1 * hidden_dim] = dp_r;
    dp_out[idx + 2 * hidden_dim] = dp_g;

    dq_out[idx + 0 * hidden_dim] = dq_z;
    dq_out[idx + 1 * hidden_dim] = dq_r;
    dq_out[idx + 2 * hidden_dim] = dq_g;

    atomicAdd(&dbx_out[row + 0 * hidden_dim], dp_z);
    atomicAdd(&dbx_out[row + 1 * hidden_dim], dp_r);
    atomicAdd(&dbx_out[row + 2 * hidden_dim], dp_g);

    atomicAdd(&dbr_out[row + 0 * hidden_dim], dq_z);
    atomicAdd(&dbr_out[row + 1 * hidden_dim], dq_r);
    atomicAdd(&dbr_out[row + 2 * hidden_dim], dq_g);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
template<typename T>
__global__
void SiLUGatingBackwardKernel(
    const int batch_dim, const int hidden_dim,
    const half* dy, const half* h_gru, const half* gate_pre,
    half* dh_gru, half* dgate_pre, half* dbg) {
    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}

template<typename T>
__global__
void GRUPointwiseBackward(
    const int batch_dim, const int hidden_dim,
    const half* h, const half* v, const half* dh_new,
    half* dbx_out, half* dbr_out, half* dh_inout, half* dp_out, half* dq_out) {
    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}
#endif

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace gru_silu {

template<typename T>
struct BackwardPass<T>::private_data {
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[3];
    cudaEvent_t event[2];
    cudaStream_t sync_stream;
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
    data_->sync_stream = stream;

    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaStreamCreate(&data_->stream[2]);
    cudaEventCreateWithFlags(&data_->event[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&data_->event[1], cudaEventDisableTiming);
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
    if (data_->sync_stream) {
        for (int i = 0; i < 3; ++i) {
            cudaEventRecord(data_->event[0], data_->stream[i]);
            cudaStreamWaitEvent(data_->sync_stream, data_->event[0], 0);
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            cudaStreamSynchronize(data_->stream[i]);
        }
    }
    cudaEventDestroy(data_->event[1]);
    cudaEventDestroy(data_->event[0]);
    cudaStreamDestroy(data_->stream[2]);
    cudaStreamDestroy(data_->stream[1]);
    cudaStreamDestroy(data_->stream[0]);
    delete data_;
}

template<typename T>
void BackwardPass<T>::Iterate(
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

    // Step 1: Backprop through SiLU gating
    // dy -> dh_gru, dgate_pre, dbg
    const int total = batch_size * hidden_size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    // Use tmp_dgate as dh_gru_from_gate temporarily
    T* dh_gru = tmp_dgate;

    SiLUGatingBackwardKernel<T><<<blocks, threads, 0, stream0>>>(
        batch_size, hidden_size, dy, h_out, gate_pre,
        dh_gru, tmp_dgate, dbg);  // Note: tmp_dgate holds both dh_gru and dgate_pre in sequence

    // Actually, let's use dh as the accumulator for dh_gru
    // and tmp_dgate for dgate_pre

    // Re-design: We need separate buffers for dh_gru_gate and dgate_pre
    // Let's allocate them properly from tmp_dgate which should be [N, H*2]
    T* dh_gru_gate = tmp_dgate;
    T* dgate_pre_val = tmp_dgate;  // Reuse since we consume dgate_pre before modifying dh_gru_gate

    SiLUGatingBackwardKernel<T><<<blocks, threads, 0, stream0>>>(
        batch_size, hidden_size, dy, h_out, gate_pre,
        dh_gru_gate, dgate_pre_val, dbg);
    cudaEventRecord(data_->event[0], stream0);

    // Step 2: Backprop through gate weights
    // dgate_pre -> dWg_x, dWg_h
    // dx += Wg_x_t @ dgate_pre
    // dh_gru_total = dh_gru_gate + Wg_h_t @ dgate_pre

    cudaStreamWaitEvent(stream1, data_->event[0], 0);
    cublasSetStream(blas_handle, stream1);

    // dWg_x += dgate_pre @ x_t (transpose to get proper shape)
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, input_size, batch_size,
        &alpha,
        dgate_pre_val, hidden_size,
        x_t, batch_size,
        &beta_sum,
        dWg_x, hidden_size);

    // dWg_h += dgate_pre @ h_out_t
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size, hidden_size, batch_size,
        &alpha,
        dgate_pre_val, hidden_size,
        h_out, hidden_size,
        &beta_sum,
        dWg_h, hidden_size);

    // dx_gate = Wg_x_t @ dgate_pre (this contributes to dx)
    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        input_size, batch_size, hidden_size,
        &alpha,
        Wg_x_t, input_size,
        dgate_pre_val, hidden_size,
        &beta_assign,
        dx, input_size);  // Will accumulate GRU's dx later

    // dh_gru_from_Wg_h = Wg_h_t @ dgate_pre
    // Add this to dh_gru_gate to get total dh_gru
    cublasSetStream(blas_handle, stream0);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size,
        &alpha,
        Wg_h_t, hidden_size,
        dgate_pre_val, hidden_size,
        &beta_sum,  // Accumulate to dh_gru_gate
        dh_gru_gate, hidden_size);

    // Step 3: GRU backward
    // dh_gru_gate now contains total gradient w.r.t. h_out (GRU output)

    // GRU pointwise backward
    const dim3 blockDim2D(32, 16);
    const dim3 gridDim2D(
        (hidden_size + blockDim2D.x - 1) / blockDim2D.x,
        (batch_size + blockDim2D.y - 1) / blockDim2D.y);

    GRUPointwiseBackward<T><<<gridDim2D, blockDim2D, 0, stream0>>>(
        batch_size, hidden_size, h, v, dh_gru_gate,
        dbx, dbr, dh, dp, dq);
    cudaEventRecord(data_->event[1], stream0);

    // dR += dq @ h_t
    cublasSetStream(blas_handle, stream0);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size * 3, hidden_size, batch_size,
        &alpha,
        dq, hidden_size * 3,
        h, hidden_size,
        &beta_sum,
        dR, hidden_size * 3);

    // dh += R_t @ dq
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size * 3,
        &alpha,
        R_t, hidden_size,
        dq, hidden_size * 3,
        &beta_sum,
        dh, hidden_size);

    // Wait for gate gradients
    cudaStreamWaitEvent(stream1, data_->event[1], 0);

    // dW += dp @ x_t
    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 3, input_size, batch_size,
        &alpha,
        dp, hidden_size * 3,
        x_t, batch_size,
        &beta_sum,
        dW, hidden_size * 3);

    // dx += W_t @ dp (accumulate to dx which already has gate contribution)
    cublasSetStream(blas_handle, stream2);
    cudaStreamWaitEvent(stream2, data_->event[1], 0);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        input_size, batch_size, hidden_size * 3,
        &alpha,
        W_t, input_size,
        dp, hidden_size * 3,
        &beta_sum,  // Accumulate to dx
        dx, input_size);

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void BackwardPass<T>::Run(
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
    T* tmp_dgate) {

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
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
    const int NH3 = batch_size * hidden_size * 3;
    const int NH4 = batch_size * hidden_size * 4;

    // Process timesteps in reverse order
    for (int i = steps - 1; i >= 0; --i) {
        // Step 1: SiLU gating backward
        const int total = batch_size * hidden_size;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;

        T* dh_gru_gate = tmp_dgate + i * NH;
        T* dgate_pre_val = tmp_dgate + i * NH;

        SiLUGatingBackwardKernel<T><<<blocks, threads, 0, stream0>>>(
            batch_size, hidden_size,
            dy + i * NH,
            h_out + i * NH,
            gate_pre + i * NH,
            dh_gru_gate, dgate_pre_val, dbg);
        cudaEventRecord(data_->event[0], stream0);

        // Step 2: Gate weight gradients (accumulated over all timesteps)
        cudaStreamWaitEvent(stream1, data_->event[0], 0);
        cublasSetStream(blas_handle, stream1);

        // dWg_x += dgate_pre @ x_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_size, input_size, batch_size,
            &alpha,
            dgate_pre_val, hidden_size,
            x_t + i * NC, batch_size,
            &beta_sum,
            dWg_x, hidden_size);

        // dWg_h += dgate_pre @ h_out_t
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            hidden_size, hidden_size, batch_size,
            &alpha,
            dgate_pre_val, hidden_size,
            h_out + i * NH, hidden_size,
            &beta_sum,
            dWg_h, hidden_size);

        // dh_gru += Wg_h_t @ dgate_pre
        cublasSetStream(blas_handle, stream0);
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_size, batch_size, hidden_size,
            &alpha,
            Wg_h_t, hidden_size,
            dgate_pre_val, hidden_size,
            &beta_sum,
            dh_gru_gate, hidden_size);

        // Step 3: GRU backward
        const dim3 blockDim2D(32, 16);
        const dim3 gridDim2D(
            (hidden_size + blockDim2D.x - 1) / blockDim2D.x,
            (batch_size + blockDim2D.y - 1) / blockDim2D.y);

        GRUPointwiseBackward<T><<<gridDim2D, blockDim2D, 0, stream0>>>(
            batch_size, hidden_size,
            h + i * NH, v + i * NH4, dh_gru_gate,
            dbx, dbr, dh, dp + i * NH3, dq + i * NH3);
        cudaEventRecord(data_->event[1], stream0);

        // dh += R_t @ dq
        cublasSetStream(blas_handle, stream0);
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_size, batch_size, hidden_size * 3,
            &alpha,
            R_t, hidden_size,
            dq + i * NH3, hidden_size * 3,
            &beta_sum,
            dh, hidden_size);
    }

    // Batch compute weight gradients
    cudaStreamWaitEvent(stream1, data_->event[1], 0);

    // dR += dq @ h_t (batched over all timesteps)
    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size * 3, hidden_size, batch_size * steps,
        &alpha,
        dq, hidden_size * 3,
        h, hidden_size,
        &beta_sum,
        dR, hidden_size * 3);

    // dW += dp @ x_t
    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 3, input_size, batch_size * steps,
        &alpha,
        dp, hidden_size * 3,
        x_t, batch_size * steps,
        &beta_sum,
        dW, hidden_size * 3);

    // dx = W_t @ dp + Wg_x_t @ dgate_pre
    cublasSetStream(blas_handle, stream2);
    cudaStreamWaitEvent(stream2, data_->event[1], 0);

    // dx = W_t @ dp
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        input_size, batch_size * steps, hidden_size * 3,
        &alpha,
        W_t, input_size,
        dp, hidden_size * 3,
        &beta_assign,
        dx, input_size);

    // dx += Wg_x_t @ dgate_pre
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

}  // namespace gru_silu
}  // namespace v0
}  // namespace haste
