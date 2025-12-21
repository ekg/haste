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

// GRU + SiLU Selectivity Gate Forward Pass
//
// This kernel fuses a standard GRU with an input-dependent output gate:
//   1. h_gru = GRU(x, h_prev)     -- Standard GRU recurrence
//   2. gate = silu(Wg_x @ x + Wg_h @ h_gru + bg)  -- SiLU selectivity gate
//   3. output = h_gru * gate      -- Gated output

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "blas.h"
#include "device_assert.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

// SiLU activation: x * sigmoid(x)
template<typename T>
__device__ __forceinline__
T silu(T x) {
    return x * sigmoid(x);
}

// Template specializations for half precision
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

// GRU pointwise operations kernel (unchanged from original)
template<typename T, bool Training>
__global__
void GRUPointwiseOperations(
    const int batch_dim,
    const int hidden_dim,
    const T* Wx,
    const T* Rh,
    const T* bx,
    const T* br,
    const T* h,
    T* h_out,
    T* v) {

    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= hidden_dim || col >= batch_dim)
        return;

    const int weight_idx = col * (hidden_dim * 3) + row;
    const int output_idx = col * hidden_dim + row;

    const int z_idx = weight_idx + 0 * hidden_dim;
    const int r_idx = weight_idx + 1 * hidden_dim;
    const int g_idx = weight_idx + 2 * hidden_dim;

    const int bz_idx = row + 0 * hidden_dim;
    const int br_idx = row + 1 * hidden_dim;
    const int bg_idx = row + 2 * hidden_dim;

    const T z = sigmoid(Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + br[bz_idx]);
    const T r = sigmoid(Wx[r_idx] + Rh[r_idx] + bx[br_idx] + br[br_idx]);
    const T g = tanh(Wx[g_idx] + r * (Rh[g_idx] + br[bg_idx]) + bx[bg_idx]);

    if (Training) {
        const int base_v_idx = col * (hidden_dim * 4) + row;
        v[base_v_idx + 0 * hidden_dim] = z;
        v[base_v_idx + 1 * hidden_dim] = r;
        v[base_v_idx + 2 * hidden_dim] = g;
        v[base_v_idx + 3 * hidden_dim] = Rh[g_idx] + br[bg_idx];
    }

    h_out[output_idx] = z * h[output_idx] + (static_cast<T>(1.0) - z) * g;
}

// SiLU gating kernel: y = h_gru * silu(gate_pre + bias)
// gate_pre already contains Wg_x @ x + Wg_h @ h_gru
template<typename T, bool Training>
__global__
void SiLUGatingKernel(
    const int batch_dim,
    const int hidden_dim,
    const T* h_gru,      // [N, H] GRU output
    const T* gate_in,    // [N, H] Wg_x @ x + Wg_h @ h_gru (accumulated)
    const T* bg,         // [H] gate bias
    T* y_out,            // [N, H] gated output
    T* gate_pre_out) {   // [N, H] pre-activation output (for backward)

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = batch_dim * hidden_dim;

    if (idx >= total)
        return;

    const int h_idx = idx % hidden_dim;

    // Compute pre-activation: gate_in + bias
    const T pre = gate_in[idx] + bg[h_idx];

    // Save pre-activation for backward pass
    if (Training && gate_pre_out != nullptr) {
        gate_pre_out[idx] = pre;
    }

    // Apply SiLU and multiply with GRU output
    y_out[idx] = h_gru[idx] * silu(pre);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
template<typename T, bool Training>
__global__
void GRUPointwiseOperations(
    const int batch_dim,
    const int hidden_dim,
    const half* Wx,
    const half* Rh,
    const half* bx,
    const half* br,
    const half* h,
    half* h_out,
    half* v) {
    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}

template<typename T, bool Training>
__global__
void SiLUGatingKernel(
    const int batch_dim,
    const int hidden_dim,
    const half* h_gru,
    const half* gate_in,
    const half* bg,
    half* y_out,
    half* gate_pre_out) {
    device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}
#endif

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace gru_silu {

template<typename T>
struct ForwardPass<T>::private_data {
    bool training;
    int batch_size;
    int input_size;
    int hidden_size;
    cublasHandle_t blas_handle;
    cudaStream_t stream[3];  // 3 streams for overlapping
    cudaEvent_t event[2];
    cudaStream_t sync_stream;
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
    data_->sync_stream = stream;

    cudaStreamCreate(&data_->stream[0]);
    cudaStreamCreate(&data_->stream[1]);
    cudaStreamCreate(&data_->stream[2]);
    cudaEventCreateWithFlags(&data_->event[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&data_->event[1], cudaEventDisableTiming);
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
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
void ForwardPass<T>::Iterate(
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
    T* tmp_gate) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream0 = data_->stream[0];
    const cudaStream_t stream1 = data_->stream[1];
    const cudaStream_t stream2 = data_->stream[2];

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    // Stream 1: Compute tmp_Wx = W @ x (GRU input projection)
    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 3, batch_size, input_size,
        &alpha,
        W, hidden_size * 3,
        x, input_size,
        &beta,
        tmp_Wx, hidden_size * 3);
    cudaEventRecord(data_->event[0], stream1);

    // Stream 2: Compute tmp_gate[0:H] = Wg_x @ x (gate input projection)
    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, input_size,
        &alpha,
        Wg_x, hidden_size,
        x, input_size,
        &beta,
        tmp_gate, hidden_size);
    cudaEventRecord(data_->event[1], stream2);

    // Stream 0: Compute tmp_Rh = R @ h (GRU recurrent projection)
    cublasSetStream(blas_handle, stream0);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 3, batch_size, hidden_size,
        &alpha,
        R, hidden_size * 3,
        h, hidden_size,
        &beta,
        tmp_Rh, hidden_size * 3);

    // Wait for tmp_Wx
    cudaStreamWaitEvent(stream0, data_->event[0], 0);

    // GRU pointwise operations
    const dim3 blockDim(32, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    if (data_->training) {
        GRUPointwiseOperations<T, true><<<gridDim, blockDim, 0, stream0>>>(
            batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, v);
    } else {
        GRUPointwiseOperations<T, false><<<gridDim, blockDim, 0, stream0>>>(
            batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, br, h, h_out, nullptr);
    }

    // Compute gate_h = Wg_h @ h_out (using h_out from GRU)
    // Note: We use tmp_gate + batch_size*hidden_size as workspace for gate_h
    T* tmp_gate_h = tmp_gate;  // Reuse tmp_gate after gate_x is consumed
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size,
        &alpha,
        Wg_h, hidden_size,
        h_out, hidden_size,
        &beta,
        tmp_gate_h, hidden_size);

    // Wait for gate_x computation
    cudaStreamWaitEvent(stream0, data_->event[1], 0);

    // SiLU gating kernel
    // Note: gate_x is in tmp_gate from stream2, gate_h is computed above
    // We need a separate buffer for gate_x, so we'll recompute inline
    // Actually, let's use a different approach: gate_x is already in tmp_gate
    // We computed gate_h into tmp_gate, overwriting gate_x... that's a bug.
    // Let me fix this by using proper temporary storage.

    // Actually, the caller provides tmp_gate which should be [N, H*2] to hold both
    // gate_x and gate_h. Let me re-design...

    // For now, let's do it sequentially: gate_x already in tmp_gate from stream2
    // Then compute gate_h and add inline. This requires modifying the kernel.

    // Simpler approach: gate_pre = Wg_x @ x + Wg_h @ h_out + bg
    // We'll use two GEMMs with beta=1 to accumulate:
    //   1. gate_pre = Wg_x @ x (beta=0)
    //   2. gate_pre += Wg_h @ h_out (beta=1)
    // Then the kernel just applies silu.

    // Let me redo this properly...
    // Actually the issue is we're overwriting tmp_gate. Let's use gate_pre as the accumulator.

    // Step 1: gate_pre = Wg_x @ x (already done in stream2 into tmp_gate)
    // Step 2: gate_pre += Wg_h @ h_out

    // Copy tmp_gate to gate_pre (or just compute directly into gate_pre)
    // Let's recompute: gate_pre = Wg_x @ x
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, input_size,
        &alpha,
        Wg_x, hidden_size,
        x, input_size,
        &beta,
        gate_pre, hidden_size);

    // gate_pre += Wg_h @ h_out
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size,
        &alpha,
        Wg_h, hidden_size,
        h_out, hidden_size,
        &alpha,  // beta=1 to accumulate
        gate_pre, hidden_size);

    // Apply SiLU gating: y = h_out * silu(gate_pre + bg)
    const int total = batch_size * hidden_size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    if (data_->training) {
        SiLUGatingKernel<T, true><<<blocks, threads, 0, stream0>>>(
            batch_size, hidden_size, h_out, gate_pre,
            bg, y_out, gate_pre);  // Save pre+bias to gate_pre
    } else {
        SiLUGatingKernel<T, false><<<blocks, threads, 0, stream0>>>(
            batch_size, hidden_size, h_out, gate_pre,
            bg, y_out, nullptr);
    }

    cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void ForwardPass<T>::Run(
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
    T* tmp_gate) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream0 = data_->stream[0];
    const cudaStream_t stream1 = data_->stream[1];

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    // Precompute all tmp_Wx = W @ x for all timesteps
    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 3, steps * batch_size, input_size,
        &alpha,
        W, hidden_size * 3,
        x, input_size,
        &beta,
        tmp_Wx, hidden_size * 3);

    // Also precompute all gate_x = Wg_x @ x for all timesteps
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, steps * batch_size, input_size,
        &alpha,
        Wg_x, hidden_size,
        x, input_size,
        &beta,
        tmp_gate, hidden_size);
    cudaEventRecord(data_->event[0], stream1);

    const int NH = batch_size * hidden_size;
    const int NH3 = batch_size * hidden_size * 3;
    const int NH4 = batch_size * hidden_size * 4;

    for (int i = 0; i < steps; ++i) {
        // Compute tmp_Rh = R @ h[i]
        cublasSetStream(blas_handle, stream0);
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_size * 3, batch_size, hidden_size,
            &alpha,
            R, hidden_size * 3,
            h + i * NH, hidden_size,
            &beta,
            tmp_Rh, hidden_size * 3);

        // Wait for precomputed projections
        if (i == 0) {
            cudaStreamWaitEvent(stream0, data_->event[0], 0);
        }

        // GRU pointwise operations
        const dim3 blockDim(32, 16);
        const dim3 gridDim(
            (hidden_size + blockDim.x - 1) / blockDim.x,
            (batch_size + blockDim.y - 1) / blockDim.y);

        T* h_out = h + (i + 1) * NH;
        T* v_t = data_->training ? v + i * NH4 : nullptr;

        if (data_->training) {
            GRUPointwiseOperations<T, true><<<gridDim, blockDim, 0, stream0>>>(
                batch_size, hidden_size,
                tmp_Wx + i * NH3, tmp_Rh, bx, br,
                h + i * NH, h_out, v_t);
        } else {
            GRUPointwiseOperations<T, false><<<gridDim, blockDim, 0, stream0>>>(
                batch_size, hidden_size,
                tmp_Wx + i * NH3, tmp_Rh, bx, br,
                h + i * NH, h_out, nullptr);
        }

        // Compute gate: gate_pre = gate_x + Wg_h @ h_out + bg
        // gate_x is already in tmp_gate + i * NH
        T* gate_pre_t = gate_pre + i * NH;

        // Copy gate_x to gate_pre_t
        cudaMemcpyAsync(gate_pre_t, tmp_gate + i * NH,
                       NH * sizeof(T), cudaMemcpyDeviceToDevice, stream0);

        // gate_pre += Wg_h @ h_out
        blas<T>::gemm(blas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_size, batch_size, hidden_size,
            &alpha,
            Wg_h, hidden_size,
            h_out, hidden_size,
            &alpha,  // beta=1 to accumulate
            gate_pre_t, hidden_size);

        // SiLU gating
        const int total = batch_size * hidden_size;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;

        T* y_t = y + i * NH;

        if (data_->training) {
            SiLUGatingKernel<T, true><<<blocks, threads, 0, stream0>>>(
                batch_size, hidden_size, h_out, gate_pre_t,
                bg, y_t, gate_pre_t);
        } else {
            SiLUGatingKernel<T, false><<<blocks, threads, 0, stream0>>>(
                batch_size, hidden_size, h_out, gate_pre_t,
                bg, y_t, nullptr);
        }
    }

    cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__half>;
template struct ForwardPass<__nv_bfloat16>;

}  // namespace gru_silu
}  // namespace v0
}  // namespace haste
