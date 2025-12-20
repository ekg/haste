// Copyright 2020 LMNT, Inc. All Rights Reserved.
// Modified 2024 for SkipElman (GRU without reset gate)
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
// SkipElman: GRU without reset gate
//   z = sigmoid(Wz @ x + Rz @ h + bx_z + bh_z)   # update gate
//   a = tanh(Wa @ x + Ra @ h + bx_a + bh_a)      # candidate (no reset gate!)
//   h_new = z * h + (1-z) * a                     # SKIP CONNECTION
//
// Simpler than GRU (2 gates instead of 3) but retains gradient highway.

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "device_assert.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

template<typename T, bool Training, bool ApplyZoneout>
__global__
void SkipElmanPointwiseOperations(const int batch_dim,
                                    const int hidden_dim,
                                    const T* Wx,
                                    const T* Rh,
                                    const T* bx,
                                    const T* bh,
                                    const T* h,
                                    T* h_out,
                                    T* v,
                                    const T zoneout_prob,
                                    const T* zoneout_mask) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_dim || col >= batch_dim)
    return;

  // SkipElman has 2 components: z (update gate), a (candidate)
  const int weight_idx = col * (hidden_dim * 2) + row;

  // Index into the `h` and `h_out` vectors
  const int output_idx = col * hidden_dim + row;

  // Indices into Wx and Rh matrices (z and a components)
  const int z_idx = weight_idx + 0 * hidden_dim;
  const int a_idx = weight_idx + 1 * hidden_dim;

  // Indices into bias vectors (separate input and hidden biases)
  const int bz_idx = row + 0 * hidden_dim;
  const int ba_idx = row + 1 * hidden_dim;

  // Update gate: z = sigmoid(Wx_z @ x + Rh_z @ h + bx_z + bh_z)
  const T z = sigmoid(Wx[z_idx] + Rh[z_idx] + bx[bz_idx] + bh[bz_idx]);

  // Candidate: a = tanh(Wx_a @ x + Rh_a @ h + bx_a + bh_a)
  // NOTE: No reset gate applied to Rh (unlike GRU)!
  const T a = tanh(Wx[a_idx] + Rh[a_idx] + bx[ba_idx] + bh[ba_idx]);

  // Store internal activations for backprop
  if (Training) {
    const int base_v_idx = col * (hidden_dim * 2) + row;
    v[base_v_idx + 0 * hidden_dim] = z;
    v[base_v_idx + 1 * hidden_dim] = a;
  }

  // SKIP CONNECTION: h_new = z * h + (1-z) * a
  T cur_h_value = z * h[output_idx] + (static_cast<T>(1.0) - z) * a;

  if (ApplyZoneout) {
    if (Training) {
      cur_h_value = (cur_h_value - h[output_idx]) * zoneout_mask[output_idx] + h[output_idx];
    } else {
      cur_h_value = (zoneout_prob * h[output_idx]) + ((static_cast<T>(1.0) - zoneout_prob) * cur_h_value);
    }
  }

  h_out[output_idx] = cur_h_value;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
template<typename T, bool Training, bool ApplyZoneout>
__global__
void SkipElmanPointwiseOperations(const int batch_dim,
                                    const int hidden_dim,
                                    const half* Wx,
                                    const half* Rh,
                                    const half* bx,
                                    const half* bh,
                                    const half* h,
                                    half* h_out,
                                    half* v,
                                    const half zoneout_prob,
                                    const half* zoneout_mask) {
  device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}
#endif

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace skip_elman {

template<typename T>
struct ForwardPass<T>::private_data {
  bool training;
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
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
  cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
  if (data_->sync_stream) {
    cudaEventRecord(data_->event, data_->stream[1]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    cudaEventRecord(data_->event, data_->stream[0]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
  } else {
    cudaStreamSynchronize(data_->stream[1]);
    cudaStreamSynchronize(data_->stream[0]);
  }
  cudaEventDestroy(data_->event);
  cudaStreamDestroy(data_->stream[1]);
  cudaStreamDestroy(data_->stream[0]);
  delete data_;
}

template<typename T>
void ForwardPass<T>::IterateInternal(
    const T* R,   // [H,H*2]
    const T* bx,  // [H*2]
    const T* bh,  // [H*2]
    const T* h,   // [N,H]
    T* h_out,     // [N,H]
    T* v,         // [N,H*2]
    T* tmp_Wx,    // [N,H*2]
    T* tmp_Rh,    // [N,H*2]
    const float zoneout_prob,
    const T* zoneout_mask) {
  static const T alpha = static_cast<T>(1.0);
  static const T beta = static_cast<T>(0.0);

  const bool training = data_->training;
  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaEvent_t event = data_->event;

  // Compute Rh = R @ h (hidden_size * 2 outputs)
  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size * 2, batch_size, hidden_size,
      &alpha,
      R, hidden_size * 2,
      h, hidden_size,
      &beta,
      tmp_Rh, hidden_size * 2);

  const dim3 blockDim(32, 16);
  const dim3 gridDim(
      (hidden_size + blockDim.x - 1) / blockDim.x,
      (batch_size + blockDim.y - 1) / blockDim.y);

  cudaStreamWaitEvent(stream1, event, 0);

  if (training) {
    if (zoneout_prob && zoneout_mask) {
      SkipElmanPointwiseOperations<T, true, true><<<gridDim, blockDim, 0, stream1>>>(
          batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, bh, h, h_out, v,
          zoneout_prob, zoneout_mask);
    } else {
      SkipElmanPointwiseOperations<T, true, false><<<gridDim, blockDim, 0, stream1>>>(
          batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, bh, h, h_out, v,
          0.0f, nullptr);
    }
  } else {
    if (zoneout_prob && zoneout_mask) {
      SkipElmanPointwiseOperations<T, false, true><<<gridDim, blockDim, 0, stream1>>>(
          batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, bh, h, h_out, nullptr,
          zoneout_prob, zoneout_mask);
    } else {
      SkipElmanPointwiseOperations<T, false, false><<<gridDim, blockDim, 0, stream1>>>(
          batch_size, hidden_size, tmp_Wx, tmp_Rh, bx, bh, h, h_out, nullptr,
          0.0f, nullptr);
    }
  }
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* W,   // [C,H*2]
    const T* R,   // [H,H*2]
    const T* bx,  // [H*2]
    const T* bh,  // [H*2]
    const T* x,   // [T*N,C]
    T* h,         // [(T+1)*N,H]
    T* v,         // [T*N,H*2]
    T* tmp_Wx,    // [T*N,H*2]
    T* tmp_Rh,    // [N,H*2]
    const float zoneout_prob,
    const T* zoneout_mask) {
  static const T alpha = static_cast<T>(1.0);
  static const T beta = static_cast<T>(0.0);

  const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
  const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

  const int batch_size = data_->batch_size;
  const int input_size = data_->input_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream2 = data_->stream[1];
  const cudaEvent_t event = data_->event;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  // Compute Wx for all timesteps: Wx = W @ x
  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size * 2, steps * batch_size, input_size,
      &alpha,
      W, hidden_size * 2,
      x, input_size,
      &beta,
      tmp_Wx, hidden_size * 2);
  cudaEventRecord(event, stream2);

  // Sequential recurrence
  const int NH = batch_size * hidden_size;
  for (int i = 0; i < steps; ++i) {
    IterateInternal(
        R,
        bx,
        bh,
        h + i * NH,
        h + (i + 1) * NH,
        v + i * NH * 2,
        tmp_Wx + i * NH * 2,
        tmp_Rh,
        zoneout_prob,
        zoneout_mask ? zoneout_mask + i * NH : nullptr);
  }

  cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<half>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;
template struct ForwardPass<__nv_bfloat16>;

}  // namespace skip_elman
}  // namespace v0
}  // namespace haste
