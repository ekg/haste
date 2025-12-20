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
// SkipElman Backward Pass
// Forward: z = sigmoid(...), a = tanh(...), h_new = z * h + (1-z) * a
// Backward:
//   dh = z * dh_new                     (gradient through skip)
//   da = (1-z) * dh_new
//   dz = (h - a) * dh_new
//   dp_a = d_tanh(a) * da
//   dp_z = d_sigmoid(z) * dz

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include "blas.h"
#include "device_assert.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

template<typename T, bool ApplyZoneout>
__global__
void SkipElmanPointwiseOperationsBackward(const int batch_dim,
                                            const int hidden_dim,
                                            const T* h,
                                            const T* v,
                                            const T* dh_new,
                                            T* dbx_out,
                                            T* dbh_out,
                                            T* dh_inout,
                                            T* dp_out,
                                            T* dq_out,
                                            const T* zoneout_mask) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_dim || col >= batch_dim)
    return;

  const int base_idx = col * hidden_dim + row;

  T dh_total = dh_new[base_idx] + dh_inout[base_idx];

  // SkipElman stores 2 values: z, a
  const int stride2_base_idx = col * (hidden_dim * 2) + row;
  const int z_idx = stride2_base_idx + 0 * hidden_dim;
  const int a_idx = stride2_base_idx + 1 * hidden_dim;

  const T z = v[z_idx];
  const T a = v[a_idx];

  if (ApplyZoneout) {
    const T mask = zoneout_mask[base_idx];
    dh_inout[base_idx] = (static_cast<T>(1.0) - mask) * dh_total;
    dh_total = mask * dh_total;
    dh_inout[base_idx] += z * dh_total;
  } else {
    // Gradient through skip connection: dh = z * dh_new
    dh_inout[base_idx] = z * dh_total;
  }

  // Gradient for candidate: da = (1-z) * dh_new
  const T da = (static_cast<T>(1.0) - z) * dh_total;

  // Gradient for gate: dz = (h - a) * dh_new
  const T dz = (h[base_idx] - a) * dh_total;

  // Through activations
  const T dp_a = d_tanh(a) * da;
  const T dp_z = d_sigmoid(z) * dz;

  // For SkipElman, dq = dp (no reset gate complicating things)
  const T dq_a = dp_a;
  const T dq_z = dp_z;

  // Output gradients (2 components: z, a)
  const int idx = col * (hidden_dim * 2) + row;

  dp_out[idx + 0 * hidden_dim] = dp_z;
  dp_out[idx + 1 * hidden_dim] = dp_a;

  dq_out[idx + 0 * hidden_dim] = dq_z;
  dq_out[idx + 1 * hidden_dim] = dq_a;

  // Accumulate bias gradients (separate input and hidden biases)
  atomicAdd(&dbx_out[row + 0 * hidden_dim], dp_z);
  atomicAdd(&dbx_out[row + 1 * hidden_dim], dp_a);

  atomicAdd(&dbh_out[row + 0 * hidden_dim], dq_z);
  atomicAdd(&dbh_out[row + 1 * hidden_dim], dq_a);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
template<typename T, bool ApplyZoneout>
__global__
void SkipElmanPointwiseOperationsBackward(const int batch_dim,
                                            const int hidden_dim,
                                            const half* h,
                                            const half* v,
                                            const half* dh_new,
                                            half* dbx_out,
                                            half* dbh_out,
                                            half* dh_inout,
                                            half* dp_out,
                                            half* dq_out,
                                            const half* zoneout_mask) {
  device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}
#endif

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace skip_elman {

template<typename T>
struct BackwardPass<T>::private_data {
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
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
  cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
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
void BackwardPass<T>::IterateInternal(
    const T* R_t,     // [H*2,H]
    const T* h,       // [N,H]
    const T* v,       // [N,H*2]
    const T* dh_new,  // [N,H]
    T* dbx,           // [H*2]
    T* dbh,           // [H*2]
    T* dh,            // [N,H]
    T* dp,            // [N,H*2]
    T* dq,            // [N,H*2]
    const T* zoneout_mask) {
  const T alpha = static_cast<T>(1.0);
  const T beta_sum = static_cast<T>(1.0);

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaEvent_t event = data_->event;

  const dim3 blockDim(32, 16);
  const dim3 gridDim(
      (hidden_size + blockDim.x - 1) / blockDim.x,
      (batch_size + blockDim.y - 1) / blockDim.y);

  if (zoneout_mask) {
    SkipElmanPointwiseOperationsBackward<T, true><<<gridDim, blockDim, 0, stream1>>>(
        batch_size, hidden_size, h, v, dh_new, dbx, dbh, dh, dp, dq, zoneout_mask);
  } else {
    SkipElmanPointwiseOperationsBackward<T, false><<<gridDim, blockDim, 0, stream1>>>(
        batch_size, hidden_size, h, v, dh_new, dbx, dbh, dh, dp, dq, nullptr);
  }
  cudaEventRecord(event, stream1);

  // dh += R^T @ dq
  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size, batch_size, hidden_size * 2,
      &alpha,
      R_t, hidden_size,
      dq, hidden_size * 2,
      &beta_sum,
      dh, hidden_size);
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* W_t,     // [H*2,C]
    const T* R_t,     // [H*2,H]
    const T* bx,      // [H*2]
    const T* bh,      // [H*2]
    const T* x_t,     // [C,T*N]
    const T* h,       // [(T+1)*N,H]
    const T* v,       // [T*N,H*2]
    const T* dh_new,  // [(T+1)*N,H]
    T* dx,            // [T*N,C]
    T* dW,            // [C,H*2]
    T* dR,            // [H,H*2]
    T* dbx,           // [H*2]
    T* dbh,           // [H*2]
    T* dh,            // [N,H]
    T* dp,            // [T*N,H*2]
    T* dq,            // [T*N,H*2]
    const T* zoneout_mask) {
  const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
  const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

  const T alpha = static_cast<T>(1.0);
  const T beta_sum = static_cast<T>(1.0);
  const T beta_assign = static_cast<T>(0.0);

  const int batch_size = data_->batch_size;
  const int input_size = data_->input_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaStream_t stream2 = data_->stream[1];
  const cudaEvent_t event = data_->event;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  const int NH = batch_size * hidden_size;
  for (int i = steps - 1; i >= 0; --i) {
    IterateInternal(
        R_t,
        h + i * NH,
        v + i * NH * 2,
        dh_new + (i + 1) * NH,
        dbx,
        dbh,
        dh,
        dp + i * NH * 2,
        dq + i * NH * 2,
        zoneout_mask ? zoneout_mask + i * NH : nullptr);
  }

  cudaStreamWaitEvent(stream2, event, 0);

  // dx = W^T @ dp
  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      input_size, batch_size * steps, hidden_size * 2,
      &alpha,
      W_t, input_size,
      dp, hidden_size * 2,
      &beta_assign,
      dx, input_size);

  // dR += dq @ h^T
  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      hidden_size * 2, hidden_size, batch_size * steps,
      &alpha,
      dq, hidden_size * 2,
      h, hidden_size,
      &beta_sum,
      dR, hidden_size * 2);

  // dW += dp @ x^T
  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size * 2, input_size, batch_size * steps,
      &alpha,
      dp, hidden_size * 2,
      x_t, batch_size * steps,
      &beta_sum,
      dW, hidden_size * 2);

  cublasSetStream(blas_handle, save_stream);
}

template struct BackwardPass<half>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
template struct BackwardPass<__nv_bfloat16>;

}  // namespace skip_elman
}  // namespace v0
}  // namespace haste
