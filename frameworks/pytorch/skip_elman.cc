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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using haste::v0::skip_elman::ForwardPass;
using haste::v0::skip_elman::BackwardPass;

using torch::Tensor;

std::vector<Tensor> skip_elman_forward(
    bool training,
    float zoneout_prob,
    Tensor x,
    Tensor h0,
    Tensor kernel,
    Tensor recurrent_kernel,
    Tensor input_bias,
    Tensor hidden_bias,
    Tensor zoneout_mask) {
  const auto time_steps = x.size(0);
  const auto batch_size = x.size(1);
  const auto input_size = x.size(2);
  const auto hidden_size = recurrent_kernel.size(0);
  const bool has_zoneout = zoneout_prob && zoneout_mask.size(0);

  CHECK_INPUT(x);
  CHECK_INPUT(h0);
  CHECK_INPUT(kernel);
  CHECK_INPUT(recurrent_kernel);
  CHECK_INPUT(input_bias);
  CHECK_INPUT(hidden_bias);
  CHECK_INPUT(zoneout_mask);

  const auto options = x.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  // SkipElman has 2 components (z, a) vs GRU's 3 (z, r, g)
  Tensor output = torch::empty({ time_steps + 1, batch_size, hidden_size }, options);
  Tensor cache = torch::empty({ time_steps, batch_size, hidden_size * 2 }, options);
  Tensor tmp_Wx = torch::empty({ time_steps, batch_size, hidden_size * 2 }, options);
  Tensor tmp_Rh = torch::empty({ batch_size, hidden_size * 2 }, options);

  output[0] = h0;

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "skip_elman_forward", ([&] {
    ForwardPass<typename native_type<scalar_t>::T> forward(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        ptr<scalar_t>(kernel),
        ptr<scalar_t>(recurrent_kernel),
        ptr<scalar_t>(input_bias),
        ptr<scalar_t>(hidden_bias),
        ptr<scalar_t>(x),
        ptr<scalar_t>(output),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(tmp_Wx),
        ptr<scalar_t>(tmp_Rh),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr);
  }));

  return { output, cache };
}

std::vector<Tensor> skip_elman_backward(
    Tensor x_t,
    Tensor kernel_t,
    Tensor recurrent_kernel_t,
    Tensor input_bias,
    Tensor hidden_bias,
    Tensor zoneout_mask,
    Tensor h,
    Tensor cache,
    Tensor dh_new) {
  const auto input_size = x_t.size(0);
  const auto time_steps = x_t.size(1);
  const auto batch_size = x_t.size(2);
  const auto hidden_size = recurrent_kernel_t.size(1);
  const bool has_zoneout = !!zoneout_mask.size(0);

  CHECK_INPUT(x_t);
  CHECK_INPUT(kernel_t);
  CHECK_INPUT(recurrent_kernel_t);
  CHECK_INPUT(input_bias);
  CHECK_INPUT(hidden_bias);
  CHECK_INPUT(h);
  CHECK_INPUT(cache);
  CHECK_INPUT(dh_new);
  CHECK_INPUT(zoneout_mask);

  const auto options = x_t.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  // SkipElman has 2 components
  Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
  Tensor dW = torch::zeros({ input_size, hidden_size * 2 }, options);
  Tensor dR = torch::zeros({ hidden_size, hidden_size * 2 }, options);
  Tensor dbx = torch::zeros({ hidden_size * 2 }, options);
  Tensor dbh = torch::zeros({ hidden_size * 2 }, options);
  Tensor dh = torch::zeros({ batch_size, hidden_size }, options);
  Tensor dp = torch::empty({ time_steps, batch_size, hidden_size * 2 }, options);
  Tensor dq = torch::empty({ time_steps, batch_size, hidden_size * 2 }, options);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x_t.scalar_type(), "skip_elman_backward", ([&] {
    BackwardPass<typename native_type<scalar_t>::T> backward(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        ptr<scalar_t>(kernel_t),
        ptr<scalar_t>(recurrent_kernel_t),
        ptr<scalar_t>(input_bias),
        ptr<scalar_t>(hidden_bias),
        ptr<scalar_t>(x_t),
        ptr<scalar_t>(h),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(dh_new),
        ptr<scalar_t>(dx),
        ptr<scalar_t>(dW),
        ptr<scalar_t>(dR),
        ptr<scalar_t>(dbx),
        ptr<scalar_t>(dbh),
        ptr<scalar_t>(dh),
        ptr<scalar_t>(dp),
        ptr<scalar_t>(dq),
        has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr);
  }));

  return { dx, dh, dW, dR, dbx, dbh };
}

}  // anonymous namespace

void skip_elman_init(py::module& m) {
  m.def("skip_elman_forward", &skip_elman_forward, "SkipElman forward", py::call_guard<py::gil_scoped_release>());
  m.def("skip_elman_backward", &skip_elman_backward, "SkipElman backward", py::call_guard<py::gil_scoped_release>());
}
