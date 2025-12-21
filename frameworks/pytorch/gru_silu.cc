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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using haste::v0::gru_silu::ForwardPass;
using haste::v0::gru_silu::BackwardPass;

using torch::Tensor;

// GRU + SiLU Selectivity Gate Forward
//
// Combines GRU recurrence with input-dependent output gating:
//   h_gru = GRU(x, h_prev)
//   gate = silu(Wg_x @ x + Wg_h @ h_gru + bg)
//   y = h_gru * gate
//
// Returns: {h (hidden states), y (gated outputs), cache (for backward), gate_pre}
std::vector<Tensor> gru_silu_forward(
    bool training,
    Tensor x,          // [T, N, C] input
    Tensor h0,         // [N, H] initial hidden state
    Tensor W,          // [C, H*3] GRU input weights
    Tensor R,          // [H, H*3] GRU recurrent weights
    Tensor bx,         // [H*3] GRU input bias
    Tensor br,         // [H*3] GRU recurrent bias
    Tensor Wg_x,       // [C, H] gate input weights
    Tensor Wg_h,       // [H, H] gate hidden weights
    Tensor bg) {       // [H] gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto hidden_size = R.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W);
    CHECK_INPUT(R);
    CHECK_INPUT(bx);
    CHECK_INPUT(br);
    CHECK_INPUT(Wg_x);
    CHECK_INPUT(Wg_h);
    CHECK_INPUT(bg);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Outputs
    Tensor h = torch::empty({ time_steps + 1, batch_size, hidden_size }, options);
    Tensor y = torch::empty({ time_steps, batch_size, hidden_size }, options);
    Tensor cache = torch::empty({ time_steps, batch_size, hidden_size * 4 }, options);
    Tensor gate_pre = torch::empty({ time_steps, batch_size, hidden_size }, options);

    // Temporaries
    Tensor tmp_Wx = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, hidden_size * 3 }, options);
    Tensor tmp_gate = torch::empty({ time_steps, batch_size, hidden_size }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "gru_silu_forward", ([&] {
        ForwardPass<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            input_size,
            hidden_size,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W),
            ptr<scalar_t>(R),
            ptr<scalar_t>(bx),
            ptr<scalar_t>(br),
            ptr<scalar_t>(Wg_x),
            ptr<scalar_t>(Wg_h),
            ptr<scalar_t>(bg),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(y),
            ptr<scalar_t>(cache),
            ptr<scalar_t>(gate_pre),
            ptr<scalar_t>(tmp_Wx),
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_gate));
    }));

    return { h, y, cache, gate_pre };
}

// GRU + SiLU Selectivity Gate Backward
std::vector<Tensor> gru_silu_backward(
    Tensor x_t,        // [C, T, N] transposed input
    Tensor W_t,        // [H*3, C] transposed GRU input weights
    Tensor R_t,        // [H*3, H] transposed GRU recurrent weights
    Tensor bx,         // [H*3] GRU input bias
    Tensor br,         // [H*3] GRU recurrent bias
    Tensor Wg_x_t,     // [H, C] transposed gate input weights
    Tensor Wg_h_t,     // [H, H] transposed gate hidden weights
    Tensor bg,         // [H] gate bias
    Tensor h,          // [T+1, N, H] hidden states from forward
    Tensor y,          // [T, N, H] gated outputs from forward (used to get h_out)
    Tensor cache,      // [T, N, H*4] GRU activations from forward
    Tensor gate_pre,   // [T, N, H] gate pre-activations from forward
    Tensor dy) {       // [T, N, H] gradient of loss w.r.t. output

    const auto input_size = x_t.size(0);
    const auto time_steps = x_t.size(1);
    const auto batch_size = x_t.size(2);
    const auto hidden_size = R_t.size(1);

    CHECK_INPUT(x_t);
    CHECK_INPUT(W_t);
    CHECK_INPUT(R_t);
    CHECK_INPUT(bx);
    CHECK_INPUT(br);
    CHECK_INPUT(Wg_x_t);
    CHECK_INPUT(Wg_h_t);
    CHECK_INPUT(bg);
    CHECK_INPUT(h);
    CHECK_INPUT(y);
    CHECK_INPUT(cache);
    CHECK_INPUT(gate_pre);
    CHECK_INPUT(dy);

    const auto options = x_t.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Gradients
    Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
    Tensor dW = torch::zeros({ input_size, hidden_size * 3 }, options);
    Tensor dR = torch::zeros({ hidden_size, hidden_size * 3 }, options);
    Tensor dbx = torch::zeros({ hidden_size * 3 }, options);
    Tensor dbr = torch::zeros({ hidden_size * 3 }, options);
    Tensor dWg_x = torch::zeros({ input_size, hidden_size }, options);
    Tensor dWg_h = torch::zeros({ hidden_size, hidden_size }, options);
    Tensor dbg = torch::zeros({ hidden_size }, options);
    Tensor dh = torch::zeros({ batch_size, hidden_size }, options);

    // Temporaries
    Tensor dp = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
    Tensor dq = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
    Tensor tmp_dgate = torch::empty({ time_steps, batch_size, hidden_size }, options);

    // h_out is h[1:], which represents the GRU output at each timestep
    Tensor h_out = h.slice(0, 1, time_steps + 1);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x_t.scalar_type(), "gru_silu_backward", ([&] {
        BackwardPass<typename native_type<scalar_t>::T> backward(
            batch_size,
            input_size,
            hidden_size,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W_t),
            ptr<scalar_t>(R_t),
            ptr<scalar_t>(bx),
            ptr<scalar_t>(br),
            ptr<scalar_t>(Wg_x_t),
            ptr<scalar_t>(Wg_h_t),
            ptr<scalar_t>(bg),
            ptr<scalar_t>(x_t),
            ptr<scalar_t>(h),
            ptr<scalar_t>(h_out),
            ptr<scalar_t>(cache),
            ptr<scalar_t>(gate_pre),
            ptr<scalar_t>(dy),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbx),
            ptr<scalar_t>(dbr),
            ptr<scalar_t>(dWg_x),
            ptr<scalar_t>(dWg_h),
            ptr<scalar_t>(dbg),
            ptr<scalar_t>(dh),
            ptr<scalar_t>(dp),
            ptr<scalar_t>(dq),
            ptr<scalar_t>(tmp_dgate));
    }));

    return { dx, dh, dW, dR, dbx, dbr, dWg_x, dWg_h, dbg };
}

}  // anonymous namespace

void gru_silu_init(py::module& m) {
    m.def("gru_silu_forward", &gru_silu_forward, "GRU+SiLU forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("gru_silu_backward", &gru_silu_backward, "GRU+SiLU backward",
          py::call_guard<py::gil_scoped_release>());
}
