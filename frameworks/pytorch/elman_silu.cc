// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for SiLU-gated Elman RNN.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using haste::v0::elman_silu::ForwardPass;
using haste::v0::elman_silu::BackwardPass;

using torch::Tensor;

std::vector<Tensor> elman_silu_forward(
    bool training,
    Tensor x,           // [T, B, D] - input sequence
    Tensor h0,          // [B, D] - initial hidden state
    Tensor Wx,          // [2D, D] - input weights (combines W and Wg input parts)
    Tensor R,           // [2D, D] - recurrent weights (combines W_h and Wg_h)
    Tensor bias) {      // [2D] - bias
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);  // input_size = hidden_size = D
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Pre-compute Wx @ x for all timesteps
    // x is [T, B, D], Wx is [2D, D]
    // Result is [T, B, 2D]
    Tensor x_flat = x.reshape({time_steps * batch_size, D});  // [T*B, D]
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());  // [T*B, 2D]
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    // Output tensors
    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);

    // Workspace
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    // Initialize h[0] with h0
    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_silu_forward", ([&] {
        ForwardPass<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,  // input_size
            D,  // hidden_size
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_silu_backward(
    Tensor x,           // [T, B, D] - input sequence
    Tensor Wx,          // [2D, D] - input weights
    Tensor R,           // [2D, D] - recurrent weights
    Tensor h,           // [T+1, B, D] - hidden states from forward
    Tensor v,           // [T, B, 2D] - saved activations
    Tensor dh_new) {    // [T+1, B, D] - gradient of output
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradient tensors
    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);

    // Workspace
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_silu_backward", ([&] {
        BackwardPass<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,  // input_size
            D,  // hidden_size
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),  // Pass R directly, not transposed
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    // Compute dx and dWx from dWx_all
    // dWx_all is [T, B, 2D]
    // dWx = dWx_all^T @ x = [2D, T*B] @ [T*B, D] = [2D, D]
    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});  // [T*B, 2D]
    Tensor x_flat = x.reshape({time_steps * batch_size, D});  // [T*B, D]
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);  // [2D, D]

    // dx = dWx_all @ Wx = [T*B, 2D] @ [2D, D] = [T*B, D]
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);  // [T*B, D]
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

}  // anonymous namespace

void elman_silu_init(py::module& m) {
    m.def("elman_silu_forward", &elman_silu_forward, "ElmanSilu forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_silu_backward", &elman_silu_backward, "ElmanSilu backward",
          py::call_guard<py::gil_scoped_release>());
}
