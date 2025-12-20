// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for Multi-head Elman RNN.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using haste::v0::multihead_elman::ForwardPass;
using haste::v0::multihead_elman::BackwardPass;

using torch::Tensor;

std::vector<Tensor> multihead_elman_forward(
    bool training,
    Tensor x,           // [T, B, nheads, headdim] - input sequence
    Tensor h0,          // [B, nheads, headdim] - initial hidden state
    Tensor R,           // [nheads, headdim, headdim] - recurrent weights per head
    Tensor Wx,          // [nheads, headdim, headdim] - input weights per head
    Tensor bias,        // [nheads, headdim] - bias per head
    int activation) {   // 0=softsign, 1=tanh_residual, 2=tanh
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto nheads = x.size(2);
    const auto headdim = x.size(3);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(R);
    CHECK_INPUT(Wx);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output tensors
    // h contains all hidden states: h[0] = h0, h[1..T] = outputs
    Tensor h = torch::empty({ time_steps + 1, batch_size, nheads, headdim }, options);
    Tensor y = torch::empty({ time_steps, batch_size, nheads, headdim }, options);
    Tensor pre_act = training
        ? torch::empty({ time_steps, batch_size, nheads, headdim }, options)
        : torch::empty({ 0 }, options);

    // Workspace tensors
    Tensor tmp_Rh = torch::empty({ batch_size, nheads, headdim }, options);
    Tensor tmp_Wxx = torch::empty({ time_steps, batch_size, nheads, headdim }, options);

    // Initialize h[0] with h0
    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "multihead_elman_forward", ([&] {
        ForwardPass<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            nheads,
            headdim,
            activation,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(Wx),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(y),
            training ? ptr<scalar_t>(pre_act) : nullptr,
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_Wxx));
    }));

    return { h, y, pre_act };
}

std::vector<Tensor> multihead_elman_backward(
    Tensor x,           // [T, B, nheads, headdim] - input sequence
    Tensor R,           // [nheads, headdim, headdim] - recurrent weights
    Tensor Wx,          // [nheads, headdim, headdim] - input weights
    Tensor h,           // [T+1, B, nheads, headdim] - hidden states from forward
    Tensor pre_act,     // [T, B, nheads, headdim] - saved pre-activations
    Tensor dy,          // [T, B, nheads, headdim] - gradient of loss w.r.t. output
    int activation) {   // 0=softsign, 1=tanh_residual, 2=tanh
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto nheads = x.size(2);
    const auto headdim = x.size(3);

    CHECK_INPUT(x);
    CHECK_INPUT(R);
    CHECK_INPUT(Wx);
    CHECK_INPUT(h);
    CHECK_INPUT(pre_act);
    CHECK_INPUT(dy);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradient tensors
    Tensor dx = torch::empty({ time_steps, batch_size, nheads, headdim }, options);
    Tensor dR = torch::zeros({ nheads, headdim, headdim }, options);
    Tensor dWx = torch::zeros({ nheads, headdim, headdim }, options);
    Tensor db = torch::zeros({ nheads, headdim }, options);
    Tensor dh0 = torch::zeros({ batch_size, nheads, headdim }, options);

    // Workspace
    Tensor tmp_dpre = torch::empty({ batch_size, nheads, headdim }, options);
    Tensor tmp_dh = torch::empty({ batch_size, nheads, headdim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "multihead_elman_backward", ([&] {
        BackwardPass<typename native_type<scalar_t>::T> backward(
            batch_size,
            nheads,
            headdim,
            activation,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(Wx),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(pre_act),
            ptr<scalar_t>(dy),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dWx),
            ptr<scalar_t>(db),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dpre),
            ptr<scalar_t>(tmp_dh));
    }));

    return { dx, dh0, dR, dWx, db };
}

}  // anonymous namespace

void multihead_elman_init(py::module& m) {
    m.def("multihead_elman_forward", &multihead_elman_forward, "MultiHeadElman forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("multihead_elman_backward", &multihead_elman_backward, "MultiHeadElman backward",
          py::call_guard<py::gil_scoped_release>());
}
