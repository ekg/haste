// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for fused Elman RNN.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using haste::v0::elman::ForwardPass;
using haste::v0::elman::BackwardPass;

using torch::Tensor;

std::vector<Tensor> elman_forward(
    bool training,
    Tensor x,           // [T, B, D] - input sequence
    Tensor h0,          // [B, D] - initial hidden state
    Tensor W1,          // [H, 2D] - first layer weights
    Tensor W2,          // [D, H] - output projection
    Tensor Wgx,         // [D, D] - gate input weights
    Tensor Wgh,         // [D, D] - gate recurrent weights
    Tensor bias) {      // [D] - gate bias
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto hidden_size = W1.size(0);
    const auto output_size = W2.size(0);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(W1);
    CHECK_INPUT(W2);
    CHECK_INPUT(Wgx);
    CHECK_INPUT(Wgh);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output tensors
    Tensor h = torch::empty({ time_steps + 1, batch_size, output_size }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, output_size * 3 }, options)
                        : torch::empty({ 0 }, options);
    Tensor hidden = training ? torch::empty({ time_steps, batch_size, hidden_size }, options)
                             : torch::empty({ 0 }, options);

    // Workspace tensors
    // tmp_Wx1 now stores W1_x @ x for ALL timesteps (pre-computed like cuDNN)
    Tensor tmp_Wx1 = torch::empty({ time_steps, batch_size, hidden_size }, options);
    Tensor tmp_gx = torch::empty({ time_steps, batch_size, output_size }, options);
    Tensor tmp_gh = torch::empty({ batch_size, output_size }, options);
    // tmp_Wg2 stores pre-computed Wgh @ W2 for the Wg2 @ hidden optimization
    Tensor tmp_Wg2 = torch::empty({ output_size, hidden_size }, options);

    // Initialize h[0] with h0
    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "elman_forward", ([&] {
        ForwardPass<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            input_size,
            hidden_size,
            output_size,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(W1),
            ptr<scalar_t>(W2),
            ptr<scalar_t>(Wgx),
            ptr<scalar_t>(Wgh),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(hidden) : nullptr,
            ptr<scalar_t>(tmp_Wx1),
            ptr<scalar_t>(tmp_gx),
            ptr<scalar_t>(tmp_gh),
            ptr<scalar_t>(tmp_Wg2));
    }));

    return { h, v, hidden };
}

std::vector<Tensor> elman_backward(
    Tensor x_t,         // [D, T*B] - transposed input
    Tensor W1_t,        // [2D, H] - transposed W1
    Tensor W2_t,        // [H, D] - transposed W2
    Tensor Wgx_t,       // [D, D] - transposed Wgx
    Tensor Wgh_t,       // [D, D] - transposed Wgh
    Tensor h,           // [(T+1)*B, D] - hidden states from forward
    Tensor hidden,      // [T*B, H] - tanh outputs from forward
    Tensor v,           // [T*B, D*3] - saved activations
    Tensor dh_new) {    // [(T+1)*B, D] - gradient of output
    const auto input_size = x_t.size(0);
    // Infer dimensions from tensor shapes
    // v is [T*B, D*3], h is [(T+1)*B, D]
    // So: h.size(0) - v.size(0) = B
    const auto batch_size = h.size(0) - v.size(0);
    const auto time_steps = v.size(0) / batch_size;
    const auto output_size = Wgh_t.size(0);
    const auto hidden_size = W2_t.size(0);

    CHECK_INPUT(x_t);
    CHECK_INPUT(W1_t);
    CHECK_INPUT(W2_t);
    CHECK_INPUT(Wgx_t);
    CHECK_INPUT(Wgh_t);
    CHECK_INPUT(h);
    CHECK_INPUT(hidden);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x_t.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output gradient tensors
    Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
    Tensor dW1 = torch::zeros({ hidden_size, input_size + output_size }, options);
    Tensor dW2 = torch::zeros({ output_size, hidden_size }, options);
    Tensor dWgx = torch::zeros({ output_size, input_size }, options);
    Tensor dWgh = torch::zeros({ output_size, output_size }, options);
    Tensor dbias = torch::zeros({ output_size }, options);
    Tensor dh = torch::zeros({ batch_size, output_size }, options);

    // Workspace tensors
    Tensor tmp_dgate = torch::empty({ batch_size, output_size }, options);
    Tensor tmp_dh_new_local = torch::empty({ batch_size, output_size }, options);
    Tensor tmp_dhidden = torch::empty({ batch_size, hidden_size }, options);
    Tensor tmp_dcombined = torch::empty({ batch_size, input_size + output_size }, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_t.scalar_type(), "elman_backward", ([&] {
        BackwardPass<typename native_type<scalar_t>::T> backward(
            batch_size,
            input_size,
            hidden_size,
            output_size,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(W1_t),
            ptr<scalar_t>(W2_t),
            ptr<scalar_t>(Wgx_t),
            ptr<scalar_t>(Wgh_t),
            ptr<scalar_t>(x_t),
            ptr<scalar_t>(h),
            ptr<scalar_t>(hidden),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dW1),
            ptr<scalar_t>(dW2),
            ptr<scalar_t>(dWgx),
            ptr<scalar_t>(dWgh),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh),
            ptr<scalar_t>(tmp_dgate),
            ptr<scalar_t>(tmp_dh_new_local),
            ptr<scalar_t>(tmp_dhidden),
            ptr<scalar_t>(tmp_dcombined));
    }));

    return { dx, dh, dW1, dW2, dWgx, dWgh, dbias };
}

}  // anonymous namespace

void elman_init(py::module& m) {
    m.def("elman_forward", &elman_forward, "Elman forward", py::call_guard<py::gil_scoped_release>());
    m.def("elman_backward", &elman_backward, "Elman backward", py::call_guard<py::gil_scoped_release>());
}
