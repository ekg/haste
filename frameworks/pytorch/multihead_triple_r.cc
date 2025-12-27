// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for Multi-head Triple R with 32× state expansion.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using haste::v0::multihead_triple_r::ForwardPass;
using haste::v0::multihead_triple_r::BackwardPass;

using torch::Tensor;

std::vector<Tensor> multihead_triple_r_forward(
    bool training,
    Tensor x,           // [T, B, nheads, headdim]
    Tensor h0,          // [B, nheads, headdim]
    Tensor R_h,         // [nheads, headdim, headdim]
    Tensor R_x,         // [nheads, headdim, headdim]
    Tensor R_delta,     // [nheads, headdim, headdim]
    Tensor W_delta,     // [nheads, headdim, headdim]
    Tensor b,           // [nheads, headdim]
    Tensor b_delta) {   // [nheads, headdim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto nheads = x.size(2);
    const auto headdim = x.size(3);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(R_h);
    CHECK_INPUT(R_x);
    CHECK_INPUT(R_delta);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Output tensors
    Tensor h = torch::empty({ time_steps + 1, batch_size, nheads, headdim }, options);
    Tensor v = training
        ? torch::empty({ time_steps, batch_size, nheads, headdim }, options)
        : torch::empty({ 0 }, options);
    Tensor delta_cache = training
        ? torch::empty({ time_steps, batch_size, nheads, headdim }, options)
        : torch::empty({ 0 }, options);

    // Workspace tensors
    Tensor tmp_Rh = torch::empty({ batch_size, nheads, headdim }, options);
    Tensor tmp_Rx = torch::empty({ batch_size, nheads, headdim }, options);
    Tensor tmp_Rdelta = torch::empty({ batch_size, nheads, headdim }, options);
    Tensor tmp_Wdelta = torch::empty({ batch_size, nheads, headdim }, options);

    // Initialize h[0] with h0
    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "multihead_triple_r_forward", ([&] {
        ForwardPass<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            nheads,
            headdim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(R_x),
            ptr<scalar_t>(R_delta),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_Rx),
            ptr<scalar_t>(tmp_Rdelta),
            ptr<scalar_t>(tmp_Wdelta));
    }));

    return { h, v, delta_cache };
}

std::vector<Tensor> multihead_triple_r_backward(
    Tensor x,           // [T, B, nheads, headdim]
    Tensor R_h,         // [nheads, headdim, headdim]
    Tensor R_x,         // [nheads, headdim, headdim]
    Tensor R_delta,     // [nheads, headdim, headdim]
    Tensor W_delta,     // [nheads, headdim, headdim]
    Tensor h,           // [T+1, B, nheads, headdim]
    Tensor v,           // [T, B, nheads, headdim]
    Tensor delta_cache, // [T, B, nheads, headdim]
    Tensor dh_new) {    // [T+1, B, nheads, headdim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto nheads = x.size(2);
    const auto headdim = x.size(3);

    CHECK_INPUT(x);
    CHECK_INPUT(R_h);
    CHECK_INPUT(R_x);
    CHECK_INPUT(R_delta);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta_cache);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Gradient tensors
    Tensor dx = torch::zeros({ time_steps, batch_size, nheads, headdim }, options);
    Tensor dR_h = torch::zeros({ nheads, headdim, headdim }, options);
    Tensor dR_x = torch::zeros({ nheads, headdim, headdim }, options);
    Tensor dR_delta = torch::zeros({ nheads, headdim, headdim }, options);
    Tensor dW_delta = torch::zeros({ nheads, headdim, headdim }, options);
    Tensor db = torch::zeros({ nheads, headdim }, options);
    Tensor db_delta = torch::zeros({ nheads, headdim }, options);
    Tensor dh = torch::zeros({ batch_size, nheads, headdim }, options);

    // Workspace tensors
    Tensor tmp_Rh = torch::empty({ batch_size, nheads, headdim }, options);
    Tensor tmp_Rx = torch::empty({ batch_size, nheads, headdim }, options);
    Tensor tmp_Rdelta = torch::empty({ batch_size, nheads, headdim }, options);
    Tensor tmp_Wdelta = torch::empty({ batch_size, nheads, headdim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "multihead_triple_r_backward", ([&] {
        BackwardPass<typename native_type<scalar_t>::T> backward(
            batch_size,
            nheads,
            headdim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(R_x),
            ptr<scalar_t>(R_delta),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dR_h),
            ptr<scalar_t>(dR_x),
            ptr<scalar_t>(dR_delta),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(dh),
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_Rx),
            ptr<scalar_t>(tmp_Rdelta),
            ptr<scalar_t>(tmp_Wdelta));
    }));

    return { dx, dh, dR_h, dR_x, dR_delta, dW_delta, db, db_delta };
}

}  // anonymous namespace

void init_multihead_triple_r(py::module& m) {
    m.def("multihead_triple_r_forward", &multihead_triple_r_forward,
        "Multi-head Triple R forward (CUDA)");
    m.def("multihead_triple_r_backward", &multihead_triple_r_backward,
        "Multi-head Triple R backward (CUDA)");
}
