// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for Diagonal Multi-Head Triple R (depth-stable variant).
// R matrices are DIAGONAL (vectors) instead of full matrices.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using haste::v0::diagonal_mhtr::ForwardPass;
using haste::v0::diagonal_mhtr::BackwardPass;

using torch::Tensor;

std::vector<Tensor> diagonal_mhtr_forward(
    bool training,
    Tensor x,           // [T, B, nheads, headdim]
    Tensor h0,          // [B, nheads, headdim]
    Tensor R_h,         // [nheads, headdim] - DIAGONAL
    Tensor R_x,         // [nheads, headdim] - DIAGONAL
    Tensor R_delta,     // [nheads, headdim] - DIAGONAL
    Tensor W_delta,     // [nheads, headdim, headdim] - still full
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

    // Workspace: pre-computed W_delta @ x for ALL timesteps
    Tensor tmp_Wdelta = torch::empty({ time_steps, batch_size, nheads, headdim }, options);

    // Initialize h[0] with h0
    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "diagonal_mhtr_forward", ([&] {
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
            ptr<scalar_t>(tmp_Wdelta));
    }));

    return { h, v, delta_cache };
}

std::vector<Tensor> diagonal_mhtr_backward(
    Tensor x,           // [T, B, nheads, headdim]
    Tensor R_h,         // [nheads, headdim] - DIAGONAL
    Tensor R_x,         // [nheads, headdim] - DIAGONAL
    Tensor R_delta,     // [nheads, headdim] - DIAGONAL
    Tensor W_delta,     // [nheads, headdim, headdim]
    Tensor h,           // [T+1, B, nheads, headdim]
    Tensor v,           // [T, B, nheads, headdim]
    Tensor delta_cache, // [T, B, nheads, headdim]
    Tensor dh_out) {    // [T, B, nheads, headdim]

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
    CHECK_INPUT(dh_out);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    // Gradient tensors - DIAGONAL for R matrices
    Tensor dx = torch::zeros_like(x);
    Tensor dR_h = torch::zeros({ nheads, headdim }, options);
    Tensor dR_x = torch::zeros({ nheads, headdim }, options);
    Tensor dR_delta = torch::zeros({ nheads, headdim }, options);
    Tensor dW_delta = torch::zeros({ nheads, headdim, headdim }, options);
    Tensor db = torch::zeros({ nheads, headdim }, options);
    Tensor db_delta = torch::zeros({ nheads, headdim }, options);

    // Workspace
    Tensor d_raw_all = torch::empty({ time_steps, batch_size, nheads, headdim }, options);
    Tensor d_delta_raw_all = torch::empty({ time_steps, batch_size, nheads, headdim }, options);
    Tensor dh_prev = torch::zeros({ batch_size, nheads, headdim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "diagonal_mhtr_backward", ([&] {
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
            ptr<scalar_t>(dh_out),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dR_h),
            ptr<scalar_t>(dR_x),
            ptr<scalar_t>(dR_delta),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(d_raw_all),
            ptr<scalar_t>(d_delta_raw_all),
            ptr<scalar_t>(dh_prev));
    }));

    return { dx, dR_h, dR_x, dR_delta, dW_delta, db, db_delta };
}

}  // anonymous namespace

void diagonal_mhtr_init(py::module& m) {
    m.def("diagonal_mhtr_forward", &diagonal_mhtr_forward,
          "Diagonal MHTR forward pass (CUDA)");
    m.def("diagonal_mhtr_backward", &diagonal_mhtr_backward,
          "Diagonal MHTR backward pass (CUDA)");
}
