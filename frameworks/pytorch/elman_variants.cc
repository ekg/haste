// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for Elman RNN variants.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using torch::Tensor;

// ============================================================================
// ElmanTanh: tanh + tanh gate
// ============================================================================

template<typename T>
using ElmanTanhForward = haste::v0::elman_tanh::ForwardPass<T>;
template<typename T>
using ElmanTanhBackward = haste::v0::elman_tanh::BackwardPass<T>;

std::vector<Tensor> elman_tanh_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,
    Tensor R,
    Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_tanh_forward", ([&] {
        ElmanTanhForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
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

std::vector<Tensor> elman_tanh_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
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

    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_tanh_backward", ([&] {
        ElmanTanhBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanSigmoid: tanh + sigmoid gate
// ============================================================================

template<typename T>
using ElmanSigmoidForward = haste::v0::elman_sigmoid::ForwardPass<T>;
template<typename T>
using ElmanSigmoidBackward = haste::v0::elman_sigmoid::BackwardPass<T>;

std::vector<Tensor> elman_sigmoid_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,
    Tensor R,
    Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_sigmoid_forward", ([&] {
        ElmanSigmoidForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
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

std::vector<Tensor> elman_sigmoid_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
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

    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_sigmoid_backward", ([&] {
        ElmanSigmoidBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanSwish: silu + silu gate
// ============================================================================

template<typename T>
using ElmanSwishForward = haste::v0::elman_swish::ForwardPass<T>;
template<typename T>
using ElmanSwishBackward = haste::v0::elman_swish::BackwardPass<T>;

std::vector<Tensor> elman_swish_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,
    Tensor R,
    Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_swish_forward", ([&] {
        ElmanSwishForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
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

std::vector<Tensor> elman_swish_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
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

    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_swish_backward", ([&] {
        ElmanSwishBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanGelu: tanh + gelu gate
// ============================================================================

template<typename T>
using ElmanGeluForward = haste::v0::elman_gelu::ForwardPass<T>;
template<typename T>
using ElmanGeluBackward = haste::v0::elman_gelu::BackwardPass<T>;

std::vector<Tensor> elman_gelu_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,
    Tensor R,
    Tensor bias) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);
    const auto gate_dim = 2 * D;

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, gate_dim});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, gate_dim }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, gate_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_gelu_forward", ([&] {
        ElmanGeluForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
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

std::vector<Tensor> elman_gelu_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
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

    Tensor dWx_all = torch::empty({ time_steps, batch_size, gate_dim }, options);
    Tensor dR = torch::zeros({ gate_dim, D }, options);
    Tensor dbias = torch::zeros({ gate_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, gate_dim }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_gelu_backward", ([&] {
        ElmanGeluBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, gate_dim});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanNoGate: tanh only, no gating (ablation baseline)
// ============================================================================

template<typename T>
using ElmanNoGateForward = haste::v0::elman_nogate::ForwardPass<T>;
template<typename T>
using ElmanNoGateBackward = haste::v0::elman_nogate::BackwardPass<T>;

std::vector<Tensor> elman_nogate_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D] - no gate dimension
    Tensor R,          // [D, D]
    Tensor bias) {     // [D]
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_nogate_forward", ([&] {
        ElmanNoGateForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
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

std::vector<Tensor> elman_nogate_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_nogate_backward", ([&] {
        ElmanNoGateBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias };
}

// ============================================================================
// ElmanLeaky: tanh + input-dependent leaky integration (Mamba2-style Δ)
// ============================================================================

template<typename T>
using ElmanLeakyForward = haste::v0::elman_leaky::ForwardPass<T>;
template<typename T>
using ElmanLeakyBackward = haste::v0::elman_leaky::BackwardPass<T>;

std::vector<Tensor> elman_leaky_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D] - no gate dimension
    Tensor R,          // [D, D]
    Tensor bias,       // [D]
    Tensor delta) {    // [T, B, D] - precomputed delta values
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);
    CHECK_INPUT(delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    // Reshape delta to match kernel expectations
    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_forward", ([&] {
        ElmanLeakyForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_leaky_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor delta,      // [T, B, D] - same delta used in forward
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor d_delta = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_backward", ([&] {
        ElmanLeakyBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(d_delta),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    // Return d_delta reshaped to match input shape
    d_delta = d_delta.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias, d_delta };
}

// ============================================================================
// ElmanLeakySilu: silu + leaky integration (NO output gate)
// ============================================================================

template<typename T>
using ElmanLeakySiluForward = haste::v0::elman_leaky_silu::ForwardPass<T>;
template<typename T>
using ElmanLeakySiluBackward = haste::v0::elman_leaky_silu::BackwardPass<T>;

std::vector<Tensor> elman_leaky_silu_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D]
    Tensor R,          // [D, D]
    Tensor bias,       // [D]
    Tensor delta) {    // [T, B, D] - precomputed delta values (sigmoid output)
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);
    CHECK_INPUT(delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_silu_forward", ([&] {
        ElmanLeakySiluForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_leaky_silu_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor delta,
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor delta_flat = delta.reshape({time_steps * batch_size, D});

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor d_delta = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_silu_backward", ([&] {
        ElmanLeakySiluBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_flat),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(d_delta),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    d_delta = d_delta.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias, d_delta };
}

// ============================================================================
// ElmanLeakySelective: tanh + Mamba2-style discretization + per-channel decay
// ============================================================================

template<typename T>
using ElmanLeakySelectiveForward = haste::v0::elman_leaky_selective::ForwardPass<T>;
template<typename T>
using ElmanLeakySelectiveBackward = haste::v0::elman_leaky_selective::BackwardPass<T>;

std::vector<Tensor> elman_leaky_selective_forward(
    bool training,
    Tensor x,
    Tensor h0,
    Tensor Wx,         // [D, D] - candidate input projection
    Tensor R,          // [D, D] - recurrence matrix
    Tensor bias,       // [D] - candidate bias
    Tensor delta_raw,  // [T, B, D] - precomputed W_delta @ x + b_delta (before softplus)
    Tensor A) {        // [D] - per-channel decay rates (log-space)
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(bias);
    CHECK_INPUT(delta_raw);
    CHECK_INPUT(A);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor Wx_all = torch::matmul(x_flat, Wx.t());
    Wx_all = Wx_all.reshape({time_steps, batch_size, D});

    // Reshape delta_raw to match kernel expectations
    Tensor delta_raw_flat = delta_raw.reshape({time_steps * batch_size, D});

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    // v stores [raw, delta_raw] for backward, so 2*D per timestep
    Tensor v = training ? torch::empty({ time_steps, batch_size, 2 * D }, options)
                        : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_selective_forward", ([&] {
        ElmanLeakySelectiveForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(bias),
            ptr<scalar_t>(Wx_all),
            ptr<scalar_t>(delta_raw_flat),
            ptr<scalar_t>(A),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            ptr<scalar_t>(tmp_Rh));
    }));

    return { h, v };
}

std::vector<Tensor> elman_leaky_selective_backward(
    Tensor x,
    Tensor Wx,
    Tensor R,
    Tensor h,
    Tensor v,
    Tensor A,          // [D] - per-channel decay rates
    Tensor dh_new) {
    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto D = x.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(Wx);
    CHECK_INPUT(R);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(A);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dWx_all = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dbias = torch::zeros({ D }, options);
    Tensor d_delta_raw = torch::empty({ time_steps, batch_size, D }, options);
    Tensor dA = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_dRh = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_leaky_selective_backward", ([&] {
        ElmanLeakySelectiveBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            D,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(A),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dWx_all),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dbias),
            ptr<scalar_t>(d_delta_raw),
            ptr<scalar_t>(dA),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_dRh));
    }));

    Tensor dWx_flat = dWx_all.reshape({time_steps * batch_size, D});
    Tensor x_flat = x.reshape({time_steps * batch_size, D});
    Tensor dWx = torch::matmul(dWx_flat.t(), x_flat);
    Tensor dx_flat = torch::matmul(dWx_flat, Wx);
    Tensor dx = dx_flat.reshape({time_steps, batch_size, D});

    // Reshape d_delta_raw to match input shape
    d_delta_raw = d_delta_raw.reshape({time_steps, batch_size, D});

    return { dx, dh0, dWx, dR, dbias, d_delta_raw, dA };
}

}  // anonymous namespace

void elman_variants_init(py::module& m) {
    // ElmanTanh
    m.def("elman_tanh_forward", &elman_tanh_forward, "ElmanTanh forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_tanh_backward", &elman_tanh_backward, "ElmanTanh backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanSigmoid
    m.def("elman_sigmoid_forward", &elman_sigmoid_forward, "ElmanSigmoid forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_sigmoid_backward", &elman_sigmoid_backward, "ElmanSigmoid backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanSwish
    m.def("elman_swish_forward", &elman_swish_forward, "ElmanSwish forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_swish_backward", &elman_swish_backward, "ElmanSwish backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanGelu
    m.def("elman_gelu_forward", &elman_gelu_forward, "ElmanGelu forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_gelu_backward", &elman_gelu_backward, "ElmanGelu backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanNoGate
    m.def("elman_nogate_forward", &elman_nogate_forward, "ElmanNoGate forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_nogate_backward", &elman_nogate_backward, "ElmanNoGate backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLeaky - input-dependent delta (Mamba2-style discretization)
    m.def("elman_leaky_forward", &elman_leaky_forward, "ElmanLeaky forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_leaky_backward", &elman_leaky_backward, "ElmanLeaky backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLeakySilu - silu + leaky integration (NO output gate)
    m.def("elman_leaky_silu_forward", &elman_leaky_silu_forward, "ElmanLeakySilu forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_leaky_silu_backward", &elman_leaky_silu_backward, "ElmanLeakySilu backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLeakySelective - Mamba2-style discretization + per-channel decay + nonlinearity
    m.def("elman_leaky_selective_forward", &elman_leaky_selective_forward, "ElmanLeakySelective forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_leaky_selective_backward", &elman_leaky_selective_backward, "ElmanLeakySelective backward",
          py::call_guard<py::gil_scoped_release>());
}
