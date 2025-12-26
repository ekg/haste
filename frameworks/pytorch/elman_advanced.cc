// Copyright 2024 Erik Garrison. Apache 2.0 License.
//
// PyTorch bindings for advanced Elman RNN variants:
// - ElmanTripleR: Three separate R matrices for different signal pathways
// - ElmanSelectiveTripleR: Triple R with input-dependent B gate (like Mamba2)
// - ElmanNeuralMemory: NTM-inspired external memory bank
// - ElmanLowRankR: R = U @ V^T + S decomposition

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using torch::Tensor;

// ============================================================================
// ElmanTripleR: Three R matrices for Input/Hidden/Delta pathways
// ============================================================================

template<typename T>
using TripleRForward = haste::v0::elman_triple_r::ForwardPass<T>;
template<typename T>
using TripleRBackward = haste::v0::elman_triple_r::BackwardPass<T>;

std::vector<Tensor> elman_triple_r_forward(
    bool training,
    Tensor x,           // [T, B, input_size]
    Tensor h0,          // [B, D]
    Tensor R_h,         // [D, D]
    Tensor R_x,         // [D, input_size]
    Tensor R_delta,     // [D, D]
    Tensor W_delta,     // [D, input_size]
    Tensor b,           // [D]
    Tensor b_delta) {   // [D]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto D = h0.size(1);

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

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor delta_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                                  : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);
    Tensor tmp_Rx = torch::empty({ batch_size, D }, options);
    Tensor tmp_Rdelta = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_triple_r_forward", ([&] {
        TripleRForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            input_size,
            D,
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
            ptr<scalar_t>(tmp_Rdelta));
    }));

    return { h, v, delta_cache };
}

std::vector<Tensor> elman_triple_r_backward(
    Tensor x,
    Tensor R_h,
    Tensor R_x,
    Tensor R_delta,
    Tensor W_delta,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor dh_new) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto D = h.size(2);

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

    Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
    Tensor dR_h = torch::zeros({ D, D }, options);
    Tensor dR_x = torch::zeros({ D, input_size }, options);
    Tensor dR_delta = torch::zeros({ D, D }, options);
    Tensor dW_delta = torch::zeros({ D, input_size }, options);
    Tensor db = torch::zeros({ D }, options);
    Tensor db_delta = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);
    Tensor tmp_Rx = torch::empty({ batch_size, D }, options);
    Tensor tmp_Rdelta = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_triple_r_backward", ([&] {
        TripleRBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            input_size,
            D,
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
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_Rx),
            ptr<scalar_t>(tmp_Rdelta));
    }));

    return { dx, dh0, dR_h, dR_x, dR_delta, dW_delta, db, db_delta };
}

// ============================================================================
// ElmanSelectiveTripleR: Triple R with input-dependent B gate (like Mamba2)
// ============================================================================

template<typename T>
using SelectiveTripleRForward = haste::v0::elman_selective_triple_r::ForwardPass<T>;
template<typename T>
using SelectiveTripleRBackward = haste::v0::elman_selective_triple_r::BackwardPass<T>;

std::vector<Tensor> elman_selective_triple_r_forward(
    bool training,
    Tensor x,           // [T, B, input_size]
    Tensor h0,          // [B, D]
    Tensor R_h,         // [D, D]
    Tensor R_x,         // [D, input_size]
    Tensor R_delta,     // [D, D]
    Tensor W_delta,     // [D, input_size]
    Tensor W_B,         // [D, input_size] - NEW: B gate projection
    Tensor b,           // [D]
    Tensor b_delta,     // [D]
    Tensor b_B) {       // [D] - NEW: B gate bias

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto D = h0.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(R_h);
    CHECK_INPUT(R_x);
    CHECK_INPUT(R_delta);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_B);
    CHECK_INPUT(b);
    CHECK_INPUT(b_delta);
    CHECK_INPUT(b_B);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor delta_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                                  : torch::empty({ 0 }, options);
    Tensor B_gate_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                                   : torch::empty({ 0 }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);
    Tensor tmp_Rx = torch::empty({ batch_size, D }, options);
    Tensor tmp_Rdelta = torch::empty({ batch_size, D }, options);
    Tensor tmp_B = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_selective_triple_r_forward", ([&] {
        SelectiveTripleRForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            input_size,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(R_x),
            ptr<scalar_t>(R_delta),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_B),
            ptr<scalar_t>(b),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(b_B),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(B_gate_cache) : nullptr,
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_Rx),
            ptr<scalar_t>(tmp_Rdelta),
            ptr<scalar_t>(tmp_B));
    }));

    return { h, v, delta_cache, B_gate_cache };
}

std::vector<Tensor> elman_selective_triple_r_backward(
    Tensor x,
    Tensor R_h,
    Tensor R_x,
    Tensor R_delta,
    Tensor W_delta,
    Tensor W_B,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor B_gate_cache,
    Tensor dh_new) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto D = h.size(2);

    CHECK_INPUT(x);
    CHECK_INPUT(R_h);
    CHECK_INPUT(R_x);
    CHECK_INPUT(R_delta);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(W_B);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta_cache);
    CHECK_INPUT(B_gate_cache);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
    Tensor dR_h = torch::zeros({ D, D }, options);
    Tensor dR_x = torch::zeros({ D, input_size }, options);
    Tensor dR_delta = torch::zeros({ D, D }, options);
    Tensor dW_delta = torch::zeros({ D, input_size }, options);
    Tensor dW_B = torch::zeros({ D, input_size }, options);
    Tensor db = torch::zeros({ D }, options);
    Tensor db_delta = torch::zeros({ D }, options);
    Tensor db_B = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);
    Tensor tmp_Rx = torch::empty({ batch_size, D }, options);
    Tensor tmp_Rdelta = torch::empty({ batch_size, D }, options);
    Tensor tmp_B = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_selective_triple_r_backward", ([&] {
        SelectiveTripleRBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            input_size,
            D,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R_h),
            ptr<scalar_t>(R_x),
            ptr<scalar_t>(R_delta),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(W_B),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(B_gate_cache),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dR_h),
            ptr<scalar_t>(dR_x),
            ptr<scalar_t>(dR_delta),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(dW_B),
            ptr<scalar_t>(db),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(db_B),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_Rx),
            ptr<scalar_t>(tmp_Rdelta),
            ptr<scalar_t>(tmp_B));
    }));

    return { dx, dh0, dR_h, dR_x, dR_delta, dW_delta, dW_B, db, db_delta, db_B };
}

// ============================================================================
// ElmanNeuralMemory: NTM-inspired external memory bank
// ============================================================================

template<typename T>
using NeuralMemoryForward = haste::v0::elman_neural_memory::ForwardPass<T>;
template<typename T>
using NeuralMemoryBackward = haste::v0::elman_neural_memory::BackwardPass<T>;

std::vector<Tensor> elman_neural_memory_forward(
    bool training,
    Tensor x,           // [T, B, input_size]
    Tensor h0,          // [B, D]
    Tensor R,           // [D, D]
    Tensor W_x,         // [D, input_size]
    Tensor b,           // [D]
    Tensor W_delta,     // [D, input_size]
    Tensor b_delta,     // [D]
    Tensor M,           // [num_slots, memory_dim] - memory bank
    Tensor W_read,      // [memory_dim, D]
    Tensor W_mem,       // [D, memory_dim]
    Tensor W_delta_mem) { // [D, memory_dim]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto D = h0.size(1);
    const auto num_slots = M.size(0);
    const auto memory_dim = M.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(R);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(b_delta);
    CHECK_INPUT(M);
    CHECK_INPUT(W_read);
    CHECK_INPUT(W_mem);
    CHECK_INPUT(W_delta_mem);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor delta_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                                  : torch::empty({ 0 }, options);
    Tensor read_weights_cache = training ? torch::empty({ time_steps, batch_size, num_slots }, options)
                                         : torch::empty({ 0 }, options);
    Tensor memory_read_cache = training ? torch::empty({ time_steps, batch_size, memory_dim }, options)
                                        : torch::empty({ 0 }, options);

    // Workspace
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);
    Tensor tmp_Wx = torch::empty({ batch_size, D }, options);
    Tensor tmp_read_key = torch::empty({ batch_size, memory_dim }, options);
    Tensor tmp_read_weights = torch::empty({ batch_size, num_slots }, options);
    Tensor tmp_memory_read = torch::empty({ batch_size, memory_dim }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_neural_memory_forward", ([&] {
        NeuralMemoryForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            input_size,
            D,
            num_slots,
            memory_dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(b),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(M),
            ptr<scalar_t>(W_read),
            ptr<scalar_t>(W_mem),
            ptr<scalar_t>(W_delta_mem),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            training ? ptr<scalar_t>(read_weights_cache) : nullptr,
            training ? ptr<scalar_t>(memory_read_cache) : nullptr,
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_Wx),
            ptr<scalar_t>(tmp_read_key),
            ptr<scalar_t>(tmp_read_weights),
            ptr<scalar_t>(tmp_memory_read));
    }));

    return { h, v, delta_cache, read_weights_cache, memory_read_cache };
}

std::vector<Tensor> elman_neural_memory_backward(
    Tensor x,
    Tensor R,
    Tensor W_x,
    Tensor W_delta,
    Tensor M,
    Tensor W_read,
    Tensor W_mem,
    Tensor W_delta_mem,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor read_weights_cache,
    Tensor memory_read_cache,
    Tensor dh_new) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto D = h.size(2);
    const auto num_slots = M.size(0);
    const auto memory_dim = M.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(R);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(M);
    CHECK_INPUT(W_read);
    CHECK_INPUT(W_mem);
    CHECK_INPUT(W_delta_mem);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta_cache);
    CHECK_INPUT(read_weights_cache);
    CHECK_INPUT(memory_read_cache);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
    Tensor dR = torch::zeros({ D, D }, options);
    Tensor dW_x = torch::zeros({ D, input_size }, options);
    Tensor db = torch::zeros({ D }, options);
    Tensor dW_delta = torch::zeros({ D, input_size }, options);
    Tensor db_delta = torch::zeros({ D }, options);
    Tensor dM = torch::zeros({ num_slots, memory_dim }, options);
    Tensor dW_read = torch::zeros({ memory_dim, D }, options);
    Tensor dW_mem = torch::zeros({ D, memory_dim }, options);
    Tensor dW_delta_mem = torch::zeros({ D, memory_dim }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);

    // Workspace
    Tensor tmp_Rh = torch::empty({ batch_size, D }, options);
    Tensor tmp_Wx = torch::empty({ batch_size, D }, options);
    Tensor tmp_d_memory_read = torch::empty({ batch_size, memory_dim }, options);
    Tensor tmp_d_read_weights = torch::empty({ batch_size, num_slots }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_neural_memory_backward", ([&] {
        NeuralMemoryBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            input_size,
            D,
            num_slots,
            memory_dim,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(R),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(M),
            ptr<scalar_t>(W_read),
            ptr<scalar_t>(W_mem),
            ptr<scalar_t>(W_delta_mem),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(read_weights_cache),
            ptr<scalar_t>(memory_read_cache),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dR),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(db),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(dM),
            ptr<scalar_t>(dW_read),
            ptr<scalar_t>(dW_mem),
            ptr<scalar_t>(dW_delta_mem),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_Rh),
            ptr<scalar_t>(tmp_Wx),
            ptr<scalar_t>(tmp_d_memory_read),
            ptr<scalar_t>(tmp_d_read_weights));
    }));

    return { dx, dh0, dR, dW_x, db, dW_delta, db_delta, dM, dW_read, dW_mem, dW_delta_mem };
}

// ============================================================================
// ElmanLowRankR: R = U @ V^T + S decomposition
// ============================================================================

template<typename T>
using LowRankRForward = haste::v0::elman_lowrank_r::ForwardPass<T>;
template<typename T>
using LowRankRBackward = haste::v0::elman_lowrank_r::BackwardPass<T>;

std::vector<Tensor> elman_lowrank_r_forward(
    bool training,
    Tensor x,           // [T, B, input_size]
    Tensor h0,          // [B, D]
    Tensor U,           // [D, rank]
    Tensor V,           // [D, rank]
    Tensor S,           // [D, D] - sparse/residual
    Tensor W_x,         // [D, input_size]
    Tensor b,           // [D]
    Tensor W_delta,     // [D, input_size]
    Tensor b_delta) {   // [D]

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto D = h0.size(1);
    const auto rank = U.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(h0);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(S);
    CHECK_INPUT(W_x);
    CHECK_INPUT(b);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(b_delta);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor h = torch::empty({ time_steps + 1, batch_size, D }, options);
    Tensor v = training ? torch::empty({ time_steps, batch_size, D }, options)
                        : torch::empty({ 0 }, options);
    Tensor delta_cache = training ? torch::empty({ time_steps, batch_size, D }, options)
                                  : torch::empty({ 0 }, options);

    // Workspace
    Tensor tmp_Vh = torch::empty({ batch_size, rank }, options);
    Tensor tmp_UVh = torch::empty({ batch_size, D }, options);
    Tensor tmp_Sh = torch::empty({ batch_size, D }, options);
    Tensor tmp_Wx = torch::empty({ batch_size, D }, options);

    h[0] = h0;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_lowrank_r_forward", ([&] {
        LowRankRForward<typename native_type<scalar_t>::T> forward(
            training,
            batch_size,
            input_size,
            D,
            rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(
            time_steps,
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(S),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(b),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(b_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            training ? ptr<scalar_t>(v) : nullptr,
            training ? ptr<scalar_t>(delta_cache) : nullptr,
            ptr<scalar_t>(tmp_Vh),
            ptr<scalar_t>(tmp_UVh),
            ptr<scalar_t>(tmp_Sh),
            ptr<scalar_t>(tmp_Wx));
    }));

    return { h, v, delta_cache };
}

std::vector<Tensor> elman_lowrank_r_backward(
    Tensor x,
    Tensor U,
    Tensor V,
    Tensor S,
    Tensor W_x,
    Tensor W_delta,
    Tensor h,
    Tensor v,
    Tensor delta_cache,
    Tensor dh_new) {

    const auto time_steps = x.size(0);
    const auto batch_size = x.size(1);
    const auto input_size = x.size(2);
    const auto D = h.size(2);
    const auto rank = U.size(1);

    CHECK_INPUT(x);
    CHECK_INPUT(U);
    CHECK_INPUT(V);
    CHECK_INPUT(S);
    CHECK_INPUT(W_x);
    CHECK_INPUT(W_delta);
    CHECK_INPUT(h);
    CHECK_INPUT(v);
    CHECK_INPUT(delta_cache);
    CHECK_INPUT(dh_new);

    const auto options = x.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
    Tensor dU = torch::zeros({ D, rank }, options);
    Tensor dV = torch::zeros({ D, rank }, options);
    Tensor dS = torch::zeros({ D, D }, options);
    Tensor dW_x = torch::zeros({ D, input_size }, options);
    Tensor db = torch::zeros({ D }, options);
    Tensor dW_delta = torch::zeros({ D, input_size }, options);
    Tensor db_delta = torch::zeros({ D }, options);
    Tensor dh0 = torch::zeros({ batch_size, D }, options);

    // Workspace
    Tensor tmp_Vh = torch::empty({ batch_size, rank }, options);
    Tensor tmp_d_raw = torch::empty({ batch_size, D }, options);
    Tensor tmp_d_delta_raw = torch::empty({ batch_size, D }, options);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "elman_lowrank_r_backward", ([&] {
        LowRankRBackward<typename native_type<scalar_t>::T> backward(
            batch_size,
            input_size,
            D,
            rank,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(
            time_steps,
            ptr<scalar_t>(U),
            ptr<scalar_t>(V),
            ptr<scalar_t>(S),
            ptr<scalar_t>(W_x),
            ptr<scalar_t>(W_delta),
            ptr<scalar_t>(x),
            ptr<scalar_t>(h),
            ptr<scalar_t>(v),
            ptr<scalar_t>(delta_cache),
            ptr<scalar_t>(dh_new),
            ptr<scalar_t>(dx),
            ptr<scalar_t>(dU),
            ptr<scalar_t>(dV),
            ptr<scalar_t>(dS),
            ptr<scalar_t>(dW_x),
            ptr<scalar_t>(db),
            ptr<scalar_t>(dW_delta),
            ptr<scalar_t>(db_delta),
            ptr<scalar_t>(dh0),
            ptr<scalar_t>(tmp_Vh),
            ptr<scalar_t>(tmp_d_raw),
            ptr<scalar_t>(tmp_d_delta_raw));
    }));

    return { dx, dh0, dU, dV, dS, dW_x, db, dW_delta, db_delta };
}

}  // anonymous namespace

void init_elman_advanced(py::module& m) {
    // ElmanTripleR - Three R matrices for different signal pathways
    m.def("elman_triple_r_forward", &elman_triple_r_forward, "ElmanTripleR forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_triple_r_backward", &elman_triple_r_backward, "ElmanTripleR backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanSelectiveTripleR - Triple R with input-dependent B gate (like Mamba2)
    m.def("elman_selective_triple_r_forward", &elman_selective_triple_r_forward, "ElmanSelectiveTripleR forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_selective_triple_r_backward", &elman_selective_triple_r_backward, "ElmanSelectiveTripleR backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanNeuralMemory - NTM-inspired external memory bank
    m.def("elman_neural_memory_forward", &elman_neural_memory_forward, "ElmanNeuralMemory forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_neural_memory_backward", &elman_neural_memory_backward, "ElmanNeuralMemory backward",
          py::call_guard<py::gil_scoped_release>());

    // ElmanLowRankR - R = U @ V^T + S decomposition
    m.def("elman_lowrank_r_forward", &elman_lowrank_r_forward, "ElmanLowRankR forward",
          py::call_guard<py::gil_scoped_release>());
    m.def("elman_lowrank_r_backward", &elman_lowrank_r_backward, "ElmanLowRankR backward",
          py::call_guard<py::gil_scoped_release>());
}
