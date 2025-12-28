// Copyright 2025 Erik Garrison. Apache 2.0 License.
//
// Level 4: Log-Storage Diagonal Elman - Signed log storage for hidden state
// Same recurrence as Level 3, but hidden state stored as (log|h|, sign(h))
// This prevents numerical underflow at depth.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cfloat>

#include "haste/elman_ladder.h"
#include "blas.h"
#include "inline_ops.h"

namespace {

// Device function: signed log addition
// Computes log|a + b| and sign(a + b) where a = sign_a * exp(log_a)
__device__ __forceinline__ void signed_log_add(
    float log_a, float sign_a,
    float log_b, float sign_b,
    float& log_result, float& sign_result) {

    float max_log = fmaxf(log_a, log_b);
    float min_log = fminf(log_a, log_b);
    float diff = min_log - max_log;  // Always <= 0

    // Determine which has max log
    bool a_is_max = log_a >= log_b;
    float sign_max = a_is_max ? sign_a : sign_b;
    float sign_min = a_is_max ? sign_b : sign_a;

    bool same_sign = sign_max * sign_min > 0;

    if (same_sign) {
        // log(exp(max) + exp(min)) = max + log(1 + exp(diff))
        log_result = max_log + log1pf(expf(diff));
    } else {
        // log(exp(max) - exp(min)) = max + log(1 - exp(diff))
        float exp_diff = expf(diff);
        if (exp_diff >= 1.0f) {
            // Complete cancellation
            log_result = -1e10f;
            sign_result = 1.0f;
            return;
        }
        log_result = max_log + log1pf(-exp_diff);
    }
    sign_result = sign_max;
}

// Device function: convert linear to log space
__device__ __forceinline__ void to_log_space(float x, float& log_x, float& sign_x) {
    sign_x = (x >= 0) ? 1.0f : -1.0f;
    log_x = logf(fabsf(x) + 1e-10f);
}

// Device function: convert log space to linear
__device__ __forceinline__ float from_log_space(float log_x, float sign_x) {
    return sign_x * expf(log_x);
}

// Kernel: Convert log/sign representation to linear
template<typename T>
__global__ void LogToLinearKernel(
    const int n,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    T* __restrict__ h_linear) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float log_val = static_cast<float>(log_h[idx]);
        float sign_val = static_cast<float>(sign_h[idx]);
        h_linear[idx] = static_cast<T>(sign_val * expf(log_val));
    }
}

// Kernel: Convert gradient w.r.t. h to gradient w.r.t. log_h
// Since h = sign * exp(log_h), by chain rule: d_log_h = d_h * h
template<typename T>
__global__ void ConvertLinearGradToLogGrad(
    const int n,
    const T* __restrict__ d_h,
    const T* __restrict__ h,
    T* __restrict__ d_log_h) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float d_h_val = static_cast<float>(d_h[idx]);
        float h_val = static_cast<float>(h[idx]);
        d_log_h[idx] = static_cast<T>(d_h_val * h_val);
    }
}

// Kernel: Compute log-space gated update with diagonal r_h
// Now stores softmax weights for TRUE log-space backward!
template<typename T>
__global__ void LogStorageGatedUpdate(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,  // [B, dim] log|h_prev|
    const T* __restrict__ sign_h_prev, // [B, dim] sign(h_prev)
    const T* __restrict__ wx_x,        // [B, dim] W_x @ x
    const T* __restrict__ r_h,         // [dim] diagonal decay
    const T* __restrict__ delta_raw,   // [B, dim] W_delta @ x
    const T* __restrict__ b,           // [dim]
    const T* __restrict__ b_delta,     // [dim]
    T* __restrict__ log_h_out,         // [B, dim] log|h_new|
    T* __restrict__ sign_h_out,        // [B, dim] sign(h_new)
    T* __restrict__ v_cache,
    T* __restrict__ delta_cache,
    T* __restrict__ weight1_cache,     // NEW: softmax weight for (1-δ)*h_prev term
    T* __restrict__ log_term1_cache,   // NEW: log|(1-δ)*h_prev|
    T* __restrict__ log_term2_cache) { // NEW: log|δ*candidate|

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Get log-space h_prev
        float log_h_prev_val = static_cast<float>(log_h_prev[idx]);
        float sign_h_prev_val = static_cast<float>(sign_h_prev[idx]);

        // Convert h_prev from log space to linear for candidate computation
        float h_prev_linear = from_log_space(log_h_prev_val, sign_h_prev_val);

        // Delta gate
        float delta_raw_val = static_cast<float>(delta_raw[idx]) + static_cast<float>(b_delta[d]);
        float delta_f = 1.0f / (1.0f + expf(-delta_raw_val));
        float one_minus_delta = 1.0f - delta_f;
        if (delta_cache) delta_cache[idx] = static_cast<T>(delta_f);

        // Log-space gate values using numerically stable formulas:
        // log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))
        // log(1 - sigmoid(x)) = -softplus(x) = -log(1 + exp(x))
        float log_delta = -log1pf(expf(-delta_raw_val));
        float log_one_minus_delta = -log1pf(expf(delta_raw_val));

        // Candidate: tanh(W_x @ x + r_h * h_prev + b)
        float v_f = static_cast<float>(wx_x[idx]) +
                    static_cast<float>(r_h[d]) * h_prev_linear +
                    static_cast<float>(b[d]);
        if (v_cache) v_cache[idx] = static_cast<T>(v_f);
        float candidate = tanhf(v_f);

        // Convert candidate to log space
        float sign_candidate = (candidate >= 0) ? 1.0f : -1.0f;
        float log_candidate = logf(fabsf(candidate) + 1e-10f);

        // Log-space terms for GRU update: h_new = (1-δ)*h_prev + δ*candidate
        // term1 = (1-δ)*h_prev:  log_term1 = log(1-δ) + log|h_prev|
        // term2 = δ*candidate:   log_term2 = log(δ) + log|candidate|
        float log_term1 = log_one_minus_delta + log_h_prev_val;
        float sign_term1 = sign_h_prev_val;  // Sign of h_prev (1-δ is positive)

        float log_term2 = log_delta + log_candidate;
        float sign_term2 = sign_candidate;  // Sign of candidate (δ is positive)

        // Add in log space using signed_log_add
        float log_h_new, sign_h_new;
        signed_log_add(log_term1, sign_term1, log_term2, sign_term2, log_h_new, sign_h_new);

        // Compute softmax weights for backward pass
        // weight1 = exp(log_term1 - log_h_new) = |(1-δ)*h_prev| / |h_new|
        // weight2 = exp(log_term2 - log_h_new) = |δ*candidate| / |h_new|
        // These are BOUNDED [0, 1] and sum to ~1 (exactly 1 when same sign)
        float weight1 = expf(log_term1 - log_h_new);
        weight1 = fminf(fmaxf(weight1, 0.0f), 1.0f);  // Clamp for safety

        log_h_out[idx] = static_cast<T>(log_h_new);
        sign_h_out[idx] = static_cast<T>(sign_h_new);

        // Store for backward
        if (weight1_cache) weight1_cache[idx] = static_cast<T>(weight1);
        if (log_term1_cache) log_term1_cache[idx] = static_cast<T>(log_term1);
        if (log_term2_cache) log_term2_cache[idx] = static_cast<T>(log_term2);
    }
}

// Kernel: Compute compete×silu output from log-space hidden state
// KEY CHANGE: Uses SIGNED log representation for softmax to preserve sign info!
// signed_log = log|h| * sign_h (positive for h>0, negative for h<0)
// This correctly differentiates positive vs negative h values in compete.
template<typename T>
__global__ void LogStorageSelectiveOutput(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,
    const T* __restrict__ w_out_log_h,  // W_out @ log_h (computed in LOG space!)
    T* __restrict__ output,
    T* __restrict__ compete_cache) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // Use SIGNED log for softmax: signed_log = log|h| * sign
    // This preserves sign information:
    // h = +5 → signed_log = +1.61 (large positive, high compete weight)
    // h = -5 → signed_log = -1.61 (large negative, low compete weight)
    float max_signed_log = -FLT_MAX;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + i]);
        float sign_h_val = static_cast<float>(sign_h[base + i]);
        float signed_log = log_h_val * sign_h_val;  // positive for h>0, negative for h<0
        max_signed_log = fmaxf(max_signed_log, signed_log);
    }
    smem[threadIdx.x] = max_signed_log;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    max_signed_log = smem[0];
    __syncthreads();

    // Compute exp sum for softmax(signed_log)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + i]);
        float sign_h_val = static_cast<float>(sign_h[base + i]);
        float signed_log = log_h_val * sign_h_val;
        sum += expf(signed_log - max_signed_log);
    }
    float* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = sum_smem[0];

    // Compute output: compete(signed_log) * silu(W_out @ log_h)
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float log_h_val = static_cast<float>(log_h[base + i]);
        float sign_h_val = static_cast<float>(sign_h[base + i]);
        float signed_log = log_h_val * sign_h_val;
        float compete = expf(signed_log - max_signed_log) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w = static_cast<float>(w_out_log_h[base + i]);  // W_out @ log_h
        float silu_val = w / (1.0f + expf(-w));
        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// Kernel: Compute compete×silu output from LINEAR hidden state
// This is the correct version that uses h (real space), not log_h
template<typename T>
__global__ void LogStorageSelectiveOutputLinear(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h_linear,    // Real-space hidden state
    const T* __restrict__ w_out_h,     // W_out @ h
    T* __restrict__ output,
    T* __restrict__ compete_cache) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // Compute softmax on h values (real space)
    float max_h = -FLT_MAX;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float h_val = static_cast<float>(h_linear[base + i]);
        max_h = fmaxf(max_h, h_val);
    }
    smem[threadIdx.x] = max_h;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    max_h = smem[0];
    __syncthreads();

    // Compute exp sum for softmax(h)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float h_val = static_cast<float>(h_linear[base + i]);
        sum += expf(h_val - max_h);
    }
    float* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = sum_smem[0];

    // Compute output: compete(h) * silu(W_out @ h)
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float h_val = static_cast<float>(h_linear[base + i]);
        float compete = expf(h_val - max_h) / sum;
        if (compete_cache) compete_cache[base + i] = static_cast<T>(compete);

        float w = static_cast<float>(w_out_h[base + i]);
        float silu_val = w / (1.0f + expf(-w));
        output[base + i] = static_cast<T>(compete * silu_val);
    }
}

// Backward through selective output for LINEAR version
// Computes gradient w.r.t. h_linear
template<typename T>
__global__ void LogStorageSelectiveOutputLinearBackward(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ h_linear,
    const T* __restrict__ w_out_h,
    const T* __restrict__ compete,
    const T* __restrict__ d_output,
    T* __restrict__ d_h_linear,       // gradient w.r.t. h
    T* __restrict__ d_w_out_h) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // First compute sum(compete * d_compete) for softmax backward
    float sum_compete_dcompete = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float d_out = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_h[base + i]);
        float silu = w / (1.0f + expf(-w));
        float d_compete = d_out * silu;  // d_loss/d_compete
        float comp = static_cast<float>(compete[base + i]);
        sum_compete_dcompete += comp * d_compete;
    }

    smem[threadIdx.x] = sum_compete_dcompete;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum_compete_dcompete = smem[0];
    __syncthreads();

    // Compute gradients
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float d_out = static_cast<float>(d_output[base + i]);
        float comp = static_cast<float>(compete[base + i]);
        float w = static_cast<float>(w_out_h[base + i]);

        // Gradient through silu: d_silu/dw = sigmoid(w) + w * sigmoid(w) * (1 - sigmoid(w))
        float sig_w = 1.0f / (1.0f + expf(-w));
        float silu = w * sig_w;
        float d_silu_dw = sig_w + w * sig_w * (1.0f - sig_w);

        // d_w_out_h = d_output * compete * d_silu_dw
        float d_w = d_out * comp * d_silu_dw;
        d_w_out_h[base + i] = static_cast<T>(d_w);

        // Softmax backward: d_h = compete * (d_compete - sum(compete * d_compete))
        float d_compete = d_out * silu;
        float d_h = comp * (d_compete - sum_compete_dcompete);
        d_h_linear[base + i] = static_cast<T>(d_h);
    }
}

// Backward through selective output - outputs gradient in LOG SPACE directly!
// KEY CHANGE: Uses signed_log = log_h * sign_h for softmax
// Gradient w.r.t. log_h = gradient w.r.t. signed_log * sign_h
template<typename T>
__global__ void LogStorageSelectiveOutputBackward(
    const int batch_size,
    const int dim,
    const int n_groups,
    const int group_size,
    const T* __restrict__ log_h,
    const T* __restrict__ sign_h,        // Used for signed_log derivative
    const T* __restrict__ w_out_log_h,   // W_out @ log_h
    const T* __restrict__ compete,
    const T* __restrict__ d_output,
    T* __restrict__ d_log_h,             // gradient flows DIRECTLY to log_h!
    T* __restrict__ d_w_out_log_h) {

    extern __shared__ float smem[];

    const int b = blockIdx.x;
    const int g = blockIdx.y;

    if (b >= batch_size || g >= n_groups) return;

    const int base = b * dim + g * group_size;

    // Gradient through silu: d_w_out_log_h
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_log_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float dsilu_dw = sig * (1.0f + w * (1.0f - sig));
        float comp = static_cast<float>(compete[base + i]);
        d_w_out_log_h[base + i] = static_cast<T>(dout * comp * dsilu_dw);
    }
    __syncthreads();

    // Compute sum for softmax backward
    float sum_compete_dcompete = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_log_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float silu_val = w * sig;
        float comp = static_cast<float>(compete[base + i]);
        sum_compete_dcompete += comp * dout * silu_val;
    }

    smem[threadIdx.x] = sum_compete_dcompete;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    sum_compete_dcompete = smem[0];

    // Compute d_log_h from softmax(signed_log) gradient
    // Forward: compete = softmax(signed_log) where signed_log = log_h * sign_h
    // Backward: d_signed_log = compete * (d_compete - sum)
    //           d_log_h = d_signed_log * d(signed_log)/d(log_h)
    //                   = d_signed_log * sign_h
    //
    // NO multiplication by |h| here! This preserves gradient magnitude.
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float dout = static_cast<float>(d_output[base + i]);
        float w = static_cast<float>(w_out_log_h[base + i]);
        float sig = 1.0f / (1.0f + expf(-w));
        float silu_val = w * sig;
        float comp = static_cast<float>(compete[base + i]);
        float d_comp = dout * silu_val;

        // Softmax backward gives d_signed_log
        float d_signed_log = comp * (d_comp - sum_compete_dcompete);

        // Chain rule: d_log_h = d_signed_log * sign_h
        // Since signed_log = log_h * sign_h, derivative is sign_h
        float sign_h_val = static_cast<float>(sign_h[base + i]);
        float d_log_h_val = d_signed_log * sign_h_val;

        d_log_h[base + i] = static_cast<T>(d_log_h_val);
    }
}

// TRUE LOG-SPACE BACKWARD through gated update
// Gradients flow w.r.t. log|h| using softmax weights from logaddexp
// This prevents gradient vanishing at depth!
//
// Key insight (Mamba2-style):
// For z = logaddexp(a, b), the gradients are:
//   dz/da = exp(a - z) = weight1 (softmax weight, bounded [0,1])
//   dz/db = exp(b - z) = weight2 (softmax weight, bounded [0,1])
//
// So gradient w.r.t. log|h_prev| is:
//   d_log_h_prev = d_log_h * weight1  (NO multiplicative decay!)
//
template<typename T>
__global__ void LogStorageGatedBackward(
    const int batch_size,
    const int dim,
    const T* __restrict__ log_h_prev,
    const T* __restrict__ sign_h_prev,
    const T* __restrict__ log_h,        // log|h_new|
    const T* __restrict__ v,
    const T* __restrict__ delta,
    const T* __restrict__ r_h,
    const T* __restrict__ weight1,      // softmax weight for (1-δ)*h_prev term
    const T* __restrict__ log_term1,    // log|(1-δ)*h_prev|
    const T* __restrict__ log_term2,    // log|δ*candidate|
    const T* __restrict__ d_log_h,      // gradient on log|h| (LOG SPACE!)
    const T* __restrict__ d_log_h_recurrent, // recurrent gradient in log space
    T* __restrict__ dv,
    T* __restrict__ d_delta_raw,
    T* __restrict__ d_log_h_prev,       // gradient on log|h_prev| in LOG SPACE!
    float* __restrict__ dr_h,
    float* __restrict__ db,
    float* __restrict__ db_delta) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * dim;

    if (idx < total) {
        const int d = idx % dim;

        // Get the incoming gradient in LOG SPACE
        float grad_log_h = static_cast<float>(d_log_h[idx]);
        if (d_log_h_recurrent) grad_log_h += static_cast<float>(d_log_h_recurrent[idx]);

        // Get stored values
        float w1 = static_cast<float>(weight1[idx]);  // softmax weight for term1
        float w2 = 1.0f - w1;  // softmax weight for term2 (they sum to ~1)

        float log_h_prev_val = static_cast<float>(log_h_prev[idx]);
        float sign_h_prev_val = static_cast<float>(sign_h_prev[idx]);
        float h_prev_linear = sign_h_prev_val * expf(log_h_prev_val);

        float del = static_cast<float>(delta[idx]);
        float one_minus_del = 1.0f - del;
        float v_val = static_cast<float>(v[idx]);
        float cand = tanhf(v_val);
        float log_h_val = static_cast<float>(log_h[idx]);

        // ==== TRUE LOG-SPACE BACKWARD ====
        // z = logaddexp(log_term1, log_term2)
        // d_log_term1 = grad_log_h * w1  (bounded gradient!)
        // d_log_term2 = grad_log_h * w2  (bounded gradient!)
        float d_log_term1 = grad_log_h * w1;
        float d_log_term2 = grad_log_h * w2;

        // term1 = log(1-δ) + log|h_prev|
        // d_log_one_minus_delta = d_log_term1
        // d_log_h_prev_from_gru = d_log_term1  <-- KEY: gradient flows without decay!

        // term2 = log(δ) + log|candidate|
        // d_log_delta = d_log_term2
        // d_log_candidate = d_log_term2

        // ==== Backward through log_sigmoid and log_one_minus_sigmoid ====
        // log_delta = -log(1 + exp(-delta_raw))
        // d_delta_raw_from_log_delta = d_log_term2 * sigmoid(delta_raw) = d_log_term2 * del
        //
        // log_one_minus_delta = -log(1 + exp(delta_raw))
        // d_delta_raw_from_log_one_minus = -d_log_term1 * sigmoid(delta_raw) = -d_log_term1 * del
        // But we need d/d(delta_raw) of log(1-sigmoid), which is -(1-del)
        // So: d_delta_raw_from_log_one_minus = d_log_term1 * (-(1-del)) = -d_log_term1 * one_minus_del
        float d_delta_raw_val = d_log_term2 * del - d_log_term1 * one_minus_del;
        d_delta_raw[idx] = static_cast<T>(d_delta_raw_val);

        // ==== Backward through log|candidate| ====
        // log|candidate| = log|tanh(v)|
        // d_v = d_log_term2 * d(log|tanh(v)|)/dv
        //     = d_log_term2 * (1 - tanh²(v)) / |tanh(v)| * sign(tanh(v))
        //     = d_log_term2 * (1 - cand²) / cand  (when cand != 0)
        float dtanh = 1.0f - cand * cand;
        float dv_val = 0.0f;
        if (fabsf(cand) > 1e-6f) {
            dv_val = d_log_term2 * dtanh / fabsf(cand);
            // Handle sign correctly
            dv_val = (cand >= 0) ? dv_val : -dv_val;
        }
        dv[idx] = static_cast<T>(dv_val);

        // ==== Gradient for log|h_prev| ====
        // From GRU recurrence: d_log_h_prev = d_log_term1 (bounded!)
        // From r_h * h_prev in candidate: need chain rule through linear->log conversion
        //
        // v = ... + r_h * h_prev_linear
        // h_prev_linear = sign * exp(log_h_prev)
        // d_log_h_prev += dv * r_h * d(h_prev_linear)/d(log_h_prev)
        //              = dv * r_h * h_prev_linear
        float d_log_h_prev_from_rh = dv_val * static_cast<float>(r_h[d]) * h_prev_linear;

        // Total: GRU recurrence path + candidate path
        float d_log_h_prev_val = d_log_term1 + d_log_h_prev_from_rh;
        d_log_h_prev[idx] = static_cast<T>(d_log_h_prev_val);

        // ==== Parameter gradients ====
        // dr_h = dv * h_prev_linear (accumulated over batch)
        float dr_h_val = dv_val * h_prev_linear;
        atomicAdd(&dr_h[d], dr_h_val);

        // db, db_delta
        atomicAdd(&db[d], dv_val);
        atomicAdd(&db_delta[d], d_delta_raw_val);
    }
}

}  // anonymous namespace


namespace haste {
namespace v0 {
namespace elman_ladder {

// =============================================================================
// Log-Storage Diagonal Elman Forward
// =============================================================================

template<typename T>
LogStorageDiagonalElmanForward<T>::LogStorageDiagonalElmanForward(
    bool training,
    int batch_size,
    int dim,
    int n_groups,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : training_(training),
      batch_size_(batch_size),
      dim_(dim),
      n_groups_(n_groups),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void LogStorageDiagonalElmanForward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* W_delta,
    const T* W_out,
    const T* b,
    const T* b_delta,
    const T* x,
    T* log_h,
    T* sign_h,
    T* output,
    T* v,
    T* delta_cache,
    T* compete_cache,
    T* weight1_cache,     // NEW: softmax weights for log-space backward
    T* log_term1_cache,   // NEW: log|(1-δ)*h_prev|
    T* log_term2_cache) { // NEW: log|δ*candidate|

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // Workspace - h_linear needed for proper output computation
    T *wx_x, *delta_tmp, *w_out_h, *h_linear;
    cudaMalloc(&wx_x, BD * sizeof(T));
    cudaMalloc(&delta_tmp, BD * sizeof(T));
    cudaMalloc(&w_out_h, BD * sizeof(T));
    cudaMalloc(&h_linear, BD * sizeof(T));  // Convert log_h to real h for output

    for (int t = 0; t < steps; ++t) {
        const T* x_t = x + t * BD;
        const T* log_h_prev = log_h + t * BD;
        const T* sign_h_prev = sign_h + t * BD;
        T* log_h_t = log_h + (t + 1) * BD;
        T* sign_h_t = sign_h + (t + 1) * BD;
        T* out_t = output + t * BD;
        T* v_t = training_ ? (v + t * BD) : nullptr;
        T* delta_t = training_ ? (delta_cache + t * BD) : nullptr;
        T* compete_t = training_ ? (compete_cache + t * BD) : nullptr;
        T* weight1_t = training_ ? (weight1_cache + t * BD) : nullptr;
        T* log_term1_t = training_ ? (log_term1_cache + t * BD) : nullptr;
        T* log_term2_t = training_ ? (log_term2_cache + t * BD) : nullptr;

        // wx_x = W_x @ x_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_x, dim_, x_t, dim_, &beta_zero, wx_x, dim_);

        // delta_tmp = W_delta @ x_t
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, x_t, dim_, &beta_zero, delta_tmp, dim_);

        // Log-space gated update (stores softmax weights for backward)
        LogStorageGatedUpdate<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, log_h_prev, sign_h_prev, wx_x, r_h, delta_tmp,
            b, b_delta, log_h_t, sign_h_t, v_t, delta_t, weight1_t, log_term1_t, log_term2_t);

        // Convert log_h to h_linear: h = sign * exp(log_h)
        // This is needed for proper output computation - log_h is a log magnitude, not a feature!
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_t, sign_h_t, h_linear);

        // Compute W_out @ h (real space, not log space!)
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, h_linear, dim_, &beta_zero, w_out_h, dim_);

        // Selective output using h_linear (softmax on real h values)
        dim3 grid(batch_size_, n_groups_);
        int smem_size = 2 * block_size * sizeof(float);
        LogStorageSelectiveOutputLinear<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, h_linear, w_out_h, out_t, compete_t);
    }

    cudaFree(wx_x);
    cudaFree(delta_tmp);
    cudaFree(w_out_h);
    cudaFree(h_linear);
}

// =============================================================================
// Log-Storage Diagonal Elman Backward
// =============================================================================

template<typename T>
LogStorageDiagonalElmanBackward<T>::LogStorageDiagonalElmanBackward(
    int batch_size,
    int dim,
    int n_groups,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream)
    : batch_size_(batch_size),
      dim_(dim),
      n_groups_(n_groups),
      blas_handle_(blas_handle),
      stream_(stream) {}

template<typename T>
void LogStorageDiagonalElmanBackward<T>::Run(
    int steps,
    const T* W_x,
    const T* r_h,
    const T* W_delta,
    const T* W_out,
    const T* x,
    const T* log_h,
    const T* sign_h,
    const T* v,
    const T* delta_cache,
    const T* compete_cache,
    const T* weight1_cache,      // NEW: softmax weights for log-space backward
    const T* log_term1_cache,    // NEW: log|(1-δ)*h_prev|
    const T* log_term2_cache,    // NEW: log|δ*candidate|
    const T* d_output,
    T* dx,
    T* dW_x,
    T* dr_h,
    T* dW_delta,
    T* dW_out,
    T* db,
    T* db_delta) {

    static const T alpha = static_cast<T>(1.0);
    static const T beta_zero = static_cast<T>(0.0);

    const int BD = batch_size_ * dim_;
    const int block_size = 256;
    const int num_blocks = (BD + block_size - 1) / block_size;
    const int group_size = dim_ / n_groups_;

    // Workspace for backward pass
    T *dv, *d_delta_raw;
    T *d_log_h, *d_log_h_recurrent;
    T *d_w_out_h, *w_out_h, *h_linear, *d_h_linear;
    cudaMalloc(&dv, BD * sizeof(T));
    cudaMalloc(&d_delta_raw, BD * sizeof(T));
    cudaMalloc(&d_log_h, BD * sizeof(T));
    cudaMalloc(&d_log_h_recurrent, BD * sizeof(T));
    cudaMalloc(&d_w_out_h, BD * sizeof(T));
    cudaMalloc(&w_out_h, BD * sizeof(T));
    cudaMalloc(&h_linear, BD * sizeof(T));
    cudaMalloc(&d_h_linear, BD * sizeof(T));
    cudaMemset(d_log_h_recurrent, 0, BD * sizeof(T));

    // Float buffers for atomic gradients
    float *dr_h_float, *db_float, *db_delta_float;
    cudaMalloc(&dr_h_float, dim_ * sizeof(float));
    cudaMalloc(&db_float, dim_ * sizeof(float));
    cudaMalloc(&db_delta_float, dim_ * sizeof(float));
    cudaMemset(dr_h_float, 0, dim_ * sizeof(float));
    cudaMemset(db_float, 0, dim_ * sizeof(float));
    cudaMemset(db_delta_float, 0, dim_ * sizeof(float));

    // Zero weight gradients
    cudaMemset(dW_x, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dW_delta, 0, dim_ * dim_ * sizeof(T));
    cudaMemset(dW_out, 0, dim_ * dim_ * sizeof(T));

    for (int t = steps - 1; t >= 0; --t) {
        const T* x_t = x + t * BD;
        const T* log_h_prev = log_h + t * BD;
        const T* sign_h_prev = sign_h + t * BD;
        const T* log_h_t = log_h + (t + 1) * BD;
        const T* sign_h_t = sign_h + (t + 1) * BD;
        const T* v_t = v + t * BD;
        const T* delta_t = delta_cache + t * BD;
        const T* compete_t = compete_cache + t * BD;
        const T* weight1_t = weight1_cache + t * BD;
        const T* log_term1_t = log_term1_cache + t * BD;
        const T* log_term2_t = log_term2_cache + t * BD;
        const T* d_out_t = d_output + t * BD;
        T* dx_t = dx + t * BD;

        // Convert log_h to h_linear for backward
        LogToLinearKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, log_h_t, sign_h_t, h_linear);

        // Recompute w_out_h = W_out @ h_linear
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, h_linear, dim_, &beta_zero, w_out_h, dim_);

        // Backward through selective output - gradient goes to d_h_linear
        dim3 grid(batch_size_, n_groups_);
        int smem_size = block_size * sizeof(float);
        LogStorageSelectiveOutputLinearBackward<T><<<grid, block_size, smem_size, stream_>>>(
            batch_size_, dim_, n_groups_, group_size, h_linear, w_out_h, compete_t,
            d_out_t, d_h_linear, d_w_out_h);

        // dW_out += d_w_out_h @ h_linear^T
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_w_out_h, dim_, h_linear, dim_, &alpha, dW_out, dim_);

        // d_h_linear += W_out^T @ d_w_out_h
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_out, dim_, d_w_out_h, dim_, &alpha, d_h_linear, dim_);

        // Convert d_h_linear to d_log_h: d_log_h = d_h * h (chain rule through exp)
        // h = sign * exp(log_h), so d_log_h = d_h * sign * exp(log_h) = d_h * h
        ConvertLinearGradToLogGrad<T><<<num_blocks, block_size, 0, stream_>>>(
            BD, d_h_linear, h_linear, d_log_h);

        // TRUE LOG-SPACE BACKWARD through gated update
        // Uses softmax weights from forward pass - gradient is BOUNDED!
        LogStorageGatedBackward<T><<<num_blocks, block_size, 0, stream_>>>(
            batch_size_, dim_, log_h_prev, sign_h_prev, log_h_t, v_t, delta_t, r_h,
            weight1_t, log_term1_t, log_term2_t,
            d_log_h,  // gradient w.r.t. log|h| (LOG SPACE!)
            (t < steps - 1) ? d_log_h_recurrent : nullptr,
            dv, d_delta_raw, d_log_h_recurrent,  // output: gradient w.r.t. log|h_prev| (LOG SPACE!)
            dr_h_float, db_float, db_delta_float);

        // dx = W_x^T @ dv + W_delta^T @ d_delta_raw
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_x, dim_, dv, dim_, &beta_zero, dx_t, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
            dim_, batch_size_, dim_, &alpha, W_delta, dim_, d_delta_raw, dim_, &alpha, dx_t, dim_);

        // Note: d_log_h_recurrent is already set by LogStorageGatedBackward

        // Weight gradients
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, dv, dim_, x_t, dim_, &alpha, dW_x, dim_);
        blas<T>::gemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
            dim_, dim_, batch_size_, &alpha, d_delta_raw, dim_, x_t, dim_, &alpha, dW_delta, dim_);
    }

    // Copy float gradients to output type
    cudaMemset(dr_h, 0, dim_ * sizeof(T));
    cudaMemset(db, 0, dim_ * sizeof(T));
    cudaMemset(db_delta, 0, dim_ * sizeof(T));

    // Copy kernel for float -> T conversion
    auto copy_float_to_T = [&](float* src, T* dst, int n) {
        // Simple device copy with type conversion
        for (int i = 0; i < n; ++i) {
            float val;
            cudaMemcpy(&val, src + i, sizeof(float), cudaMemcpyDeviceToHost);
            T tval = static_cast<T>(val);
            cudaMemcpy(dst + i, &tval, sizeof(T), cudaMemcpyHostToDevice);
        }
    };
    copy_float_to_T(dr_h_float, dr_h, dim_);
    copy_float_to_T(db_float, db, dim_);
    copy_float_to_T(db_delta_float, db_delta, dim_);

    cudaFree(dv);
    cudaFree(d_delta_raw);
    cudaFree(d_log_h);
    cudaFree(d_log_h_recurrent);
    cudaFree(d_w_out_h);
    cudaFree(w_out_h);
    cudaFree(h_linear);
    cudaFree(d_h_linear);
    cudaFree(dr_h_float);
    cudaFree(db_float);
    cudaFree(db_delta_float);
}

// Explicit template instantiations
template struct LogStorageDiagonalElmanForward<__half>;
template struct LogStorageDiagonalElmanForward<__nv_bfloat16>;
template struct LogStorageDiagonalElmanForward<float>;
template struct LogStorageDiagonalElmanForward<double>;

template struct LogStorageDiagonalElmanBackward<__half>;
template struct LogStorageDiagonalElmanBackward<__nv_bfloat16>;
template struct LogStorageDiagonalElmanBackward<float>;
template struct LogStorageDiagonalElmanBackward<double>;

}  // namespace elman_ladder
}  // namespace v0
}  // namespace haste
