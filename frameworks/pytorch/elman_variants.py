"""
Elman RNN variants with different activation combinations.

All gated variants follow the same architecture:
  raw = R @ h + Wx[t] + b          -- [B, 2D] single matmul
  [h_cand_raw, gate_raw] = split(raw)
  h_candidate = ACT1(h_cand_raw)   -- [B, D]
  gate = ACT2(gate_raw)            -- [B, D]
  h_new = h_candidate * gate       -- [B, D] elementwise

Variants:
  ElmanTanh:    ACT1=tanh, ACT2=tanh
  ElmanSigmoid: ACT1=tanh, ACT2=sigmoid
  ElmanSwish:   ACT1=silu, ACT2=silu
  ElmanGelu:    ACT1=tanh, ACT2=gelu
  ElmanNoGate:  ACT1=tanh, no gate (ablation baseline)
"""

import torch
import torch.nn as nn
from torch.autograd import Function

import haste_pytorch_lib as lib

__all__ = [
    'ElmanTanh',
    'ElmanSigmoid',
    'ElmanSwish',
    'ElmanGelu',
    'ElmanNoGate',
    'ElmanLeaky',
    'ElmanLeakySelective',
]


# ============================================================================
# ElmanTanh: tanh + tanh gate
# ============================================================================

class ElmanTanhFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias):
        h, v = lib.elman_tanh_forward(training, x, h0, Wx, R, bias)

        if training:
            ctx.save_for_backward(x, Wx, R, h, v)

        # Return hidden states [1:] to match input length
        return h[1:]

    @staticmethod
    def backward(ctx, grad_h):
        x, Wx, R, h, v = ctx.saved_tensors

        # Pad gradient with zeros for h[0]
        B = x.size(1)
        D = x.size(2)
        dh_new = torch.zeros(grad_h.size(0) + 1, B, D,
                           dtype=grad_h.dtype, device=grad_h.device)
        dh_new[1:] = grad_h

        dx, dh0, dWx, dR, dbias = lib.elman_tanh_backward(x, Wx, R, h, v, dh_new)

        return None, dx, dh0, dWx, dR, dbias


class ElmanTanh(nn.Module):
    """
    Elman RNN with tanh + tanh gate.

    Architecture:
        raw = R @ h + Wx @ x + b
        h_candidate = tanh(raw[:D])
        gate = tanh(raw[D:])
        h_new = h_candidate * gate

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden state.

    Input shape: [T, B, D] (time-first)
    Output shape: [T, B, D]
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices
        self.Wx = nn.Parameter(torch.empty(2 * hidden_size, input_size))
        self.R = nn.Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(2 * hidden_size))

        self._init_weights()

    def _init_weights(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        nn.init.uniform_(self.Wx, -std, std)
        nn.init.uniform_(self.R, -std, std)
        nn.init.zeros_(self.bias)

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        return ElmanTanhFunction.apply(
            self.training, x, h0, self.Wx, self.R, self.bias
        )


# ============================================================================
# ElmanSigmoid: tanh + sigmoid gate
# ============================================================================

class ElmanSigmoidFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias):
        h, v = lib.elman_sigmoid_forward(training, x, h0, Wx, R, bias)

        if training:
            ctx.save_for_backward(x, Wx, R, h, v)

        return h[1:]

    @staticmethod
    def backward(ctx, grad_h):
        x, Wx, R, h, v = ctx.saved_tensors

        B = x.size(1)
        D = x.size(2)
        dh_new = torch.zeros(grad_h.size(0) + 1, B, D,
                           dtype=grad_h.dtype, device=grad_h.device)
        dh_new[1:] = grad_h

        dx, dh0, dWx, dR, dbias = lib.elman_sigmoid_backward(x, Wx, R, h, v, dh_new)

        return None, dx, dh0, dWx, dR, dbias


class ElmanSigmoid(nn.Module):
    """
    Elman RNN with tanh + sigmoid gate.

    Architecture:
        raw = R @ h + Wx @ x + b
        h_candidate = tanh(raw[:D])
        gate = sigmoid(raw[D:])
        h_new = h_candidate * gate
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wx = nn.Parameter(torch.empty(2 * hidden_size, input_size))
        self.R = nn.Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(2 * hidden_size))

        self._init_weights()

    def _init_weights(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        nn.init.uniform_(self.Wx, -std, std)
        nn.init.uniform_(self.R, -std, std)
        nn.init.zeros_(self.bias)

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        return ElmanSigmoidFunction.apply(
            self.training, x, h0, self.Wx, self.R, self.bias
        )


# ============================================================================
# ElmanSwish: silu + silu gate
# ============================================================================

class ElmanSwishFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias):
        h, v = lib.elman_swish_forward(training, x, h0, Wx, R, bias)

        if training:
            ctx.save_for_backward(x, Wx, R, h, v)

        return h[1:]

    @staticmethod
    def backward(ctx, grad_h):
        x, Wx, R, h, v = ctx.saved_tensors

        B = x.size(1)
        D = x.size(2)
        dh_new = torch.zeros(grad_h.size(0) + 1, B, D,
                           dtype=grad_h.dtype, device=grad_h.device)
        dh_new[1:] = grad_h

        dx, dh0, dWx, dR, dbias = lib.elman_swish_backward(x, Wx, R, h, v, dh_new)

        return None, dx, dh0, dWx, dR, dbias


class ElmanSwish(nn.Module):
    """
    Elman RNN with SiLU/Swish + SiLU/Swish gate.

    Architecture:
        raw = R @ h + Wx @ x + b
        h_candidate = silu(raw[:D])
        gate = silu(raw[D:])
        h_new = h_candidate * gate
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wx = nn.Parameter(torch.empty(2 * hidden_size, input_size))
        self.R = nn.Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(2 * hidden_size))

        self._init_weights()

    def _init_weights(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        nn.init.uniform_(self.Wx, -std, std)
        nn.init.uniform_(self.R, -std, std)
        nn.init.zeros_(self.bias)

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        return ElmanSwishFunction.apply(
            self.training, x, h0, self.Wx, self.R, self.bias
        )


# ============================================================================
# ElmanGelu: tanh + gelu gate
# ============================================================================

class ElmanGeluFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias):
        h, v = lib.elman_gelu_forward(training, x, h0, Wx, R, bias)

        if training:
            ctx.save_for_backward(x, Wx, R, h, v)

        return h[1:]

    @staticmethod
    def backward(ctx, grad_h):
        x, Wx, R, h, v = ctx.saved_tensors

        B = x.size(1)
        D = x.size(2)
        dh_new = torch.zeros(grad_h.size(0) + 1, B, D,
                           dtype=grad_h.dtype, device=grad_h.device)
        dh_new[1:] = grad_h

        dx, dh0, dWx, dR, dbias = lib.elman_gelu_backward(x, Wx, R, h, v, dh_new)

        return None, dx, dh0, dWx, dR, dbias


class ElmanGelu(nn.Module):
    """
    Elman RNN with tanh + GELU gate.

    Architecture:
        raw = R @ h + Wx @ x + b
        h_candidate = tanh(raw[:D])
        gate = gelu(raw[D:])
        h_new = h_candidate * gate
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wx = nn.Parameter(torch.empty(2 * hidden_size, input_size))
        self.R = nn.Parameter(torch.empty(2 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(2 * hidden_size))

        self._init_weights()

    def _init_weights(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        nn.init.uniform_(self.Wx, -std, std)
        nn.init.uniform_(self.R, -std, std)
        nn.init.zeros_(self.bias)

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        return ElmanGeluFunction.apply(
            self.training, x, h0, self.Wx, self.R, self.bias
        )


# ============================================================================
# ElmanNoGate: tanh only, no gating (ablation baseline)
# ============================================================================

class ElmanNoGateFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias):
        h, v = lib.elman_nogate_forward(training, x, h0, Wx, R, bias)

        if training:
            ctx.save_for_backward(x, Wx, R, h, v)

        return h[1:]

    @staticmethod
    def backward(ctx, grad_h):
        x, Wx, R, h, v = ctx.saved_tensors

        B = x.size(1)
        D = x.size(2)
        dh_new = torch.zeros(grad_h.size(0) + 1, B, D,
                           dtype=grad_h.dtype, device=grad_h.device)
        dh_new[1:] = grad_h

        dx, dh0, dWx, dR, dbias = lib.elman_nogate_backward(x, Wx, R, h, v, dh_new)

        return None, dx, dh0, dWx, dR, dbias


class ElmanNoGate(nn.Module):
    """
    Simple Elman RNN without gating (ablation baseline).

    Architecture:
        raw = R @ h + Wx @ x + b
        h_new = tanh(raw)

    This is the classic Elman network without any gating mechanism,
    provided as a baseline for ablation studies.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # No gate, so weights are [D, D] instead of [2D, D]
        self.Wx = nn.Parameter(torch.empty(hidden_size, input_size))
        self.R = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))

        self._init_weights()

    def _init_weights(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        nn.init.uniform_(self.Wx, -std, std)
        nn.init.uniform_(self.R, -std, std)
        nn.init.zeros_(self.bias)

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        return ElmanNoGateFunction.apply(
            self.training, x, h0, self.Wx, self.R, self.bias
        )


# ============================================================================
# ElmanLeaky: tanh + input-dependent leaky integration (Mamba2-style Δ)
# ============================================================================

class ElmanLeakyFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias, delta):
        # delta: [T, B, D] - precomputed delta values (sigmoid output)
        h, v = lib.elman_leaky_forward(training, x, h0, Wx, R, bias, delta)

        if training:
            ctx.save_for_backward(x, Wx, R, h, v, delta)

        return h[1:]

    @staticmethod
    def backward(ctx, grad_h):
        x, Wx, R, h, v, delta = ctx.saved_tensors

        B = x.size(1)
        D = x.size(2)
        dh_new = torch.zeros(grad_h.size(0) + 1, B, D,
                           dtype=grad_h.dtype, device=grad_h.device)
        dh_new[1:] = grad_h

        dx, dh0, dWx, dR, dbias, d_delta = lib.elman_leaky_backward(
            x, Wx, R, h, v, delta, dh_new)

        return None, dx, dh0, dWx, dR, dbias, d_delta


class ElmanLeaky(nn.Module):
    """
    Elman RNN with input-dependent leaky integration (Mamba2-style discretization).

    Architecture:
        candidate = tanh(R @ h + Wx @ x + b)
        delta = sigmoid(W_delta @ x + b_delta)  -- input-dependent delta
        h_new = (1 - delta) * h + delta * candidate  -- leaky integration

    Key difference from standard gating: delta controls how much of the candidate
    state is blended into the previous state. This is the discretized continuous
    dynamics: dh/dt = -h + f(x), discretized as h[t] = (1-Δ)*h[t-1] + Δ*f(x[t],h[t-1]).

    The recurrence matrix R sees the properly blended state h[t-1], not the
    raw Elman output. This is critical for correct discretized dynamics.

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden state.

    Input shape: [T, B, D] (time-first)
    Output shape: [T, B, D]
    """

    def __init__(self, input_size: int, hidden_size: int, delta_init: float = -2.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_init = delta_init

        # Main Elman weights (no gate, so [D, D])
        self.Wx = nn.Parameter(torch.empty(hidden_size, input_size))
        self.R = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))

        # Delta projection weights (input-dependent delta)
        self.W_delta = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_delta = nn.Parameter(torch.empty(hidden_size))

        self._init_weights()

    def _init_weights(self):
        std = 1.0 / (self.hidden_size ** 0.5)
        nn.init.uniform_(self.Wx, -std, std)
        nn.init.uniform_(self.R, -std, std)
        nn.init.zeros_(self.bias)

        # Initialize delta projection to produce values near sigmoid(delta_init)
        nn.init.uniform_(self.W_delta, -std * 0.1, std * 0.1)
        nn.init.constant_(self.b_delta, self.delta_init)  # sigmoid(-2) ≈ 0.12

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        # Compute input-dependent delta (can be parallelized over time)
        x_flat = x.reshape(T * B, D)
        delta_raw = torch.mm(x_flat, self.W_delta.t()) + self.b_delta
        delta = torch.sigmoid(delta_raw).reshape(T, B, self.hidden_size)

        return ElmanLeakyFunction.apply(
            self.training, x, h0, self.Wx, self.R, self.bias, delta
        )


# ============================================================================
# ElmanLeakySelective: tanh + Mamba2-style discretization + per-channel decay
# ============================================================================

class ElmanLeakySelectiveFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias, W_delta, b_delta, A, W_gate_x, W_gate_h, b_gate):
        T, B, D = x.shape

        # Compute delta_raw in parallel (before softplus - kernel handles softplus)
        x_flat = x.reshape(T * B, D)
        delta_raw = (torch.mm(x_flat, W_delta.t()) + b_delta).reshape(T, B, D)

        # Run the CUDA kernel
        h, v = lib.elman_leaky_selective_forward(training, x, h0, Wx, R, bias, delta_raw, A)

        # h is [T+1, B, D], we want [T, B, D] for hidden states h[1:]
        h_out = h[1:]  # [T, B, D]

        # Compute h+x selective output gate OUTSIDE the kernel (parallel over T)
        # gate = silu(W_gate_x @ x + W_gate_h @ h + b_gate)
        # output = h * gate
        h_flat = h_out.reshape(T * B, D)
        gate_raw = torch.mm(x_flat, W_gate_x.t()) + torch.mm(h_flat, W_gate_h.t()) + b_gate
        gate = torch.nn.functional.silu(gate_raw)  # [T*B, D]
        output = h_flat * gate  # [T*B, D]
        output = output.reshape(T, B, D)

        if training:
            ctx.save_for_backward(x, Wx, R, h, v, delta_raw, A, W_delta,
                                  W_gate_x, W_gate_h, b_gate, gate_raw, h_out)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (x, Wx, R, h, v, delta_raw, A, W_delta,
         W_gate_x, W_gate_h, b_gate, gate_raw, h_out) = ctx.saved_tensors

        T, B, D = x.shape

        # Backward through output = h * gate where gate = silu(gate_raw)
        x_flat = x.reshape(T * B, D)
        h_flat = h_out.reshape(T * B, D)
        grad_out_flat = grad_output.reshape(T * B, D)

        # silu(x) = x * sigmoid(x)
        # d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = torch.sigmoid(gate_raw)
        gate = gate_raw * sig  # silu
        d_silu = sig * (1 + gate_raw * (1 - sig))

        # Backward through output = h * gate
        d_h_from_output = grad_out_flat * gate
        d_gate = grad_out_flat * h_flat

        # Backward through gate = silu(gate_raw)
        d_gate_raw = d_gate * d_silu

        # Backward through gate_raw = W_gate_x @ x + W_gate_h @ h + b_gate
        d_W_gate_x = torch.mm(d_gate_raw.t(), x_flat)
        d_W_gate_h = torch.mm(d_gate_raw.t(), h_flat)
        d_b_gate = d_gate_raw.sum(0)
        d_x_from_gate = torch.mm(d_gate_raw, W_gate_x)
        d_h_from_gate = torch.mm(d_gate_raw, W_gate_h)

        # Total gradient on h[1:] (from output gate and from next layers)
        d_h_out = (d_h_from_output + d_h_from_gate).reshape(T, B, D)

        # Pad to [T+1, B, D] for kernel backward
        dh_new = torch.zeros(T + 1, B, D, dtype=x.dtype, device=x.device)
        dh_new[1:] = d_h_out

        # Call kernel backward
        dx, dh0, dWx, dR, dbias, d_delta_raw, dA = lib.elman_leaky_selective_backward(
            x, Wx, R, h, v, A, dh_new)

        # Add gradient from gate computation to dx
        dx = dx + d_x_from_gate.reshape(T, B, D)

        # Gradient through delta_raw = W_delta @ x + b_delta
        # delta_raw = x @ W_delta.t() + b_delta
        # So: dW_delta = d_delta_raw.t() @ x, db_delta = d_delta_raw.sum(0)
        # And: dx += d_delta_raw @ W_delta
        d_delta_raw_flat = d_delta_raw.reshape(T * B, D)
        d_W_delta = torch.mm(d_delta_raw_flat.t(), x_flat)
        d_b_delta = d_delta_raw_flat.sum(0)
        dx = dx + torch.mm(d_delta_raw_flat, W_delta).reshape(T, B, D)

        return (None, dx, dh0, dWx, dR, dbias, d_W_delta, d_b_delta, dA,
                d_W_gate_x, d_W_gate_h, d_b_gate)


class ElmanLeakySelective(nn.Module):
    """
    Elman RNN with Mamba2-style discretization + per-channel decay + h+x output gate.

    Architecture:
        candidate = tanh(R @ h + Wx @ x + b)           -- NONLINEAR (our innovation!)
        delta_raw = W_delta @ x + b_delta              -- input-dependent
        dt = softplus(delta_raw)                       -- positive timestep

        # Mamba-style log parameterization for numerical stability:
        decay_rate = exp(-exp(A_log))                  -- ALWAYS in (0, 1)!
        alpha = exp(-dt * decay_rate)                  -- per-channel blend factor

        h_new = alpha * h + (1 - alpha) * candidate    -- exponential blend
        gate = silu(W_gate_x @ x + W_gate_h @ h + b_gate)  -- h+x selective output
        output = h * gate

    Key features:
    - Log-space A parameterization (decay_rate = exp(-exp(A_log)) is ALWAYS stable!)
    - Per-channel decay rates (like Mamba2's diagonal A matrix)
    - Input-dependent timestep via softplus (like Mamba2's Δ)
    - NONLINEAR candidate (tanh) - our innovation beyond Mamba2!
    - h+x output gate for selective output (computed outside kernel)

    This is the target architecture for achieving Mamba2 parity with nonlinearity.

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden state.
        delta_init: Initial value for delta bias (default -2.0 -> softplus(-2)~0.13)
        A_init_range: Range for A_log init (default (0.5, 2.0) for slow decay)

    Input shape: [T, B, D] (time-first)
    Output shape: [T, B, D]
    """

    def __init__(self, input_size: int, hidden_size: int,
                 delta_init: float = 3.0,  # softplus(3) ≈ 3.0 for meaningful timestep
                 A_init_range: tuple = (-0.5, 0.5)):  # LOG-SPACE: decay_rate = exp(-exp(A))
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_init = delta_init

        # Main Elman weights (candidate computation)
        self.Wx = nn.Parameter(torch.empty(hidden_size, input_size))
        self.R = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))

        # Delta projection weights (input-dependent timestep)
        self.W_delta = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_delta = nn.Parameter(torch.empty(hidden_size))

        # Per-channel decay rates (Mamba2's diagonal A)
        self.A = nn.Parameter(torch.empty(hidden_size))

        # h+x output gate weights
        self.W_gate_x = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_gate_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_gate = nn.Parameter(torch.empty(hidden_size))

        self._init_weights(A_init_range)

    def _init_weights(self, A_init_range):
        std = 1.0 / (self.hidden_size ** 0.5)

        # Candidate projection
        nn.init.uniform_(self.Wx, -std, std)
        nn.init.uniform_(self.R, -std, std)
        nn.init.zeros_(self.bias)

        # Delta projection (small weights, bias controls initial delta)
        nn.init.uniform_(self.W_delta, -std * 0.1, std * 0.1)
        nn.init.constant_(self.b_delta, self.delta_init)

        # Per-channel decay rate A (LOG-SPACE formula for stability)
        # decay_rate = exp(-exp(A_log)) -- ALWAYS in (0, 1) no matter what A is!
        # alpha = exp(-dt * decay_rate)
        #
        # A_log in (-0.5, 0.5):
        #   A=-0.5 → decay_rate = exp(-0.6) = 0.55, alpha = exp(-1.65) = 0.19 → 81% candidate
        #   A= 0.0 → decay_rate = exp(-1.0) = 0.37, alpha = exp(-1.11) = 0.33 → 67% candidate
        #   A= 0.5 → decay_rate = exp(-1.65) = 0.19, alpha = exp(-0.57) = 0.57 → 43% candidate
        nn.init.uniform_(self.A, A_init_range[0], A_init_range[1])

        # Output gate
        nn.init.uniform_(self.W_gate_x, -std, std)
        nn.init.uniform_(self.W_gate_h, -std, std)
        nn.init.zeros_(self.b_gate)

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        return ElmanLeakySelectiveFunction.apply(
            self.training, x, h0, self.Wx, self.R, self.bias,
            self.W_delta, self.b_delta, self.A,
            self.W_gate_x, self.W_gate_h, self.b_gate
        )


# ============================================================================
# ElmanMamba: CLEANER Mamba2-style - input-only output selectivity
# ============================================================================

class ElmanMambaFunction(Function):
    """
    Same CUDA kernel as ElmanLeakySelective, but with INPUT-ONLY output gate.
    This is closer to Mamba2's actual architecture where C depends only on x.
    """
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias, W_delta, b_delta, A, W_gate, b_gate, use_gate):
        T, B, D = x.shape

        # Compute delta_raw (input-dependent timestep)
        x_flat = x.reshape(T * B, D)
        delta_raw = (torch.mm(x_flat, W_delta.t()) + b_delta).reshape(T, B, D)

        # Run CUDA kernel - gets raw h_new (discretized state)
        h, v = lib.elman_leaky_selective_forward(training, x, h0, Wx, R, bias, delta_raw, A)

        # h is [T+1, B, D], we want h[1:]
        h_out = h[1:]  # [T, B, D]
        h_flat = h_out.reshape(T * B, D)

        # OUTPUT: INPUT-ONLY gate (like Mamba2's C) or no gate
        if use_gate:
            # gate = silu(W_gate @ x + b_gate) - depends ONLY on x!
            gate_raw = torch.mm(x_flat, W_gate.t()) + b_gate
            gate = torch.nn.functional.silu(gate_raw)
            output = h_flat * gate
        else:
            # No gate - just return raw h
            gate_raw = None
            gate = None
            output = h_flat

        output = output.reshape(T, B, D)

        if training:
            ctx.save_for_backward(x, Wx, R, h, v, delta_raw, A, W_delta, W_gate, b_gate, gate_raw, h_out)
            ctx.use_gate = use_gate

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, Wx, R, h, v, delta_raw, A, W_delta, W_gate, b_gate, gate_raw, h_out = ctx.saved_tensors
        use_gate = ctx.use_gate

        T, B, D = x.shape
        x_flat = x.reshape(T * B, D)
        h_flat = h_out.reshape(T * B, D)
        grad_out_flat = grad_output.reshape(T * B, D)

        if use_gate:
            # Backward through output = h * gate where gate = silu(W_gate @ x)
            sig = torch.sigmoid(gate_raw)
            gate = gate_raw * sig  # silu
            d_silu = sig * (1 + gate_raw * (1 - sig))

            d_h_from_output = grad_out_flat * gate
            d_gate = grad_out_flat * h_flat
            d_gate_raw = d_gate * d_silu

            # Backward through gate_raw = W_gate @ x + b_gate (NO h term!)
            d_W_gate = torch.mm(d_gate_raw.t(), x_flat)
            d_b_gate = d_gate_raw.sum(0)
            d_x_from_gate = torch.mm(d_gate_raw, W_gate)
        else:
            # No gate - gradient flows directly
            d_h_from_output = grad_out_flat
            d_W_gate = torch.zeros_like(W_gate)
            d_b_gate = torch.zeros_like(b_gate)
            d_x_from_gate = torch.zeros_like(x_flat)

        # Gradient on h
        d_h_out = d_h_from_output.reshape(T, B, D)

        # Pad for kernel backward
        dh_new = torch.zeros(T + 1, B, D, dtype=x.dtype, device=x.device)
        dh_new[1:] = d_h_out

        # Call kernel backward
        dx, dh0, dWx, dR, dbias, d_delta_raw, dA = lib.elman_leaky_selective_backward(
            x, Wx, R, h, v, A, dh_new)

        # Add gradient from gate computation to dx
        dx = dx + d_x_from_gate.reshape(T, B, D)

        # Gradient through delta_raw
        d_delta_raw_flat = d_delta_raw.reshape(T * B, D)
        d_W_delta = torch.mm(d_delta_raw_flat.t(), x_flat)
        d_b_delta = d_delta_raw_flat.sum(0)
        dx = dx + torch.mm(d_delta_raw_flat, W_delta).reshape(T, B, D)

        return (None, dx, dh0, dWx, dR, dbias, d_W_delta, d_b_delta, dA,
                d_W_gate, d_b_gate, None)


class ElmanMamba(nn.Module):
    """
    CLEANER Mamba2-style RNN: nonlinear candidate + input-only output selectivity.

    This is a CLOSER CRIB of Mamba2 than ElmanLeakySelective:
    - Input-dependent Δ (timestep) via softplus
    - Per-channel decay in log-space: decay_rate = exp(-exp(A_log))
    - INPUT-ONLY output gate (NOT h+x!) - like Mamba2's C matrix
    - Nonlinear candidate with tanh (our innovation beyond Mamba2)

    Architecture:
        candidate = tanh(R @ h + Wx @ x + b)     -- NONLINEAR
        dt = softplus(W_delta @ x + b_delta)    -- input-dependent
        decay_rate = exp(-exp(A_log))           -- log-space, ALWAYS in (0,1)
        alpha = exp(-dt * decay_rate)
        h_new = alpha * h + (1 - alpha) * candidate

        # Output (if use_gate=True):
        gate = silu(W_gate @ x + b_gate)        -- INPUT-ONLY! No h!
        output = h_new * gate

        # Or (if use_gate=False):
        output = h_new                          -- simplest

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden state.
        delta_init: Initial delta bias (3.0 for meaningful timestep)
        A_init_range: Range for A_log initialization
        use_gate: Whether to use input-only output gate (default True)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 delta_init: float = 3.0,
                 A_init_range: tuple = (-0.5, 0.5),
                 use_gate: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_init = delta_init
        self.use_gate = use_gate

        # Candidate computation weights
        self.Wx = nn.Parameter(torch.empty(hidden_size, input_size))
        self.R = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))

        # Delta projection (input-dependent timestep)
        self.W_delta = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_delta = nn.Parameter(torch.empty(hidden_size))

        # Per-channel decay rate (log-space A)
        self.A = nn.Parameter(torch.empty(hidden_size))

        # Input-only output gate (like Mamba2's C)
        self.W_gate = nn.Parameter(torch.empty(hidden_size, input_size))
        self.b_gate = nn.Parameter(torch.empty(hidden_size))

        self._init_weights(A_init_range)

    def _init_weights(self, A_init_range):
        std = 1.0 / (self.hidden_size ** 0.5)

        # Candidate projection
        nn.init.uniform_(self.Wx, -std, std)
        nn.init.uniform_(self.R, -std, std)
        nn.init.zeros_(self.bias)

        # Delta projection
        nn.init.uniform_(self.W_delta, -std * 0.1, std * 0.1)
        nn.init.constant_(self.b_delta, self.delta_init)

        # Per-channel decay rate A (log-space)
        # decay_rate = exp(-exp(A_log))
        # A in (-0.5, 0.5) gives decay_rate in (0.19, 0.55)
        # With dt=3: alpha in (0.19, 0.57) -> 43-81% candidate influence
        nn.init.uniform_(self.A, A_init_range[0], A_init_range[1])

        # Output gate (input-only)
        nn.init.uniform_(self.W_gate, -std, std)
        nn.init.zeros_(self.b_gate)

    def forward(self, x, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        return ElmanMambaFunction.apply(
            self.training, x, h0, self.Wx, self.R, self.bias,
            self.W_delta, self.b_delta, self.A,
            self.W_gate, self.b_gate, self.use_gate
        )
