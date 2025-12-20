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
