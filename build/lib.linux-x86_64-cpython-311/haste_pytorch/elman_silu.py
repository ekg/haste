# Copyright 2024 Erik Garrison. Apache 2.0 License.
"""SiLU-gated Elman RNN for PyTorch - cuDNN-level performance."""

import torch
import torch.nn as nn
from torch.autograd import Function

import haste_pytorch_lib as lib


class ElmanSiluFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, Wx, R, bias):
        # x: [T, B, D]
        # h0: [B, D]
        # Wx: [2D, D] - input weights for both state candidate and gate
        # R: [2D, D] - recurrent weights for both state candidate and gate
        # bias: [2D]
        h, v = lib.elman_silu_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            Wx.contiguous(),
            R.contiguous(),
            bias.contiguous()
        )
        if training:
            ctx.save_for_backward(x, h, v, Wx, R)

        # h is [T+1, B, D], return h[1:] as output
        return h[1:]  # [T, B, D]

    @staticmethod
    def backward(ctx, grad_output):
        x, h, v, Wx, R = ctx.saved_tensors

        T, B, D = grad_output.shape

        # Create dh_new with zeros for initial state gradient
        dh_new = torch.zeros(T + 1, B, D, dtype=grad_output.dtype, device=grad_output.device)
        dh_new[1:] = grad_output

        dx, dh0, dWx, dR, dbias = lib.elman_silu_backward(
            x, Wx, R, h, v, dh_new
        )

        return None, dx, dh0, dWx, dR, dbias


class ElmanSilu(nn.Module):
    """
    SiLU-gated Elman RNN designed for cuDNN-level performance.

    Architecture per timestep (1 matmul for recurrent connection):
        raw = R @ h + Wx @ x + b          -- [B, 2D] single recurrent matmul
        [h_candidate, gate_logit] = split(raw)
        h_candidate = tanh(h_candidate)   -- [B, D]
        gate = silu(gate_logit)           -- [B, D]
        h_new = h_candidate * gate        -- [B, D] elementwise

    Key insight: By combining state candidate and gate into a single matmul,
    we get exactly 1 recurrent matmul per timestep, matching cuDNN's GRU structure.
    The SiLU gate provides selectivity similar to GRU's update gate.

    This should achieve cuDNN-level performance with proper kernel fusion.
    """

    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Gate dimension is 2 * hidden_size (state candidate + gate)
        gate_dim = 2 * hidden_size

        # Input weights: projects input to [h_candidate; gate_logit]
        self.Wx = nn.Parameter(torch.empty(gate_dim, input_size))

        # Recurrent weights: projects h to [h_candidate; gate_logit]
        self.R = nn.Parameter(torch.empty(gate_dim, hidden_size))

        # Bias for both state candidate and gate
        self.bias = nn.Parameter(torch.zeros(gate_dim))

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization
        nn.init.xavier_uniform_(self.Wx)
        nn.init.xavier_uniform_(self.R)
        nn.init.zeros_(self.bias)

    def forward(self, x, h0=None):
        """
        Args:
            x: input tensor of shape [T, B, D] (or [B, T, D] if batch_first)
            h0: initial hidden state [B, D], defaults to zeros

        Returns:
            output: [T, B, D] (or [B, T, D] if batch_first)
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)

        output = ElmanSiluFunction.apply(
            self.training, x, h0,
            self.Wx, self.R, self.bias
        )

        if self.batch_first:
            output = output.transpose(0, 1)

        return output

    def extra_repr(self):
        return f'input_size={self.input_size}, hidden_size={self.hidden_size}'
