# Copyright 2024 Erik Garrison. Apache 2.0 License.
"""Multi-head Elman RNN with per-head recurrence matrices.

Architecture per timestep per head:
    h_new[i] = activation(R[i] @ h[i] + Wx[i] @ x[i] + b[i])

Key insight: Each head has its own headdim×headdim R matrix,
giving 2048x more expressive recurrence than Mamba2's scalar decays.
"""

import torch
import torch.nn as nn
from torch.autograd import Function

import haste_pytorch_lib as lib


class MultiHeadElmanFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, R, Wx, bias, activation):
        # x: [T, B, nheads, headdim]
        # h0: [B, nheads, headdim]
        # R: [nheads, headdim, headdim] - recurrent weights per head
        # Wx: [nheads, headdim, headdim] - input weights per head
        # bias: [nheads, headdim]
        # activation: 0=softsign, 1=tanh_residual, 2=tanh
        h, y, pre_act = lib.multihead_elman_forward(
            training,
            x.contiguous(),
            h0.contiguous(),
            R.contiguous(),
            Wx.contiguous(),
            bias.contiguous(),
            activation
        )
        if training:
            ctx.save_for_backward(x, R, Wx, h, pre_act)
            ctx.activation = activation

        # y is [T, B, nheads, headdim]
        return y, h[-1]  # Return output and final hidden state

    @staticmethod
    def backward(ctx, grad_output, grad_h_final):
        x, R, Wx, h, pre_act = ctx.saved_tensors
        activation = ctx.activation

        T, B, nheads, headdim = grad_output.shape

        # Combine gradient from output and final hidden state
        dy = grad_output.contiguous()
        # Add grad_h_final to dy[-1]
        if grad_h_final is not None:
            dy = dy.clone()
            dy[-1] = dy[-1] + grad_h_final

        dx, dh0, dR, dWx, db = lib.multihead_elman_backward(
            x, R, Wx, h, pre_act, dy, activation
        )

        return None, dx, dh0, dR, dWx, db, None


class MultiHeadElman(nn.Module):
    """
    Multi-head Elman RNN with per-head recurrence matrices.

    Unlike Mamba2 which uses scalar decays per head (total: 64 parameters),
    this uses full headdim×headdim R matrices per head, giving 4096x more
    expressive recurrence (262,144 recurrence parameters per layer).

    Architecture per timestep per head:
        h_new[i] = activation(R[i] @ h[i] + Wx[i] @ x[i] + b[i])

    Activation options:
        0 = softsign: x / (1 + |x|) - gradient-friendly, non-saturating
        1 = tanh_residual: x + tanh(x) - preserves gradients via skip
        2 = tanh: standard tanh
    """

    def __init__(self, input_size, hidden_size, nheads=64, activation=0, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nheads = nheads
        self.headdim = hidden_size // nheads
        self.activation = activation
        self.batch_first = batch_first

        assert hidden_size % nheads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by nheads ({nheads})"

        # Per-head weights
        # R: [nheads, headdim, headdim] - recurrent weights
        self.R = nn.Parameter(torch.empty(nheads, self.headdim, self.headdim))

        # Wx: [nheads, headdim, headdim] - input weights
        self.Wx = nn.Parameter(torch.empty(nheads, self.headdim, self.headdim))

        # bias: [nheads, headdim]
        self.bias = nn.Parameter(torch.zeros(nheads, self.headdim))

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.R)
        nn.init.xavier_uniform_(self.Wx)
        nn.init.zeros_(self.bias)

    def forward(self, x, h0=None):
        """
        Args:
            x: input tensor [T, B, D] or [B, T, D] if batch_first
            h0: initial hidden state [B, nheads, headdim], defaults to zeros

        Returns:
            output: [T, B, D] or [B, T, D] if batch_first
            h_final: [B, nheads, headdim] - final hidden state
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        T, B, D = x.shape

        # Reshape input to [T, B, nheads, headdim]
        x_heads = x.view(T, B, self.nheads, self.headdim)

        if h0 is None:
            h0 = torch.zeros(B, self.nheads, self.headdim,
                           dtype=x.dtype, device=x.device)

        output, h_final = MultiHeadElmanFunction.apply(
            self.training, x_heads, h0,
            self.R, self.Wx, self.bias, self.activation
        )

        # Reshape output back to [T, B, D]
        output = output.view(T, B, D)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_final

    def extra_repr(self):
        return (f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
                f'nheads={self.nheads}, headdim={self.headdim}, '
                f'activation={self.activation}')
