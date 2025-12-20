# Copyright 2024 Erik Garrison. Apache 2.0 License.
"""Fused Elman RNN with SiLU gating for PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import haste_pytorch_lib as lib


class ElmanFunction(Function):
    @staticmethod
    def forward(ctx, training, x, h0, W1, W2, Wgx, Wgh, bias):
        # x: [T, B, D]
        # h0: [B, D]
        # W1: [H, 2D]
        # W2: [D, H]
        # Wgx: [D, D]
        # Wgh: [D, D]
        # bias: [D]
        h, v, hidden = lib.elman_forward(training, x.contiguous(), h0.contiguous(),
                                          W1.contiguous(), W2.contiguous(),
                                          Wgx.contiguous(), Wgh.contiguous(),
                                          bias.contiguous())
        if training:
            ctx.save_for_backward(x, h, hidden, v, W1, W2, Wgx, Wgh)

        # h is [(T+1)*B, D], return h[1:] as output
        return h[1:]  # [T, B, D]

    @staticmethod
    def backward(ctx, grad_output):
        x, h, hidden, v, W1, W2, Wgx, Wgh = ctx.saved_tensors

        T, B, D = grad_output.shape
        H = W1.shape[0]

        # Transpose inputs for backward pass
        x_t = x.permute(2, 0, 1).contiguous().view(D, T * B)
        W1_t = W1.t().contiguous()
        W2_t = W2.t().contiguous()
        Wgx_t = Wgx.t().contiguous()
        Wgh_t = Wgh.t().contiguous()

        # Reshape tensors for backward pass
        h_flat = h.view((T + 1) * B, D)
        hidden_flat = hidden.view(T * B, H)
        v_flat = v.view(T * B, D * 3)

        # Create dh_new with zeros for initial state gradient
        dh_new = torch.zeros(T + 1, B, D, dtype=grad_output.dtype, device=grad_output.device)
        dh_new[1:] = grad_output
        dh_new_flat = dh_new.view((T + 1) * B, D)

        dx, dh0, dW1, dW2, dWgx, dWgh, dbias = lib.elman_backward(
            x_t, W1_t, W2_t, Wgx_t, Wgh_t,
            h_flat, hidden_flat, v_flat, dh_new_flat)

        dx = dx.view(T, B, D)

        return None, dx, dh0, dW1, dW2, dWgx, dWgh, dbias


class Elman(nn.Module):
    """
    Fused Elman RNN with SiLU output gating.

    Architecture per timestep:
        hidden = tanh(W1 @ [x_t, h])
        h_new = W2 @ hidden
        gate = silu(Wgx @ x_t + Wgh @ h_new + bias)
        out = h_new * gate

    This is a simple MLP-based RNN with learned output gating, designed to
    test the hypothesis that selectivity (via SiLU gate) is sufficient for
    good language modeling without the complex gate structure of GRU/LSTM.
    """

    def __init__(self, input_size, hidden_size, expansion=2.0, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.expanded_size = int(hidden_size * expansion)
        self.batch_first = batch_first

        # First layer: expand [x, h] to hidden
        # W1 layout: [H, D+D] where first D cols are for input, second D for recurrent
        self.W1 = nn.Parameter(torch.empty(self.expanded_size, 2 * hidden_size))

        # Second layer: project back to hidden size
        self.W2 = nn.Parameter(torch.empty(hidden_size, self.expanded_size))

        # Gate weights
        self.Wgx = nn.Parameter(torch.empty(hidden_size, input_size))
        self.Wgh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization scaled for stability
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.Wgx)
        nn.init.xavier_uniform_(self.Wgh)
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

        output = ElmanFunction.apply(
            self.training, x, h0,
            self.W1, self.W2, self.Wgx, self.Wgh, self.bias
        )

        if self.batch_first:
            output = output.transpose(0, 1)

        return output
