# Copyright 2020 LMNT, Inc. All Rights Reserved.
# LSTM + SiLU Selectivity Gate extension by Erik Gaasedelen, 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""LSTM with SiLU Selectivity Gate

Combines LSTM recurrence with input-dependent output gating:
  c, h_lstm = LSTM(x, c_prev, h_prev)
  gate = silu(Wg_x @ x + Wg_h @ h_lstm + bg)
  output = h_lstm * gate

This matches the selectivity mechanism used in Mamba2 and provides
significant improvement over vanilla LSTM.
"""

import haste_pytorch_lib as LIB
import torch
import torch.nn as nn


__all__ = [
    'LSTM_SiLU'
]


class LSTM_SiLU_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, training, x, h0, c0, W, R, b, Wg_x, Wg_h, bg):
        # Forward pass returns: h (hidden states), c (cell states), y (gated outputs), cache, gate_pre
        h, c, y, cache, gate_pre = LIB.lstm_silu_forward(
            training, x, h0, c0, W, R, b, Wg_x, Wg_h, bg
        )
        ctx.save_for_backward(x, W, R, b, Wg_x, Wg_h, bg, h, c, y, cache, gate_pre)
        ctx.training = training
        return h, c, y

    @staticmethod
    def backward(ctx, grad_h, grad_c, grad_y):
        if not ctx.training:
            raise RuntimeError('LSTM_SiLU backward can only be called in training mode')

        x, W, R, b, Wg_x, Wg_h, bg, h, c, y, cache, gate_pre = ctx.saved_tensors

        # Transpose inputs for backward pass
        x_t = x.permute(2, 0, 1).contiguous()  # [T, N, C] -> [C, T, N]
        W_t = W.permute(1, 0).contiguous()     # [C, H*4] -> [H*4, C]
        R_t = R.permute(1, 0).contiguous()     # [H, H*4] -> [H*4, H]
        Wg_x_t = Wg_x.permute(1, 0).contiguous()  # [C, H] -> [H, C]
        Wg_h_t = Wg_h.permute(1, 0).contiguous()  # [H, H] -> [H, H]

        # The output y is what the model uses, so grad_y is the main gradient
        dy = grad_y.contiguous()

        dx, dh0, dc0, dW, dR, db, dWg_x, dWg_h, dbg = LIB.lstm_silu_backward(
            x_t, W_t, R_t, b, Wg_x_t, Wg_h_t, bg,
            h, c, y, cache, gate_pre, dy
        )

        return None, dx, dh0, dc0, dW, dR, db, dWg_x, dWg_h, dbg


class LSTM_SiLU(nn.Module):
    """
    LSTM with SiLU Selectivity Gate layer.

    This layer combines a standard LSTM with an input-dependent output gate:
      1. c, h_lstm = LSTM(x, c_prev, h_prev)  -- Standard LSTM recurrence
      2. gate = silu(Wg_x @ x + Wg_h @ h_lstm + bg)  -- SiLU selectivity gate
      3. output = h_lstm * gate  -- Gated output

    This matches the selectivity mechanism used in Mamba2 and provides
    significant improvement over vanilla LSTM.

    Supports BF16 for efficient training on modern GPUs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = False,
        forget_bias: float = 1.0,
    ):
        """
        Initialize the parameters of the LSTM+SiLU layer.

        Arguments:
          input_size: int, the feature dimension of the input.
          hidden_size: int, the feature dimension of the hidden/output.
          batch_first: (optional) bool, if True, input/output tensors are
            (batch, seq, feature) instead of (seq, batch, feature).
          forget_bias: (optional) float, sets the initial bias of the forget gate.

        Variables:
          kernel: LSTM input projection weight [input_size, hidden_size * 4]
          recurrent_kernel: LSTM recurrent projection [hidden_size, hidden_size * 4]
          bias: LSTM bias [hidden_size * 4]
          gate_kernel_x: selectivity gate input weights [input_size, hidden_size]
          gate_kernel_h: selectivity gate hidden weights [hidden_size, hidden_size]
          gate_bias: selectivity gate bias [hidden_size]
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.forget_bias = forget_bias

        # LSTM weights (i, g, f, o gates)
        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 4))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.empty(hidden_size * 4))

        # SiLU selectivity gate weights
        self.gate_kernel_x = nn.Parameter(torch.empty(input_size, hidden_size))
        self.gate_kernel_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.gate_bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        hidden_size = self.hidden_size

        # Initialize LSTM weights
        for i in range(4):
            nn.init.xavier_uniform_(self.kernel[:, i*hidden_size:(i+1)*hidden_size])
            nn.init.orthogonal_(self.recurrent_kernel[:, i*hidden_size:(i+1)*hidden_size])
        nn.init.zeros_(self.bias)
        # Set forget gate bias
        nn.init.constant_(self.bias[hidden_size*2:hidden_size*3], self.forget_bias)

        # Initialize gate weights
        nn.init.xavier_uniform_(self.gate_kernel_x)
        nn.init.orthogonal_(self.gate_kernel_h)
        nn.init.zeros_(self.gate_bias)

    def forward(self, input, state=None):
        """
        Runs a forward pass of the LSTM+SiLU layer.

        Arguments:
          input: Tensor, a batch of input sequences.
            Shape (seq_len, batch_size, input_size) if batch_first=False,
            or (batch_size, seq_len, input_size) if batch_first=True.
          state: (optional) tuple of (h0, c0), initial hidden and cell states.
            h0 shape: (1, batch_size, hidden_size) or (batch_size, hidden_size).
            c0 shape: (1, batch_size, hidden_size) or (batch_size, hidden_size).

        Returns:
          output: Tensor, the gated output of the layer.
            Shape (seq_len, batch_size, hidden_size) if batch_first=False,
            or (batch_size, seq_len, hidden_size) if batch_first=True.
          (h_n, c_n): tuple of final hidden and cell states.
            Each has shape (1, batch_size, hidden_size).
        """
        # Handle batch_first
        if self.batch_first:
            input = input.permute(1, 0, 2)  # (B, T, C) -> (T, B, C)

        time_steps, batch_size, _ = input.shape

        # Handle initial state
        if state is None:
            h0 = torch.zeros(
                batch_size, self.hidden_size,
                dtype=input.dtype, device=input.device
            )
            c0 = torch.zeros(
                batch_size, self.hidden_size,
                dtype=input.dtype, device=input.device
            )
        else:
            h0, c0 = state
            if h0.dim() == 3:
                h0 = h0[0]  # (1, B, H) -> (B, H)
            if c0.dim() == 3:
                c0 = c0[0]  # (1, B, H) -> (B, H)

        # Run forward pass
        h, c, y = LSTM_SiLU_Function.apply(
            self.training,
            input.contiguous(),
            h0.contiguous(),
            c0.contiguous(),
            self.kernel.contiguous(),
            self.recurrent_kernel.contiguous(),
            self.bias.contiguous(),
            self.gate_kernel_x.contiguous(),
            self.gate_kernel_h.contiguous(),
            self.gate_bias.contiguous(),
        )

        # y is the gated output [T, B, H]
        # h is the hidden states [T+1, B, H], h[0] is h0, h[1:] are the states
        # c is the cell states [T+1, B, H], c[0] is c0, c[1:] are the states

        # Get final states
        h_n = h[-1].unsqueeze(0)  # [1, B, H]
        c_n = c[-1].unsqueeze(0)  # [1, B, H]

        # Handle batch_first for output
        if self.batch_first:
            y = y.permute(1, 0, 2)  # (T, B, H) -> (B, T, H)

        return y, (h_n, c_n)

    def extra_repr(self):
        return f'input_size={self.input_size}, hidden_size={self.hidden_size}, batch_first={self.batch_first}'
