# Copyright 2020 LMNT, Inc. All Rights Reserved.
# GRU + SiLU Selectivity Gate extension by Erik Gaasedelen, 2024.
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

"""GRU with SiLU Selectivity Gate

Combines GRU recurrence with input-dependent output gating:
  h_gru = GRU(x, h_prev)
  gate = silu(Wg_x @ x + Wg_h @ h_gru + bg)
  output = h_gru * gate

This matches the selectivity mechanism used in Mamba2 and provides
significant improvement over vanilla GRU.
"""

import haste_pytorch_lib as LIB
import torch
import torch.nn as nn


__all__ = [
    'GRU_SiLU'
]


class GRU_SiLU_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, training, x, h0, W, R, bx, br, Wg_x, Wg_h, bg):
        # Forward pass returns: h (hidden states), y (gated outputs), cache, gate_pre
        h, y, cache, gate_pre = LIB.gru_silu_forward(
            training, x, h0, W, R, bx, br, Wg_x, Wg_h, bg
        )
        ctx.save_for_backward(x, W, R, bx, br, Wg_x, Wg_h, bg, h, y, cache, gate_pre)
        ctx.training = training
        return h, y

    @staticmethod
    def backward(ctx, grad_h, grad_y):
        if not ctx.training:
            raise RuntimeError('GRU_SiLU backward can only be called in training mode')

        x, W, R, bx, br, Wg_x, Wg_h, bg, h, y, cache, gate_pre = ctx.saved_tensors

        # Transpose inputs for backward pass
        x_t = x.permute(2, 0, 1).contiguous()  # [T, N, C] -> [C, T, N]
        W_t = W.permute(1, 0).contiguous()     # [C, H*3] -> [H*3, C]
        R_t = R.permute(1, 0).contiguous()     # [H, H*3] -> [H*3, H]
        Wg_x_t = Wg_x.permute(1, 0).contiguous()  # [C, H] -> [H, C]
        Wg_h_t = Wg_h.permute(1, 0).contiguous()  # [H, H] -> [H, H]

        # Combine gradients: we receive grad_h for hidden states and grad_y for outputs
        # The output y is what the model uses, so grad_y is the main gradient
        # grad_h[:, 1:] could also contribute if someone wants hidden state gradients
        dy = grad_y.contiguous()

        dx, dh0, dW, dR, dbx, dbr, dWg_x, dWg_h, dbg = LIB.gru_silu_backward(
            x_t, W_t, R_t, bx, br, Wg_x_t, Wg_h_t, bg,
            h, y, cache, gate_pre, dy
        )

        return None, dx, dh0, dW, dR, dbx, dbr, dWg_x, dWg_h, dbg


class GRU_SiLU(nn.Module):
    """
    GRU with SiLU Selectivity Gate layer.

    This layer combines a standard GRU with an input-dependent output gate:
      1. h_gru = GRU(x, h_prev)     -- Standard GRU recurrence
      2. gate = silu(Wg_x @ x + Wg_h @ h_gru + bg)  -- SiLU selectivity gate
      3. output = h_gru * gate      -- Gated output

    This matches the selectivity mechanism used in Mamba2 and provides
    significant improvement over vanilla GRU.

    Supports BF16 for efficient training on modern GPUs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = False,
    ):
        """
        Initialize the parameters of the GRU+SiLU layer.

        Arguments:
          input_size: int, the feature dimension of the input.
          hidden_size: int, the feature dimension of the hidden/output.
          batch_first: (optional) bool, if True, input/output tensors are
            (batch, seq, feature) instead of (seq, batch, feature).

        Variables:
          kernel: GRU input projection weight [input_size, hidden_size * 3]
          recurrent_kernel: GRU recurrent projection [hidden_size, hidden_size * 3]
          bias: GRU input bias [hidden_size * 3]
          recurrent_bias: GRU recurrent bias [hidden_size * 3]
          gate_kernel_x: selectivity gate input weights [input_size, hidden_size]
          gate_kernel_h: selectivity gate hidden weights [hidden_size, hidden_size]
          gate_bias: selectivity gate bias [hidden_size]
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # GRU weights (z, r, h gates)
        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))

        # SiLU selectivity gate weights
        self.gate_kernel_x = nn.Parameter(torch.empty(input_size, hidden_size))
        self.gate_kernel_h = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.gate_bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        hidden_size = self.hidden_size

        # Initialize GRU weights
        for i in range(3):
            nn.init.xavier_uniform_(self.kernel[:, i*hidden_size:(i+1)*hidden_size])
            nn.init.orthogonal_(self.recurrent_kernel[:, i*hidden_size:(i+1)*hidden_size])
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.recurrent_bias)

        # Initialize gate weights
        nn.init.xavier_uniform_(self.gate_kernel_x)
        nn.init.orthogonal_(self.gate_kernel_h)
        nn.init.zeros_(self.gate_bias)

    def forward(self, input, state=None):
        """
        Runs a forward pass of the GRU+SiLU layer.

        Arguments:
          input: Tensor, a batch of input sequences.
            Shape (seq_len, batch_size, input_size) if batch_first=False,
            or (batch_size, seq_len, input_size) if batch_first=True.
          state: (optional) Tensor, initial hidden state.
            Shape (1, batch_size, hidden_size) or (batch_size, hidden_size).

        Returns:
          output: Tensor, the gated output of the layer.
            Shape (seq_len, batch_size, hidden_size) if batch_first=False,
            or (batch_size, seq_len, hidden_size) if batch_first=True.
          h_n: Tensor, the final hidden state.
            Shape (1, batch_size, hidden_size).
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
        elif state.dim() == 3:
            h0 = state[0]  # (1, B, H) -> (B, H)
        else:
            h0 = state  # (B, H)

        # Run forward pass
        h, y = GRU_SiLU_Function.apply(
            self.training,
            input.contiguous(),
            h0.contiguous(),
            self.kernel.contiguous(),
            self.recurrent_kernel.contiguous(),
            self.bias.contiguous(),
            self.recurrent_bias.contiguous(),
            self.gate_kernel_x.contiguous(),
            self.gate_kernel_h.contiguous(),
            self.gate_bias.contiguous(),
        )

        # y is the gated output [T, B, H]
        # h is the hidden states [T+1, B, H], h[0] is h0, h[1:] are the states

        # Get final hidden state
        h_n = h[-1].unsqueeze(0)  # [1, B, H]

        # Handle batch_first for output
        if self.batch_first:
            y = y.permute(1, 0, 2)  # (T, B, H) -> (B, T, H)

        return y, h_n

    def extra_repr(self):
        return f'input_size={self.input_size}, hidden_size={self.hidden_size}, batch_first={self.batch_first}'
