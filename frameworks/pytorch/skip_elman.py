# Copyright 2020 LMNT, Inc. All Rights Reserved.
# Modified 2024 for SkipElman
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

"""SkipElman: Elman RNN with skip connection and separate input/hidden biases.

Formula:
  z = sigmoid(Wz @ x + Rz @ h + bx_z + bh_z)   # update gate
  a = tanh(Wa @ x + Ra @ h + bx_a + bh_a)      # candidate
  h_new = z * h + (1-z) * a                     # skip connection!

Simpler than GRU (2 components, no reset gate) but retains gradient highway.
"""


import haste_pytorch_lib as LIB
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_rnn import BaseRNN


__all__ = [
    'SkipElman'
]


#@torch.jit.script
def SkipElmanScript(
    training: bool,
    zoneout_prob: float,
    input,
    h0,
    kernel,
    recurrent_kernel,
    input_bias,
    hidden_bias,
    zoneout_mask):
  """Pure PyTorch fallback for CPU."""
  time_steps = input.shape[0]
  batch_size = input.shape[1]
  hidden_size = recurrent_kernel.shape[0]

  h = [h0]
  Wx = input @ kernel  # [T, B, H*2]
  for t in range(time_steps):
    Rh = h[t] @ recurrent_kernel  # [B, H*2]
    # Split into z (update gate) and a (candidate)
    vx = torch.chunk(Wx[t], 2, 1)
    vh = torch.chunk(Rh, 2, 1)
    vbx = torch.chunk(input_bias, 2, 0)
    vbh = torch.chunk(hidden_bias, 2, 0)

    z = torch.sigmoid(vx[0] + vh[0] + vbx[0] + vbh[0])
    a = torch.tanh(vx[1] + vh[1] + vbx[1] + vbh[1])

    # Skip connection: h_new = z * h + (1-z) * a
    h.append(z * h[t] + (1 - z) * a)
    if zoneout_prob:
      if training:
        h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
      else:
        h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]

  h = torch.stack(h)
  return h


class SkipElmanFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, training, zoneout_prob, *inputs):
    h, cache = LIB.skip_elman_forward(training, zoneout_prob, *inputs)
    ctx.save_for_backward(inputs[0], *inputs[2:], h, cache)
    ctx.mark_non_differentiable(inputs[-1])
    ctx.training = training
    return h

  @staticmethod
  def backward(ctx, grad_h):
    if not ctx.training:
      raise RuntimeError('SkipElman backward can only be called in training mode')

    saved = [*ctx.saved_tensors]
    saved[0] = saved[0].permute(2, 0, 1).contiguous()
    saved[1] = saved[1].permute(1, 0).contiguous()
    saved[2] = saved[2].permute(1, 0).contiguous()
    grads = LIB.skip_elman_backward(*saved, grad_h.contiguous())
    return (None, None, *grads, None)


class SkipElman(BaseRNN):
  """
  SkipElman layer: Elman RNN with skip connection.

  This layer implements a simplified GRU without the reset gate:
    z = sigmoid(Wz @ x + Rz @ h + bx_z + bh_z)   # update gate
    a = tanh(Wa @ x + Ra @ h + bx_a + bh_a)      # candidate
    h_new = z * h + (1-z) * a                     # skip connection

  The skip connection (z * h term) provides a gradient highway similar to GRU,
  enabling better long-range learning compared to standard Elman.

  Key differences from GRU:
    - No reset gate (simpler, fewer parameters)
    - Separate input and hidden biases (like cuDNN)

  This layer has built-in support for DropConnect and Zoneout regularization.
  """

  def __init__(self,
      input_size,
      hidden_size,
      batch_first=False,
      dropout=0.0,
      zoneout=0.0,
      return_state_sequence=False):
    """
    Initialize the parameters of the SkipElman layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.
      return_state_sequence: (optional) bool, if `True`, the forward pass will
        return the entire state sequence instead of just the final state.

    Variables:
      kernel: input projection weight matrix. Dimensions (input_size, hidden_size * 2)
        with `z,a` gate layout. Initialized with Xavier uniform.
      recurrent_kernel: recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 2) with `z,a` gate layout. Initialized orthogonal.
      input_bias: input projection bias vector. Dimensions (hidden_size * 2).
      hidden_bias: recurrent projection bias vector. Dimensions (hidden_size * 2).
    """
    super().__init__(input_size, hidden_size, batch_first, zoneout, return_state_sequence)

    if dropout < 0 or dropout > 1:
      raise ValueError('SkipElman: dropout must be in [0.0, 1.0]')
    if zoneout < 0 or zoneout > 1:
      raise ValueError('SkipElman: zoneout must be in [0.0, 1.0]')

    self.dropout = dropout

    # SkipElman has 2 components (z, a) instead of GRU's 3 (z, r, h)
    self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 2))
    self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 2))
    self.input_bias = nn.Parameter(torch.empty(hidden_size * 2))
    self.hidden_bias = nn.Parameter(torch.empty(hidden_size * 2))
    self.reset_parameters()

  def reset_parameters(self):
    """Resets this layer's parameters to their initial values."""
    hidden_size = self.hidden_size
    # Initialize each gate separately
    for i in range(2):
      nn.init.xavier_uniform_(self.kernel[:, i*hidden_size:(i+1)*hidden_size])
      nn.init.orthogonal_(self.recurrent_kernel[:, i*hidden_size:(i+1)*hidden_size])
    nn.init.zeros_(self.input_bias)
    nn.init.zeros_(self.hidden_bias)

  def forward(self, input, state=None, lengths=None):
    """
    Runs a forward pass of the SkipElman layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the SkipElman.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      state: (optional) Tensor, the initial hidden state. Dimensions
        (1, batch_size, hidden_size).
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size).

    Returns:
      output: Tensor, the output of the SkipElman layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`.
      h_n: the hidden state for the last sequence item. Dimensions
        (1, batch_size, hidden_size).
    """
    input = self._permute(input)
    state_shape = [1, input.shape[1], self.hidden_size]
    h0 = self._get_state(input, state, state_shape)
    h = self._impl(input, h0[0], self._get_zoneout_mask(input))
    state = self._get_final_state(h, lengths)
    output = self._permute(h[1:])
    return output, state

  def _impl(self, input, state, zoneout_mask):
    if self._is_cuda():
      return SkipElmanFunction.apply(
          self.training,
          self.zoneout,
          input.contiguous(),
          state.contiguous(),
          self.kernel.contiguous(),
          F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
          self.input_bias.contiguous(),
          self.hidden_bias.contiguous(),
          zoneout_mask.contiguous())
    else:
      return SkipElmanScript(
          self.training,
          self.zoneout,
          input.contiguous(),
          state.contiguous(),
          self.kernel.contiguous(),
          F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
          self.input_bias.contiguous(),
          self.hidden_bias.contiguous(),
          zoneout_mask.contiguous())
