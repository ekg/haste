# Copyright 2020 LMNT, Inc. All Rights Reserved.
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

"""
Haste: a fast, simple, and open RNN library.
"""

import torch as _

from ._version import __version__  # generated in setup.py
from .elman import Elman
from .elman_silu import ElmanSilu
from .elman_variants import ElmanTanh, ElmanSigmoid, ElmanSwish, ElmanGelu, ElmanNoGate
from .gru import GRU
from .skip_elman import SkipElman
from .indrnn import IndRNN
from .lstm import LSTM
from .layer_norm_gru import LayerNormGRU
from .layer_norm_indrnn import LayerNormIndRNN
from .layer_norm_lstm import LayerNormLSTM
from .multihead_elman import MultiHeadElman

__all__ = [
    'Elman',
    'ElmanSilu',
    'ElmanTanh',
    'ElmanSigmoid',
    'ElmanSwish',
    'ElmanGelu',
    'ElmanNoGate',
    'GRU',
    'SkipElman',
    'IndRNN',
    'LSTM',
    'LayerNormGRU',
    'LayerNormIndRNN',
    'LayerNormLSTM',
    'MultiHeadElman'
]
