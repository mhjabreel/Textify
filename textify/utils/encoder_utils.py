# Copyright 2019 Mohammed Jabreel. All Rights Reserved.
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

import tensorflow as tf
from collections import Sequence

def concat_reduce_fn(inputs, axis=-1):

    return tf.concat(inputs, axis)

def add_reducer_fn(inputs):

    return tf.add_n(inputs)

def zip_and_reduce(x, y, reduce_fn=concat_reduce_fn):
    """Zips :obj:`x` with :obj:`y` and reduces all elements."""
    if tf.contrib.framework.nest.is_sequence(x):
        tf.contrib.framework.nest.assert_same_structure(x, y)

        x_flat = tf.contrib.framework.nest.flatten(x)
        y_flat = tf.contrib.framework.nest.flatten(y)

        flat = []
        for x_i, y_i in zip(x_flat, y_flat):
            flat.append(reduce_fn([x_i, y_i]))

        return tf.contrib.framework.nest.pack_sequence_as(x, flat)
    else:
        return reduce_fn([x, y])

def last_encoding_from_state(state):
    if isinstance(state, Sequence):
        state = state[-1]
    if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
        return state.h
    return state

def build_cell(cell_type, num_units, num_layers, residual, dropout, training):
    
    cells = []

    for l in range(num_layers):
        cell = cell_type(num_units)
        if training and dropout > 0:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1 - dropout)
        if residual and l > 0:
            cell = tf.nn.rnn_cell.ResidualWrapper(cell)
        
        cells.append(cell)
    
    return cells[0] if num_layers == 1 else tf.nn.rnn_cell.MultiRNNCell(cells)


class EncoderException(Exception):

    def __init__(self, message, error_code=None, errors=None):
        super(EncoderException, self).__init__(message)
        self._error_code = error_code

    @property
    def error_code(self):
        return self._error_code