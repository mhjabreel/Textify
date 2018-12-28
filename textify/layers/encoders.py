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
import abc
import six

from textify.utils.encoder_utils import zip_and_reduce, concat_reduce_fn, build_cell

@six.add_metaclass(abc.ABCMeta)
class Encoder(tf.layers.Layer):

    def __init__(self, training=True):
        super(Encoder, self).__init__()
        self._training = training

    def build(self):
        super(Encoder, self).build(None)
        self.built = True
    
    @abc.abstractmethod
    def call(self, inputs, sequence_length=None):
        raise NotImplementedError() 


class MeanEncoder(Encoder):
        
    def call(self, inputs, sequence_length=None):

        state = tf.reduce_mean(inputs, axis=1)
        return inputs, state, sequence_length
    

class RNNEncoder(Encoder):

    def __init__(self,
            num_units,
            num_layers=1,
            cell_type=tf.nn.rnn_cell.LSTMCell,
            bidirectional=False,
            reduce_fn=None,
            residual=False,
            training=True,
            dropout=0.1):

        super(RNNEncoder, self).__init__(training)

        self._num_units = num_units
        self._num_layers = num_layers
        self._cell_type = cell_type
        self._bidirectional = bidirectional
        self._residual = residual
        self._dropout = dropout
        self._reduce_fn = reduce_fn
    
    def call(self, inputs, sequence_length=None):

        if not self._bidirectional:
            cell = build_cell(self._cell_type,
                self._num_units,
                self._num_layers,
                self._residual,
                self._dropout,
                self._training)
            
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell,
                inputs,
                sequence_length=sequence_length,
                dtype=inputs.dtype)  
        else:

            reduce_fn = self._reduce_fn or concat_reduce_fn

            cell_fw = build_cell(self._cell_type,
                self._num_units,
                self._num_layers,
                self._residual,
                self._dropout,
                self._training)

            cell_bw = build_cell(self._cell_type,
                self._num_units,
                self._num_layers,
                self._residual,
                self._dropout,
                self._training)

            encoder_outputs_tup, encoder_state_tup = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                inputs,
                sequence_length=sequence_length,
                dtype=inputs.dtype)
            
            encoder_outputs = zip_and_reduce(encoder_outputs_tup[0], encoder_outputs_tup[1], reduce_fn)
            encoder_state = zip_and_reduce(encoder_state_tup[0], encoder_state_tup[1], reduce_fn)            

        return encoder_outputs, encoder_state, sequence_length


class UnidirectionalRNNEncoder(RNNEncoder):
    def __init__(self,
            num_units,
            num_layers=1,
            cell_type=tf.nn.rnn_cell.LSTMCell,
            residual=False,
            training=True,
            dropout=0.1):
        
        super(UnidirectionalRNNEncoder, self).__init__(num_units,
                        num_layers,
                        cell_type,
                        bidirectional=False,
                        residual=residual,
                        training=training,
                        dropout=dropout)


class BidirectionalRNNEncoder(RNNEncoder):
    def __init__(self,
            num_units,
            num_layers=1,
            cell_type=tf.nn.rnn_cell.LSTMCell,
            reduce_fn=concat_reduce_fn,
            residual=False,
            training=True,
            dropout=0.1):
        
        super(BidirectionalRNNEncoder, self).__init__(num_units,
                        num_layers,
                        cell_type,
                        bidirectional=True,
                        reduce_fn=reduce_fn,
                        residual=residual,
                        training=training,
                        dropout=dropout)