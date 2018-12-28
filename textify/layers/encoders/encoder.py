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

from textify.utils.encoder_utils import zip_and_reduce, concat_reduce_fn, build_cell, EncoderException

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