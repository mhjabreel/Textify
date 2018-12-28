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
from collections import OrderedDict


class Embedding(tf.layers.Layer):

    def __init__(self, embedding_spec):

        super(Embedding, self).__init__()
        self._name = embedding_spec.name
        self._vocab_size = embedding_spec.vocab_size
        self._embedding_size = embedding_spec.embedding_size
        
        self._pretrained_weights = embedding_spec.pretrained_weights
        self._trainable = embedding_spec.trainable or True
        self._dtype = embedding_spec.dtype or tf.float32

    def build(self, _):
        shape = None
        if self._pretrained_weights is None:
            initializer = tf.random_normal_initializer(0., self._embedding_size ** -0.5)
            shape = [self._vocab_size, self._embedding_size]         
        else:
            _weights = self._pretrained_weights.astype(self.dtype.as_numpy_dtype())
            initializer = tf.constant_initializer(_weights, dtype=self._dtype) 
            shape = _weights.shape
        
        self._embedding_weights = tf.get_variable("weights_%s" % self._name,
                shape=shape,
                initializer=initializer,
                dtype=self.dtype,
                trainable=self._trainable)            

        self.built = True
    
    def call(self, x):

        return tf.nn.embedding_lookup(self._embedding_weights, x)


class MultipleEmbedding(tf.layers.Layer):

    def __init__(self, embedding_specs):
        super(MultipleEmbedding, self).__init__()

        self._embeddings = OrderedDict()
        for emb_spec in embedding_specs:
            self._embeddings[emb_spec.name] = Embedding(emb_spec)
    
    def build(self, _):

        for emb_name in self._embeddings:
            self._embeddings[emb_name].build(_)           
        self.built = True
    
    def call(self, inputs):
        embeddings = {}
        for k in inputs:
            if k in self._embeddings:
                x = inputs[k]
                embeddings[k] = self._embeddings[k].call(x)
        return embeddings    