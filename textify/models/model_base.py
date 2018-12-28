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

import six
import abc

from textify.layers.embeddings import Embedding


@six.add_metaclass(abc.ABCMeta)
class Model:

    def __init__(self, params, scope=None):

        self._params = params
        self._scope = scope       

    @abc.abstractmethod
    def _get_embeddings(self, features):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def _encode(self, embeddings, lengths):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_logits(self, inputs):
        raise NotImplementedError()
    
    def __call__(self, features):
        with tf.variable_scope(self._scope or "textify"):
            embeddings = self._get_embeddings(features['ids'])
            encoded = self._encode(embeddings, features['length'])
            logits = self._get_logits(encoded)
            return logits

    @abc.abstractmethod
    def get_predictions(self, logits):
        raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class _Classifier(Model):

    def __init__(self, params, reverse_target_vocab=None, scope=None):
        super(_Classifier, self).__init__(params, scope)
        self._reverse_target_vocab = reverse_target_vocab

    def _get_logits(self, inputs):

        with tf.variable_scope("Output"):
            num_classes = self._params.get("num_classes", 2)
            hidden_layers = self._params.get("hidden_layers", None)
            activation = self._params.get("activations", tf.nn.relu)

            x = inputs
            if not hidden_layers is None:
                for _, h in enumerate(hidden_layers):
                    x = tf.layers.dense(x, h, activation=activation) 
                    
            return tf.layers.dense(inputs, units=(1 if num_classes == 2 else num_classes)) 

    def get_predictions(self, logits):
        
        num_classes = self._params.get("num_classes", 2)
        if num_classes == 2:
            proba = tf.sigmoid(logits)
            predictions = tf.round(proba)
        else:
            proba = tf.nn.softmax(logits)
            predictions = tf.arg_max(proba, dimension=1)
        
        if not self._reverse_target_vocab is None:
            labels = self._reverse_target_vocab.lookup(tf.to_int64(predictions))
            return {'Predictions': predictions, 'Probabilities': proba, 'Labels': labels}
        
        return {'Predictions': predictions, 'Probabilities': proba}

@six.add_metaclass(abc.ABCMeta)
class _WordEmbeddingBasedModel(Model):

    def _get_embeddings(self, features):

        with tf.device("/cpu:0"):
            embedding = Embedding(self._params['embedding_specs'])
            embedding.build(None)
            return embedding.call(features)

