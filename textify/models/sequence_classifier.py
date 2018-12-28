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

from textify.models.model_base import _Classifier, _WordEmbeddingBasedModel
from textify.utils.encoder_utils import last_encoding_from_state

import tensorflow as tf

class SequenceClassifier(_Classifier, _WordEmbeddingBasedModel):

    def __init__(self, params, encoder, encoding_mode='average', reverse_target_vocab=None, scope=None):
        super(SequenceClassifier, self).__init__(params, reverse_target_vocab, scope)
        self._encoder = encoder
        self._encoding_mode = encoding_mode.lower()

    def _encode(self, embeddings, lengths):

        encoder_outputs, encoder_state, _ = self._encoder.call(embeddings, sequence_length=lengths)       
        if self._encoding_mode == "average":
            encoded = tf.reduce_mean(encoder_outputs, axis=1)
        elif self._encoding_mode == "last":
            encoded = last_encoding_from_state(encoder_state)        
        return encoded