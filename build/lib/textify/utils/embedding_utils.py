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

import os
import codecs
import numpy as np
from collections import namedtuple
import tensorflow as tf

class EmbeddingSpec(namedtuple("EmbeddingSpec",
                                ["name",
                                "embedding_size",
                                "vocab_size",
                                "pretrained_weights",
                                "trainable",
                                "dtype"])):
    
    def __new__(cls,
                name,
                embedding_size=None,
                vocab_size=None,
                pretrained_weights=None,
                trainable=True,
                dtype=tf.float32):

        
        return super(EmbeddingSpec, cls).__new__(cls,
                name,
                embedding_size,
                vocab_size,
                pretrained_weights,
                trainable,
                dtype)



def load_embedding(emb_path, vocab, dim=300, with_header=True, separator=' '):

    print("Load embedding {} ..".format(emb_path))

    vectors = np.zeros((len(vocab), dim), dtype=np.float32)
    found = 0
    with codecs.getreader("utf-8")(tf.gfile.GFile(emb_path, "rb")) as f:
        if with_header:
            next(f)
        for _, line in enumerate(f):
            word, vect = line.rstrip().split(separator, 1)
            
            if word in vocab:
                found += 1
                vect = np.fromstring(vect, sep=separator)
                idx = vocab[word]
                vectors[idx] = vect

    tf.logging.info("Found %d/%d (%.3f)", found, len(vocab), found / len(vocab))
    return vectors
