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

import codecs
import os
from collections import OrderedDict
import tensorflow as tf
import numpy as np

UNK = "<unk>"
UNK_ID = 0

def load_vocab(vocab_file):
    vocab = OrderedDict()
    vocab_size = 0
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        for word in f:
            vocab[word.strip()] = vocab_size
            vocab_size += 1
    return vocab, vocab_size