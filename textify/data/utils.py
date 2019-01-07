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
import tensorflow as tf

def count_lines(file_name):
    lines = 0
    with codecs.getreader("utf-8")(tf.gfile.GFile(file_name, "rb")) as f:
        for _ in f:
            lines += 1
    return lines

def make_tokenized_data(source, tokenizer, vocab, max_len, num_parallel_calls, prefix_key=None):
    dataset = tf.data.TextLineDataset(source)
    dataset = dataset.map(lambda text: tokenizer(text),
        num_parallel_calls=num_parallel_calls)
    if max_len:
        dataset = dataset.map(lambda tokens: tokens[:max_len],
            num_parallel_calls=num_parallel_calls)        
    dataset = dataset.map(lambda tokens: tf.cast(vocab.lookup(tokens), tf.int32),
        num_parallel_calls=num_parallel_calls) 

    ids_key = 'ids' if prefix_key is None else '{}_ids'.format(prefix_key)
    length_key = 'length' if prefix_key is None else '{}_length'.format(prefix_key)
    dataset = dataset.map(lambda token_ids: {ids_key: token_ids, length_key: tf.size(token_ids)},
        num_parallel_calls=num_parallel_calls) 
    return dataset