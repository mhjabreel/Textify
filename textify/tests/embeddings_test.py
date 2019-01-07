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
import io

import tensorflow as tf
import numpy as np

from textify.layers.embeddings import Embedding
from textify.layers.embeddings import MultipleEmbedding
from textify.utils.embedding_utils import EmbeddingSpec,load_embedding

class EmbeddingTest(tf.test.TestCase):

    def testDefaultEmbedding(self):
        embedding_size = 30
        vocab_size = 100
        embedding_spec = EmbeddingSpec("TestEmb1", embedding_size=embedding_size, vocab_size=vocab_size)
        embedding = Embedding(embedding_spec)
        embedding.build(None)

        x_input = tf.placeholder(dtype=tf.int32, shape=[None, None])
        emb_out = embedding.call(x_input)
        x = np.random.randint(0, vocab_size, size=(2, 10), dtype=np.int32)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            
            y = sess.run(emb_out, feed_dict={x_input: x})

            self.assertTrue(y.ndim == 3)
            self.assertEqual(x.shape, y.shape[:-1])
            self.assertEqual(y.shape[-1], embedding_size)

    def testPretrainedEmbedding(self):

        pretrained_weights = np.random.rand(100, 30)
        vocab_size, embedding_size = pretrained_weights.shape
        embedding_spec = EmbeddingSpec("TestEmb2", pretrained_weights=pretrained_weights, trainable=False, dtype=tf.float64)
        embedding = Embedding(embedding_spec)
        embedding.build(None)

        x_input = tf.placeholder(dtype=tf.int32, shape=[None, None])
        emb_out = embedding.call(x_input)
        x = np.random.randint(0, vocab_size, size=(2, 10), dtype=np.int32)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            
            y = sess.run(emb_out, feed_dict={x_input: x})

            self.assertTrue(y.ndim == 3)
            self.assertEqual(x.shape, y.shape[:-1])
            self.assertEqual(y.shape[-1], embedding_size)
            self.assertTrue(y.dtype == np.float64)
    

    def testMultipleEmbedding(self):
        pretrained_weights = np.random.rand(100, 30)
        vocab_size, embedding_size = pretrained_weights.shape
        
        embedding_specs = [
            EmbeddingSpec("input", pretrained_weights=pretrained_weights, trainable=False),
            EmbeddingSpec("key", embedding_size=embedding_size, vocab_size=11)
        ]

        embedding = MultipleEmbedding(embedding_specs)
        embedding.build(None)

        x_input = {
            'input': tf.placeholder(dtype=tf.int32, shape=[None, None]),
            'key': tf.placeholder(dtype=tf.int32, shape=[None, None])
        }

        emb_out = embedding.call(x_input)
        
        x = {
            'input': np.random.randint(0, vocab_size, size=(2, 10), dtype=np.int32),
            'key': np.random.randint(0, 11, size=(2, 1), dtype=np.int32)
        }
        
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            y = sess.run(emb_out, feed_dict={x_input['input']: x['input'], x_input['key']: x ['key']})
            
            self.assertTrue(isinstance(y, dict))

            self.assertTrue('input' in y)
            self.assertTrue('key' in y)
            self.assertEqual(x['input'].shape, y['input'].shape[:-1])
            self.assertEqual(x['key'].shape, y['key'].shape[:-1])
            self.assertEqual(embedding_size, y['input'].shape[-1])
            self.assertEqual(embedding_size, y['key'].shape[-1])
       

if __name__ == '__main__':
    tf.test.main()