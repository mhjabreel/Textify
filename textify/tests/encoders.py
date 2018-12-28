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

from textify.layers.encoders import MeanEncoder
from textify.layers.encoders import UnidirectionalRNNEncoder
from textify.layers.encoders import BidirectionalRNNEncoder
from textify.utils.encoder_utils import last_encoding_from_state, add_reducer_fn, EncoderException

class EncoderTest(tf.test.TestCase):

    def testMeanEncoder(self):
        encoder = MeanEncoder()
        #encoder.build()

        x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        
        out_op, state_op, _ = encoder.call(x_input)
        input_rank_op = tf.rank(x_input)
        out_rank_op = tf.rank(out_op)

        x = np.random.rand(2, 10, 5).astype(np.float32)

        with self.test_session() as sess:
            
            sess.run(tf.global_variables_initializer())

            input_rank = sess.run(input_rank_op)
            out_rank = sess.run(out_rank_op)
            self.assertEqual(input_rank, out_rank)

            out, state = sess.run([out_op, state_op], feed_dict={x_input: x})
            last_sate = last_encoding_from_state(state)
            self.assertAllEqual(out, x)
            self.assertAllEqual(last_sate, np.mean(x, axis=1))
            
    def _rnnTestCase(self, num_units, num_layers=1, cell_type=tf.nn.rnn_cell.LSTMCell, residual=False):
        
        depth = 5
        batch_size = 2
        seq_len = 10
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, depth])

        encoder = UnidirectionalRNNEncoder(num_units=num_units, num_layers=num_layers, cell_type=cell_type, residual=residual)
        encoder.build()

        out_op, state_op, _ = encoder.call(x_input)
        input_rank_op = tf.rank(x_input)
        out_rank_op = tf.rank(out_op)

        x = np.random.rand(batch_size, seq_len, depth).astype(np.float32)

        with self.test_session() as sess:
            
            sess.run(tf.global_variables_initializer())

            input_rank = sess.run(input_rank_op)
            out_rank = sess.run(out_rank_op)
            self.assertEqual(input_rank, out_rank)

            out, state = sess.run([out_op, state_op], feed_dict={x_input: x})
            self.assertAllEqual(out.shape[:-1], x.shape[:-1])
            self.assertEqual(out.shape[-1], num_units)
            last_sate = last_encoding_from_state(state)
            self.assertAllEqual(last_sate.shape, [batch_size, num_units])

    def _biRNNTestCase(self, num_units, num_layers=1, reduce_fn=None, cell_type=tf.nn.rnn_cell.LSTMCell, residual=False):
        
        depth = 5
        batch_size = 2
        seq_len = 10
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, depth])

        if reduce_fn == add_reducer_fn:
            expected_out_units = num_units
        else:
            if residual:
                expected_out_units = num_units
            else:
                expected_out_units = 2 * num_units

        encoder = BidirectionalRNNEncoder(num_units=num_units, num_layers=num_layers, cell_type=cell_type, reduce_fn=reduce_fn, residual=residual)
        encoder.build()

        out_op, state_op, _ = encoder.call(x_input)
        input_rank_op = tf.rank(x_input)
        out_rank_op = tf.rank(out_op)

        x = np.random.rand(batch_size, seq_len, depth).astype(np.float32)

        with self.test_session() as sess:
            
            sess.run(tf.global_variables_initializer())

            input_rank = sess.run(input_rank_op)
            out_rank = sess.run(out_rank_op)
            self.assertEqual(input_rank, out_rank)

            out, state = sess.run([out_op, state_op], feed_dict={x_input: x})
            self.assertAllEqual(out.shape[:-1], x.shape[:-1])
            self.assertEqual(out.shape[-1], expected_out_units)
            last_sate = last_encoding_from_state(state)
            self.assertAllEqual(last_sate.shape, [batch_size, expected_out_units])

    def testUnidirectionalRNNEncoder(self):
        self._rnnTestCase(num_units=10)
        self._rnnTestCase(num_units=10, num_layers=2)
        self._rnnTestCase(num_units=10, cell_type=tf.nn.rnn_cell.GRUCell)
        self._rnnTestCase(num_units=10, num_layers=3, cell_type=tf.nn.rnn_cell.GRUCell)

    def testResidualUnidirectionalRNNEncoder(self):
        self._rnnTestCase(num_units=10, num_layers=3, residual=True)  
        self._rnnTestCase(num_units=10, num_layers=3, cell_type=tf.nn.rnn_cell.GRUCell, residual=True)   
    
    def testBidirectionalRNNEncoder(self):
        self._biRNNTestCase(num_units=10)
        self._biRNNTestCase(num_units=10, num_layers=2)

        self._biRNNTestCase(num_units=10, cell_type=tf.nn.rnn_cell.GRUCell)
        self._biRNNTestCase(num_units=10, num_layers=2, cell_type=tf.nn.rnn_cell.GRUCell)
    
    def testAddReducerBidirectionalRNNEncoder(self):
        self._biRNNTestCase(num_units=10, reduce_fn=add_reducer_fn)
        self._biRNNTestCase(num_units=10, num_layers=2, reduce_fn=add_reducer_fn)         
        self._biRNNTestCase(num_units=10, reduce_fn=add_reducer_fn, cell_type=tf.nn.rnn_cell.GRUCell)
        self._biRNNTestCase(num_units=10, num_layers=2, reduce_fn=add_reducer_fn, cell_type=tf.nn.rnn_cell.GRUCell)


    def testResidualBidirectionalRNNEncoder(self):
        self._biRNNTestCase(num_units=10, num_layers=3, residual=True)  
        self._biRNNTestCase(num_units=10, num_layers=3, cell_type=tf.nn.rnn_cell.GRUCell, residual=True)  

    def testResidualAddReducerBidirectionalRNNEncoder(self):
        self._biRNNTestCase(num_units=10, num_layers=3, residual=True, reduce_fn=add_reducer_fn)  
        self._biRNNTestCase(num_units=10, num_layers=3, cell_type=tf.nn.rnn_cell.GRUCell, residual=True, reduce_fn=add_reducer_fn)          

    def testRaiseResidualBidirectionalRNNEncoder(self):
        
        depth = 5
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, None, depth])
        encoder = BidirectionalRNNEncoder(num_units=3, num_layers=2, residual=True)
        encoder.build()
        
        with self.assertRaises(EncoderException) as cm:
            _ = encoder.call(x_input)
        
        ex = cm.exception
        self.assertEqual(ex.error_code, 200)       

if __name__ == "__main__":
    tf.test.main()