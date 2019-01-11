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

def precision_score(labels,
        predictions,
        num_classes,
        average='micro',
        name=None):

    # Flatten the input if its rank > 1.
    if labels.get_shape().ndims > 1:
        labels = tf.reshape(labels, [-1])

    if predictions.get_shape().ndims > 1:
        predictions = tf.reshape(predictions, [-1])

    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if average == 'micro':
        global_labels = []
        global_predictions = []
        for c in range(num_classes):
            global_labels.append(tf.equal(labels, c))
            global_predictions.append(tf.equal(predictions, c))
        
        global_labels = tf.concat(global_labels, axis=0)
        global_predictions = tf.concat(global_predictions, axis=0)
        p, update_op = tf.metrics.precision(global_labels, global_predictions)
    
    elif average == 'macro':

        precisions = []
        update_ops = []

        for c in range(num_classes):

            p, u = tf.metrics.precision(tf.equal(labels, c), tf.equal(predictions, c))
            precisions.append(p)
            update_ops.append(u)
        
        p = tf.reduce_mean(precisions)
        update_op = tf.reduce_mean(update_ops)

    return p, update_op

def recall_score(labels,
        predictions,
        num_classes,
        average='micro',
        name=None):

    # Flatten the input if its rank > 1.
    if labels.get_shape().ndims > 1:
        labels = tf.reshape(labels, [-1])

    if predictions.get_shape().ndims > 1:
        predictions = tf.reshape(predictions, [-1])

    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if average == 'micro':
        global_labels = []
        global_predictions = []
        for c in range(num_classes):
            global_labels.append(tf.equal(labels, c))
            global_predictions.append(tf.equal(predictions, c))
        
        global_labels = tf.concat(global_labels, axis=0)
        global_predictions = tf.concat(global_predictions, axis=0)
        r, update_op = tf.metrics.recall(global_labels, global_predictions)
    elif average == 'macro':

        recalls = []
        update_ops = []

        for c in range(num_classes):

            r, u = tf.metrics.recall(tf.equal(labels, c), tf.equal(predictions, c))
            recalls.append(r)
            update_ops.append(u)
        
        r = tf.reduce_mean(recalls)
        update_op = tf.reduce_mean(update_ops)

    return r, update_op
