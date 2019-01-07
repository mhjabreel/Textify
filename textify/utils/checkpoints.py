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
import six
import numpy as np
import tensorflow as tf


def avearge_checkpoints(model_dir, max_count):
    checkpoint_paths = tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths
    if len(checkpoint_paths) > max_count:
        checkpoint_paths = checkpoint_paths[-max_count:]
    num_checkpoints = len(checkpoint_paths)

    tf.logging.info("Averaging %d checkpoints..." % num_checkpoints)
    tf.logging.info("Listing variables...")

    var_list = tf.train.list_variables(checkpoint_paths[0])
    avg_values = {}

    for name, shape in var_list:
        if not name.startswith("global_step"):
            avg_values[name] = np.zeros(shape, dtype=np.float32)
        
    for checkpoint_path in checkpoint_paths:
        tf.logging.info("Loading checkpoint {}".format( checkpoint_path))
        reader = tf.train.load_checkpoint(checkpoint_path)

        for name in avg_values:
            avg_values[name] += reader.get_tensor(name)

    for name in avg_values:
        avg_values[name] /= num_checkpoints 

    return avg_values    


def clone_checkpoint(checkpoint_path_src, output_dir, model_name='model'):

    tf.logging.info("Listing variables...")

    var_list = tf.train.list_variables(checkpoint_path_src)
    reader = tf.train.load_checkpoint(checkpoint_path_src)
    variables = {}
    latest_step = None
    for name, _ in var_list:
        
        variables[name] = reader.get_tensor(name)
        if name.startswith("global_step"):
            latest_step = variables[name]
    latest_step = latest_step or 0
    
    export_as_checkpoint(variables, latest_step, output_dir, model_name)


def export_as_checkpoint(variables, latest_step, output_dir, model_name='model'):

    if "global_step" in variables:
        del variables["global_step"]
    
    g = tf.Graph()
    with g.as_default():
        tf_vars = []
        for name, value in six.iteritems(variables):
            trainable = True
            dtype = tf.as_dtype(value.dtype)
            tf_vars.append(tf.get_variable(
                name,
                shape=value.shape,
                dtype=dtype))

        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
        
        out_base_file = os.path.join(output_dir, model_name)

        global_step = tf.get_variable("global_step",
            initializer=tf.constant(latest_step, dtype=tf.int64),
            trainable=False)
        
        tf_vars.append(global_step)
        
        saver = tf.train.Saver(tf_vars, save_relative_paths=True)

    with tf.Session(graph=g) as sess:
        sess.run(tf.variables_initializer(tf_vars))
        for p, assign_op, value in zip(placeholders, assign_ops, six.itervalues(variables)):
            sess.run(assign_op, {p: value})
        tf.logging.info("\t\tSaving new checkpoint to %s" % output_dir)
        saver.save(sess, out_base_file)

    return output_dir