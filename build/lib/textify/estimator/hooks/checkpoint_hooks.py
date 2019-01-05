import os
import six
import tensorflow as tf
import numpy as np

class Evaluator(tf.train.CheckpointSaverListener):

    def __init__(self, eval_fn, callbacks=None, **kwargs):

        self._eval_fn = eval_fn
        self._callbacks = callbacks
    
    def after_save(self, session, global_step_value):
        tf.logging.info("Step[%d]: Done writing checkpoint.", global_step_value)
        tf.logging.info("Step[%d]: Start evaluating the model.", global_step_value)
        res = self._eval_fn()
        if not self._callbacks is None:
            for clpk in self._callbacks:
                clpk(res, global_step_value)


class CheckpointAveragator(tf.train.CheckpointSaverListener):
    """
    Inspired by:
    github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_avg_all.py
    """
    def __init__(self, model_variables, model_dir, max_ckpts, output_dir=None):
        
        if model_dir == output_dir:
            raise ValueError("Model and output directory must be different")

        self._model_dir = model_dir
        self._max_ckpts = max_ckpts
        self._output_dir = output_dir
        self._model_variables = model_variables
        self._variable_names = set()
        for v in model_variables:
            self._variable_names.add(v.name)

    def after_save(self, session, global_step_value):
        tf.logging.info("Step[%d]: Done writing checkpoint.", global_step_value)
        tf.logging.info("Step[%d]: Start avergaging the lasd %d checkpoints.", global_step_value, self._max_ckpts)
        self._average_checkpoints()


    def _average_checkpoints(self):
        max_count = self._max_ckpts
        checkpoint_paths = tf.train.get_checkpoint_state(self._model_dir).all_model_checkpoint_paths
        if len(checkpoint_paths) > max_count:
            checkpoint_paths = checkpoint_paths[-max_count:]
        num_checkpoints = len(checkpoint_paths)

        tf.logging.info("\tAveraging %d checkpoints..." % num_checkpoints)
        tf.logging.info("\tListing variables...")
        
        new_variables = {}
        for i, checkpoint_path in enumerate(checkpoint_paths):
            tf.logging.info("\t\tLoading checkpoint %s" % checkpoint_path)
            variables = tf.train.list_variables(checkpoint_path)
            for name, value in variables:
                if name in self._variable_names:
                    if self._model_variables[name].trainable and value.dtype not in (np.int32, np.int64):
                        scaled_value = value / num_checkpoints
                        if name in new_variables:
                            new_variables[name] += scaled_value
                        else:
                            new_variables[name] = scaled_value
                    else:
                        new_variables[name] = value

        self._export_to_ckpt(new_variables)

    def _export_to_ckpt(self, variables):

        tf_vars = []

        if 'global_step' in variables:
            del variables['global_step']

        for name in six.iterkeys(variables):
            old_var = self._model_variables[name]
            var = tf.get_variable(name, shape=old_var.shape, dtype=old_var.dtype, trainable=old_var.trainable)
            tf_vars.append(var)

        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

        output_dir = self._output_dir
        out_base_file = os.path.join(output_dir, "model.ckpt")
        global_step = tf.get_variable(
            "global_step",
            initializer=tf.constant(1, dtype=tf.int64),
            trainable=False)
        saver = tf.train.Saver(tf_vars + [global_step], save_relative_paths=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for p, assign_op, value in zip(placeholders, assign_ops, six.itervalues(variables)):
                sess.run(assign_op, {p: value})
            tf.logging.info("\t\tSaving new checkpoint to %s" % output_dir)
            saver.save(sess, out_base_file, global_step=global_step)

        return output_dir