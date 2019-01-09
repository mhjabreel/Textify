import os
import six
import tensorflow as tf
import numpy as np

from textify.utils.checkpoints import avearge_checkpoints
from textify.utils.checkpoints import clone_checkpoint
from textify.utils.checkpoints import export_as_checkpoint


class Evaluator(tf.train.CheckpointSaverListener):

    def __init__(self, eval_fn, estimator, callbacks=None, **kwargs):

        self._eval_fn = eval_fn
        self._callbacks = callbacks
        self._estimator = estimator

    
    def after_save(self, session, global_step_value):
        tf.logging.info("Step[%d]: Done writing checkpoint.", global_step_value)
        tf.logging.info("Step[%d]: Start evaluating the model.", global_step_value)
        res = self._eval_fn()
        if not self._callbacks is None:
            for clpk in self._callbacks:
                clpk(res, global_step_value, self._estimator.latest_checkpoint())


class CheckpointAveragator(tf.train.CheckpointSaverListener):
    """
    Inspired by:
    github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_avg_all.py
    """
    def __init__(self, model_dir, max_ckpts, output_dir, callbacks=None):
        
        if model_dir == output_dir:
            raise ValueError("Model and output directory must be different")

        self._model_dir = model_dir
        self._max_ckpts = max_ckpts
        self._output_dir = output_dir

        self._callbacks = callbacks

    def after_save(self, session, global_step_value):
        tf.logging.info("Step[%d]: Done writing checkpoint.", global_step_value)
        tf.logging.info("Step[%d]: Start avergaging the last %d checkpoints.", global_step_value, self._max_ckpts)
        
        variables = avearge_checkpoints(self._model_dir, self._max_ckpts)
        
        ckpt_dir_ = export_as_checkpoint(variables, global_step_value, self._output_dir)

        if not self._callbacks is None:
            for cbk in self._callbacks:
                cbk(ckpt_dir_)

class BestCheckpointExporter:

    def __init__(self, output_dir, monitor='F1', compare_fn=lambda x, y: y is None or x > y):

        self._monitor = monitor
        self._compare_fn = compare_fn
        self._output_dir = output_dir
        self._best_val = None

    def __call__(self, results, global_step_value, check_pointpath):
        val = results[self._monitor]
        if self._compare_fn(val, self._best_val):
            old_val = "INF" if self._best_val is None else "%.4f" % (self._best_val)
            tf.logging.info("Step[%d]: Improvment in %s from %s to %.4f.", global_step_value, self._monitor, old_val, val)
            tf.logging.info("Step[%d]: Export the best checkpoint to %s.", global_step_value, self._output_dir)
            model_name = "model_%s_%.5f" % (self._monitor, val)
            clone_checkpoint(check_pointpath, self._output_dir, id_=val * 10000)
            
            self._best_val = val
