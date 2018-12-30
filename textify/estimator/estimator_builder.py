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
import six
import abc

from textify.utils.metrics import precision_score, recall_score

@six.add_metaclass(abc.ABCMeta)
class EstimatorBuilder:

    def __init__(self, model, params):

        # self._model_creator = model_creator
        self._params = params
        self._model = model #model_creator(params, None)
    
    def model_fn(self, scope=None):
        params = self._params
        def model_fn_impl(features, labels, mode):
            self._global_step = tf.train.get_or_create_global_step()
            logits = self._model(features)
            predictions = self._model.get_predictions(logits)
            eval_metric_ops = None
            if mode in {tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL}:
                loss = self._get_loss(logits, labels)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    optimizer = self._get_optimizer(params)
                    variables = tf.trainable_variables()
                    gradients = tf.gradients(loss, variables)
                    #TODO: add gradient clip
                    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self._global_step)
                else:
                    train_op = None    
                    eval_metric_ops = self._evaluate(labels, predictions)  
                predictions = None          
            else:
                loss = None
            
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops)
        
        return model_fn_impl
        

    def _get_optimizer(self, params):
        
        print(params)
        learning_rate = params.get("learning_rate")
        print(learning_rate)
        self._learning_rate = tf.constant(learning_rate)
        
        # decay
        self._learning_rate = self._get_learning_rate_decay(params)        
        optimizer_name = params.get("optimizer")

        if optimizer_name == 'rms':
            optimizer = tf.train.RMSPropOptimizer(self._learning_rate,
                                                    params.get("rmsprop_decay"),
                                                    params.get("momentum"),
                                                    params.get("rmsprop_epsilon", 1e-6))
        elif optimizer_name == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self._learning_rate,
                                                    params.get("momentum"))
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self._learning_rate)
        else:
            raise ValueError('Invalid value of optimizer: %s' % optimizer_name)

        return optimizer

    def _get_decay_info(self, params):
        """Return decay info based on decay_scheme."""
        decay_scheme = params.get("decay_scheme", None)
        num_train_steps = params.get("num_train_steps", 1000)
        if decay_scheme in ["luong5", "luong10", "luong234"]:
            decay_factor = 0.8
            if decay_scheme == "luong5":
                start_decay_step = int(num_train_steps / 2)
                decay_times = 5
            elif decay_scheme == "luong10":
                start_decay_step = int(num_train_steps / 2)
                decay_times = 10
            elif decay_scheme == "luong234":
                start_decay_step = int(num_train_steps * 2 / 3)
                decay_times = 4
            remain_steps = num_train_steps - start_decay_step
            decay_steps = int(remain_steps / decay_times)
        elif not decay_scheme:  # no decay
            start_decay_step = num_train_steps
            decay_steps = 0
            decay_factor = 1.0
        elif decay_scheme:
            raise ValueError("Unknown decay scheme %s" % decay_scheme)
        return start_decay_step, decay_steps, decay_factor

    def _get_learning_rate_decay(self, params):
        """Get learning rate decay."""
        decay_scheme = params.get("decay_scheme", None)
        start_decay_step, decay_steps, decay_factor = self._get_decay_info(params)
        tf.logging.info("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                        "decay_factor %g" , decay_scheme,
                                                start_decay_step,
                                                decay_steps,
                                                decay_factor)

        return tf.cond(
            self._global_step < start_decay_step,
            lambda: self._learning_rate,
            lambda: tf.train.exponential_decay(
                self._learning_rate,
                (self._global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True), 
                name="learning_rate_decay_cond")        


    @abc.abstractmethod
    def _get_loss(self, logits, labels):
        raise NotImplementedError()

    @abc.abstractmethod
    def _evaluate(self, y_true, y_pred):
        raise NotImplementedError()        
    

class ClassifierBuilder(EstimatorBuilder):

    def __init__(self, model_creator, params):
        super(ClassifierBuilder, self).__init__(model_creator, params)
        self._num_classes = params.get("num_classes", 2)

    def _get_loss(self, logits, labels):
        if self._num_classes == 2:
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=tf.squeeze(logits, 1))
        else:
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(labels, tf.float32), logits=logits)
        
        class_weights = self._params.get("class_weights", None)

        if class_weights is None:
            loss = tf.reduce_mean(losses)
        else:
            class_weights = tf.constant(class_weights)
            weights = tf.nn.embedding_lookup(class_weights, labels)
            loss = tf.losses.compute_weighted_loss(losses, weights=weights, reduction=tf.losses.Reduction.MEAN)
        
        weight_decay = self._params.get("weight_decay", None)

        if weight_decay and weight_decay > 0:

            loss_wd = (
                weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            )

            tf.summary.scalar('loss_wd', loss_wd)
            loss += loss_wd
        
        tf.summary.scalar("loss", loss)

        return loss

    def _evaluate(self, y_true, y_pred):
        
        if isinstance(y_pred, dict):
            y_pred = y_pred['Predictions']
        
        accuracy = tf.metrics.accuracy(y_true, y_pred)
        precision = precision_score(y_true, y_pred, self._num_classes)
        recall = recall_score(y_true, y_pred, self._num_classes)
        f1 = (2 * precision[0] * recall[0]) / (recall[0] + precision[0])
        f1_update = (2 * precision[1] * recall[1]) / (recall[1] + precision[1])

        eval_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': (f1, f1_update)  
        } 

        return eval_metrics        


class BinaryClassifierBuilder(ClassifierBuilder):
    def __init__(self, model_creator, params):
        super(BinaryClassifierBuilder, self).__init__(model_creator, params)
        self._num_classes = 2

    def _evaluate(self, y_true, y_pred):
        
        if isinstance(y_pred, dict):
            y_pred = y_pred['Predictions']
        
        accuracy = tf.metrics.accuracy(y_true, y_pred)
        precision = tf.metrics.precision(y_true, y_pred)
        recall = tf.metrics.recall(y_true, y_pred)
        f1 = (2 * precision[0] * recall[0]) / (recall[0] + precision[0])

        eval_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': (f1, f1)  
        } 

        return eval_metrics

