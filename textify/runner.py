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

tf.logging.set_verbosity(tf.logging.INFO)

class Runner:

    def __init__(self,            
            estimator_builder,
            config,
            session_config=None,
            gpu_allow_growth=False,
            seed=None):
            
        self._estimator_builder = estimator_builder
        self._config = config
        
        session_config_base = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(allow_growth=gpu_allow_growth)
        )
        
        if session_config is not None:
            session_config_base.MergeFrom(session_config)

        run_config = tf.estimator.RunConfig(
            model_dir=config["model_dir"],
            session_config=session_config_base,
            tf_random_seed=seed)

        self._estimator = tf.estimator.Estimator(
            model_fn=estimator_builder.model_fn(),
            config=run_config
        )


    def _build_train_spec(self, data_layer, checkpoint_path=None):
        train_hooks = None
        train_spec = tf.estimator.TrainSpec(
            input_fn=data_layer.input_fn(repeat=True),
            max_steps=self._config["train"].get("train_steps"),
            hooks=train_hooks)
        return train_spec

    def _build_eval_spec(self, data_layer, checkpoint_path=None):
        eval_hooks = None
        eval_spec = tf.estimator.EvalSpec(
            input_fn=data_layer.input_fn(),
            steps=100,
            hooks=eval_hooks)
        return eval_spec             
    
    def train(self, data_layer, checkpoint_path=None):
        """Runs the training loop.
        Args:
            checkpoint_path: The checkpoint path to load the model weights from it.
        """
        if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        train_spec = self._build_train_spec(data_layer, checkpoint_path)
        self._estimator.train(train_spec.input_fn, hooks=train_spec.hooks, max_steps=train_spec.max_steps)
    
    def evaluate(self, data_layer, checkpoint_path=None):
        eval_spec = self._build_eval_spec(data_layer, checkpoint_path)
        self._estimator.evaluate(eval_spec.input_fn, hooks=eval_spec.hooks, steps=eval_spec.steps, checkpoint_path=checkpoint_path)
    

    def train_and_evaluate(self, train_data_layer, eval_data_layer, checkpoint_path=None):
        if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        train_spec = self._build_train_spec(train_data_layer, checkpoint_path)
        eval_spec = self._build_eval_spec(eval_data_layer)
        tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)

    def predict(self, checkpoint_path=None):
        pass

    