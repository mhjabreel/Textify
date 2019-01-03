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

import six
import abc
import tensorflow as tf

from sklearn.metrics import f1_score, recall_score, precision_score, \
    accuracy_score, jaccard_similarity_score, accuracy_score

from textify.utils import get_dict_from_collection

@six.add_metaclass(abc.ABCMeta)
class ExternalEvaluatorHook(tf.train.SessionRunHook):

    def __init__(self, name, labels=None, predictions=None):
        self._labels = labels
        self._predictions = predictions
        self._name = name

    def begin(self):
        if self._predictions is None:
            self._predictions = get_dict_from_collection("predictions")

        if not self._predictions:
            raise RuntimeError("The model did not define any predictions.")

        if self._labels is None:
            self._labels = get_dict_from_collection("labels")

        if not self._labels:
            raise RuntimeError("The model did not define any labels.")

        self._global_step = tf.train.get_global_step()
        if self._global_step is None:
            raise RuntimeError("Global step should be created to use ExternalEvaluatorHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs([self._labels, self._predictions, self._global_step])    
    


    def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
        labels, predictions, current_step = run_values.results
        self._evaluate(labels, predictions, current_step)
        

    def end(self, session):
        tf.logging.info("External evaluater %s is done.", self._name)


    @abc.abstractmethod
    def _evaluate(self, labels, predictions, step):
        raise NotImplementedError()



class MultiLabelExternalEvaluatorHook(ExternalEvaluatorHook):

    def __init__(self, labels=None, predictions=None):
        super(MultiLabelExternalEvaluatorHook, self).__init__("MultiLabel Evaluator", labels, predictions)


    def _evaluate(self, labels, predictions, step):

        results = {
            "Jacard": jaccard_similarity_score(labels,
                                                predictions),
            "F1-macro": f1_score(labels, predictions,
                                    average='macro'),
            "F1-micro": f1_score(labels, predictions,
                                    average='micro'),
        }

        tf.logging.info("Step[%d]: Jacard=%.3f\tF1-macro=%.3f\tF1-micro:%.3f", step, results['Jacard'], results['F1-macro'], results['F1-micro'])

        return results