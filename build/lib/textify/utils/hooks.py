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

class AverageMeter:

    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

@six.add_metaclass(abc.ABCMeta)
class ExternalEvaluatorHook(tf.train.SessionRunHook):

    def __init__(self, name, labels=None, predictions=None):
        self._labels = labels
        self._predictions = predictions
        self._name = name
        


    def begin(self):
        if self._predictions is None:
            self._predictions = get_dict_from_collection("predictions")

        if self._predictions is None:
            raise RuntimeError("The model did not define any predictions.")

        if self._labels is None:
            self._labels = get_dict_from_collection("labels")

        if self._labels is None:
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
    def __init__(self, name="MultiLabel Evaluator", labels=None, predictions=None):
        super(MultiLabelExternalEvaluatorHook, self).__init__(name, labels, predictions)
        self._avg_results = {
            'Jacard': AverageMeter(),
            'F1-macro': AverageMeter(),
            'F1-micro': AverageMeter()
        }
        self.all_labels = []
        self.all_predictions = []

    def begin(self):
        super(MultiLabelExternalEvaluatorHook, self).begin()
        for k in self._avg_results.keys():
            self._avg_results[k].reset()
        self.all_labels = []
        self.all_predictions = []            

    def _evaluate(self, labels, predictions, step):

        results = {
            "Jacard": jaccard_similarity_score(labels,
                                                predictions),
            "F1-macro": f1_score(labels, predictions,
                                    average='macro'),
            "F1-micro": f1_score(labels, predictions,
                                    average='micro'),
        }

        n = labels.shape[0]
        for k in results.keys():
            self._avg_results[k].update(results[k], n)

        self.all_labels.extend(labels.tolist())
        self.all_predictions.extend(predictions.tolist())
        
        info = "Step[{step}]:\tJacard=({Jacard.val:.3f}/{Jacard.avg:.3f})\t" \
                "F1-macro=({F1Macro.val:.3f}/{F1Macro.avg:.3f})\t" \
                "F1-micro=({F1Maicro.val:.3f}/{F1Maicro.avg:.3f})".format(
                    step=step,
                    Jacard=self._avg_results['Jacard'],
                    F1Macro=self._avg_results['F1-macro'],
                    F1Maicro=self._avg_results['F1-micro'])

        tf.logging.info(info)

        return results

    def end(self, session):
        
        tf.logging.info("Jacard=%.3f\tF1-macro=%.3f\tF1-micro:%.3f",
                self._avg_results['Jacard'].avg,
                self._avg_results['F1-macro'].avg,
                self._avg_results['F1-micro'].avg) 

        results = {
            "Jacard": jaccard_similarity_score(self.all_labels, self.all_predictions),
            "F1-macro": f1_score(self.all_labels, self.all_predictions,
                                    average='macro'),
            "F1-micro": f1_score(self.all_labels, self.all_predictions,
                                    average='micro'),
        }   

        print(results)     

        tf.logging.info("External evaluater %s is done.", self._name)   