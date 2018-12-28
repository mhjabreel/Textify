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


from textify.estimator.estimator_builder import EstimatorBuilder
from textify.estimator.estimator_builder import ClassifierBuilder
from textify.estimator.estimator_builder import BinaryClassifierBuilder

class Runner:

    def __init__(self, data_layer, estimator_builder):

        self._data_layer = data_layer
        self._estimator_builder = estimator_builder