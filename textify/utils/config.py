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
import yaml

class Configuration(object):

    def __init__(self, config_paths, config={}):
        
        config = config or {}

        for config_path in config_paths:
            with tf.gfile.Open(config_path, mode="rb") as config_file:
                subconfig = yaml.load(config_file.read())
                Configuration.merge_dict(config, subconfig)
        self._config = config       

    def __len__(self):
        return len(self._config)
    

    def __getitem__(self, key, default_value=None):
        return self._config.get(key, default_value)
    
    def __setitem__(self, key, value):
        self._config[key] = value

    def __repr__(self):
        return repr(self._config)
    
    def __str__(self):
        return str(self._config)

    @staticmethod
    def merge_dict(dict1, dict2):
        for k, v in dict2.items():
            if isinstance(v, dict):
                dict1[k] = Configuration.merge_dict(dict1.get(k, {}), v)
            else:
                dict1[k] = v
        
        return dict1   

