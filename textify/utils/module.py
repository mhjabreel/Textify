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
import sys
import inspect
from importlib import import_module



class DynamicImporter:

    def __init__(self, module_path, public_only=True):

        self._module_path = module_path
        self._classes = {}
        self._functions = {}
        self._public_only = public_only
        self._load_modules()
    
    def _load_modules(self):
        
        dirname, filename = os.path.split(self._module_path)
        module_name, _ = os.path.splitext(filename)
        sys.path.insert(0, os.path.abspath(dirname))
        module = import_module(module_name)
        sys.path.pop(0)
        
        del sys.path_importer_cache[os.path.dirname(module.__file__)]
        del sys.modules[module.__name__]    

        classes = inspect.getmembers(module, inspect.isclass)
        
        if self._public_only:
            classes = list(filter(lambda m: not m[0].startswith("_"), classes))  
        
        self._classes = dict(classes)

        functions = inspect.getmembers(module, inspect.isfunction)
        if self._public_only:
            functions = list(filter(lambda m: not m[0].startswith("_"), functions))
        
        self._functions = dict(functions)

    def get_function(self, func_name):
        return self._functions.get(func_name, None)
    
    def get_class(self, cls_name):
        return self._classes.get(cls_name, None)
    
    def get_first_class_of(self, cls_type):

        for _, cls_v in self._classes.items():
            if issubclass(cls_v, cls_type):
                return cls_v
    
    def classes(self, cls_type):
        for cls_name, cls_v in self._classes.items():
            if cls_v != cls_type and issubclass(cls_v, cls_type):
                yield (cls_name, cls_v)
    
    def get_last_class_of(self, cls_type):
        cls_ = None
        for _, cls_v in self._classes.items():
            if issubclass(cls_v, cls_type):
                cls_ = cls_v        
        
        return cls_