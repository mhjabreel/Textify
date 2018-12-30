from importlib import import_module
import os
import pickle
import sys
import tensorflow as tf
import yaml
from sentideep.utils.misc import merge_dict
import inspect
from sentideep.model import Model
from sentideep.data import DataLayer

def load_config(config_paths, config=None):
    """Loads configuration files.
    Args:
        config_paths: A list of configuration files.
        config: A (possibly non empty) config dictionary to fill.
    Returns:
        The configuration dictionary.
    """
    if config is None:
        config = {}

    for config_path in config_paths:
        with tf.gfile.Open(config_path, mode="rb") as config_file:
            subconfig = yaml.load(config_file.read())
            # Add or update section in main configuration.
            merge_dict(config, subconfig)

    return config

def load_model_module(path):
    """Loads a model configuration file.
    Args:
        path: The relative path to the configuration file.
    Returns:
        A Python module.
    """
    dirname, filename = os.path.split(path)
    module_name, _ = os.path.splitext(filename)
    sys.path.insert(0, os.path.abspath(dirname))
    module = import_module(module_name)
    sys.path.pop(0)
    
    del sys.path_importer_cache[os.path.dirname(module.__file__)]
    del sys.modules[module.__name__] 

    return module


def list_classes(module, public_only=True):
    classes = inspect.getmembers(module, inspect.isclass)
    if public_only:
        classes = list(filter(lambda m: not m[0].startswith("_"), classes))
    return classes

def get_model(model_path):

    module = load_model_module(model_path)
    classes = list_classes(module)
    models = list(filter(lambda m: issubclass(m[1], Model), classes))
    model = models[0][1]
    return model

def get_data_layer(path):
    module = load_model_module(path)
    classes = list_classes(module)
    models = list(filter(lambda m: issubclass(m[1], DataLayer)), classes)
    model = models[0][1]
    return model 

