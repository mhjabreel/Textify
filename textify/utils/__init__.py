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


from textify.utils.module import DynamicImporter
from textify.utils.config import Configuration


import tensorflow as tf
import six


def add_dict_to_collection(collection_name, dict_):
    """Adds a dictionary to a graph collection.
    Args:
        collection_name: The name of the collection to add the dictionary to
    dict_: A dictionary of string keys to tensor values
    """
    key_collection = collection_name + "_keys"
    value_collection = collection_name + "_values"
    for key, value in six.iteritems(dict_):
        tf.add_to_collection(key_collection, key)
        tf.add_to_collection(value_collection, value)


def get_dict_from_collection(collection_name):
    """Gets a dictionary from a graph collection.
    Args:
        collection_name: A collection name to read a dictionary from
    Returns:
        A dictionary with string keys and tensor values
    """
    key_collection = collection_name + "_keys"
    value_collection = collection_name + "_values"
    keys = tf.get_collection(key_collection)
    values = tf.get_collection(value_collection)
    return dict(zip(keys, values))