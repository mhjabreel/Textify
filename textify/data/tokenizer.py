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


def space_tokenizer(text):
    if tf.contrib.framework.is_tensor(text):
        return tf.string_split([text], delimiter=" ").values
    return text.split(" ")


def character_tokenizer(text):

    if tf.contrib.framework.is_tensor(text):
        text = tf.py_func(
                lambda x: "\0".join(character_tokenizer(x)), [text], tf.string)
        
        return tf.string_split([text], delimiter="\0").values        
    
    text = tf.compat.as_text(text)
    return list(text.replace(" ", u"▁"))
