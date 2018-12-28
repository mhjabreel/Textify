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

import abc
import six
import tensorflow as tf

from textify.data.utils import count_lines, make_tokenized_data
from textify.data.tokenizer import space_tokenizer, character_tokenizer

from collections import defaultdict

class DataException(Exception):
    def __init__(self, message, errors=None):

        # Call the base class constructor with the parameters it needs
        super(DataException, self).__init__(message)

        # Now for your custom code...
        self.errors = errors

@six.add_metaclass(abc.ABCMeta)
class DataLayer(object):

    def __init__(self, features_source, labels_source=None, tokenizer=space_tokenizer, init_params={},  **kwargs):
        
        self._batch_size = kwargs.get("batch_size", 32)
        self._num_parallel_calls = kwargs.get("num_parallel_calls", 4)
        self._shuffle = kwargs.get("shuffle", True)
        self._buffer_size = kwargs.get("buffer_size", None)
        self._repeat = kwargs.get("repeat", None)
        self._filter_fn = kwargs.get("filter_fn", None)
        self._tokenizer = tokenizer

        self._features = None
        self._labels = None
        
        self._initialized = False

        self._features_source = features_source
        self._labels_source = labels_source
        self._init_params = init_params

    def build(self):
        
        buffer_size = self._buffer_size or self._batch_size * 100

        if not self._initialized:
            raise DataException("This data layer is not initialized yet.")
        
        features_dataset = self._build_features_dataset(self._features_source)
        if not self._labels_source is None:
            labels_dataset = self._build_labels_dataset(self._labels_source)
            dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
        else:
            dataset = features_dataset
        
        if not self._filter_fn is None:
            dataset = self._filter_fn(dataset)

        if self._shuffle:
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True) 

        padded_shapes = self._get_padded_shapes()   # labels is 1D
        padding_values = self._get_padding_values() # labels is 1D

        dataset = dataset.padded_batch(self._batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        
        if not self._repeat is None and self._repeat > 1:
            dataset = dataset.repeat(self._repeat)

        dataset = dataset.prefetch(buffer_size)
        
        iterator = dataset.make_initializable_iterator()

        self._initializer = iterator.initializer
        next_batch = iterator.get_next()
        
        if not self._labels_source is None:
            self._features = next_batch[0]
            self._labels = next_batch[1]
        else:
            self._features = next_batch
            self._labels = None 

        data_size = self._get_data_size(self._features_source)
        nb_btaches, r = divmod(data_size, self._batch_size)
        if r != 0:
            nb_btaches += 1
        
        self._nb_btaches = nb_btaches
    

    def __len__(self):
        return self._nb_btaches
    
    @abc.abstractmethod
    def _get_data_size(self, features_source):
        raise NotImplementedError()     

    @abc.abstractmethod
    def _build_features_dataset(self, features_source):
        raise NotImplementedError() 

    @abc.abstractmethod
    def _build_labels_dataset(self, labels_source):
        raise NotImplementedError()  

    def _get_padded_shapes(self):
        features_padded_shapes = self._get_features_padded_shapes()
        labels_padded_shapes = self._get_labels_padded_shapes()
        if labels_padded_shapes is None:
            return features_padded_shapes
        return features_padded_shapes, labels_padded_shapes
                 

    def _get_padding_values(self):
        features_padding_values = self._get_features_padding_values()
        labels_padding_values = self._get_labels_padding_values()
        if labels_padding_values is None:
            return features_padding_values
        return features_padding_values, labels_padding_values
    
    @abc.abstractmethod
    def _get_features_padded_shapes(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_labels_padded_shapes(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_features_padding_values(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_labels_padding_values(self):
        raise NotImplementedError()

    def initialize(self):
        self._initialized = True
    
    @property
    def initializer(self):
        return self._initializer
    
    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    def input_fn(self, repeat=None):
        
        self._repeat = repeat
        self.initialize()
        self.build()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, self.initializer)
        if not self._labels_source is None:
            return self.features, self.labels      
        return self.features


@six.add_metaclass(abc.ABCMeta)
class _OneLabelDataLayer(DataLayer):

    def __init__(self, features_source, labels_source=None, tokenizer=space_tokenizer, init_params={}, **kwargs):
        
        super(_OneLabelDataLayer, self).__init__(features_source, labels_source, tokenizer, init_params, **kwargs)
    
    def initialize(self):
        super(_OneLabelDataLayer, self).initialize()
        labels_vocab = self._init_params['labels_vocab']
        self._labels_vocab = tf.contrib.lookup.index_table_from_tensor(labels_vocab, default_value=-1)
        
    def _build_labels_dataset(self, labels_source):
        
        labels_dataset = tf.data.TextLineDataset(labels_source)
        labels_dataset = labels_dataset.map(lambda labels: tf.cast(self._labels_vocab.lookup(labels), tf.float32),
                num_parallel_calls=self._num_parallel_calls) 
        
        return labels_dataset  

    def _get_labels_padded_shapes(self):
        return []

    def _get_labels_padding_values(self):
        return 0.0


@six.add_metaclass(abc.ABCMeta)
class _MultiLabelDataLayer(DataLayer):

    def __init__(self, features_source, labels_source=None, tokenizer=space_tokenizer, init_params={}, num_labels=2, **kwargs):
        
        super(_MultiLabelDataLayer, self).__init__(features_source, labels_source, tokenizer, init_params, **kwargs)
        self._num_labels = num_labels
    
    def initialize(self):
        super(_MultiLabelDataLayer, self).initialize()
        self._separtor = self._init_params.get("labels_separator", ",")
        
    def _build_labels_dataset(self, labels_source):
        
        labels_dataset = tf.data.TextLineDataset(labels_source)
        labels_dataset = labels_dataset.map(lambda labels: tf.strings.split([labels], sep=self._separtor).values,
                num_parallel_calls=self._num_parallel_calls) 
        
        labels_dataset = labels_dataset.map(lambda labels: tf.string_to_number(labels),
                num_parallel_calls=self._num_parallel_calls) 
        
        return labels_dataset  

    def _get_labels_padded_shapes(self):
        return [None]

    def _get_labels_padding_values(self):
        return 0.0


@six.add_metaclass(abc.ABCMeta)
class _SingleFeatureDataLayer(DataLayer):

    def __init__(self, features_source, labels_source=None, tokenizer=space_tokenizer, init_params={}, **kwargs):
        
        super(_SingleFeatureDataLayer, self).__init__(features_source, labels_source, tokenizer, init_params, **kwargs)

    def initialize(self):
        words_vocab = self._init_params['vocab']
        unk_value = self._init_params['unk_id']
        self._max_len = self._init_params.get('max_len', None)
        self._words_vocab = tf.contrib.lookup.index_table_from_file(words_vocab, default_value=unk_value)
        super(_SingleFeatureDataLayer, self).initialize()

    def _build_features_dataset(self, features_source):
        return make_tokenized_data(features_source, self._tokenizer, self._words_vocab, self._max_len, self._num_parallel_calls)
    
    def _get_data_size(self, features_source):
        return count_lines(features_source)

    def _get_features_padded_shapes(self):
        return {'ids': [None], 'length': []}           

    def _get_features_padding_values(self):
        return {'ids': 0, 'length': 0}


# TODO: add multiple tokenizers one for each source
@six.add_metaclass(abc.ABCMeta)
class _MultiFeatureDataLayer(DataLayer):

    def __init__(self, features_source, labels_source=None, tokenizer=space_tokenizer, init_params={}, **kwargs):
        
        super(_MultiFeatureDataLayer, self).__init__(features_source, labels_source, tokenizer, init_params, **kwargs)
        self._feature_names = kwargs['feature_names']

    def initialize(self):

        init_params = self._init_params
        self._vocabs = {}
        self._max_lengths = {}

        for feature in self._feature_names:
            vocab = init_params['{}_vocab'.format(feature)]
            unk_value = init_params['{}_unk_id'.format(feature)]
            max_len = init_params.get('{}_max_len'.format(feature), None)
            self._vocabs[feature] = tf.contrib.lookup.index_table_from_file(vocab, default_value=unk_value) 
            self._max_lengths[feature] = max_len

        super(_MultiFeatureDataLayer, self).initialize()

    def _build_features_dataset(self, features_source):

        """
        Args:
            features_source: a dictionary of feature sources, {'feature_1': source_1, ... 'feature_n': source_n}
        """

        datasets = []
        for feature in self._feature_names:
            source = features_source[feature]
            vocab = self._vocabs[feature]
            max_len = self._max_lengths[feature]

            dataset = make_tokenized_data(source, self._tokenizer, vocab, max_len, self._num_parallel_calls, feature)
            datasets.append(dataset)
        
        features_dataset = tf.data.Dataset.zip(tuple(datasets))

        def _process(data):
            processed_data = {}
            for sub_data in data:
                for key, value in six.iteritems(sub_data):
                    prefix, key = key.split("_")
                    if not key in processed_data:
                        processed_data[key] = {}
                    processed_data[key][prefix] = value
            return processed_data        
            
        features_dataset = features_dataset.map(lambda *data: _process(data), num_parallel_calls=self._num_parallel_calls)
        return features_dataset

    def _get_data_size(self, features_source):
        return count_lines(features_source[self._feature_names[0]])

    def _get_features_padded_shapes(self):
        padded_shapes = defaultdict(dict)
        for feature in self._feature_names:
            padded_shapes['ids'][feature] = [None]
            padded_shapes['length'][feature] = []
        return padded_shapes          

    def _get_features_padding_values(self):
        padded_values = defaultdict(dict)
        for feature in self._feature_names:
            padded_values['ids'][feature] = 0
            padded_values['length'][feature] = 0
        return padded_values     


class _CharacterBasedDataLayer(DataLayer):
    def __init__(self, features_source, labels_source=None, init_params={}, **kwargs):
        super(_CharacterBasedDataLayer, self).__init__(features_source, labels_source, character_tokenizer, init_params, **kwargs)



class DefaultDataLayer(_SingleFeatureDataLayer, _OneLabelDataLayer):
    def __init__(self, features_source, labels_source=None, init_params={}, **kwargs):
        super(DefaultDataLayer, self).__init__(features_source, labels_source, space_tokenizer, init_params, **kwargs)
 

class MultiInputDataLayer(_MultiFeatureDataLayer, _OneLabelDataLayer):

    def __init__(self, features_source, labels_source=None, init_params={}, **kwargs):
        super(MultiInputDataLayer, self).__init__(features_source, labels_source, space_tokenizer, init_params, **kwargs)        

class MultiOutputDataLayer(_SingleFeatureDataLayer, _MultiLabelDataLayer):

    def __init__(self, features_source, labels_source=None, init_params={}, **kwargs):
        super(MultiOutputDataLayer, self).__init__(features_source, labels_source, space_tokenizer, init_params, **kwargs)

class MultiIODataLayer(_MultiFeatureDataLayer, _MultiLabelDataLayer):
    def __init__(self, features_source, labels_source=None, init_params={}, **kwargs):
        super(MultiIODataLayer, self).__init__(features_source, labels_source, space_tokenizer, init_params, **kwargs)


class CharacterBasedDataLayer(_SingleFeatureDataLayer, _OneLabelDataLayer):
    def __init__(self, features_source, labels_source=None, init_params={}, **kwargs):
        super(CharacterBasedDataLayer, self).__init__(features_source, labels_source, character_tokenizer, init_params, **kwargs)