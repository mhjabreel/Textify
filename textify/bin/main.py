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
import argparse
import json
import os
import six
import copy
import tensorflow as tf
import collections

from textify import Runner
from textify.utils import Configuration
from textify.utils import DynamicImporter
from textify.models import Model
from textify.data import DataLayer
from textify.estimator import EstimatorBuilder
from textify.estimator import ClassifierBuilder
from textify.estimator import BinaryClassifierBuilder
from textify.utils.vocab_utils import load_vocab
from textify.utils.embedding_utils import EmbeddingSpec
from textify.utils.embedding_utils import load_embedding
from textify.data import DefaultDataLayer
from textify.data import MultiInputDataLayer
from textify.data import MultiOutputDataLayer
from textify.data import MultiIODataLayer
from textify.utils.hooks import ExternalEvaluatorHook

def main():


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", choices=["train_and_eval", "train", "eval", "predict"],
                        help="Run type.")
    parser.add_argument("--config", required=True, nargs="+",
                        help="List of configuration files.")       
    parser.add_argument("--run_dir", default="",
                        help="If set, model_dir will be created relative to this location.")
    parser.add_argument("--features_file", default=[], nargs="+",
                        help="Run inference on this file.")
    parser.add_argument("--predictions_file", default="",
                        help=("File used to save predictions. If not set, predictions are printed "
                            "on the standard output."))
    parser.add_argument("--checkpoint_path", default=None,
                        help=("Checkpoint or directory to use for inference or export "
                            "(when a directory is set, the latest checkpoint is used).")) 
    
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
                        help="Logs verbosity.")    


    args = parser.parse_args()

    tf.logging.set_verbosity(getattr(tf.logging, args.log_level))

    config = Configuration(args.config)
    if args.run_dir:
        config["model_dir"] = os.path.join(args.run_dir, config["model_dir"])

    #print(config)
    model_config = config['model']
    data_config = config['data'] 
    
    data_kwargs = {}
    if 'others' in data_config:
        for k in data_config['others']:
            data_kwargs[k] = data_config['others'][k]    

    data_init_params = {}

    features_count = 1

    data_layer = data_config.get('data_layer', None)
    
    if not data_layer is None:
        data_layer_and_ref = data_layer.split('#')     
        data_layer =  data_layer_and_ref[0]  
        data_layer_importer = DynamicImporter(data_layer)

        if len(data_layer_and_ref) == 2:
            cls_data_layer = data_layer_importer.get_class(data_layer_and_ref[1])
        else:
            cls_data_layer = data_layer_importer.get_last_class_of(DataLayer)
    else:
        feature_names = data_config.get("feature_names", None)
        if not feature_names is None:
            features_count = len(feature_names)

        if features_count > 1:
            
            cls_data_layer = MultiInputDataLayer
            data_kwargs['feature_names'] = feature_names
            data_init_params['features'] = {}
            for feature in feature_names:
                data_init_params['features'][feature] = {
                    'vocab': data_config['vocabs'][feature],
                    'unk_id': data_config['unk_ids'][feature],
                }
                if 'max_lengths' in data_config:
                    data_init_params['features'][feature]['max_len'] = \
                        data_config['max_lengths'].get(feature, None)
        else:
            cls_data_layer = DefaultDataLayer
    
    if 'init_params' in data_config:
        data_init_params.update(data_config['init_params'])


    
    model_type_and_ref = model_config['model_type'].split('#')

    model_type = model_type_and_ref[0]
    
    model_importer = DynamicImporter(model_type)

    model_params = model_config['model_params']

    #model_type

    if len(model_type_and_ref) == 2:
        model_creator = model_importer.get_class(model_type_and_ref[1])
    else:
        model_creator = model_importer.get_last_class_of(Model)
    
    if 'estimator' in model_params:
        estimator = model_params['estimator']
        
        estimator_builder = model_importer.get_class(estimator)
        if estimator_builder is None:
            estimator = model_importer.get_last_class_of(EstimatorBuilder)
        
        if estimator_builder is None:
            raise RuntimeError("Unknown estimator builder %s." % estimator )
    else:
        num_classes = model_params.get("num_classes", 2)
 
        if num_classes == 2:
            estimator_builder = BinaryClassifierBuilder
        else:
            estimator_builder = ClassifierBuilder
    
    vocabs = None

    if features_count == 1:
        
        vocab_input = data_init_params['vocab']
        if not isinstance(vocab_input, list):
            vocabs = load_vocab(vocab_input)
        else:
            vocabs = vocab_input, len(vocab_input)        
    else:
        vocabs = {}
        for feature in feature_names:
            vocab_source = data_init_params['features'][feature]['vocab']
            if not isinstance(vocab_source, list):
                vocabs[feature] = load_vocab(vocab_source)
            else:
                vocabs[feature] = vocab_source, len(vocab_source)            

    
    embeddings_specs = []
    embedding_config = model_params.get('embeddings', None)

    if not embedding_config is None:
        if features_count == 1:

            if 'path' in embedding_config:
                pretrained_weights = load_embedding(embedding_config['path'],
                                            vocabs[0],
                                            with_header=embedding_config['header'],
                                            dim=embedding_config['size'],
                                            separator=embedding_config.get('separator', ' '))
                emb_spec = EmbeddingSpec(
                    name="input_emb",
                    pretrained_weights=pretrained_weights,
                    vocab_size=pretrained_weights.shape[0],
                    embedding_size=pretrained_weights.shape[1],
                    trainable=embedding_config.get("trainable", False)
                ) 
            else:
                if 'vocab_size' in embedding_config:
                    vocb_size = embedding_config['vocab_size']
                else:
                    vocb_size = vocabs[1] 

                embedding_size = embedding_config['size'] 

                emb_spec = EmbeddingSpec(
                    name="input_emb",
                    vocab_size=vocb_size,
                    embedding_size=embedding_size,
                    trainable=True
                ) 
            
            embeddings_specs.append(emb_spec)
        else:
            for k in embedding_config:
                if k in vocabs:
                    vocab = vocabs[k][0]
                    if 'path' in embedding_config[k]:
                        pretrained_weights = load_embedding(embedding_config[k]['path'],
                                                    vocab,
                                                    with_header=embedding_config[k]['header'],
                                                    dim=embedding_config[k]['size'],
                                                    separator=embedding_config[k].get('separator', ' '))
                        
                        emb_spec = EmbeddingSpec(
                            name=k,
                            pretrained_weights=pretrained_weights,
                            trainable=embedding_config[k].get("trainable", False)
                        )
                        
                    else:
                        if 'vocab_size' in embedding_config[k]:
                            vocb_size = embedding_config[k]['vocab_size']
                        else:
                            vocb_size = vocabs[k][1]                         
                        emb_spec = EmbeddingSpec(
                            name=k,
                            vocab_size=vocabs[k][1],
                            embedding_size=embedding_config[k]['size'],
                            trainable=True
                        )   
                    embeddings_specs.append(emb_spec)             
                else:
                    raise RuntimeError("Got an embedding %s without vocabulary. Each embbeding must be linked to a vocabulary." % k)    

    if len(embeddings_specs) == 1:
        embeddings_specs = embeddings_specs[0]

    model_params['embedding_specs'] = embeddings_specs


    model = model_creator(model_params)
    train_config = config['train']
    train_config['num_classes'] = num_classes

    model = model_creator(model_params)

    estimator = estimator_builder(model, train_config)

    if args.run == 'train' or args.run == 'train_and_eval':
        
        train_features_source = data_config['train']['features_source']
        train_labels_source = data_config['train']['labels_source']

        _kwargs = copy.deepcopy(data_kwargs)

        _kwargs['batch_size'] = train_config['batch_size']
        _kwargs['repeat'] = train_config.get('repeat', True)

        train_data_layer = cls_data_layer(train_features_source,
                    train_labels_source,
                    data_init_params,
                    **_kwargs)
        
        dev_data_layer = None

        if args.run == 'train_and_eval':
            
            dev_features_source = data_config['dev']['features_source']
            dev_labels_source = data_config['dev']['labels_source']

            _kwargs = copy.deepcopy(data_kwargs)

            _kwargs['batch_size'] = train_config.get('dev_batch_size', 100)
            
            dev_data_layer = cls_data_layer(dev_features_source,
                        dev_labels_source,
                        data_init_params,
                        **_kwargs) 
        
        if 'eval_hooks' in train_config:
            eval_hooks = None
        else:
            eval_hooks = None
        
        if 'external_eval_hooks' in train_config:
            external_eval_hooks = []
            eeh_importer = DynamicImporter(train_config['external_eval_hooks'])
            for eeh in eeh_importer.classes(ExternalEvaluatorHook):
                external_eval_hooks.append(eeh)
        else:
            external_eval_hooks = None


        runner = Runner(estimator, train_config, eval_hooks=eval_hooks, external_eval_hooks=external_eval_hooks)
        if args.run == 'train_and_eval':
            runner.train_and_evaluate(train_data_layer, dev_data_layer)
        else:
            runner.train(train_data_layer)

    elif args.run == 'eval':
        checkpoint_path = args.checkpoint_path
        dev_features_source = data_config['dev']['features_source']
        dev_labels_source = data_config['dev']['labels_source']

        _kwargs = copy.deepcopy(data_kwargs)

        _kwargs['batch_size'] = train_config.get('dev_batch_size', 100)
        
        
        dev_data_layer = cls_data_layer(dev_features_source,
                    dev_labels_source,
                    data_init_params,
                    **_kwargs) 
        
        runner = Runner(estimator, train_config)
        runner.evaluate(dev_data_layer, checkpoint_path)

    elif args.run == 'predict':
        checkpoint_path = args.checkpoint_path
        dev_features_source = data_config['dev']['features_source']#args.features_file
        dev_labels_source = data_config['dev']['labels_source']
        out_file = args.predictions_file

        _kwargs = copy.deepcopy(data_kwargs)

        _kwargs['batch_size'] = train_config.get('dev_batch_size', 100)
        _kwargs['shuffle'] = False
        
        data_layer = cls_data_layer(dev_features_source,
                    dev_labels_source,
                    data_init_params,
                    **_kwargs) 
        
        runner = Runner(estimator, train_config)
        runner.predict(data_layer, checkpoint_path)        

if __name__ == "__main__":
    main()