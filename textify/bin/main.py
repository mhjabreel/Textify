import six
import argparse
import json
import os
import six
import copy
import tensorflow as tf
import collections
from sentideep.runner import Runner
from sentideep.config import load_config
from sentideep.utils.misc import prefix_paths, merge_dict
from sentideep.config import get_model, get_data_layer
from sentideep.utils.vocab_utils import load_vocab
from sentideep.utils.embedding_utils import EmbeddingSpec, load_embedding
from sentideep.data import MultiInputTextClassificationDataLayer, TextClassificationDataLayer

def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", choices=["train_and_eval", "train", "eval", "predict", "score"],
                        help="Run type.")
    parser.add_argument("--config", required=True, nargs="+",
                        help="List of configuration files.")    
    parser.add_argument("--model", default="", help="Custom model configuration file.")    
    parser.add_argument("--run_dir", default="",
                        help="If set, model_dir will be created relative to this location.")
    parser.add_argument("--data_dir", default="",
                        help="If set, data files are expected to be relative to this location.")
    parser.add_argument("--features_file", default=[], nargs="+",
                        help="Run inference on this file.")
    parser.add_argument("--predictions_file", default="",
                        help=("File used to save predictions. If not set, predictions are printed "
                            "on the standard output."))
    parser.add_argument("--log_prediction_time", default=False, action="store_true",
                        help="Logs some prediction time metrics.")
    parser.add_argument("--checkpoint_path", default=None,
                        help=("Checkpoint or directory to use for inference or export "
                            "(when a directory is set, the latest checkpoint is used).")) 
    
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
                        help="Logs verbosity.")    


    args = parser.parse_args()

    tf.logging.set_verbosity(getattr(tf.logging, args.log_level))

    config = load_config(args.config)
    if args.run_dir:
        config["model_dir"] = os.path.join(args.run_dir, config["model_dir"])
    if args.data_dir:
        config["data"] = prefix_paths(args.data_dir, config["data"]) 

    print(config)
    model_config = config['model']
    data_config = config['data'] 

    
    data_kwargs = {}
    if 'others' in data_config:
        for k in data_config['others']:
            data_kwargs[k] = data_config['others'][k]    

    data_init_params = {}

    data_layer = data_config['data_layer']
    if not data_layer is None:
        cls_data_layer = get_data_layer(data_layer)
    else:
        if data_config['features']['count'] > 1:
            cls_data_layer = MultiInputTextClassificationDataLayer
            data_kwargs['feature_names'] = data_config['features']['names']
            for feature in data_config['features']['names']:
                data_init_params['%s_vocab' % feature] = data_config['vocabs'][feature]
                data_init_params['%s_unk_id' % feature] = data_config['unk_ids'][feature]

        else:
            cls_data_layer = TextClassificationDataLayer
            data_init_params['vocab'] = data_config['vocabs']['input']
            data_init_params['unk_id'] = data_config['unk_id']
    
    data_init_params.update(data_config['init_params'])
    

    model_creator = get_model(model_config['model_type'])
    model_params = model_config['model_params']

    vocbas = {}

    for k in data_config['vocabs']:
        vocsb_source = data_config['vocabs'][k]
        if not isinstance(vocsb_source, list):
            vocbas[k] = load_vocab(vocsb_source)
        else:
            vocbas[k] = vocsb_source, len(vocsb_source)
    
    embeddings_specs = []
    for k in data_config['embeddings']:
        if k in vocbas:
            
            vocab = vocbas[k][0]
            print(len(vocab))
            if 'path' in data_config['embeddings'][k] and not data_config['embeddings'][k] is None:
                pretrained_weights = load_embedding(data_config['embeddings'][k]['path'],
                                            vocab,
                                            with_header=data_config['embeddings'][k]['header'],
                                            dim=data_config['embeddings'][k]['embedding_size'])
                
                emb_spec = EmbeddingSpec(
                    name=k,
                    pretrained_weights=pretrained_weights,
                    trainable=data_config['embeddings'][k].get("trainable", False)
                )
                
            else:
                emb_spec = EmbeddingSpec(
                    name=k,
                    vocab_size=vocbas[k][1],
                    embedding_size=data_config['embeddings'][k]['embedding_size'],
                    trainable=True
                )   
            embeddings_specs.append(emb_spec)             
        else:
            raise RuntimeError("Got an embedding %s without vocabulary. Each embbeding must be linked to a vocabulary." % k)    

    if len(embeddings_specs) == 1:
        embeddings_specs = embeddings_specs[0]

    model_params['embedding_specs'] = embeddings_specs

    print(model_params)

    if args.run == 'train' or args.run == 'train_and_eval':
        
        train_config = config['train']

        train_features_source = data_config['train']['features_source']
        train_labels_source = data_config['train']['labels_source']

        _kwargs = copy.deepcopy(data_kwargs)

        _kwargs['batch_size'] = train_config['batch_size']
        _kwargs['epochs'] = train_config.get('epochs')

        train_data_layer = cls_data_layer(train_features_source,
                    train_labels_source,
                    data_init_params,
                    **_kwargs)

        if args.run == 'train_and_eval':
            
            dev_features_source = data_config['dev']['features_source']
            dev_labels_source = data_config['dev']['labels_source']

            _kwargs = copy.deepcopy(data_kwargs)

            _kwargs['batch_size'] = train_config.get('dev_batch_size', 100)
            

            dev_data_layer = cls_data_layer(dev_features_source,
                        dev_labels_source,
                        data_init_params,
                        **_kwargs) 

            model_params['dev_data_layer'] = dev_data_layer
        
        model_params = merge_dict(model_params, train_config)
        
        runner = Runner(model_creator, model_params)  
        runner.train(train_data_layer, epochs=train_config.get('epochs', 1))

    elif args.run == 'eval':
        pass
    elif args.run == 'predict':
        pass

if __name__ == "__main__":
    main()