import os
from textify.models import SequenceClassifier
from textify.estimator import ClassifierBuilder, BinaryClassifierBuilder
from textify.runner import Runner
from textify.data.data_layer import DefaultDataLayer
from textify.utils.embedding_utils import EmbeddingSpec
from textify.layers.encoders import UnidirectionalRNNEncoder

from collections import defaultdict


encoder = UnidirectionalRNNEncoder(150)

model_crator = lambda _params, _scope=None: SequenceClassifier(_params, encoder, scope=_scope)


params = defaultdict(
    embedding_specs=EmbeddingSpec(
        name="emb1",
        embedding_size=300,
        vocab_size=90000
    ),
    learning_rate=0.0001,
    optimizer='adam',
    num_classes=2
)

vocab_file = os.path.join("data", "imdb.vocab")
features_file = os.path.join("data", "reviews.txt")
labels_file = os.path.join("data", "labels.txt")
         

init_params = {}
init_params['vocab'] = vocab_file
init_params['unk_id'] = 0
init_params['labels_vocab'] = ["pos", "neg"]

data_layer = DefaultDataLayer(features_file, labels_file, init_params, batch_size=10)

config = {'model_dir': 'tmp', 'train': {'train_steps': 2000}}

estimator_builder = ClassifierBuilder(model_crator, params)
runner = Runner(data_layer, estimator_builder, config)
runner.train()
