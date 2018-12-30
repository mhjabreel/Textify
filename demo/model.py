from textify.models import SequenceClassifier
from textify.layers.encoders import UnidirectionalRNNEncoder

class Model(SequenceClassifier):

    def __init__(self, params, scope=None):

        encoder = UnidirectionalRNNEncoder(150)
        super(Model, self).__init__(params, encoder, scope=scope)