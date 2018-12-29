import codecs
import os
from collections import OrderedDict
import tensorflow as tf
import numpy as np

UNK = "<unk>"
UNK_ID = 0

def load_vocab(vocab_file):
    vocab = OrderedDict()
    vocab_size = 0
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        for word in f:
            vocab[word.strip()] = vocab_size
            vocab_size += 1
    return vocab, vocab_size