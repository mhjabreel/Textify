import tensorflow as tf
from collections import Sequence


def concat_reducer_fn(inputs, axis=-1):

    return tf.concat(inputs, axis)


def add_reducer_fn(inputs):

    return tf.add_n(inputs)


def zip_and_reduce(x, y, reduce_fn=concat_reducer_fn):
    """Zips :obj:`x` with :obj:`y` and reduces all elements."""
    if tf.contrib.framework.nest.is_sequence(x):
        tf.contrib.framework.nest.assert_same_structure(x, y)

        x_flat = tf.contrib.framework.nest.flatten(x)
        y_flat = tf.contrib.framework.nest.flatten(y)

        flat = []
        for x_i, y_i in zip(x_flat, y_flat):
            flat.append(reduce_fn([x_i, y_i]))

        return tf.contrib.framework.nest.pack_sequence_as(x, flat)
    else:
        return reduce_fn([x, y])


def last_encoding_from_state(state):
    if isinstance(state, Sequence):
        state = state[-1]
    if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
        return state.h
    return state