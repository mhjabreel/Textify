[![Build Status](https://travis-ci.org/mhjabreel/Textify.svg?branch=master)](https://travis-ci.org/mhjabreel/Textify) 
[![codecov](https://codecov.io/gh/mhjabreel/Textify/branch/master/graph/badge.svg)](https://codecov.io/gh/mhjabreel/Textify)
# Textify

Textify (comes from the prefix of "Text" and the suffix of "Classify") is a high-level framework using TensorFlow for text classification. While text classification is the main task of this toolkit, it is, also, has been designed to support different NLP tasks such as:

   * Text tagging.
   * Neural machine translation.


## Example:
Textify is used to implement the following models:
1. Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). NIPS 2015. You can find the implementation in this repo, [CharCNN](https://github.com/mhjabreel/CharCNN/).

## Documentation

Textify provides a framework consisting of two main API layers:

![Alt text](imgs/textify.png?raw=true "Textify framework.")

### Data Layer

The textify.data module enables you to build input pipelines from simple, reusable pieces. As Textify is desgined to mainly suuport text classification and other NLP tasks, we provide some predifined [textify.data.DataLayer]s. First, we describe the abstract class DataLayer.

   * textify.data.DataLayer: this default data layer is designed to build input pipeline for word-based text classification. 
      ```python
         __init__(features_source,
                  labels_source=None,
                  tokenizer=space_tokenizer,
                  init_params={},
                  **kwargs)
      ```
      * Parameters:	

         * features_source: A tf.string tensor containing one or more filenames. Each line in the file represents one sample. 
         * labels_source (Optional): If None [default], the dtat layer only works in the inference mode. Otherwise, the input pipline will be      prepared as labeled data pipeline. In this case the data layer is supposed to be used in train mode or eval mode. The labels_source must be text file(s), each line represents the class that the corresponding sample in the features_source file belongs to.
         * tokenizer: is a function takes as an input a text and tokenises it. The default tokenizer is [space_tokenizer]. It uses to split the text before send it to the vocabulary lookup table.
         * init_params: a dictionary containing initialization parameters of the data layer, e.g. the vocabulary file, maximum length, unknown id, labels, etc. 
         * **kwargs: provides extra name and value params, e.g. batch_size=100.
      
   All other data layers should subclass it. All subclasses should override ```_build_features_dataset ```, that builds input dataset, and ```_build_labels_dataset ```, building the labels dataset.

   Optionally, other data layers can override ```_get_features_padded_shapes ```, ```_get_labels_padded_shapes ```, ```_get_features_padding_values ```, and ```_get_labels_padding_values ```, if it is necessary to make batch padding.

   Example:

   ```python
      import tensorflow as tf
      from textify.data import DataLayer

      class DemoDataLayer(DataLayer):

         def _build_features_dataset(self, features_source):

            features_dataset =  tf.data.TextLineDataset(features_source) # read the data line by line
            features_dataset = features_dataset.map(lambda text: self._tokenizer(text)) # tokenize it.

            return features_dataset
         
         def _build_labels_dataset(self, labels_source):
            labels_dataset =  tf.data.TextLineDataset(labels_source) # read the data line by line
            labels_dataset = labels_dataset.map(tf.string_to_number) # tokenize it.

            return labels_dataset

      data_layer = DemoDataLayer('features.txt', 'labels.txt', padding=False, batch_size=2)
      next_batch = data_layer.input_fn(None)()

      with tf.Session() as sess:
         sess.run(tf.tables_initializer())
         output = sess.run(next_batch)

         print(output)
   ```

   If the file 'features.txt' contains the following tow lines:
   ```text
      Welcome to textify . <pad> <pad> <pad> <pad>
      This is a text data layer demo .
   ```

   And the file 'labels.txt' contains:
   ```text
   1
   2
   ```

   Then the output of the example is:
   ```bash
   (array([[b'This', b'is', b'a', b'text', b'data', b'layer', b'demo', b'.'],
       [b'Welcome', b'to', b'textify', b'.', b'<pad>', b'<pad>',
        b'<pad>', b'<pad>']], dtype=object), array([2., 1.], dtype=float32))
   ```