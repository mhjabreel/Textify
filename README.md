[![Build Status](https://travis-ci.org/mhjabreel/Textify.svg?branch=master)](https://travis-ci.org/mhjabreel/Textify) 

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

The textify.data module enables you to build input pipelines from simple, reusable pieces. As Textify is desgined to mainly suuport text classification and other NLP tasks, we provide some predifined [textify.data.DataLayer]s:

   * textify.data.DefaultDataLayer: this default data layer is designed to build input pipeline for word-based text classification. 
      ```python
         __init__(features_source,
                  labels_source=None,
                  nit_params={},
                  **kwargs)
      ```
      Parameters:	

      * features_source: A tf.string tensor containing one or more filenames. Each line in the file represents one sample. 
      * labels_source (Optional): If None [default], the dtat layer only works in the inference mode. Otherwise, the input pipline will be prepared as labeled data pipeline. In this case the data layer is supposed to be used in train mode or eval mode. The labels_source must be text file(s), each line represents the class that the corresponding sample in the features_source file belongs to.