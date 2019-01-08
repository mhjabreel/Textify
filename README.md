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
![Alt text](imgs\textif.png?raw=true "Textify framework.")

### Data Layer

The textify.data module enables you to build input pipelines from simple, reusable pieces. As Textify is desgined to mainly suuport text classification and other NLP tasks, we provide some predifined [textify.data.DataLayer]s. 


