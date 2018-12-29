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

from textify.data import (DefaultDataLayer,
                        MultiInputDataLayer,
                        MultiOutputDataLayer,
                        CharacterBasedDataLayer)

import tensorflow as tf
import os
import io

class DataLayerTest(tf.test.TestCase):

    def testDefaultDataLayer(self):

        vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
        features_file = os.path.join(self.get_temp_dir(), "data.txt")
        labels_file = os.path.join(self.get_temp_dir(), "labels.txt")

        with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
            vocab.write(u"<unk>\n"
                        u"the\n"
                        u"world\n"
                        u"hello\n"
                        u"toto\n")
        with io.open(features_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"hello world !\n")

        with io.open(labels_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"positive\n")            

        init_params = {}
        init_params['vocab'] = vocab_file
        init_params['unk_id'] = 0
        init_params['labels_vocab'] = ["positive", "negative"]

        data_layer = DefaultDataLayer(features_file, labels_file, init_params, batch_size=1)

        features, labels = data_layer.input_fn()

        with self.test_session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            
            f, l = sess.run([features, labels])
            self.assertAllEqual([3], f["length"])
            self.assertAllEqual([[3, 2, 0]], f["ids"])
            self.assertAllEqual([0], l)    

    def testMultiInputDataLayer(self):

        vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
        features_file = os.path.join(self.get_temp_dir(), "data.txt")
        labels_file = os.path.join(self.get_temp_dir(), "labels.txt")

        with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
            vocab.write(u"<unk>\n"
                        u"the\n"
                        u"world\n"
                        u"hello\n"
                        u"toto\n")
        with io.open(features_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"hello world !\n")

        with io.open(labels_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"positive\n")            

        init_params = {}
        init_params['input_vocab'] = vocab_file
        init_params['input_unk_id'] = 0

        init_params['key_vocab'] = vocab_file
        init_params['key_unk_id'] = 0
        
        init_params['labels_vocab'] = ["positive", "negative"]

        data_layer = MultiInputDataLayer({'input': features_file, 'key': features_file}, labels_file, init_params, batch_size=1, feature_names=['input', 'key'])
        features, labels = data_layer.input_fn()

        with self.test_session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            f, l = sess.run([features, labels])
            self.assertEqual(set(f.keys()), {"length", "ids"})
            self.assertTrue(len(f['ids']) == 2)
            self.assertTrue('input' in f['ids'])
            self.assertTrue('key' in f['ids'])
            self.assertTrue('input' in f['length'])
            self.assertTrue('key' in f['length'])            
            self.assertAllEqual([3], f["length"]['input'])
            self.assertAllEqual([3], f["length"]['key'])
            self.assertAllEqual([[3, 2, 0]], f["ids"]['input'])
            self.assertAllEqual([[3, 2, 0]], f["ids"]['key'])            
            self.assertAllEqual([0], l)  

    def testMultiOutputDataLayer(self):

        vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
        features_file = os.path.join(self.get_temp_dir(), "data.txt")
        labels_file = os.path.join(self.get_temp_dir(), "labels.txt")

        with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
            vocab.write(u"<unk>\n"
                        u"the\n"
                        u"world\n"
                        u"hello\n"
                        u"toto\n")
        with io.open(features_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"hello world !\n")
            # fp.write(u"the world is toto\n")

        with io.open(labels_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"0, 1, 0\n")            
            # fp.write(u"1, 1, 0\n") 

        init_params = {}
        init_params['vocab'] = vocab_file
        init_params['unk_id'] = 0
        init_params['labels_vocab'] = ["positive", "negative"]

        data_layer = MultiOutputDataLayer(features_file, labels_file, init_params, num_labels=3, batch_size=1)

        features, labels = data_layer.input_fn()

        with self.test_session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            
            f, l = sess.run([features, labels])
            
            self.assertAllEqual([3], f["length"])
            self.assertAllEqual([[3, 2, 0]], f["ids"])
            self.assertAllEqual([[0.0, 1.0, 0.0]], l)    

    def testCharacterBasedDataLayer(self):
        vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
        features_file = os.path.join(self.get_temp_dir(), "data.txt")
        labels_file = os.path.join(self.get_temp_dir(), "labels.txt")

        with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
            vocab.write(u"<unk>\n"
                        u"▁\n"
                        u"w\n"
                        u"r\n"
                        u"d\n"
                        u"!\n"
                        u"h\n"
                        u"e\n"
                        u"l\n"
                        u"o\n")

        with io.open(features_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"hello world !\n")
            # fp.write(u"the world is toto\n")

        with io.open(labels_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"positive\n")            

        init_params = {}
        init_params['vocab'] = vocab_file
        init_params['unk_id'] = 0
        init_params['labels_vocab'] = ["positive", "negative"]            


        data_layer = CharacterBasedDataLayer(features_file, labels_file, init_params, num_labels=3, batch_size=1)

        features, labels = data_layer.input_fn()

        with self.test_session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            
            f, l = sess.run([features, labels])
            n = len("hello world !")
            self.assertAllEqual([n], f["length"])
            self.assertAllEqual([[6, 7, 8, 8, 9, 1, 2, 9, 3, 8, 4, 1, 5]], f["ids"])
            self.assertAllEqual([0], l)   

if __name__ == '__main__':
    tf.test.main()