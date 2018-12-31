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

import io
import os
import unittest
import tempfile
from textify.utils import Configuration

class ConfigureTest(unittest.TestCase):

    _tempdir = None
    @classmethod
    def get_temp_dir(cls):

        if not cls._tempdir:
            cls._tempdir = tempfile.mkdtemp()
        return cls._tempdir

    def testConfig(self):
        demo_file = os.path.join(self.get_temp_dir(), "demo.yml")
        
        with io.open(demo_file, encoding="utf-8", mode="w") as fp:
            fp.write(
                u"model:\n"
                 "    name: SeqClassifier\n"
                 "    params:\n"
                 "        num_layers: 2\n"
            
            )   
        print(demo_file)
        config = Configuration([demo_file])
        layers = config['model']['params']['num_layers']
        self.assertEqual(layers, 2)
        config['model']['params']['num_layers'] = 3
        layers = config['model']['params']['num_layers']
        self.assertEqual(layers, 3)

if __name__ == "__main__":
    unittest.main()