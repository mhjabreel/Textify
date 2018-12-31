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
from textify.utils import DynamicImporter
from textify.models import Model, SequenceClassifier

class DynamicImporterTest(unittest.TestCase):

    _tempdir = None
    @classmethod
    def get_temp_dir(cls):

        if not cls._tempdir:
            cls._tempdir = tempfile.mkdtemp()
        return cls._tempdir

    def testClassImportingByName(self):
        demo_file = os.path.join(self.get_temp_dir(), "demo.py")
        
        with io.open(demo_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"from textify.models import SequenceClassifier\n")   

        importer = DynamicImporter(demo_file)
        c = importer.get_class("SequenceClassifier")
        self.assertTrue(not c is None)
    

    def testClassImportingByType(self):
        demo_file = os.path.join(self.get_temp_dir(), "demo.py")
        
        with io.open(demo_file, encoding="utf-8", mode="w") as fp:
            fp.write(u"class Dummy:\n\tpass\n\nfrom textify.models import SequenceClassifier\n")   

        importer = DynamicImporter(demo_file)
        c = importer.get_first_class_of(Model)
        self.assertEqual(c, SequenceClassifier)
             

if __name__ == "__main__":
    unittest.main()