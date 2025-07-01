import unittest
import numpy as np
from neural_network.loaders.rses_loader import RSESLoader
import tempfile
import os


class TestRSESLoader(unittest.TestCase):

    def test_valid_file_load(self):
        content = "x x d\n0 1 0\n1 0 1\n"
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp.write(content)
            tmp_name = tmp.name

        X, y = RSESLoader.load(tmp_name)
        os.remove(tmp_name)

        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y[1], 1.0)

    def test_non_numeric_handling(self):
        content = "x x d\n1 0 x\n"
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp.write(content)
            tmp_name = tmp.name

        X, y = RSESLoader.load(tmp_name)
        os.remove(tmp_name)

        self.assertIsNone(X)
        self.assertIsNone(y)
