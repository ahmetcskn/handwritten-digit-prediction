"""Run with python -m unittest discover -s tests -v after creating the model."""
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf


class ModelTest(unittest.TestCase):
    def test_model_output_shape(self):
        path = Path(__file__).resolve().parents[1] / 'digit_classifier.h5'
        self.assertTrue(path.exists(), 'Run python train_model.py first')
        model = tf.keras.models.load_model(path, compile=False)
        output = model.predict(np.zeros((1, 28, 28, 1), dtype='float32'), verbose=0)
        self.assertEqual(output.shape, (1, 10))
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output.sum(axis=1), 1, atol=1e-5)
