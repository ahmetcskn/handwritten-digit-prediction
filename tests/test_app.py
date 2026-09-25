import base64
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import app as application


def drawing():
    image = Image.new('L', (28, 28))
    ImageDraw.Draw(image).line([(8, 5), (20, 5), (12, 24)], fill=255, width=4)
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


class AppTest(unittest.TestCase):
    def setUp(self):
        application.app.config.update(TESTING=True, SECRET_KEY='test-only')
        self.client = application.app.test_client()

    def test_prediction_confidence(self):
        response = self.client.post('/predict', data={'image': drawing()})
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json['digit'], range(10))
        self.assertGreaterEqual(response.json['confidence'], 0)
        self.assertLessEqual(response.json['confidence'], 1)

    def test_prediction_error_is_generic(self):
        with patch.object(application.model, 'predict', side_effect=RuntimeError('private detail')):
            response = self.client.post('/predict', data={'image': drawing()})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json['message'], 'Error processing image.')
        self.assertNotIn(b'private detail', response.data)

    def test_training_error_is_generic(self):
        self.client.get('/train')
        with patch.object(application, 'decode_image', side_effect=RuntimeError('private detail')):
            response = self.client.post('/train', data={'image': drawing(), 'digit': '0'})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(b'private detail', response.data)

    def test_training_sequence_and_persistence(self):
        self.client.get('/train')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_data.pkl'
            with patch.object(application, 'TRAINING_DATA_FILE', path), patch.object(application, 'training_data', []):
                response = self.client.post('/train', data={'image': drawing(), 'digit': '0'})
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'Draw digit 1', response.data)
                self.assertTrue(path.exists())
                self.assertEqual(application.training_data[0][0].shape, (28, 28, 1))
                response = self.client.post('/train', data={'image': drawing(), 'digit': '0'})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(len(application.training_data), 1)

    def test_full_retraining_shows_comparison(self):
        # Isolate both model and generated files from the user's saved demo.
        import numpy as np
        import tensorflow as tf
        model = tf.keras.models.clone_model(application.model)
        model.set_weights(application.model.get_weights())
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'static').mkdir()
            with (patch.object(application, 'BASE_DIR', root),
                  patch.object(application, 'MODEL_PATH', root / 'digit_classifier.h5'),
                  patch.object(application, 'TRAINING_DATA_FILE', root / 'training_data.pkl'),
                  patch.object(application, 'training_data', []),
                  patch.object(application, 'model', model)):
                self.client.get('/train')
                for digit in range(10):
                    sample = images[np.flatnonzero(labels == digit)[0]]
                    stream = io.BytesIO()
                    Image.fromarray(sample).save(stream, format='PNG')
                    image = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
                    response = self.client.post('/train', data={'digit': digit, 'image': image})
                    self.assertEqual(response.status_code, 200, response.data)
                self.assertIn(b'MNIST test accuracy:', response.data)
                self.assertIn(b'training_accuracy.png?v=', response.data)
                self.assertTrue((root / 'digit_classifier.h5').exists())
                self.assertTrue((root / 'static/training_accuracy.png').exists())
                with self.client.session_transaction() as session:
                    self.assertFalse(session['training_active'])

    def test_thin_one_is_accepted_but_empty_canvas_is_rejected(self):
        for blank in (False, True):
            image = Image.new('L', (300, 300))
            if not blank:
                ImageDraw.Draw(image).line([(155, 50), (140, 250)], fill=255, width=15)
            stream = io.BytesIO()
            image.save(stream, format='PNG')
            url = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
            response = self.client.post('/predict', data={'image': url})
            self.assertEqual(response.status_code, 400 if blank else 200)
