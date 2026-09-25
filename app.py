"""Single-user MNIST drawing and fine-tuning demo."""
import logging
import os
from pathlib import Path
import pickle
import time

from flask import Flask, jsonify, render_template, request, session
import numpy as np
import tensorflow as tf

from utils import calculate_training_stats, decode_image, plot_training_results, validate_image, normalize_digit

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'digit_classifier.h5'
TRAINING_DATA_FILE = BASE_DIR / 'training_data.pkl'
logging.basicConfig(filename=BASE_DIR / 'training.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

if not MODEL_PATH.exists():
    print('Model not found. Training for the first time. This may take a few minutes.', flush=True)
    from train_model import DigitClassification
    clf = DigitClassification()
    clf.train()
    clf.save_model(MODEL_PATH)
    del clf

# Recompile with a fresh optimizer for reliable fine-tuning after HDF5 loading.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Process-global, shared across sessions; restored only from the local pickle.
# This is a single-user demo, not designed for multi-user/production deployment.
training_data = []


def save_training_data():
    with open(TRAINING_DATA_FILE, 'wb') as stream:
        pickle.dump(training_data, stream)


def load_training_data():
    global training_data
    try:
        # Only load this application's own trusted local file.
        with open(TRAINING_DATA_FILE, 'rb') as stream:
            training_data = pickle.load(stream)
    except FileNotFoundError:
        training_data = []


load_training_data()


def page(message='', **kwargs):
    return render_template('index.html', current_digit=session.get('current_digit', 0),
                           training_active=session.get('training_active', False),
                           message=message, **kwargs)


@app.route('/')
def index():
    session['training_active'] = False
    return page()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_array = decode_image(request.form['image'])
        valid, message = validate_image(image_array)
        if not valid:
            return jsonify({'message': message}), 400
        image_array = normalize_digit(image_array)
        prediction = model.predict(image_array[None, ...], verbose=0)
        digit = int(np.argmax(prediction[0]))
        return jsonify(digit=digit, confidence=float(prediction[0][digit]))
    except Exception:
        logger.exception('Error processing prediction')
        return jsonify({'message': 'Error processing image.'}), 400


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        session['current_digit'] = 0
        session['training_active'] = True
        return page('Draw digit 0. Save your drawing when ready.')
    try:
        digit = int(request.form['digit'])
        if (not session.get('training_active') or not 0 <= digit <= 9
                or digit != session.get('current_digit')):
            return page('Start training and draw the requested digit.'), 400
        image_array = decode_image(request.form['image'])
        valid, message = validate_image(image_array)
        if not valid:
            return page(message), 400
        training_data.append((image_array, digit))
        save_training_data()
        if digit < 9:
            session['current_digit'] = digit + 1
            return page(f'Draw digit {digit + 1}. Save your drawing when ready.')

        start_time = time.monotonic()
        # Ten drawings can overfit badly. Measure generalization on all 10,000
        # MNIST test images before and after; never train on this test set.
        (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_test = X_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
        _, before = model.evaluate(X_test, y_test, verbose=0)
        # Normalize old and new saved drawings consistently with prediction.
        X_train = np.array([normalize_digit(item[0]) for item in training_data])
        y_train = np.array([item[1] for item in training_data])
        history = model.fit(X_train, y_train, epochs=5, verbose=2)
        _, after = model.evaluate(X_test, y_test, verbose=0)
        model.save(MODEL_PATH)
        plot_training_results(history, before, after, BASE_DIR / 'static/training_accuracy.png')
        stats = calculate_training_stats(training_data)
        elapsed = time.monotonic() - start_time
        logger.info('Retrained in %.2fs; stats=%s; MNIST before=%.4f after=%.4f', elapsed, stats, before, after)
        session['current_digit'] = 0
        session['training_active'] = False
        message = (f'Training completed in {elapsed:.2f}s using {stats["total_digits"]} drawings. '
                   f'MNIST test accuracy: {before:.2%} → {after:.2%}. '
                   'Fine-tuning on few drawings may reduce general accuracy.')
        return page(message, chart_version=time.time_ns())
    except Exception:
        logger.exception('Error during training')
        return page('Error during training. Please try again.'), 500


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False') == 'True',
            port=int(os.environ.get('PORT', '5000')))
