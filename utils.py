"""Image preprocessing and reporting shared by the demo routes."""
import base64
import io
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def decode_image(data_url):
    data = base64.b64decode(data_url.split(',', 1)[1], validate=True)
    with Image.open(io.BytesIO(data)) as image:
        # Composite transparent canvas pixels onto MNIST's black background.
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'black')
        image = Image.alpha_composite(background, rgba).convert('L').resize((28, 28))
        return np.asarray(image, dtype='float32').reshape(28, 28, 1) / 255.0


def normalize_digit(image_array):
    """Fit the ink into a 20px box and center its mass, matching MNIST layout.

    Use after validation, for both prediction and saved fine-tuning samples.
    Padding and position should not change the meaning of a drawn digit.
    """
    pixels = np.clip(np.asarray(image_array).reshape(28, 28) * 255, 0, 255).astype('uint8')
    ys, xs = np.where(pixels > 25)
    if not len(xs):
        return np.zeros((28, 28, 1), dtype='float32')
    image = Image.fromarray(pixels)
    ink = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = 20 / max(ink.size)
    size = tuple(max(1, round(length * scale)) for length in ink.size)
    ink = ink.resize(size, Image.Resampling.LANCZOS)
    centered = Image.new('L', (28, 28))
    centered.paste(ink, ((28 - ink.width) // 2, (28 - ink.height) // 2))
    weights = np.asarray(centered, dtype='float32')
    yy, xx = np.indices(weights.shape)
    offset = (round(13.5 - float((xx * weights).sum() / weights.sum())),
              round(13.5 - float((yy * weights).sum() / weights.sum())))
    shifted = Image.new('L', (28, 28))
    shifted.paste(centered, offset)
    return np.asarray(shifted, dtype='float32').reshape(28, 28, 1) / 255.0


def validate_image(image_array):
    total_pixels = np.sum(image_array)
    logger.debug('Validating image, total pixels: %s', total_pixels)
    # Keep thin but valid digits (especially 1) while rejecting empty canvases.
    if total_pixels < 20:
        return False, 'Image is too empty.'
    pixel_density = np.mean(image_array > 0.1)
    logger.debug('Pixel density: %s', pixel_density)
    if pixel_density < 0.03:
        return False, 'Image density too low.'
    return True, 'Image validated.'


def calculate_training_stats(training_data):
    return {'total_digits': len(training_data),
            'unique_digits': len({item[1] for item in training_data})}


def plot_training_results(history, before, after, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'])
    axes[0].set(xlabel='Epoch', ylabel='Accuracy', title='Drawing training accuracy', ylim=(0, 1))
    axes[1].bar(['Before', 'After'], [before, after], color=['#a7b2a2', '#365f4d'])
    axes[1].set(title='MNIST test accuracy', ylim=(0, 1))
    for i, value in enumerate([before, after]):
        axes[1].text(i, value / 2, f'{value:.2%}', ha='center')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
