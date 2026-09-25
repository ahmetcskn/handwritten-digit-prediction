import unittest

import numpy as np
from PIL import Image, ImageDraw

from utils import normalize_digit


class PreprocessingTest(unittest.TestCase):
    @staticmethod
    def four(offset=(0, 0)):
        image = Image.new('L', (28, 28))
        draw = ImageDraw.Draw(image)
        x, y = offset
        draw.line([(6+x, 4+y), (6+x, 12+y), (16+x, 12+y)], fill=255, width=2)
        draw.line([(13+x, 4+y), (13+x, 21+y)], fill=255, width=2)
        return np.asarray(image, dtype='float32')[..., None] / 255

    def test_offset_four_has_identical_normalized_input(self):
        centered = normalize_digit(self.four())
        moved = normalize_digit(self.four((5, 3)))
        np.testing.assert_array_equal(centered, moved)

    def test_normalized_ink_is_centered_and_padded(self):
        result = normalize_digit(self.four((5, 3)))
        self.assertEqual(result.shape, (28, 28, 1))
        self.assertEqual(result.dtype, np.float32)
        a = result[..., 0]
        yy, xx = np.indices(a.shape)
        self.assertAlmostEqual(float((xx*a).sum()/a.sum()), 13.5, delta=0.51)
        self.assertAlmostEqual(float((yy*a).sum()/a.sum()), 13.5, delta=0.51)
        ys, xs = np.where(a > 0.1)
        self.assertLessEqual(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1), 20)
        self.assertGreater(ys.min(), 0)
        self.assertLess(ys.max(), 27)

    def test_empty_input_remains_empty(self):
        result = normalize_digit(np.zeros((28, 28, 1), dtype='float32'))
        self.assertFalse(result.any())
