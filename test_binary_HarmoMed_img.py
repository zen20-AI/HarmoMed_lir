import unittest
import cv2
import numpy as np
from binary_HarmoMed_img import (
    detection, color_transfer, orange_region_boost, analyze_rgb_and_orange_shift,
    advanced_color_match, deltaE_cie2000, iterative_color_match, apply_color_to_full_image
)

class TestBinaryHarmoMedImg(unittest.TestCase):
    def setUp(self):
        # สร้าง dummy images สำหรับทดสอบ
        self.ref = np.ones((100, 100, 3), dtype=np.uint8) * 200
        self.target = np.ones((100, 100, 3), dtype=np.uint8) * 100
        self.full_img = np.ones((100, 100, 3), dtype=np.uint8) * 150
        cv2.imwrite('test_ref.jpg', self.ref)
        cv2.imwrite('test_target.jpg', self.target)

    def test_detection(self):
        result = detection('test_target.jpg')
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[2], 3)

    def test_color_transfer(self):
        result = color_transfer(self.ref, self.target)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.target.shape)

    def test_orange_region_boost(self):
        result = orange_region_boost(self.target)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.target.shape)

    def test_analyze_rgb_and_orange_shift(self):
        result = analyze_rgb_and_orange_shift(self.ref, self.target)
        self.assertIn('rgb_diff', result)
        self.assertIn('orange_shift', result)

    def test_advanced_color_match(self):
        result = advanced_color_match(self.ref, self.target)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.target.shape)

    def test_deltaE_cie2000(self):
        diff = deltaE_cie2000(self.ref, self.target)
        self.assertIsInstance(diff, float)

    def test_iterative_color_match(self):
        matched, history, orange_history = iterative_color_match(self.ref, self.target, max_iter=3)
        self.assertIsNotNone(matched)
        self.assertTrue(len(history) > 0)

    def test_apply_color_to_full_image(self):
        result = apply_color_to_full_image(self.ref, self.target, self.full_img)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.full_img.shape)

if __name__ == '__main__':
    unittest.main()
