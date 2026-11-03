"""Reject the answer/embedding false positives found in the evidence audit."""
import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location(
    "cpu_image_evidence", Path(__file__).resolve().parents[1] / "tools/check_cpu_image_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class EvidenceTest(unittest.TestCase):
    def test_labeled_colors_allow_recorded_parallel_suffix(self):
        evidence.check_colors("LEFT=red; RIGHT=blue\nred\nblue", "red", "blue")

    def test_color_word_order_is_not_a_side_assignment(self):
        for text in ("red, blue", "LEFT=blue; RIGHT=red. red, blue", "LEFT=red; RIGHT=blue; LEFT=blue", "LEFT=red; RIGHT=blue; LEFT=green"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evidence.check_colors(text, "red", "blue")

    def test_encoder_valid_shape(self):
        evidence.check_embedding(struct.pack("<HH", 0x3F80, 0x4000), rows=1, hidden=2)

    def test_encoder_truncation_is_not_a_smaller_valid_tensor(self):
        with self.assertRaises(ValueError):
            evidence.check_embedding(struct.pack("<H", 0x3F80), rows=1, hidden=2)

    def test_encoder_rejects_nan_infinity_and_empty_signal(self):
        for word in (0x7F80, 0xFF80, 0x7FC0, 0):
            with self.subTest(word=word), self.assertRaises(ValueError):
                evidence.check_embedding(struct.pack("<H", word), rows=1, hidden=1)


if __name__ == "__main__":
    unittest.main()
