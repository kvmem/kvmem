import sys
import tempfile
import unittest
from unittest.mock import patch
import errno
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from linux_probe_metrics import discover_amd_cards, sample_amd_cards, read_counter


class MetricsTest(unittest.TestCase):
    def test_discovery_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, vendor in [('card2', '0x1002'), ('card0', '0x10de'), ('card2-DP-1', '0x1002')]:
                dev = root / name / 'device'
                dev.mkdir(parents=True)
                (dev / 'vendor').write_text(vendor)
                (dev / 'mem_info_vram_total').write_text('17179869184')
                (dev / 'mem_info_vram_used').write_text('123456')
            cards = discover_amd_cards(root)
            self.assertEqual(cards, [root / 'card2' / 'device'])
            values = sample_amd_cards(cards)[str(cards[0])]
            self.assertEqual(values['mem_info_vram_used'], 123456)
            self.assertEqual(values['mem_info_vram_total'], 16 * 1024 ** 3)

    def test_missing_counters_are_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                sample_amd_cards([Path(tmp)])

    def test_optional_busy_is_unavailable_not_zero(self):
        with patch.object(Path, 'read_text', side_effect=OSError(errno.EBUSY, 'busy')):
            value, error = read_counter(Path('/fake/gpu_busy_percent'))
            self.assertIsNone(value)
            self.assertIn('busy', error)

    def test_required_busy_retries_then_fails(self):
        with patch.object(Path, 'read_text', side_effect=OSError(errno.EBUSY, 'busy')) as read, \
                patch('linux_probe_metrics.time.sleep'):
            with self.assertRaises(RuntimeError):
                read_counter(Path('/fake/mem_info_vram_used'), required=True)
            self.assertEqual(read.call_count, 3)

    def test_required_transient_busy_recovers(self):
        with patch.object(Path, 'read_text', side_effect=[OSError(errno.EBUSY, 'busy'), '123']), \
                patch('linux_probe_metrics.time.sleep'):
            self.assertEqual(read_counter(Path('/fake/mem_info_vram_used'), True), (123, None))


if __name__ == '__main__':
    unittest.main()
