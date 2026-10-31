import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from draft_control import resolve, verify_log
from inspect_rad import inspect


def container(path, names, metadata=()):
    """Small valid directory fixture; no real model payload or GPU needed."""
    header = struct.Struct("<II30Q")
    entry = struct.Struct("<3Q2I6q3iI5Q")
    strings = bytearray(b"\0")

    def string(value):
        offset = len(strings)
        strings.extend(value.encode() + b"\0")
        return offset

    h = [0] * 32
    h[:2] = [0x31444152, 2]
    h[3], h[4], h[5] = string("qwen35"), string("test"), string("fp8_e4m3")
    entries = []
    for name in names:
        values = [0] * 20
        values[0], values[4], values[5], values[11] = string(name), 1, 1, -1
        entries.append(entry.pack(*values))
    meta = []
    for key, value in metadata:
        key_offset = string(key)
        kind = 2 if isinstance(value, str) else 1
        meta.append(struct.pack("<QIIQ", key_offset, kind, 0,
                                string(value) if kind == 2 else value))
    h[8], h[9] = header.size, len(strings)
    h[10], h[11] = h[8] + h[9], len(meta)
    h[12], h[13] = h[10] + 24 * len(meta), len(entries)
    h[2] = h[12] + len(entries) * entry.size
    path.write_bytes(header.pack(*h) + strings + b"".join(meta) + b"".join(entries))


class DraftControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name) / "model.rad"

    def dflash(self, names=("dflash.fc.weight",), block=8):
        container(self.model, names, (("draft.architectures", "DFlash2Model"),
                                     ("draft.dflash_config.block_size", block),
                                     ("mtp_num_hidden_layers", 1)))

    def test_dflash_depth_and_nextn_without_weights(self):
        self.dflash()
        self.assertEqual(resolve(self.model, "dflash")["depth"], 7)
        # Retained source metadata cannot turn missing MTP tensors into weights.
        with self.assertRaisesRegex(ValueError, "no mtp"):
            resolve(self.model, "mtp")
        with self.assertRaisesRegex(ValueError, "trained block"):
            resolve(self.model, "dflash", 3)

    def test_mtp_uses_existing_native_path(self):
        container(self.model, ["mtp.fc.weight"], [("mtp_num_hidden_layers", 1)])
        self.assertEqual(resolve(self.model, "mtp")["depth"], 3)
        self.assertEqual(resolve(self.model, "mtp", 2)["depth"], 2)
        with self.assertRaisesRegex(ValueError, "positive depth"):
            resolve(self.model, "mtp", 0)

    def test_native_converter_string_integers(self):
        container(self.model, ["mtp.fc.weight"], [("mtp_num_hidden_layers", "1")])
        self.assertEqual(resolve(self.model, "mtp")["depth"], 3)
        self.dflash(block="8")
        self.assertEqual(resolve(self.model, "dflash")["depth"], 7)
        self.dflash(block="8.0")
        with self.assertRaisesRegex(ValueError, "integer"):
            resolve(self.model, "dflash")

    def test_both_drafters_cannot_silently_choose_dflash(self):
        self.dflash(("dflash.fc.weight", "mtp.fc.weight"))
        with self.assertRaisesRegex(ValueError, "prioritizes DFlash"):
            resolve(self.model, "mtp")

    def test_missing_declaration_and_invalid_block(self):
        container(self.model, ["mtp.fc.weight"])
        with self.assertRaisesRegex(ValueError, "no NextN"):
            resolve(self.model, "mtp")
        self.dflash(block=7)
        with self.assertRaisesRegex(ValueError, "block size"):
            resolve(self.model, "dflash")

    def test_off_really_sets_zero_and_auto_is_backwards_compatible(self):
        self.assertEqual(resolve(self.model, "off")["depth"], 0)
        with self.assertRaisesRegex(ValueError, "zero"):
            resolve(self.model, "off", 7)
        self.assertEqual(resolve(self.model, "auto")["depth"], 0)
        self.assertEqual(resolve(self.model, "auto", 7)["depth"], 7)

    def test_verify_actual_runtime_not_just_requested_name(self):
        container(self.model, ["mtp.fc.weight"], [("mtp_num_hidden_layers", 1)])
        selected = resolve(self.model, "mtp")
        self.assertTrue(verify_log(selected, "I speculation: 'mtp' proposes 3 token(s)")["pass"])
        self.assertFalse(verify_log(selected, "I speculation: 'dflash2' proposes 7 token(s)")["pass"])
        self.assertFalse(verify_log(selected, "I serving on 127.0.0.1")["pass"])
        self.assertFalse(verify_log(resolve(self.model, "off"), "I speculation: 'mtp' proposes 3")["pass"])
        self.assertTrue(verify_log(resolve(self.model, "off"), "I serving on 127.0.0.1")["pass"])

    def test_directory_metadata_and_truncation(self):
        self.dflash()
        data = inspect(self.model)
        self.assertEqual(data["entries_by_name_prefix"], {"dflash": 1})
        self.assertEqual(data["draft_metadata"]["draft.architectures"], "DFlash2Model")
        self.model.write_bytes(self.model.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "complete"):
            inspect(self.model)


if __name__ == "__main__":
    unittest.main()
