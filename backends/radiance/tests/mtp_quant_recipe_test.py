import fnmatch
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mtp_quant_recipe import draft_names, recipe
from compare_mtp_quant import judge


class RecipeTest(unittest.TestCase):
    def config(self, layers=64):
        return {"text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": layers,
                                "mtp_num_hidden_layers": 1}}

    def test_index_is_after_all_target_layers_not_last_target(self):
        for layers in (32, 64, 80):
            names = draft_names(self.config(layers))
            self.assertEqual(len(names), 6)
            for index in range(layers):
                self.assertFalse(any(fnmatch.fnmatch(f"blk.{index}.attn_qg.weight", p) for p in names))
            self.assertIn(f"blk.{layers}.attn_qg.weight", names)

    def test_unsupported_maps_fail_closed(self):
        for key, value in (("model_type", "other"), ("num_hidden_layers", True),
                           ("num_hidden_layers", "64"), ("mtp_num_hidden_layers", 2),
                           ("mtp_num_hidden_layers", 0), ("num_experts", 8)):
            cfg = self.config()
            cfg["text_config"][key] = value
            with self.assertRaises(ValueError):
                draft_names(cfg)

    def test_base_preserved_fc_head_and_norms_not_requantized(self):
        base = "mtp.draft_head.weight rtn codes=u2 zero=u8 group=128 scale=f16\n"
        for precision in ("mxfp4", "fp8"):
            result = recipe(self.config(), base, precision)
            self.assertTrue(result.endswith(base))
            rules = [x.split()[0] for x in result.splitlines() if x and not x.startswith("#")]
            self.assertEqual(rules[:-1], draft_names(self.config()))
            self.assertNotIn("mtp.fc.weight", rules)
            self.assertFalse(any("norm" in x for x in rules))

    def test_audit_requires_exact_changed_set_and_no_missing_extra(self):
        allowed = draft_names(self.config())
        report = {"missing": [], "candidate_extra": [], "equal_count": 1033,
                  "different": [{"name": n, "reference_bytes": 128, "candidate_bytes": 32}
                                for n in allowed]}
        self.assertTrue(judge(report, allowed)["pass"])
        self.assertEqual(judge(report, allowed)["changed_payload_bytes_saved"], 576)
        report["different"].append({"name": "blk.63.attn_qg.weight", "reference_bytes": 128, "candidate_bytes": 32})
        self.assertFalse(judge(report, allowed)["pass"])
        report["different"].pop()
        report["different"].pop()
        self.assertFalse(judge(report, allowed)["pass"])
        report["missing"] = ["output.weight"]
        self.assertFalse(judge(report, allowed)["pass"])


if __name__ == "__main__":
    unittest.main()
