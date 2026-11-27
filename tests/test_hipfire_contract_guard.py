import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hipfire_contract", ROOT / "tests/test_hipfire_contract.py")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


class HipfireContractGuardTest(unittest.TestCase):
    def setUp(self):
        self.header = contract.HEADER.read_text(encoding="utf-8")
        self.source = contract.SOURCE.read_text(encoding="utf-8")

    def test_real_header_and_shim_include_closure_pass(self):
        self.assertEqual(contract.header_guard_errors(self.header, self.source), [])

    def test_removing_layout_assertions_is_rejected(self):
        mutated = "\n".join(
            line for line in self.header.splitlines() if "static_assert(" not in line
        )
        errors = contract.header_guard_errors(mutated, self.source)
        self.assertTrue(any("sizeof(hk_raw_store_config)" in error for error in errors))
        self.assertTrue(any("offsetof(hk_kv_store_config" in error for error in errors))

    def test_engine_include_is_rejected(self):
        errors = contract.header_guard_errors(
            self.header, self.source + '\n#include "llama.h"\n'
        )
        self.assertTrue(any("llama.h" in error for error in errors))

    def test_missing_abi_entrypoint_guard_is_rejected(self):
        text = "ensure_abi()?;\n" * 4
        self.assertTrue(contract.rust_entrypoint_errors(text))
        self.assertEqual(contract.rust_entrypoint_errors(text + "ensure_abi()?;\n"), [])


if __name__ == "__main__":
    unittest.main()
