#!/usr/bin/env python3
"""Run the prepared NInfer backend's model-free KVMem task/window contract.

This checks selection semantics, not CUDA execution or model-answer quality.
Frontend/media and real-model checks remain in NInfer's normal test bundle.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    backend = ROOT / "backends/ninfer"
    test = backend / "tests/models/qwen3_5/test_memory_context.cpp"
    if not test.is_file():
        raise SystemExit("Prepare the pinned backend first: python scripts/prepare-backends.py --backend ninfer")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="kvmem-ninfer-context-") as temporary:
        executable = Path(temporary) / "context-test"
        subprocess.run(compiler + ["-std=c++20", "-O2", "-ffunction-sections", "-fdata-sections",
            "-Wl,--gc-sections", "-I" + str(backend / "src"), "-I" + str(ROOT / "kvmem/include"),
            str(test), str(ROOT / "kvmem/src/host/kvmem_store.cpp"),
            str(ROOT / "kvmem/src/host/memory_contract.cpp"), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
