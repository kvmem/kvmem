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
import json
from concurrent.futures import ThreadPoolExecutor

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
        shapes = json.loads((ROOT / "tests/ninfer_context_budget_shapes.json").read_text())
        rows = []
        for case in shapes["cases"]:
            def spans(name):
                return "std::vector<Span>{" + ",".join("{" + str(b) + "," + str(n) + "}" for b,n in case[name]) + "}"
            for field, expected in (("instruction_prose",case["expected_mandatory_pages"]),
                                    ("published_instruction_block",case["expected_published_pages"])):
                rows.append("{const auto p=memory_context_pages(" + str(case["prompt_tokens"]) + "," +
                    spans(field) + "," + spans("active_users") + ",{},2,80,512," +
                    str(case["prompt_tokens"]+shapes["generation_extra_tokens"]) + ",36864);" +
                    "if(memory_window_mandatory_pages(" + str(case["prompt_tokens"]) + ",p,2,80).size()!=" +
                    str(expected) + ")return 1;}")
        shape_source = Path(temporary) / "budget-shapes.cpp"
        shape_source.write_text('#include "models/qwen3_5/program/storage/memory_context.h"\n'
            '#include <iostream>\nstruct Span{std::size_t begin,count;};\n'
            'using namespace ninfer::models::qwen3_5::detail;\nint main(){\n' + "\n".join(rows) +
            '\nstd::cout<<"anonymous budget shapes: PASS\\n";}\n')
        shape_test = Path(temporary) / "budget-shapes"
        subprocess.run(compiler + ["-std=c++20", "-O2", "-I" + str(backend / "src"),
            str(shape_source), "-o", str(shape_test)], check=True)
        subprocess.run([str(shape_test)], check=True)
        checkpoint = Path(temporary) / "checkpoint-test"
        subprocess.run(compiler + ["-std=c++20", "-O2", "-I" + str(backend / "src"),
            str(backend / "tests/models/qwen3_5/test_checkpoint_identity.cpp"),
            "-o", str(checkpoint)], check=True)
        subprocess.run([str(checkpoint)], check=True)

        # Compile the actual source-proof interpreter without CUDA or model data.
        # Its oracle compares tagged and ordinary rendering for mixed values,
        # nested mutations and separators; the checkpoint oracle checks immutable
        # query/count/pin identity independently of the mutable history directory.
        include = ["-I" + str(backend / name) for name in
            ("src", "include", "third_party", "third_party/llama-jinja")]
        sources = [backend / "third_party/llama-jinja/jinja" / (name + ".cpp")
            for name in ("json", "lexer", "parser", "runtime", "string", "value", "unicode")]
        sources += [backend / "src/text/jinja.cpp", backend / "src/text/unicode.cpp",
            backend / "tests/text/test_jinja_source_proof.cpp"]
        objects = [Path(temporary) / ("proof-" + str(i) + ".o") for i in range(len(sources))]
        def compile_unit(pair):
            source, obj = pair
            subprocess.run(compiler + ["-std=c++20", "-O2", "-DUTF8PROC_STATIC"] + include +
                ["-c", str(source), "-o", str(obj)], check=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(compile_unit, zip(sources, objects)))
        unicode_object = Path(temporary) / "utf8proc.o"
        c_compiler = shlex.split(os.environ.get("CC", "cc"))
        subprocess.run(c_compiler + ["-O2", "-DUTF8PROC_STATIC", "-c",
            str(backend / "third_party/utf8proc/utf8proc.c"), "-o", str(unicode_object)], check=True)
        proof = Path(temporary) / "source-proof-test"
        subprocess.run(compiler + [str(p) for p in objects] + [str(unicode_object), "-o", str(proof)], check=True)
        subprocess.run([str(proof)], check=True)


if __name__ == "__main__":
    main()
