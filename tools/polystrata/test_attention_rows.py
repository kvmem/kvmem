"""Compile and execute actual Qwen35 row dispatch and graph-key code, without a GPU.

Two small host test executables only; no engine or GPU kernel compilation.
--emit-dir writes the translation unit for an independently inspected Ninja build.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def function(text, signature):
    start = text.index(signature)
    body = text.index("{", start)
    depth, at = 1, body + 1
    while depth:
        depth += (text[at] == "{") - (text[at] == "}")
        at += 1
    return text[start:at]


def generate(source, output):
    qwen = (source / "src/poly/qwen35.cpp").read_text(encoding="utf-8")
    engine = (source / "src/poly/engine.cpp").read_text(encoding="utf-8")
    step = function(engine, "void Engine::step(const int* seqs, const int32_t* tokens, int n, const int32_t* guesses)")
    key_start = step.index("const int64_t keep = pos_;")
    key_end = step.index("rows_of_ = rows.data();", key_start)
    code = Path(__file__).with_name("attention_rows_harness.cpp.in").read_text(encoding="utf-8")
    for marker, value in {
        "@ATTN_ROWS@": function(qwen, "void Qwen35::attn_rows(Block& B, int n)"),
        "@HEAD_KEY@": function(qwen, "int Qwen35::head_key(int il, int n) const"),
        "@ROOM_METHODS@": function(engine, "int Engine::room(int n) const") + "\n" +
                          function(engine, "int Engine::room_key(int n) const"),
        "@GRAPH_KEY@": step[key_start:key_end],
    }.items():
        assert code.count(marker) == 1, marker
        code = code.replace(marker, value)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "attention_rows.cpp"
    path.write_text(code, encoding="utf-8", newline="\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(os.environ.get("POLYSTRATA_SOURCE", ROOT / "backends/polystrata")))
    parser.add_argument("--emit-dir", type=Path)
    args = parser.parse_args()
    if args.emit_dir:
        print(generate(args.source, args.emit_dir))
        return
    compiler = shutil.which(os.environ.get("CXX", "cl" if os.name == "nt" else "c++"))
    if not compiler:
        parser.error("C++ compiler not found; run in a developer shell or set CXX")
    msvc = Path(compiler).name.lower() in ("cl", "cl.exe", "clang-cl", "clang-cl.exe")
    with tempfile.TemporaryDirectory(prefix="poly-attention-rows-") as directory:
        output = Path(directory)
        source = generate(args.source, output)
        for memory in (False, True):
            binary = output / ("kvmem" if memory else "native")
            if os.name == "nt":
                binary = binary.with_suffix(".exe")
            if msvc:
                command = [compiler, "/nologo", "/std:c++17", "/EHsc", "/O2", "/W4", "/Fe" + str(binary),
                           "/Fo" + str(binary.with_suffix(".obj")), str(source)]
            else:
                command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(source), "-o", str(binary)]
            if memory:
                command += ["/DSTRATA_KVMEM" if msvc else "-DSTRATA_KVMEM"]
            subprocess.run(command, cwd=output, check=True)
            subprocess.run([str(binary)], cwd=output, check=True)


if __name__ == "__main__":
    main()
