#!/usr/bin/env bash
# Host-only paired renderer/capacity/cache-predicate comparison. Requires the
# prepared NInfer source and its built generic text/Jinja/core/runtime libraries.
set -euo pipefail
build=${1:?Usage: test_ninfer_query_origin_side_effects.sh BUILD CUDA_INCLUDE NVTX_INCLUDE}
cuda_include=${2:?CUDA include directory required}
nvtx_include=${3:?NVTX include directory required}
tests=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$tests/.." && pwd)
source="$repo/backends/ninfer"
out=$(mktemp -d /tmp/ninfer-query-origin-side-effects.XXXXXX)
python3 "$tests/ninfer_query_origin_source_audit.py" "$repo" "$out"
flags=(-O1 -std=c++20 -ffunction-sections -fdata-sections -I"$out"
  -I"$source/include" -I"$source/src" -I"$source/third_party"
  -I"$cuda_include" -I"$nvtx_include")
objects=()
for unit in prompt_layout tokenizer processor media_cache; do
  c++ "${flags[@]}" -c "$source/src/models/qwen3_5/frontend/$unit.cpp" -o "$out/$unit.o"
  objects+=("$out/$unit.o")
done
c++ "${flags[@]}" -c "$source/src/models/qwen3_5/frontend/chat_template.cpp" -o "$out/current.o"
c++ "${flags[@]}" -c "$out/baseline/src/models/qwen3_5/frontend/chat_template.cpp" -o "$out/baseline.o"
libs=("$build/src/text/libninfer_text.a" "$build/third_party/llama-jinja/libninfer_jinja.a"
  "$build/src/core/libninfer_core.a" "$build/src/runtime/libninfer_runtime_support.a")
c++ "${flags[@]}" -Wl,--gc-sections "$tests/ninfer_query_origin_side_effects.cpp" "$out/current.o" \
  "${objects[@]}" "${libs[@]}" -lcrypto -pthread -o "$out/current"
c++ "${flags[@]}" -DORIGINAL_QUERY_LOGIC -Wl,--gc-sections \
  "$tests/ninfer_query_origin_side_effects.cpp" "$out/baseline.o" "${objects[@]}" "${libs[@]}" \
  -lcrypto -pthread -o "$out/baseline-probe"
"$out/current" "$out/current-unmarked.json"
"$out/baseline-probe" "$out/baseline-unmarked.json"
python3 - "$out" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
a = json.loads((p/'current-unmarked.json').read_text())
b = json.loads((p/'baseline-unmarked.json').read_text())
assert a == b, 'unmarked snapshots differ from the #115 renderer baseline'
print(f'unmarked baseline/current snapshots: PASS ({len(a)})')
PY
printf 'Host-only side effects: PASS; artifacts=%s\n' "$out"
