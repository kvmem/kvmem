#!/bin/bash
# Build kvmem-bonsai for Ada Lovelace (sm_89) — e.g. RTX 4060 Ti
# Usage: bash build-sm89.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# --- nvcc detection ---
if [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
    export PATH="$CUDA_HOME/bin:${PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
else
    CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-nvcc}"
fi
CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-nvcc}"

echo "=== nvcc ==="
"$CMAKE_CUDA_COMPILER" --version

echo ""
echo "=== source sanity check ==="
test -f llama.cpp/src/llama-kvmem-factory.h && echo "kvmem adapter patch OK" || echo "FATAL: apply patches first"
test -f llama.cpp/tools/server/server-context.cpp && echo "server sources OK"

echo ""
echo "=== cmake configure (CMAKE_CUDA_ARCHITECTURES=89, Ada Lovelace) ==="
rm -rf build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DKVMEM_BUILD_LLAMA=ON \
    -DLLAMA_KVMEM=ON \
    -DLLAMA_KVMEM_ROOT="$ROOT"

echo ""
echo "=== cmake build ==="
cmake --build build -j"${NPROC:-$(nproc)}"

echo ""
echo "=== artifacts ==="
ls -lh build/bin/llama-kvmem-server
ls -lh build/bin/llama-kvmem-cli 2>/dev/null || true
find build -name "*.so*" -type f 2>/dev/null || echo "(no .so files)"

echo ""
echo "=== sm_89 marker check ==="
if strings build/bin/llama-kvmem-server 2>/dev/null | grep -q "sm_89"; then
    echo "✅ sm_89 build confirmed"
else
    echo "⚠️  sm_89 marker not found in binary (may still work, check cuobjdump)"
fi

echo ""
echo "=== ldd check ==="
ldd build/bin/llama-kvmem-server 2>/dev/null | grep -E "not found|cudart|cuda|cublas|ggml" || true
