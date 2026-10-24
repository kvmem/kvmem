#!/usr/bin/env bash
# Configure + build kvmem tests, libllama, and llama-kvmem-server.
# Default: Blackwell sm_120a. Override with ARCH=89 for Ada (RTX 4060 Ti), ARCH=90 for Hopper.
#   Example: ARCH=89 bash scripts/build-cuda.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- nvcc detection ---
if [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
    export PATH="$CUDA_HOME/bin:${PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-$CUDA_HOME/bin/nvcc}"
fi
# Fallback to developer-specific CU13 env var, then system nvcc
CU13="${CU13:-/home/leye/kvmem_qw3/.cu13-env}"
if [[ -z "${CMAKE_CUDA_COMPILER:-}" && -x "$CU13/bin/nvcc" ]]; then
    export PATH="$CU13/bin:${PATH:-}"
    export LD_LIBRARY_PATH="$CU13/lib:${LD_LIBRARY_PATH:-}"
    export CUDA_HOME="$CU13"
    CMAKE_CUDA_COMPILER="$CU13/bin/nvcc"
fi
CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-nvcc}"

# --- architecture selection ---
# ARCH env var shortcut: 89 → Ada, 90 → Hopper, 120 → Blackwell
ARCH="${ARCH:-}"
if [[ -n "$ARCH" && -z "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    case "$ARCH" in
        89)  CMAKE_CUDA_ARCHITECTURES="89" ;;
        90)  CMAKE_CUDA_ARCHITECTURES="90" ;;
        120) CMAKE_CUDA_ARCHITECTURES="120a-real" ;;
        *)   CMAKE_CUDA_ARCHITECTURES="$ARCH" ;;
    esac
fi
ARCH_LABEL="${CMAKE_CUDA_ARCHITECTURES:-120a-real}"

CMAKE="${CMAKE:-$ROOT/.venv/bin/cmake}"
if [[ ! -x "$CMAKE" ]]; then
    CMAKE="$(command -v cmake)"
fi

BUILD="${BUILD_DIR:-$ROOT/build}"
TYPE="${CMAKE_BUILD_TYPE:-Release}"

echo "=== nvcc ==="
"$CMAKE_CUDA_COMPILER" --version
echo "=== architecture: $ARCH_LABEL ==="

"$CMAKE" -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE="$TYPE" \
    -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH_LABEL" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DKVMEM_BUILD_LLAMA=ON \
    -DLLAMA_KVMEM=ON \
    -DLLAMA_KVMEM_ROOT="$ROOT"

"$CMAKE" --build "$BUILD" -j"${NPROC:-$(nproc)}"
echo "=== binaries under $BUILD/bin ==="
ls -lh "$BUILD/bin/" | grep -E "llama|kvmem" || true
echo "=== build complete (arch=$ARCH_LABEL) ==="
