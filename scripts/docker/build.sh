#!/usr/bin/env bash
# Build the KVMem NInfer worker image.
#
#   scripts/docker/build.sh                       # tag kvmem/ninfer:<VERSION>-<arch>
#   TAG=kvmem/ninfer:dev CUDA_ARCH=86 scripts/docker/build.sh
#
# The engine submodule must be prepared (KVMem integration patches applied) before the image
# build; this script runs scripts/prepare-backends.py --backend ninfer, which is idempotent.
# Requires git, python3 and docker on the host, and the NVIDIA Container Toolkit only to run.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CUDA_VERSION="${CUDA_VERSION:-13.5.0}"
UBUNTU="${UBUNTU:-ubuntu26.04}"
CUDA_ARCH="${CUDA_ARCH:-}"   # empty: derive from the GPUs of this host
JOBS="${JOBS:-}"
CUBLAS_PACKAGE="${CUBLAS_PACKAGE:-}"

# The Dockerfile has no default target: an image built here should run on the cards this host
# has, so the tiers come from the driver unless CUDA_ARCH pins them.
arch_tier() {
    case "$1" in
        7.0) echo 70 ;;
        7.5) echo 75 ;;
        8.0) echo 80 ;;
        8.6) echo 86 ;;
        8.9) echo 89 ;;
        12.0) echo 120a ;;
        *) return 1 ;;
    esac
}

if [ -z "$CUDA_ARCH" ]; then
    caps=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
    fi
    if [ -z "$caps" ]; then
        printf 'No GPU detected; pass CUDA_ARCH=70|75|80|86|89|120a, or several as "86;89".\n' >&2
        exit 2
    fi
    tiers=""
    for cap in $caps; do
        tier="$(arch_tier "$cap")" || {
            printf 'Unsupported compute capability %s (supported: 7.0, 7.5, 8.0, 8.6, 8.9, 12.0)\n' "$cap" >&2
            exit 2
        }
        case ";$tiers;" in *";$tier;"*) ;; *) tiers="${tiers:+$tiers;}$tier" ;; esac
    done
    CUDA_ARCH="$tiers"
    printf '==> GPU tiers on this host: %s\n' "$CUDA_ARCH"
fi

arch_list='^(70|75|80|86|89|120a)(;(70|75|80|86|89|120a))*$'
if ! [[ "$CUDA_ARCH" =~ $arch_list ]]; then
    printf 'CUDA_ARCH must be 70, 75, 80, 86, 89 or 120a (or a semicolon-separated list); got %s\n' "$CUDA_ARCH" >&2
    exit 2
fi
if [[ -n "$JOBS" ]] && ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'JOBS must be a positive integer\n' >&2
    exit 2
fi
if ! [[ "$CUDA_VERSION" =~ ^([1-9][0-9]*)\.([0-9]|[1-9][0-9]*)\.[0-9]+$ ]]; then
    printf 'CUDA_VERSION must be a CUDA image release such as 13.5.0\n' >&2
    exit 2
fi
cuda_major="${BASH_REMATCH[1]}"
cuda_minor="${BASH_REMATCH[2]}"
IFS=';' read -r -a arch_elements <<< "$CUDA_ARCH"

# The engine's own configure step rejects these combinations; fail before uploading the context.
for element in "${arch_elements[@]}"; do
    if [ "$element" = 70 ] && [ "$cuda_major" -ge 13 ]; then
        printf 'sm_70 (Tesla V100) needs CUDA 12.x; %s cannot compile it\n' "$CUDA_VERSION" >&2
        exit 2
    fi
    if [ "$element" = 120a ] && (( cuda_major < 13 || (cuda_major == 13 && cuda_minor < 1) )); then
        printf 'sm_120a needs CUDA 13.1 or newer, not %s\n' "$CUDA_VERSION" >&2
        exit 2
    fi
done

TAG_ARCH="$(printf '%s' "$CUDA_ARCH" | tr ';' '-')"
TAG="${TAG:-kvmem/ninfer:$(tr -d '[:space:]' < VERSION)-${TAG_ARCH}}"

command -v docker >/dev/null || { printf 'docker is required\n' >&2; exit 2; }
# The build uses a BuildKit cache mount for ccache, so the classic builder is not enough.
if ! docker buildx version >/dev/null 2>&1; then
    cat >&2 <<'EOF'
BuildKit is required: the image build keeps its ccache in a BuildKit cache mount.
  Arch/CachyOS: sudo pacman -S --needed docker-buildx
  Debian/Ubuntu: sudo apt-get install -y docker-buildx-plugin
EOF
    exit 2
fi
command -v python3 >/dev/null || { printf 'python3 is required\n' >&2; exit 2; }

# cuBLAS ships as a separate package in the CUDA runtime image; the package name follows the
# CUDA minor release (libcublas-13-5 for 13.5).
if [[ -z "$CUBLAS_PACKAGE" ]]; then
    CUBLAS_PACKAGE="libcublas-${cuda_major}-${cuda_minor}"
fi

# The context carries no .git, so record the exact source state in the image.
revision="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    revision="${revision}-dirty"
fi

printf '==> preparing backends/ninfer\n'
python3 scripts/prepare-backends.py --backend ninfer
python3 scripts/prepare-backends.py --backend ninfer --check

build_args=(
    --build-arg "CUDA_VERSION=$CUDA_VERSION"
    --build-arg "UBUNTU=$UBUNTU"
    --build-arg "CUDA_ARCH=$CUDA_ARCH"
    --build-arg "CUBLAS_PACKAGE=$CUBLAS_PACKAGE"
    --build-arg "KV_SRC_COMMIT=$revision"
)
[[ -n "$JOBS" ]] && build_args+=(--build-arg "JOBS=$JOBS")

printf '==> docker build -t %s (CUDA %s, arch %s, source %s)\n' "$TAG" "$CUDA_VERSION" "$CUDA_ARCH" "$revision"
docker build -f Dockerfile "${build_args[@]}" -t "$TAG" .

printf '\nimage: %s\n' "$TAG"
docker image inspect --format 'size: {{.Size}} bytes' "$TAG"
