#!/usr/bin/env bash
# Portable same-budget pair; launch only AFTER real GPU native tests pass.
# No downloads, installation, cloud start or automatic capacity/33 run.
set -euo pipefail
if (( $# != 9 )); then
  echo 'Usage: page-recycle-pair.sh BASE_EXE NEW_EXE RAD_HOME MODEL PLAN ROCM_SDK OUTPUT_DIR BUDGET linux|wsl' >&2
  exit 2
fi
base=$1 candidate=$2 rad_home=$3 model=$4 plan=$5 sdk=$6 out=$7 budget=$8 platform=$9
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../tools" && pwd)
case "$budget" in *[!0-9]*|'') exit 2;; esac
(( budget >= 256 && budget <= 36864 && budget % 128 == 0 )) || exit 2
test -x "$base" && test -x "$candidate" && test -f "$model" && test -f "$plan"
test -d "$rad_home" && test -d "$sdk/lib"
test ! -e "$out"
case "$platform" in
  linux) counters=(--linux-gpu-counters);;
  wsl) counters=(--windows-gpu-counters); export HSA_ENABLE_DXG_DETECTION=1;;
  *) exit 2;;
esac
# One shared lane for these paired runs. Other GPU users must be checked before launch.
exec 9>/tmp/kvmem-radiance-page-recycle.lock
flock -n 9 || exit 90
mkdir -p -- "$out"
exec > >(tee "$out/driver.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" >"$out/driver.exit"' EXIT
export ROCM_PATH="$sdk" LD_LIBRARY_PATH="$sdk/lib:$sdk/llvm/lib" OMP_NUM_THREADS=2
unset RADIANCE_NO_DRAFT
args=(--home "$rad_home" --sdk "$sdk" --model "$model" --plan "$plan"
  --embedding-placement ram --gpu-headroom-mib 512 --min-system-available-mib 2048
  --max-process-swap-mib 64 --observe-statistics --batch 128 --context 262144
  --kv-cache-dtype fp8 --replay-tokens 1024 --host-archive-mib 9216
  --sink-tokens 128 --recent-tokens 768 --drafter mtp --startup-timeout 600
  --selection-budget "$budget" "${counters[@]}")
for lane in reference candidate; do
  printf '%s\n' "$lane" >"$out/driver.phase"
  exe="$base"; [[ "$lane" == candidate ]] && exe="$candidate"
  python3 -u "$tools/replay_requests.py" "${args[@]}" --exe "$exe" --output "$out/$lane"
done
printf 'compare\n' >"$out/driver.phase"
python3 "$tools/compare_adapter_pair.py" "$out/reference" "$out/candidate" >"$out/comparison.json"
printf 'same-budget-screen-complete-review-before-capacity\n' >"$out/driver.phase"
