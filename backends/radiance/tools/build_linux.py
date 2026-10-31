"""Configure/build the pinned, patched Radiance with an explicitly selected SDK."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from source_identity import source_revision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--rocm", required=True, type=Path)
    p.add_argument("--arch", action="append")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--configure-only", action="store_true")
    a = p.parse_args()
    if os.name != "posix" or a.jobs < 1:
        p.error("use Linux/WSL with a positive job count")
    a.source, a.build, a.rocm = a.source.resolve(), a.build.resolve(), a.rocm.resolve()
    metadata = json.loads((Path(__file__).resolve().parents[1] / "upstream.json").read_text())
    head = source_revision(a.source)
    if head != metadata["revision"]:
        p.error(f"checkout revision differs from the adapter pin: {head}")
    hipcc = a.rocm / "bin/hipcc"
    clang = a.rocm / "llvm/bin/clang++"
    hip = a.rocm / "lib/libamdhip64.so"
    hsa = a.rocm / "lib/libhsa-runtime64.so"
    for file in (hipcc, clang, hip, hsa):
        if not file.is_file():
            p.error(f"missing SDK component: {file}")
    env = os.environ.copy()
    env.update(ROCM_PATH=str(a.rocm), HIP_PATH=str(a.rocm),
               LD_LIBRARY_PATH=f"{a.rocm}/lib:{a.rocm}/llvm/lib")
    if Path("/dev/dxg").exists():
        env["HSA_ENABLE_DXG_DETECTION"] = "1"
    arch = a.arch
    if not arch:
        detector = a.rocm / "llvm/bin/amdgpu-arch"
        if not detector.is_file():
            p.error("no architecture detector in SDK; pass --arch explicitly")
        arch = subprocess.check_output([str(detector)], env=env, text=True).split()
    arch = sorted(set(arch))
    if not arch or any(not re.fullmatch(r"gfx[0-9a-f]+", x) for x in arch):
        p.error(f"invalid/missing AMD architectures: {arch}")
    if not any(x.startswith("gfx12") for x in arch):
        p.error("pinned libr4d has RDNA4 kernels only; a new ISA needs a kernel plugin")
    a.build.mkdir(parents=True, exist_ok=True)
    command = ["cmake", "-S", str(a.source), "-B", str(a.build), "-G", "Ninja",
               "-DCMAKE_BUILD_TYPE=Release", "-DRAD_WITH_HIP=ON", "-DRAD_WITH_FFMPEG=OFF",
               f"-DRAD_HIPCC={hipcc}", f"-DCMAKE_HIP_COMPILER={clang}",
               f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={a.rocm}", f"-DCMAKE_PREFIX_PATH={a.rocm}",
               f"-DRAD_GPU_TARGETS={';'.join(arch)}", f"-DCMAKE_HIP_ARCHITECTURES={';'.join(arch)}",
               f"-DRAD_HIP_RUNTIME={hip}", f"-DRAD_HSA_RUNTIME={hsa}",
               f"-DRAD_KVMEM_SOURCE_DIR={Path(__file__).resolve().parents[3]}"]
    inputs = {"revision": head, "source": str(a.source), "sdk": str(a.rocm),
              "architectures": arch, "configure": command}
    (a.build / "kvmem-build-inputs.json").write_text(json.dumps(inputs, indent=2))
    subprocess.run(command, env=env, check=True)
    if not a.configure_only:
        subprocess.run(["cmake", "--build", str(a.build), "--parallel", str(a.jobs),
                        "--target", "radiance", "rad-kbench", "rad-info", "rad-convert", "libr4d",
                        "libref", "libquant", "libavx", "llama_fp8", "llama_dense_fp8", "qwen35_bf16", "qwen35_fp8",
                        "sched_test", "server_test", "radiance_pool_geometry_test",
                        "radiance_native_pages_test", "radiance_execution_pages_test", "radiance_query_identity_test",
                        "radiance_page_transfer_test", "radiance_page_transfer_gpu_test", "radiance_native_memory_gpu_test",
                        "radiance_statistics_capture_gpu_test", "kvmem-radiance-observer-probe"], env=env, check=True)


if __name__ == "__main__":
    main()
