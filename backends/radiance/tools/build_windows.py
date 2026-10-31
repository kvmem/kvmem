"""Build the pinned native Windows adapter from an x64 VS developer shell."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from source_identity import source_revision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--rocm', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--configure-only', action='store_true')
    a = p.parse_args()
    if os.name != 'nt' or a.jobs < 1:
        p.error('requires Windows and a positive job count')
    source, build, sdk = a.source.resolve(), a.build.resolve(), a.rocm.resolve()
    root = Path(__file__).resolve().parents[3]
    meta = json.loads((root / 'backends/radiance/upstream.json').read_text(encoding='utf-8'))
    head = source_revision(source)
    if head != meta['revision']:
        p.error('native checkout differs from ABI15 pin; run prepare_native.py')
    for file in (sdk/'bin/clang.exe', sdk/'bin/clang++.exe', sdk/'bin/hipcc.exe', sdk/'lib/amdhip64.lib'):
        if not file.is_file():
            p.error(f'missing SDK component: {file}')
    sdk_path = sdk.as_posix()  # CMake cache/script strings must not contain \P escapes.
    command = ['cmake', '-S', source.as_posix(), '-B', build.as_posix(), '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_C_COMPILER={sdk_path}/bin/clang.exe',
               f'-DCMAKE_CXX_COMPILER={sdk_path}/bin/clang++.exe',
               '-DRAD_WITH_HIP=ON', '-DRAD_WITH_FFMPEG=OFF', '-DRAD_GPU_TARGETS=gfx1200',
               '-DCMAKE_HIP_ARCHITECTURES=gfx1200', f'-DCMAKE_HIP_COMPILER={sdk_path}/bin/clang++.exe',
               f'-DCMAKE_HIP_COMPILER_ROCM_ROOT={sdk_path}',
               f'-DRAD_HIPCC={sdk_path}/bin/hipcc.exe', f'-DRAD_HIP_RUNTIME={sdk_path}/lib/amdhip64.lib',
               f'-DRAD_KVMEM_SOURCE_DIR={root.as_posix()}']
    build.mkdir(parents=True, exist_ok=True)
    (build/'kvmem-build-inputs.json').write_text(json.dumps({'revision':head, 'configure':command}, indent=2), encoding='utf-8')
    subprocess.run(command, check=True)
    if not a.configure_only:
        subprocess.run(['cmake', '--build', str(build), '--parallel', str(a.jobs), '--target',
                        'kvmem-radiance-observer-probe', 'radiance', 'rad-info', 'rad-convert',
                        'libr4d', 'libref', 'libquant', 'qwen35_fp8'], check=True)


if __name__ == '__main__':
    main()
