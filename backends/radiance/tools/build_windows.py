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
    p.add_argument('--gpu-target', choices=('gfx1200', 'gfx1201'), default='gfx1200')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--ffmpeg', choices=['AUTO','ON','OFF'], default='AUTO')
    p.add_argument('--ffmpeg-prefix', type=Path)
    p.add_argument('--ffmpeg-static', action='store_true')
    p.add_argument('--configure-only', action='store_true')
    p.add_argument('--target', action='append', help='Specific audited Ninja target; repeat for multiple targets (for IQR-V4 use qwen35_w8a8)')
    a = p.parse_args()
    if os.name != 'nt' or a.jobs < 1:
        p.error('requires Windows and a positive job count')
    if not a.configure_only and not a.target:
        p.error('pass --configure-only, review the Ninja dry run, then pass explicit --target values')
    source, build, sdk = a.source.resolve(), a.build.resolve(), a.rocm.resolve()
    root = Path(__file__).resolve().parents[3]
    meta = json.loads((root / 'backends/radiance/upstream.json').read_text(encoding='utf-8'))
    head = source_revision(source)
    if head != meta['revision']:
        p.error('native checkout differs from the adapter pin; run prepare_native.py')
    compiler_dir = sdk/'bin'
    if not (compiler_dir/'clang++.exe').is_file():
        compiler_dir = sdk/'lib/llvm/bin'
    for file in (compiler_dir/'clang.exe', compiler_dir/'clang++.exe', sdk/'bin/hipcc.exe', sdk/'lib/amdhip64.lib'):
        if not file.is_file():
            p.error(f'missing SDK component: {file}')
    sdk_path = sdk.as_posix()  # CMake cache/script strings must not contain \P escapes.
    command = ['cmake', '-S', source.as_posix(), '-B', build.as_posix(), '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_C_COMPILER={compiler_dir.as_posix()}/clang.exe',
               f'-DCMAKE_CXX_COMPILER={compiler_dir.as_posix()}/clang++.exe',
               '-DRAD_WITH_HIP=ON', f'-DRAD_WITH_FFMPEG={a.ffmpeg}', f'-DRAD_GPU_TARGETS={a.gpu_target}',
               f'-DCMAKE_HIP_ARCHITECTURES={a.gpu_target}', f'-DCMAKE_HIP_COMPILER={compiler_dir.as_posix()}/clang++.exe',
               f'-DCMAKE_HIP_COMPILER_ROCM_ROOT={sdk_path}',
               f'-DRAD_HIPCC={sdk_path}/bin/hipcc.exe', f'-DRAD_HIP_RUNTIME={sdk_path}/lib/amdhip64.lib',
               f'-DRAD_KVMEM_SOURCE_DIR={root.as_posix()}']
    bitcode = sdk/'lib/llvm/amdgcn/bitcode'
    if (bitcode/'ocml.bc').is_file():
        command.append(f'-DCMAKE_HIP_FLAGS=--rocm-path="{sdk_path}" --rocm-device-lib-path="{bitcode.as_posix()}"')
    build.mkdir(parents=True, exist_ok=True)
    env=os.environ.copy()
    if a.ffmpeg_prefix:
        pc=a.ffmpeg_prefix.resolve()/'lib/pkgconfig'
        if not (pc/'libavcodec.pc').is_file():p.error('missing FFmpeg pkg-config files')
        env['PKG_CONFIG_PATH']=str(pc)
    if a.ffmpeg_static:command.append('-DPKG_CONFIG_ARGN=--static')
    (build/'kvmem-build-inputs.json').write_text(json.dumps({'revision':head, 'configure':command, 'targets':a.target or []}, indent=2), encoding='utf-8')
    subprocess.run(command, env=env, check=True)
    if not a.configure_only:
        subprocess.run(['cmake', '--build', str(build), '--parallel', str(a.jobs), '--target',
                        *a.target], env=env, check=True)


if __name__ == '__main__':
    main()
