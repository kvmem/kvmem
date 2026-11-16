"""Assemble a portable RDNA3 Windows preview from explicit, already-built inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import zipfile
from patch_series import verified_patches
from source_identity import source_revision


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_gpu_bundles(path, targets):
    """Validate each uncompressed clang offload bundle's device code objects."""
    data = path.read_bytes()
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    pos = 0
    checked = 0
    while True:
        at = data.find(magic, pos)
        if at < 0:
            break
        pos = at + len(magic)
        cursor = at + 32
        try:
            count = struct.unpack_from('<Q', data, at + 24)[0]
            if not 0 < count < 16:
                continue  # A copy of the magic inside code/string data, not a bundle.
            found = set()
            for _ in range(count):
                off, size, length = struct.unpack_from('<QQQ', data, cursor)
                cursor += 24
                if length > 512 or cursor + length > len(data):
                    raise ValueError('invalid offload triple')
                triple = data[cursor:cursor+length].decode('ascii')
                cursor += length
                if 'amdgcn-amd-amdhsa--' not in triple:
                    continue
                target = triple.split('amdgcn-amd-amdhsa--', 1)[1].split(':', 1)[0]
                if size <= 0 or at + off + size > len(data) or data[at+off:at+off+4] != b'\x7fELF':
                    raise ValueError(f'{path.name}: invalid {target} code object')
                found.add(target)
            if found:
                if found != set(targets):
                    raise ValueError(f'{path.name}: bundle targets {sorted(found)} do not match {targets}')
                checked += 1
        except (UnicodeDecodeError, struct.error) as e:
            raise ValueError(f'{path.name}: malformed offload bundle') from e
    if not checked:
        raise ValueError(f'{path.name}: no uncompressed offload bundles found')
    return checked


def anonymize_source_paths(data, profile):
    """Replace a build user's name only in PE .rdata source-file strings.

    Equal-length substitution leaves executable code, device code, sections,
    offsets and relocation data untouched. Refuse any other occurrence.
    """
    profile = profile.replace('\\', '/').rstrip('/')
    user = profile.rsplit('/', 1)[-1]
    if not user or not profile.lower().startswith(tuple(f'{c}:/users/' for c in 'abcdefghijklmnopqrstuvwxyz')):
        raise ValueError('cannot establish the Windows build-user profile')
    old = profile.encode('utf8')
    new = (profile[:-len(user)] + 'x'*len(user.encode('utf8'))).encode('utf8')
    if len(old) != len(new):
        raise ValueError('profile replacement must have identical byte length')
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    if data[pe:pe+4] != b'PE\0\0':
        raise ValueError('invalid Windows executable')
    count = struct.unpack_from('<H', data, pe+6)[0]
    opt = struct.unpack_from('<H', data, pe+20)[0]
    sections=[]
    for i in range(count):
        at=pe+24+opt+40*i
        name=data[at:at+8].rstrip(b'\0').decode('ascii')
        size,offset=struct.unpack_from('<II',data,at+16)
        sections.append((name,offset,offset+size))
    out=bytearray(data)
    lower=data.lower();start=0;changed=0
    while True:
        at=lower.find(old.lower(),start)
        if at<0:break
        start=at+len(old)
        if not any(name=='.rdata' and lo<=at and at+len(old)<=hi for name,lo,hi in sections):
            raise ValueError('build-user path appears outside the read-only data section')
        end=data.find(b'\0',at)
        if end<0 or not re.search(rb'\.(?:cpp|hpp|hip|h|cc|c)$',data[at:end],re.I):
            raise ValueError('build-user path is not a source-file diagnostic string')
        out[at:at+len(old)]=new
        changed+=1
    for name,lo,hi in sections:
        if name!='.rdata' and data[lo:hi]!=out[lo:hi]:
            raise ValueError(f'non-data section {name} changed during path anonymization')
    return bytes(out),changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'sdk', 'native', 'json-include', 'overlay', 'stage', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    build, sdk, native, include, overlay, stage, output = (getattr(a, n).resolve() for n in
        ('build', 'sdk', 'native', 'json_include', 'overlay', 'stage', 'output'))
    if stage.exists() or output.exists():
        p.error('stage and output must be new paths; existing files are never replaced')
    _, specs, _ = verified_patches()
    source_revision(native)
    manifest = json.loads((native/'.radiance-source-manifest.json').read_text(encoding='utf8'))
    targets = ('gfx1100', 'gfx1101', 'gfx1102', 'gfx1103')
    payload = {}
    def add(src, dest):
        if not src.is_file():
            raise FileNotFoundError(src)
        if dest in payload:
            raise ValueError('duplicate destination: '+dest)
        payload[dest] = src

    for name in ('radiance-kvmem-server.exe', 'rad_core.dll', 'r3di_iqr_gpu_test.exe'):
        add(build/'bin'/name, 'bin/'+name)
    for name in ('amdhip64_7.dll', 'amd_comgr.dll', 'rocm_kpack.dll'):
        add(sdk/'bin'/name, 'bin/'+name)
    add(build/'radiance_home/kernels.rkb', 'radiance_home/kernels.rkb')
    add(build/'radiance_home/architectures/qwen35_w8a8.dll', 'radiance_home/architectures/qwen35_w8a8.dll')
    # Match the qualified device-only plugin set. The optional CPU plugins
    # have a separate Visual Studio OpenMP runtime dependency.
    for name in ('libr3d.dll', 'libr3diqr.dll'):
        add(build/'radiance_home/kernels'/name, 'radiance_home/kernels/'+name)
    bundles = {name:check_gpu_bundles(build/'radiance_home/kernels'/name, targets)
               for name in ('libr3d.dll', 'libr3diqr.dll')}
    for f in overlay.iterdir():
        if f.is_file(): add(f, f.name)
    for name in ('LICENSE', 'NOTICE', 'vendor/llama/LICENSE', 'vendor/yyjson/LICENSE',
                 'vendor/cpp-httplib/LICENSE', 'vendor/rapidyaml/LICENSE.txt', 'dashboard/THIRD-PARTY-LICENSES.txt'):
        add(native/name, 'licenses/radiance/'+name)
    add(include/'nlohmann/LICENSE.MIT', 'licenses/nlohmann/LICENSE.MIT')
    add(sdk/'share/doc/rocm-core/LICENSE.md', 'licenses/rocm-core/LICENSE.md')
    add(sdk/'share/doc/amd_comgr/LICENSE.txt', 'licenses/amd_comgr/LICENSE.txt')
    add(sdk/'lib/llvm/include/llvm/Support/LICENSE.TXT', 'licenses/llvm/LICENSE.TXT')
    stage.mkdir(parents=True)
    (stage/'models').mkdir()
    redacted={}
    project_binaries={'bin/radiance-kvmem-server.exe','bin/rad_core.dll','bin/r3di_iqr_gpu_test.exe',
                      'radiance_home/architectures/qwen35_w8a8.dll',
                      'radiance_home/kernels/libr3d.dll','radiance_home/kernels/libr3diqr.dll'}
    for dest, src in payload.items():
        out = stage/dest
        out.parent.mkdir(parents=True, exist_ok=True)
        if dest == 'README.txt':
            out.write_text(src.read_text(encoding='utf8'), encoding='utf-8-sig')
        elif dest in project_binaries:
            original=src.read_bytes()
            anonymized,count=anonymize_source_paths(original,os.environ.get('USERPROFILE',str(Path.home())))
            out.write_bytes(anonymized)
            redacted[dest]={'source_path_strings':count,'compiled_sha256':hashlib.sha256(original).hexdigest(),
                            'packaged_sha256':hashlib.sha256(anonymized).hexdigest()}
        else:
            shutil.copyfile(src, out)
    buildinfo = {'status':'preview', 'platform':'windows-x86_64', 'radiance_version':'2.0.0',
        'plugin_version':'0.5.0-rdna3-fusion', 'targets':targets, 'validated_offload_bundles':bundles,
        'upstream_revision':manifest['revision'], 'native_patch_sha256':manifest['patches'][0],
        'native_patch_series':specs,
        'runtime_evidence':'Windows RX 7800M/gfx1101 only; other GPUs not yet qualified',
        'model_evidence':'Windows RX 7800M: Qwen3.8-27B IQ2R-L, IQ3R and IQ4R-S v4 text TP1/C1, including short IQ3R/IQ4R DFlash2 tests; no model included',
        'sdk_version':'ROCm Core 10.1.0 poly bundle used by this build',
        'source_path_anonymization':redacted,
        'note':'Test builds only; GPU model, VRAM, driver, workload and operating system affect compatibility.'}
    (stage/'build-info.json').write_text(json.dumps(buildinfo,indent=2)+'\n',encoding='utf8')
    files = sorted(f for f in stage.rglob('*') if f.is_file())
    (stage/'SHA256SUMS.txt').write_text(''.join(f'{digest(f)}  {f.relative_to(stage).as_posix()}\n' for f in files),encoding='utf8')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(stage.rglob('*')):
            if f.is_file(): z.write(f, f.relative_to(stage).as_posix())
        z.writestr('models/',b'')
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'sha256':digest(output),
                      'files':len(files)+1,'stage':str(stage)},indent=2),flush=True)


if __name__ == '__main__':
    main()
