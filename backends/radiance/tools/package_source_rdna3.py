"""Create a source-only snapshot matching the RDNA3 Radiance preview."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from patch_series import verified_patches
from source_identity import source_revision


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--json-include',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    native=a.native.resolve();include=a.json_include.resolve();output=a.output.resolve()
    if output.exists():p.error('output already exists; choose a new path')
    meta,specs,_=verified_patches(root/'backends/radiance')
    source_revision(native)
    manifest=json.loads((native/'.radiance-source-manifest.json').read_text(encoding='utf8'))
    assert manifest['revision']==meta['revision'] and manifest['patches']==[s['sha256'] for s in specs]
    actual={f.relative_to(native).as_posix() for f in native.rglob('*') if f.is_file() and f.name!='.radiance-source-manifest.json' and '.git' not in f.relative_to(native).parts}
    assert actual==set(manifest['files']),'native source inventory differs from manifest'
    for name,digest in manifest['files'].items():
        assert sha((native/name).read_bytes())==digest,name
    tracked=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z']).decode('utf8').split('\0')
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    files={};omitted=[]
    for name in sorted(set(tracked)):
        if not name:continue
        path=root/name
        if not path.is_file():
            omitted.append(name);continue  # Includes the three unpopulated gitlinks.
        parts=Path(name).parts
        if parts[0]=='docs' and ('benchmarks' in parts or 'evidence' in parts):
            omitted.append(name);continue
        if path.suffix.lower() in ('.rad','.safetensors','.exe','.dll','.so','.zip'):
            omitted.append(name);continue
        data=path.read_bytes()
        if parts[0]=='docs' and re.search(rb'(?i)(?:[a-z]:)?[\\/]+users[\\/]+[^\\/\s]+',data):
            omitted.append(name);continue
        files['kvmem-source/'+name]=data
    for required in ('kvmem-source/CMakeLists.txt','kvmem-source/kvmem/CMakeLists.txt',
                     'kvmem-source/backends/radiance/CMakeLists.txt','kvmem-source/backends/patches/radiance-v2-kvmem.patch'):
        assert required in files,required
    for name in manifest['files']:
        files['radiance-native/'+name]=(native/name).read_bytes()
    files['radiance-native/.radiance-source-manifest.json']=(native/'.radiance-source-manifest.json').read_bytes()
    for name in ('json.hpp','LICENSE.MIT'):
        files['third_party/include/nlohmann/'+name]=(include/'nlohmann'/name).read_bytes()
    files['README.txt']=(root/'backends/radiance/package/rdna3-source/README.txt').read_text(encoding='utf8').encode('utf-8-sig')
    info={'kind':'source-only preview','kvmem_base_commit':commit,'kvmem_uncommitted_changes_included':True,
          'radiance_revision':manifest['revision'],'native_patch_sha256':meta['native_patch_sha256'],
          'native_patch_series':specs,
          'native_files_verified':len(manifest['files']),'nlohmann_json_sha256':sha(files['third_party/include/nlohmann/json.hpp']),
          'models_or_toolchains_included':False,'unpopulated_submodules_omitted':['backends/llamacpp','backends/ninfer','backends/polystrata'],
          'private_test_evidence_omitted':True,'omitted_repository_paths':omitted,
          'hardware_evidence':'Windows RX7800M gfx1101 only; additional RDNA3 targets need runtime testing'}
    files['source-info.json']=(json.dumps(info,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    checksums=''.join(f'{sha(data)}  {name}\n' for name,data in sorted(files.items()))
    files['SHA256SUMS.txt']=checksums.encode('utf8')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(files.items()):z.writestr(name,data)
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'sha256':sha(output.read_bytes()),
                      'files':len(files),'native_files':len(manifest['files']),'omitted':len(omitted)},indent=2))


if __name__=='__main__':main()
