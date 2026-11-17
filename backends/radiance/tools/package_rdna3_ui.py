"""Add the checked KVMem console and bundled Python to the RDNA3 preview."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_capabilities import for_profile

def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','stage','ui-dir','python-archive','psutil-wheel','runtime-licenses','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    binary,stage,ui,py,wheel,extra,output=(getattr(a,n).resolve() for n in
        ('binary','stage','ui_dir','python_archive','psutil_wheel','runtime_licenses','output'))
    if stage.exists() or output.exists():p.error('stage and output must be new paths')
    manifest=json.loads((ui/'ui-build.json').read_text(encoding='utf8'))
    expected={f.name:sha(f.read_bytes().replace(b'\r\n',b'\n')) for f in (ROOT/'ui/lightweight').iterdir() if f.is_file()}
    if manifest.get('mode')!='lightweight' or manifest.get('console_chat_protocol')!=1 or manifest.get('entry_files')!=expected:
        raise ValueError('Prebuilt chat UI does not match the current lightweight sources')
    with zipfile.ZipFile(binary) as z:
        if z.testzip() is not None:raise ValueError('damaged input archive')
        for n in z.namelist():
            if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('unsafe archive member')
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            h,n=line.split('  ',1)
            if sha(z.read(n))!=h:raise ValueError('input checksum differs: '+n)
        z.extractall(stage)
    home=stage/'radiance_home';dest=stage/'bin/radiance_home'
    if not home.is_dir() or dest.exists():raise ValueError('unexpected native package layout')
    home.rename(dest)
    overlay=ROOT/'backends/radiance/package/rdna3-windows'
    for name in ('start.ps1','check-gpu.ps1','README.txt'):
        if name=='README.txt':(stage/name).write_text((overlay/name).read_text(encoding='utf8'),encoding='utf-8-sig')
        else:shutil.copyfile(overlay/name,stage/name)
    powershell=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    cmd=[str(powershell),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/windows/package-console.ps1'),
         '-OutputDir',str(stage),'-SourceDir',str(ROOT),'-PythonArchive',str(py),'-PythonSha256',sha(py.read_bytes()),
         '-PsutilWheel',str(wheel),'-PsutilSha256',sha(wheel.read_bytes()),'-ChatUiDir',str(ui),'-Backend','radiance']
    subprocess.run(cmd,check=True)
    allowed={'bin/msvcp140.dll','bin/vcruntime140.dll','bin/vcruntime140_1.dll','licenses/Microsoft-runtime.txt',
             'licenses/AMD-clr-LICENSE.md','licenses/AMD-hip-LICENSE.md','licenses/rocm-kpack-LICENSE.txt'}
    with zipfile.ZipFile(extra) as z:
        if set(z.namelist())!=allowed or z.testzip() is not None:raise ValueError('unexpected runtime/license inputs')
        for name in sorted(allowed):
            path=stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
    capfile=stage/'scripts/launcher-capabilities.json'
    caps=json.loads(capfile.read_text(encoding='utf-8-sig'))
    caps['backends']={'radiance':for_profile(caps,'qwen35-iqr-rdna3')}
    capfile.write_text(json.dumps(caps,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    consolefile=stage/'console-build.json';console=json.loads(consolefile.read_text(encoding='utf-8-sig'))
    console['controller_files']['scripts/launcher-capabilities.json']=sha(capfile.read_bytes())
    consolefile.write_text(json.dumps(console,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    infofile=stage/'build-info.json';info=json.loads(infofile.read_text(encoding='utf8'))
    info['backend']='radiance';info['ui']='KVMem local console and lightweight chat; offline embedded Python'
    info['source_path_anonymization']={(('bin/'+k) if k.startswith('radiance_home/') else k):v for k,v in info['source_path_anonymization'].items()}
    infofile.write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (stage/'START-HERE.txt').write_text('双击 START-HERE-CN.bat 打开 KVMem 图形控制台。\n模型需另行准备，详情见 README.txt。\nDouble-click START-HERE-EN.bat for English.\n',encoding='utf-8-sig')
    sums=stage/'SHA256SUMS.txt';sums.unlink()
    # Python imports during package checks may create bytecode embedding the
    # builder's absolute path. Ship source modules, never generated caches.
    files=sorted(x for x in stage.rglob('*') if x.is_file()
                 and '__pycache__' not in x.relative_to(stage).parts and x.suffix.lower()!='.pyc')
    sums.write_text(''.join(f'{sha(f.read_bytes())}  {f.relative_to(stage).as_posix()}\n' for f in files),encoding='utf8')
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,f.relative_to(stage).as_posix())
        z.write(sums,sums.relative_to(stage).as_posix())
        z.writestr('models/',b'')
    print(json.dumps({'archive':str(output),'sha256':sha(output.read_bytes()),'bytes':output.stat().st_size,'files':len(files)+1},indent=2))

if __name__=='__main__':main()
