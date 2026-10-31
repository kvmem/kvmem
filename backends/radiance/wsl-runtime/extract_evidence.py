"""Recreate this evidence bundle from the locally retained historical sources (no GPU)."""
import hashlib
import json
from pathlib import Path
import subprocess

out = Path(__file__).resolve().parent
historical = Path('E:/Codex_work/diagnostics/radiance-proposal-20261019')
source = historical / 'rocdxg-source-v'
build = historical / 'rocdxg-reserve-v'
revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert revision == 'fbd6f28765809e485d63a6a3edeb3fdbf76dc63c'
changed = subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).splitlines()
assert changed == ['src/wddm/queue.cpp'], changed
patch = subprocess.check_output(['git','-C',str(source),'diff','--binary','--','src/wddm/queue.cpp'])
patchfile = out / 'rocdxg-v1.2.0-scratch-reserve.patch'
patchfile.write_bytes(patch)
subprocess.run(['git','-C',str(source),'apply','--reverse','--check',str(patchfile)],check=True)
numstat = subprocess.check_output(['git','apply','--numstat',str(patchfile)],text=True).strip()
assert numstat == '10\t1\tsrc/wddm/queue.cpp', numstat
license_bytes = (source/'LICENSE.md').read_bytes()
(out/'UPSTREAM-LICENSE.md').write_bytes(license_bytes)
library = build/'librocdxg.so.1.2.0'
library_sha = hashlib.sha256(library.read_bytes()).hexdigest()
assert library_sha == '2b2a082ede46324a5e3771b4b14f2850fdb9bd38348abfbeedee83966bbd6088'
base_queue = subprocess.check_output(['git','-C',str(source),'show','HEAD:src/wddm/queue.cpp'])
queue = (source/'src/wddm/queue.cpp').read_bytes()
sha = lambda value: hashlib.sha256(value).hexdigest()
provenance = {
    'upstream_url':'https://github.com/ROCm/librocdxg.git',
    'upstream_tag':'v1.2.0', 'upstream_commit':revision,
    'patch_file':patchfile.name, 'patch_sha256':sha(patch),
    'changed_files':changed, 'added_lines':10, 'removed_lines':1,
    'base_queue_sha256':sha(base_queue), 'patched_queue_sha256':sha(queue),
    'license_file':'UPSTREAM-LICENSE.md', 'license_sha256':sha(license_bytes),
    'license_note':'MIT source except separately licensed AMD binary src/thunk_proxy/libthunk_proxy.a; preserve complete upstream license.',
    'historical_library_name':library.name,
    'historical_library_sha256':library_sha,
    'historical_library_included':False,
    'rebuild_status':'Commands reconstructed from preserved CMakeCache; no new build or GPU execution in this bundle task.'
}
(out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
cache = (build/'CMakeCache.txt').read_text()
keys = ['CMAKE_BUILD_TYPE','CMAKE_C_COMPILER','CMAKE_CXX_COMPILER','CMAKE_C_COMPILER_AR',
        'CMAKE_CXX_COMPILER_AR','CMAKE_GENERATOR','WIN_SDK','CMAKE_INSTALL_PREFIX','CMAKE_MAKE_PROGRAM']
facts = {line.split(':',1)[0]:line.split('=',1)[1] for line in cache.splitlines()
         if any(line.startswith(key+':') for key in keys)}
logroot = historical/'evidence/full-target-agent33-x'
maps = [line for line in (logroot/'loaded-maps.txt').read_text().splitlines() if 'rocdxg-reserve-v/librocdxg' in line]
scratch = [line for line in (logroot/'server.stderr.log').read_text().splitlines() if 'SCRATCH_RESERVE_V' in line]
assert maps and scratch
evidence = {
    'evidence_scope':'Historical final-X WSL run; not a fresh integration qualification.',
    'historical_sdk':'/opt/rocm-7.2.3', 'historical_hardware':'RX9060XT/gfx1200',
    'historical_library_sha256':library_sha, 'cmake_cache_selected_values':facts,
    'actual_loaded_library_paths':sorted(set(line.split()[-1] for line in maps)),
    'scratch_log_lines':scratch,
    'source_records':{
        'run_script':'diagnostics/radiance-proposal-20261019/run-final-x.sh',
        'cmake_cache':'diagnostics/radiance-proposal-20261019/rocdxg-reserve-v/CMakeCache.txt',
        'loaded_maps':'diagnostics/radiance-proposal-20261019/evidence/full-target-agent33-x/loaded-maps.txt',
        'stderr':'diagnostics/radiance-proposal-20261019/evidence/full-target-agent33-x/server.stderr.log'
    },
    'library_selection':'Child-only LD_LIBRARY_PATH prepend; HSA_ENABLE_DXG_DETECTION=1; LD_PRELOAD unset; no system installation.',
    'known_remaining_issue':'Historical teardown destroy_allocation -22 was retained, not claimed fixed.'
}
(out/'historical-build-evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'patch_sha256':sha(patch),'library_sha256':library_sha,'reverse_apply_check':True,'numstat':numstat},indent=2))
