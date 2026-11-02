"""Inspect a specific Ninja build before compiling; optionally reject CUDA work."""
import argparse
import collections
import json
import hashlib
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--ninja', default='ninja')
parser.add_argument('--target', action='append', required=True)
parser.add_argument('--expect-no-cuda', action='store_true')
parser.add_argument('--resolve-regeneration', action='store_true', help='audit and update only build.ninja before inspecting object work')
parser.add_argument('--output', type=Path, help='save JSON and adjacent .log evidence')
args = parser.parse_args()
build = args.build.resolve()
if not (build/'build.ninja').is_file():
    parser.error('the existing build directory must contain build.ninja')
command = [args.ninja, '-C', str(build), '-n', '-d', 'explain', *args.target]
original_command=list(command)
run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
text = run.stdout.decode('utf-8', errors='replace')
if args.resolve_regeneration and 'Re-running CMake' in text:
    # GLOB verification or changed CMake inputs may hide the real object plan.
    # build.ninja is a metadata-only target: audit its closure before running it,
    # then repeat the requested dry run. Never compile objects to fix metadata.
    metadata_command=[args.ninja,'-C',str(build),'-n','-d','explain','build.ninja']
    metadata=subprocess.run(metadata_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    metadata_text=metadata.stdout.decode('utf-8',errors='replace')
    metadata_jobs=re.findall(r'^\[\d+/\d+\] (.+)$',metadata_text,re.MULTILINE)
    allowed=('Re-checking globbed directories','Re-running CMake')
    print(metadata_text)
    if metadata.returncode or not metadata_jobs or any(not job.startswith(allowed) for job in metadata_jobs):
        sys.exit('STOP: metadata-only target contains unexpected work; no build started')
    update=subprocess.run([args.ninja,'-C',str(build),'build.ninja'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(update.stdout.decode('utf-8',errors='replace'))
    if update.returncode:sys.exit(update.returncode)
    run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    text=run.stdout.decode('utf-8',errors='replace')
    if 'Re-running CMake' in text:
        # CONFIGURE_DEPENDS has a deliberately missing phony force input. Even
        # after a real successful glob/CMake refresh, Ninja -n cannot evaluate
        # its restat result and shows only regeneration. Inspect an identical
        # manifest snapshot so automatic self-regeneration cannot hide the
        # object plan. No Ninja records, timestamps, rules or dependencies are
        # altered; the actual build continues using the original manifest.
        remaining=[line for line in text.splitlines() if line.startswith('ninja explain: ')]
        if not remaining or any(not any(marker in line for marker in
                ('VerifyGlobs.cmake_force','cmake.verify_globs','build.ninja is dirty')) for line in remaining):
            print(text,file=sys.stderr)
            sys.exit('STOP: non-glob regeneration remains after metadata refresh')
        snapshot=build/'kvmem-inspection.ninja'
        data=(build/'build.ninja').read_bytes();snapshot.write_bytes(data)
        assert hashlib.sha256(snapshot.read_bytes()).digest()==hashlib.sha256(data).digest()
        command=[args.ninja,'-C',str(build),'-f',snapshot.name,'-n','-d','explain',*args.target]
        run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        text=run.stdout.decode('utf-8',errors='replace')
jobs = re.findall(r'^\[\d+/\d+\] (.+)$', text, re.MULTILINE)
counts = collections.Counter()
for job in jobs:
    if 'Building CUDA object' in job: counts['cuda_compile'] += 1
    elif 'Building HIP object' in job: counts['hip_compile'] += 1
    elif 'Building CXX object' in job or 'Building C object' in job: counts['host_compile'] += 1
    elif job.startswith('Linking '): counts['link'] += 1
    else: counts['other'] += 1
reasons = [line.removeprefix('ninja explain: ') for line in text.splitlines() if line.startswith('ninja explain: ')]
missing = [line for line in reasons if 'command line not found in log' in line]
regenerate = any('Re-running CMake' in job for job in jobs)
blocked = bool(args.expect_no_cuda and counts['cuda_compile'])
report = dict(build=str(build), targets=args.target, command=command, original_command=original_command, exit_code=run.returncode,
              job_counts=dict(counts), missing_command_records=len(missing),
              missing_record_examples=missing[:8], regeneration_required=regenerate,
              cuda_guard_blocked=blocked)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    args.output.with_suffix('.log').write_text(text, encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
if missing:
    print('Review missing command records before treating existing objects as requiring recompilation.', file=sys.stderr)
if regenerate:
    print('\n'.join(reasons[-20:]),file=sys.stderr)
    print('The plan needs CMake regeneration; inspect the plan again afterwards.', file=sys.stderr)
if blocked:
    print('STOP: CUDA compilation was scheduled despite --expect-no-cuda. No compilation was performed.', file=sys.stderr)
sys.exit(run.returncode or (3 if blocked else 2 if regenerate else 0))
