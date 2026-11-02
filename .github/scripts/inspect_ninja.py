"""Inspect a specific Ninja build before compiling; optionally reject CUDA work."""
import argparse
import collections
import json
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--ninja', default='ninja')
parser.add_argument('--target', action='append', required=True)
parser.add_argument('--expect-no-cuda', action='store_true')
parser.add_argument('--output', type=Path, help='save JSON and adjacent .log evidence')
args = parser.parse_args()
build = args.build.resolve()
if not (build/'build.ninja').is_file():
    parser.error('the existing build directory must contain build.ninja')
command = [args.ninja, '-C', str(build), '-n', '-d', 'explain', *args.target]
run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
text = run.stdout.decode('utf-8', errors='replace')
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
report = dict(build=str(build), targets=args.target, command=command, exit_code=run.returncode,
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
    print('The plan needs CMake regeneration; inspect the plan again afterwards.', file=sys.stderr)
if blocked:
    print('STOP: CUDA compilation was scheduled despite --expect-no-cuda. No compilation was performed.', file=sys.stderr)
sys.exit(run.returncode or (3 if blocked else 2 if regenerate else 0))
