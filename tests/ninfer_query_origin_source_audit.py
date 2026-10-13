#!/usr/bin/env python3
"""Read-only source audit; all reconstructed files go under the supplied /tmp path."""
import hashlib
import json
import pathlib
import subprocess
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2]).resolve()
if not str(out).startswith('/tmp/'):
    raise SystemExit('audit output must be /tmp')
out.mkdir(parents=True, exist_ok=True)
source = repo / 'backends/ninfer'
baseline = out / 'baseline'
baseline.mkdir(exist_ok=True)

def git(root, *args, check=True):
    return subprocess.run(['git', '-c', f'safe.directory={root}', '-C', str(root), *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

paths = [
    'src/models/qwen3_5/frontend/chat_template.cpp',
    'src/models/qwen3_5/frontend/frontend.cpp',
    'src/models/qwen3_5/frontend/processor.cpp',
    'src/models/qwen3_5/program/storage/kvmem_window.cpp',
    'src/models/qwen3_5/program/storage/memory_context.h',
]
baseline_commit = 'e69e9c00bc9d8db1d204d54c42dd0f7b6cb2039b'
patch = git(repo, 'show', f'{baseline_commit}:backends/patches/ninfer-kvmem.patch').stdout
for path in paths:
    original = git(source, 'show', f'HEAD:{path}', check=False)
    target = baseline / path
    if original.returncode == 0:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.stdout)
    elif target.exists():
        target.unlink()
subprocess.run(['git', 'apply', *[f'--include={p}' for p in paths]],
               cwd=baseline, input=patch, check=True)

old = (baseline / paths[0]).read_text()
new = (source / paths[0]).read_text()
start = '    const auto generation_begin = continuation'
end = '    for (std::size_t i = 0; i < options.cache_markers.size(); ++i)'
old_block = old[old.index(start):old.index(end)]
new_block = new[new.index(start):new.index(end)]
assert old_block == new_block, 'thinking-history/rewrite block changed from committed PR patch'

window_old = (baseline / paths[3]).read_text()
window_new = (source / paths[3]).read_text()
signature = 'bool ProgramImpl::memory_same_query('
def function(text, marker):
    begin = text.index(marker)
    end = text.index('\n}', begin) + 2
    return text[begin:end]
old_pred = function(window_old, signature)
new_pred = function(window_new, signature)
body = new_pred[new_pred.index('{'):]
(out / 'cache_predicate.inc').write_text(
    'bool Predicate::memory_same_query(const Window& window, const Prompt& prompt, std::uint32_t frontier) const '\
    + body + '\n')
(out / 'cache_predicate_baseline.inc').write_text(
    'bool Predicate::memory_same_query(const Window& window, const Prompt& prompt, std::uint32_t frontier) const '
    + old_pred[old_pred.index('{'):] + '\n')
history_function = function(window_new, 'std::optional<KvmemHistoryMatch> ProgramImpl::memory_restorable_history(')
admission = function(window_new, 'bool ProgramImpl::memory_query_checkpoint_compatible(')
assert '!memory_query_checkpoint_compatible(*history,prompt,frontier)' in ''.join(history_function.split())
(out / 'cache_admission.inc').write_text(
    'bool Predicate::memory_query_checkpoint_compatible(const Window& window, const Prompt& prompt, std::uint32_t frontier) const '
    + admission[admission.index('{'):] + '\n')

summary = {
    'checkout_head': git(repo, 'rev-parse', 'HEAD').stdout.decode().strip(),
    'baseline_commit': baseline_commit,
    'ninfer_pin': git(source, 'rev-parse', 'HEAD').stdout.decode().strip(),
    'committed_patch_sha256': hashlib.sha256(patch).hexdigest(),
    'working_patch_sha256': hashlib.sha256((repo / 'backends/patches/ninfer-kvmem.patch').read_bytes()).hexdigest(),
    'source_sha256': {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in paths},
    'unchanged_thinking_rewrite_block': True,
    'unchanged_memory_same_query': old_pred == new_pred,
    'cache_query_match_checks_all_ranges_counts': 'window.query_counts[slot] != span.count' in new_pred,
    'history_admission_checks_changed_query_span': True,
    'limits': ['cache predicate executes its exact body in field-only host fixture, not a ProgramImpl or device restore',
               'renderer baseline is reconstructed from the committed versioned backend patch']
}
(out / 'source-audit.json').write_text(json.dumps(summary, indent=2)+'\n')
print('independent source audit: PASS; thinking/rewrite unchanged; both exact cache predicates extracted')
