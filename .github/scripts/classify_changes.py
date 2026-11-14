"""Skip build/test jobs only for known documentation changes; otherwise run CI."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

DOC_SUFFIXES = {'.md', '.rst', '.txt', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.pdf'}
WORKFLOW = '.github/workflows/server-compat.yml'
CUDA_JOBS = ('cuda-build-linux', 'polystrata-cuda-build-linux', 'polystrata-hip-build-linux')
NON_CUDA_PATHS = {
    WORKFLOW, '.github/scripts/classify_changes.py', 'tests/test_ci_changes.py',
    'tests/test_ninfer_dflash_conditioning.py',
    'tests/query-pressure-regression.py',
    'scripts/windows/start-backend.ps1', 'scripts/windows/start-ninfer.ps1',
    'scripts/windows/start-server.ps1', 'scripts/windows/test-backend-launcher.ps1',
    'scripts/windows/test-launcher.ps1',
}
CONSOLE_PATHS = {
    '.github/scripts/classify_changes.py', 'tests/test_ci_changes.py',
    'tests/test_radiance_launcher.py',
    'tests/test_layerwise_launcher.py', 'tests/test_layerwise_product.py',
    'tests/test_windows_worker_args.py', 'scripts/windows/start-backend.ps1',
    'START-HERE-CN.sh', 'START-HERE-EN.sh', 'START-HERE-CN.bat', 'START-HERE-EN.bat',
    'scripts/local-console.py', 'scripts/launcher-cli.py', 'scripts/launcher-capabilities.json',
    'scripts/console_chat.py', 'scripts/console_radiance.py', 'scripts/console_state.py', 'scripts/console_hardware.py', 'scripts/model_health.py',
    'scripts/requirements-console.txt', 'scripts/speculative-capabilities.json',
    'scripts/linux/start-ui.sh', 'scripts/linux/package-console.py',
    'scripts/windows/start-ui.ps1', 'scripts/windows/package-console.ps1',
    'scripts/windows/package-polystrata.ps1',
    'scripts/windows_amd_hardware.py', 'scripts/windows_gpu_hardware.py',
    'tests/test_windows_amd_hardware.py', 'tests/test_windows_gpu_hardware.py',
    'scripts/windows/package-unified.py', 'scripts/windows/native-process.ps1',
    'scripts/windows/describe-launch-json.ps1', 'scripts/install-console-dependency.py',
    'scripts/windows/start-polystrata.ps1', 'scripts/windows/select-model.ps1',
    'ui/package.json', 'ui/package-lock.json', 'ui/README.md',
}

# These files audit archived results; none is an input to a native target.
EVIDENCE_PATHS = {
    'backends/radiance/README.md',
    'backends/radiance/tools/check_cpu_image_evidence.py',
    'backends/radiance/tests/cpu_image_evidence_test.py',
    'docs/benchmarks/radiance-cpu-image-20261103.json',
}
EVIDENCE_TEST_STEP = (
    '      - name: Test offline CPU image evidence checks\n'
    "        if: steps.classify.outputs.run_ci == 'true'\n"
    '        run: python3 backends/radiance/tests/cpu_image_evidence_test.py\n'
)
CONSOLE_CONTRACT_TEST_STEP = (
    '      - name: Test model-free recall and worker contracts\n'
    '        shell: bash\n'
    '        run: |\n'
    '          python -m unittest discover -s tests -p "test_layerwise_launcher.py"\n'
    '          python -m unittest discover -s tests -p "test_layerwise_product.py"\n'
    '          python -m unittest discover -s tests -p "test_windows_worker_args.py"\n'
)
HARDWARE_TEST_STEP = (
    '      - name: Test hardware telemetry contracts without a GPU\n'
    '        shell: bash\n'
    '        run: |\n'
    '          python -m unittest discover -s tests -p "test_console_hardware.py"\n'
    '          python -m unittest discover -s tests -p "test_console_radiance.py"\n'
    '          python -m unittest discover -s tests -p "test_windows_*hardware.py"\n'
)


def is_console_path(path):
    if '..' in PurePosixPath(path).parts or path.startswith('/'):
        return False
    return (path in CONSOLE_PATHS or path.startswith(('scripts/launcher/', 'scripts/model_health_ninfer/', 'ui/monitor/', 'ui/lightweight/'))
            or bool(re.fullmatch(r'tests/test_(launcher[^/]*|local_console|console_[^/]+|one_click|model_health)\.py', path))
            or bool(re.fullmatch(r'ui/test-[^/]+\.cjs', path)))


def requires_native(paths, before=None, after=None, cwd=None):
    if radiance_only_change(paths, before, after, cwd):
        return False
    # Unknown paths and CI recipe changes stay conservative. Pure UI changes
    # still run Python/browser checks, without configuring or building engines.
    return not paths or any(
        not is_documentation(path) and not is_console_path(path) and path not in EVIDENCE_PATHS
        and not (path == WORKFLOW and only_test_step_added(before, after,
                 (EVIDENCE_TEST_STEP, CONSOLE_CONTRACT_TEST_STEP, HARDWARE_TEST_STEP), cwd))
        for path in paths)


def radiance_only_change(paths, before=None, after=None, cwd=None):
    """Route isolated Radiance work to its own native checks, retaining unknown-path fallback."""
    if not paths or not any(path not in EVIDENCE_PATHS and path.startswith(('backends/radiance/', 'backends/patches/radiance-'))
                            for path in paths):
        return False
    if WORKFLOW in paths and not only_radiance_workflow_changed(before, after, cwd):
        return False
    for path in paths:
        parts = PurePosixPath(path).parts
        if '..' in parts or path.startswith('/'):
            return False
        if (is_documentation(path) or is_console_path(path) or path == WORKFLOW
                or path.startswith('backends/radiance/')
                or re.fullmatch(r'backends/patches/radiance-[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.patch', path)
                or re.fullmatch(r'docs/benchmarks/radiance-[A-Za-z0-9_-]+\.json', path)):
            continue
        if path == '.gitattributes' and before and after:
            try:
                old = git('show', before + ':.gitattributes', cwd=cwd).decode().replace('\r\n', '\n')
                new = git('show', after + ':.gitattributes', cwd=cwd).decode().replace('\r\n', '\n')
            except (OSError, UnicodeError, subprocess.CalledProcessError):
                return False
            addition = ('\n# These files retain their existing CRLF byte layout.\n'
                        'backends/radiance/upstream.json whitespace=cr-at-eol\n'
                        'ui/test-monitor.cjs whitespace=cr-at-eol\n'
                        '.github/workflows/server-compat.yml whitespace=cr-at-eol\n')
            chat_attributes = '\n# Preserve existing lightweight chat CRLF blobs.\nui/lightweight/+page.svelte whitespace=cr-at-eol\nui/lightweight/labels.ts whitespace=cr-at-eol\n'
            if new in (old + addition, old + chat_attributes):
                continue
        return False
    return True


def only_radiance_workflow_changed(before, after, cwd=None):
    if not before or not after:
        return False
    try:
        old = git('show', before + ':' + WORKFLOW, cwd=cwd).decode().replace('\r\n', '\n')
        new = git('show', after + ':' + WORKFLOW, cwd=cwd).decode().replace('\r\n', '\n')
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        return False
    def sections(text):
        start = text.find('\njobs:\n')
        if start < 0:
            return None, {}
        matches = list(re.finditer(r'^  ([A-Za-z0-9_-]+):[ \t]*$', text[start:], re.MULTILINE))
        return text[:start], {m[1]: text[start+m.start():start+(matches[i+1].start()
            if i+1<len(matches) else len(text)-start)] for i,m in enumerate(matches)}
    prefix_old, jobs_old = sections(old)
    prefix_new, jobs_new = sections(new)
    if prefix_old is None or prefix_old != prefix_new or jobs_old.keys() != jobs_new.keys():
        return False
    if ('      run_radiance: ${{ steps.classify.outputs.run_radiance }}\n' in jobs_old.get('changes', '') and
            '          RUN_RADIANCE: ${{ needs.changes.outputs.run_radiance }}\n' in jobs_old.get('host-and-server-units-linux', '')):
        # Routing already exists: only its own job body may change. Retain the
        # bootstrap transformation below for the original routing migration.
        return all(body == jobs_old[name] for name, body in jobs_new.items()
                   if name != 'radiance-host-contracts')
    output = '      run_native: ${{ steps.classify.outputs.run_native }}\n'
    expected_changes = jobs_old.get('changes', '').replace(output,
        output + '      run_radiance: ${{ steps.classify.outputs.run_radiance }}\n')
    expected_gate = jobs_old.get('host-and-server-units-linux', '')
    env = '          RUN_NATIVE: ${{ needs.changes.outputs.run_native }}\n'
    expected_gate = expected_gate.replace(env,
        env + '          RUN_RADIANCE: ${{ needs.changes.outputs.run_radiance }}\n')
    expected_gate = expected_gate.replace('            test "$RADIANCE_RESULT" = success\n', '')
    expected_gate = expected_gate.replace('            test "$RADIANCE_RESULT" = skipped\n', '')
    expected_gate = expected_gate.replace('          if [ "$RUN_CUDA" = true ]; then',
        '          if [ "$RUN_RADIANCE" = true ]; then\n'
        '            test "$RADIANCE_RESULT" = success\n'
        '          else\n'
        '            test "$RUN_RADIANCE" = false\n'
        '            test "$RADIANCE_RESULT" = skipped\n'
        '          fi\n'
        '          if [ "$RUN_CUDA" = true ]; then')
    for name, body in jobs_new.items():
        if name == 'radiance-host-contracts':
            continue
        expected = expected_changes if name == 'changes' else expected_gate if name == 'host-and-server-units-linux' else jobs_old[name]
        if body != expected:
            return False
    return True


def is_documentation(path):
    parts = PurePosixPath(path).parts
    if not parts or '..' in parts or path.startswith('/'):
        return False
    name = parts[-1]
    suffix = PurePosixPath(name).suffix.lower()
    if len(parts) == 1:
        return suffix in {'.md', '.rst'} or name in {'LICENSE', 'NOTICE', 'COPYING', 'AUTHORS'}
    if parts[0] == 'docs':
        return suffix in DOC_SUFFIXES
    return parts[0] == 'scripts' and name.startswith('README') and suffix in {'.md', '.rst'}


def requires_ci(paths):
    return not paths or not all(is_documentation(path) for path in paths)


def git(*args, cwd=None):
    return subprocess.check_output(['git', *args], cwd=cwd, stderr=subprocess.PIPE)


def only_evidence_test_step_changed(before, after, cwd=None):
    return only_test_step_added(before, after, (EVIDENCE_TEST_STEP,), cwd)


def only_test_step_added(before, after, steps, cwd=None):
    # Only an exact known model-free test addition can avoid native builds.
    # Other recipe edits retain the conservative native-build policy.
    if not all(isinstance(ref, str) and re.fullmatch(r'[0-9a-f]{40}', ref) for ref in (before, after)):
        return False
    try:
        old, new = (git('show', ref + ':' + WORKFLOW, cwd=cwd).decode('utf-8').replace('\r\n', '\n')
                    for ref in (before, after))
        return any(new.count(step) == 1 and new.replace(step, '', 1) == old for step in steps)
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        return False


def cuda_build_definition(text):
    """Keep globals and every native CUDA recipe; ignore only job eligibility."""
    text = text.replace('\r\n', '\n')
    jobs = re.search(r'^jobs:[ \t]*$', text, re.MULTILINE)
    sections = list(re.finditer(r'^  ([a-zA-Z0-9_-]+):[ \t]*$', text, re.MULTILINE))
    if jobs is None:
        return None
    recipes = {}
    for index, section in enumerate(sections):
        if section.group(1) in CUDA_JOBS:
            end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
            body = text[section.start():end]
            body = '\n'.join(line for line in body.splitlines()
                             if not re.match(r'^    (needs|if):', line))
            recipes[section.group(1)] = body
    if any(name not in recipes for name in CUDA_JOBS):
        return None
    return text[:jobs.start()], tuple(recipes[name] for name in CUDA_JOBS)


def ninfer_only_backend_change(paths, before, after, cwd=None):
    # The CUDA/HIP jobs build llama.cpp and PolyStrata, not NInfer. A manifest
    # update may skip those jobs only when every other backend record is equal.
    if 'backends/versions.json' not in paths:
        return False
    for path in paths:
        if is_documentation(path) or path in NON_CUDA_PATHS or is_console_path(path):
            continue
        # Published NInfer measurements accompany its patch manifest, but are
        # not inputs to the llama.cpp/PolyStrata CUDA or HIP builds.
        if re.fullmatch(r'docs/benchmarks/ninfer-[A-Za-z0-9_-]+\.json', path):
            continue
        if path in {'backends/versions.json', 'backends/ninfer'}:
            continue
        if re.fullmatch(r'backends/patches/ninfer-[^/]+\.patch', path):
            continue
        return False
    try:
        old, new = (json.loads(git('show', ref + ':backends/versions.json', cwd=cwd))
                    for ref in (before, after))
        if not all(isinstance(value, dict) and isinstance(value.get('ninfer'), dict)
                   for value in (old, new)):
            return False
        return ({key: value for key, value in old.items() if key != 'ninfer'} ==
                {key: value for key, value in new.items() if key != 'ninfer'})
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError):
        return False


def requires_cuda(paths, before, after, cwd=None):
    if not paths or any(not is_documentation(path) and path not in NON_CUDA_PATHS and not is_console_path(path) and path not in EVIDENCE_PATHS for path in paths):
        if not (ninfer_only_backend_change(paths, before, after, cwd)
                or radiance_only_change(paths, before, after, cwd)):
            return True
    if WORKFLOW in paths:
        try:
            old = cuda_build_definition(git('show', before + ':' + WORKFLOW, cwd=cwd).decode('utf-8'))
            new = cuda_build_definition(git('show', after + ':' + WORKFLOW, cwd=cwd).decode('utf-8'))
        except (OSError, UnicodeError, subprocess.CalledProcessError):
            return True
        if old is None or new is None or old != new:
            return True
    return False


def classify(event_name, event, cwd=None):
    if event_name == 'workflow_dispatch':
        return True, True, 'manual dispatch always runs full CI', []
    if event_name == 'pull_request':
        request = event.get('pull_request', {})
        before = request.get('base', {}).get('sha', '')
        after = request.get('head', {}).get('sha', '')
    elif event_name == 'push':
        before, after = event.get('before', ''), event.get('after', '')
    else:
        return True, True, 'unrecognized event runs full CI', []
    if any(not re.fullmatch(r'[0-9a-f]{40}', sha) or sha == '0' * 40 for sha in (before, after)):
        return True, True, 'missing comparison runs full CI', []
    try:
        if event_name == 'pull_request':
            before = git('merge-base', before, after, cwd=cwd).decode().strip()
        # A source-to-document rename must include the deleted source path.
        paths = [os.fsdecode(path) for path in git('diff', '--no-renames', '--name-only', '-z',
                                                  before, after, cwd=cwd).split(b'\0') if path]
    except (OSError, subprocess.CalledProcessError):
        return True, True, 'comparison unavailable runs full CI', []
    full = requires_ci(paths)
    cuda = requires_cuda(paths, before, after, cwd=cwd) if full else False
    return full, cuda, 'source/configuration or empty change' if full else 'documentation only', paths


def main():
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text(encoding='utf-8'))
    full, cuda, reason, paths = classify(os.environ['GITHUB_EVENT_NAME'], event)
    if os.environ['GITHUB_EVENT_NAME'] == 'pull_request':
        request = event.get('pull_request', {})
        before, after = request.get('base', {}).get('sha'), request.get('head', {}).get('sha')
    else:
        before, after = event.get('before'), event.get('after')
    native = full and requires_native(paths, before, after)
    radiance = full and (native or radiance_only_change(paths, before, after))
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        output.write('run_ci=' + str(full).lower() + '\n')
        output.write('run_cuda=' + str(cuda).lower() + '\n')
        output.write('run_native=' + str(native).lower() + '\n')
        output.write('run_radiance=' + str(radiance).lower() + '\n')
    print(json.dumps({'run_ci': full, 'run_cuda': cuda, 'run_native': native,
                      'run_radiance': radiance, 'reason': reason, 'paths': paths}, ensure_ascii=True))


if __name__ == '__main__':
    main()
