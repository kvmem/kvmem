"""Skip build/test jobs only for known documentation changes; otherwise run CI."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

DOC_SUFFIXES = {'.md', '.rst', '.txt', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.pdf'}
WORKFLOW = '.github/workflows/server-compat.yml'
CUDA_JOBS = ('cuda-build-linux', 'polystrata-cuda-build-linux')
NON_CUDA_PATHS = {
    WORKFLOW, '.github/scripts/classify_changes.py', 'tests/test_ci_changes.py',
    'scripts/windows/start-backend.ps1', 'scripts/windows/start-ninfer.ps1',
    'scripts/windows/start-server.ps1', 'scripts/windows/test-backend-launcher.ps1',
    'scripts/windows/test-launcher.ps1',
}
CONSOLE_PATHS = {
    'START-HERE-CN.sh', 'START-HERE-EN.sh', 'START-HERE-CN.bat', 'START-HERE-EN.bat',
    'scripts/local-console.py', 'scripts/launcher-cli.py', 'scripts/launcher-capabilities.json',
    'scripts/console_chat.py', 'scripts/console_hardware.py', 'scripts/model_health.py',
    'scripts/requirements-console.txt', 'scripts/speculative-capabilities.json',
    'scripts/linux/start-ui.sh', 'scripts/linux/package-console.py',
    'scripts/windows/start-ui.ps1', 'scripts/windows/package-console.ps1',
    'scripts/windows/describe-launch-json.ps1', 'scripts/install-console-dependency.py',
    'ui/package.json', 'ui/package-lock.json',
}


def is_console_path(path):
    if '..' in PurePosixPath(path).parts or path.startswith('/'):
        return False
    return (path in CONSOLE_PATHS or path.startswith(('scripts/launcher/', 'scripts/model_health_ninfer/', 'ui/monitor/'))
            or bool(re.fullmatch(r'tests/test_(launcher[^/]*|local_console|console_[^/]+|one_click|model_health)\.py', path))
            or bool(re.fullmatch(r'ui/test-[^/]+\.cjs', path)))


def requires_native(paths):
    # Unknown paths and CI recipe changes stay conservative. Pure UI changes
    # still run Python/browser checks, without configuring or building engines.
    return not paths or any(not is_documentation(path) and not is_console_path(path) for path in paths)


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


def requires_cuda(paths, before, after, cwd=None):
    if not paths or any(not is_documentation(path) and path not in NON_CUDA_PATHS and not is_console_path(path) for path in paths):
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
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        output.write('run_ci=' + str(full).lower() + '\n')
        output.write('run_cuda=' + str(cuda).lower() + '\n')
        output.write('run_native=' + str(full and requires_native(paths)).lower() + '\n')
    print(json.dumps({'run_ci': full, 'run_cuda': cuda, 'run_native': full and requires_native(paths), 'reason': reason, 'paths': paths}, ensure_ascii=True))


if __name__ == '__main__':
    main()
