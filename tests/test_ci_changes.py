import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('ci_changes', Path(__file__).resolve().parents[1] /
                                            '.github/scripts/classify_changes.py')
changes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(changes)


class DocumentationChangesTest(unittest.TestCase):
    def test_documentation_and_images_skip_builds(self):
        self.assertFalse(changes.requires_ci(['README.md', 'docs/guide.md', 'docs/assets/flow.svg',
                                             'scripts/windows/README-quantizer.md', 'LICENSE']))

    def test_code_configurations_and_submodules_always_run(self):
        for path in ('src/main.cpp', 'docs/example.py', 'docs/config.json', 'CMakeLists.txt',
                     'VERSION', '.github/workflows/server-compat.yml',
                     '.github/scripts/classify_changes.py', 'backends/ninfer',
                     'backends/versions.json', 'scripts/windows/start-ninfer.ps1',
                     'tests/fixtures/answer.md'):
            with self.subTest(path=path):
                self.assertTrue(changes.requires_ci(['README.md', path]))

    def test_empty_unknown_and_manual_events_run(self):
        self.assertTrue(changes.requires_ci([]))
        self.assertTrue(changes.classify('workflow_dispatch', {})[0])
        self.assertTrue(changes.classify('unknown', {})[0])
        self.assertTrue(changes.classify('pull_request', {})[0])
        self.assertTrue(changes.classify('push', {'before': '0' * 40, 'after': '1' * 40})[0])
        self.assertTrue(changes.classify('workflow_dispatch', {})[1])

    def test_console_changes_run_host_scripts_without_engine_builds(self):
        paths = ['scripts/local-console.py', 'scripts/launcher/plan.py', 'scripts/linux/start-ui.sh',
                 'ui/monitor/launcher.js', 'tests/test_launcher_services.py', 'docs/linux-launcher.md']
        self.assertTrue(changes.requires_ci(paths))
        self.assertFalse(changes.requires_native(paths))
        self.assertFalse(changes.requires_cuda(paths, 'a', 'b'))
        for extra in ('kvmem/src/host/memory_contract.cpp', 'backends/ninfer', '.github/workflows/server-compat.yml',
                      'scripts/linux/build-engine.sh', 'scripts/launcher/../../CMakeLists.txt'):
            self.assertTrue(changes.requires_native(paths + [extra]))

    def test_launchers_do_not_recompile_unchanged_cuda(self):
        paths = ['scripts/windows/start-ninfer.ps1', 'scripts/windows/test-backend-launcher.ps1', 'README.md']
        self.assertTrue(changes.requires_ci(paths))
        self.assertFalse(changes.requires_cuda(paths, 'a', 'b'))
        for path in ('src/adapter/llama-memory-kvmem.cpp', 'backends/ninfer', 'CMakeLists.txt',
                     'scripts/windows/build-backends.ps1', 'backends/patches/ninfer-kvmem.patch',
                     'backends/polystrata', 'backends/versions.json',
                     'backends/patches/polystrata-attention-feedback-performance.patch',
                     'kvmem/src/host/memory_contract.cpp'):
            with self.subTest(path=path):
                self.assertTrue(changes.requires_cuda(paths + [path], 'a', 'b'))

    def test_workflow_eligibility_is_separate_from_cuda_recipe(self):
        original = ('name: Test\non: pull_request\njobs:\n  cuda-build-linux:\n'
                    "    needs: changes\n    if: needs.changes.outputs.run_ci == 'true'\n"
                    '    runs-on: ubuntu-latest\n    steps:\n      - run: cmake --build build\n'
                    '  polystrata-cuda-build-linux:\n'
                    "    needs: changes\n    if: needs.changes.outputs.run_ci == 'true'\n"
                    '    runs-on: ubuntu-latest\n    steps:\n      - run: cmake --build build-polystrata\n'
                    '  other:\n    runs-on: ubuntu-latest\n')
        self.assertIsNotNone(changes.cuda_build_definition(original))
        eligibility = original.replace('outputs.run_ci', 'outputs.run_cuda')
        self.assertEqual(changes.cuda_build_definition(original), changes.cuda_build_definition(eligibility))
        recipe = original.replace('cmake --build build', 'cmake --build another-build')
        self.assertNotEqual(changes.cuda_build_definition(original), changes.cuda_build_definition(recipe))
        poly_recipe = original.replace('--build build-polystrata', '--build different-polystrata')
        self.assertNotEqual(changes.cuda_build_definition(original), changes.cuda_build_definition(poly_recipe))
        removed = original.replace('  polystrata-cuda-build-linux:', '  removed-build:')
        self.assertIsNone(changes.cuda_build_definition(removed))
        global_env = original.replace('jobs:', 'env:\n  CXXFLAGS: new-flags\njobs:')
        self.assertNotEqual(changes.cuda_build_definition(original), changes.cuda_build_definition(global_env))
        self.assertIsNone(changes.cuda_build_definition('jobs:\n  host-linux:\n'))

    def test_workflow_only_polystrata_recipe_change_runs_cuda(self):
        # Exercise git comparison too: workflow-only changes must not bypass
        # the new job just because the llama.cpp recipe is unchanged.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workflow = root / changes.WORKFLOW
            workflow.parent.mkdir(parents=True)
            original = (Path(__file__).resolve().parents[1] / changes.WORKFLOW).read_text(encoding='utf-8')

            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE).decode().strip()

            def commit(text):
                workflow.write_text(text, encoding='utf-8')
                git('add', '.')
                git('-c', 'user.name=CI test', '-c', 'user.email=ci-test@example.invalid',
                    'commit', '-qm', 'workflow')
                return git('rev-parse', 'HEAD')

            git('init', '-q')
            before = commit(original)
            modified = original.replace('--build build-polystrata', '--build changed-polystrata')
            self.assertNotEqual(original, modified)
            after = commit(modified)
            self.assertTrue(changes.requires_cuda([changes.WORKFLOW], before, after, root))
            renamed = commit(modified.replace('name: Server and UI compatibility', 'name: Renamed CI'))
            self.assertTrue(changes.requires_cuda([changes.WORKFLOW], after, renamed, root))

    def test_real_git_diff_renames_and_diverged_pull_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE).decode().strip()

            def commit(message):
                git('add', '.')
                git('-c', 'user.name=CI test', '-c', 'user.email=ci-test@example.invalid',
                    'commit', '-qm', message)
                return git('rev-parse', 'HEAD')

            git('init', '-q', '-b', 'test-base')
            (root / 'README.md').write_text('base\n')
            (root / 'main.cpp').write_text('int main() {}\n')
            base = commit('base')
            (root / 'README.md').write_text('updated documentation\n')
            docs = commit('documentation')
            event = {'pull_request': {'base': {'sha': base}, 'head': {'sha': docs}}}
            self.assertFalse(changes.classify('pull_request', event, root)[0])
            self.assertFalse(changes.classify('pull_request', event, root)[1])
            self.assertFalse(changes.classify('push', {'before': base, 'after': docs}, root)[0])
            git('checkout', '-q', '-b', 'base-update', base)
            (root / 'other.cpp').write_text('int other;\n')
            advanced_base = commit('base advances independently')
            event['pull_request']['base']['sha'] = advanced_base
            self.assertFalse(changes.classify('pull_request', event, root)[0])
            git('checkout', '-q', 'test-base')
            (root / 'main.cpp').rename(root / 'example.md')
            renamed = commit('move source into markdown')
            full, cuda, _, paths = changes.classify('push', {'before': docs, 'after': renamed}, root)
            self.assertTrue(full)
            self.assertTrue(cuda)
            self.assertIn('main.cpp', paths)
            self.assertIn('example.md', paths)
            self.assertTrue(changes.classify('push', {'before': 'f' * 40, 'after': renamed}, root)[0])


if __name__ == '__main__':
    unittest.main()
