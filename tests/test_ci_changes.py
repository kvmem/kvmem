import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('ci_changes', Path(__file__).resolve().parents[1] /
                                            '.github/scripts/classify_changes.py')
changes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(changes)


class DocumentationChangesTest(unittest.TestCase):
    def test_query_launcher_contracts_run_without_engine_builds(self):
        from unittest.mock import patch
        paths = ['scripts/launcher/worker.py', 'scripts/windows/start-backend.ps1',
                 'tests/test_layerwise_launcher.py', 'tests/test_layerwise_product.py',
                 'tests/test_windows_worker_args.py', 'tests/test_launcher_export.py',
                 '.github/scripts/classify_changes.py', 'tests/test_ci_changes.py']
        self.assertTrue(changes.requires_ci(paths))
        self.assertFalse(changes.requires_native(paths))
        self.assertFalse(changes.requires_cuda(paths, 'a' * 40, 'b' * 40))
        current = (Path(__file__).resolve().parents[1] / changes.WORKFLOW).read_text(encoding='utf-8')
        self.assertEqual(current.count(changes.CONSOLE_CONTRACT_TEST_STEP), 1)
        before = current.replace(changes.CONSOLE_CONTRACT_TEST_STEP, '', 1)
        for after, expected in ((current, False), (current + '\n# unrelated edit\n', True),
                                (current.replace('cmake --build build', 'cmake --build other'), True)):
            with patch.object(changes, 'git', side_effect=lambda _, ref, **kw:
                              (before if ref.startswith('a' * 40) else after).encode()):
                self.assertEqual(changes.requires_native(paths + [changes.WORKFLOW], 'a' * 40, 'b' * 40), expected)
        for extra in ('kvmem/src/host/kvmem_store.cpp', 'backends/ninfer', 'CMakeLists.txt',
                      'scripts/windows/build-backends.ps1', 'tests/unreviewed_native_test.py'):
            self.assertTrue(changes.requires_native(paths + [extra]))
            self.assertTrue(changes.requires_cuda(paths + [extra], 'a' * 40, 'b' * 40))

    def test_chat_crlf_attributes_are_not_native_build_inputs(self):
        from unittest.mock import patch
        old = b'*.md text\n'
        addition = '\n# Preserve existing lightweight chat CRLF blobs.\nui/lightweight/+page.svelte whitespace=cr-at-eol\nui/lightweight/labels.ts whitespace=cr-at-eol\n'
        paths = ['backends/patches/radiance-win-single-stream.patch', '.gitattributes']
        for suffix, expected in [(addition, False), (addition + '* eol=crlf\n', True)]:
            with patch.object(changes, 'git', side_effect=lambda _, ref, **kw:
                              old if ref.startswith('old:') else old + suffix.encode()):
                self.assertEqual(changes.requires_native(paths, 'old', 'new'), expected)

    def test_radiance_console_and_chat_do_not_build_unrelated_engines(self):
        paths = ['scripts/console_radiance.py', 'scripts/console_state.py',
                 'ui/lightweight/+page.svelte', 'ui/lightweight/labels.ts',
                 'ui/monitor/launcher.js', 'tests/test_console_state.py']
        self.assertTrue(changes.requires_ci(paths))
        self.assertFalse(changes.requires_native(paths))
        self.assertFalse(changes.requires_cuda(paths, 'a', 'b'))
        paths += ['backends/patches/radiance-win-single-stream.patch', 'backends/radiance/upstream.json']
        self.assertTrue(changes.radiance_only_change(paths))
        self.assertFalse(changes.requires_native(paths))
        self.assertFalse(changes.requires_cuda(paths, 'a', 'b'))
        self.assertTrue(changes.requires_native(paths + ['src/unknown.cpp']))

    def test_versioned_radiance_patch_uses_only_radiance_native_checks(self):
        paths = ['backends/patches/radiance-1.17-kvmem.patch',
                 'backends/radiance/adapter/window_checkpoint.hpp',
                 '.github/scripts/classify_changes.py', 'tests/test_ci_changes.py']
        self.assertTrue(changes.radiance_only_change(paths))
        self.assertFalse(changes.requires_native(paths))
        self.assertFalse(changes.requires_cuda(paths, 'a', 'b'))
        for extra in ['backends/patches/ninfer-new.patch', 'kvmem/src/host/kvmem_store.cpp',
                      'backends/patches/radiance-1..17-kvmem.patch']:
            self.assertTrue(changes.requires_native(paths + [extra]))

    def test_existing_radiance_workflow_can_add_a_focused_unit(self):
        from unittest.mock import patch
        old = (Path(__file__).resolve().parents[1] / changes.WORKFLOW).read_bytes()
        new = old.replace(b'engine_bringup engine_mm',
                          b'engine_bringup mem/pools engine_mm')
        self.assertNotEqual(new, old)
        paths = [changes.WORKFLOW, 'backends/patches/radiance-1.17-kvmem.patch']
        with patch.object(changes, 'git', side_effect=lambda _, ref, **kw:
                          old if ref.startswith('old:') else new):
            self.assertTrue(changes.radiance_only_change(paths, 'old', 'new'))
            self.assertFalse(changes.requires_native(paths, 'old', 'new'))
        unsafe = new.replace(b'cmake --build build --parallel 2',
                             b'cmake --build unrelated --parallel 2', 1)
        self.assertNotEqual(unsafe, new)
        with patch.object(changes, 'git', side_effect=lambda _, ref, **kw:
                          old if ref.startswith('old:') else unsafe):
            self.assertTrue(changes.requires_native(paths, 'old', 'new'))

    def test_offline_evidence_changes_keep_python_checks_without_native_builds(self):
        paths = [*changes.EVIDENCE_PATHS, 'docs/radiance-cpu-image-validation-20261103.md',
                 '.github/scripts/classify_changes.py', 'tests/test_ci_changes.py']
        self.assertTrue(changes.requires_ci(paths))
        self.assertFalse(changes.requires_native(paths))
        self.assertFalse(changes.requires_cuda(paths, 'a', 'b'))
        for path in ('backends/radiance/adapter/native_memory.cpp', 'CMakeLists.txt',
                     'backends/radiance/tools/new_tool.py', 'backends/radiance/upstream.json',
                     'docs/benchmarks/unreviewed.json', '../backends/radiance/README.md'):
            scoped = path.startswith('backends/radiance/')
            self.assertEqual(changes.requires_native(paths + [path]), not scoped)
            self.assertEqual(changes.requires_cuda(paths + [path], 'a', 'b'), not scoped)
            self.assertEqual(changes.radiance_only_change(paths + [path]), scoped)

    def test_radiance_workflow_routing_keeps_other_engine_recipes_fixed(self):
        from unittest.mock import patch
        old = b'name: CI\njobs:\n  changes:\n    outputs:\n      run_native: ${{ steps.classify.outputs.run_native }}\n  host-linux:\n    run: cmake --build build --parallel 2\n  radiance-host-contracts:\n    if: old\n  host-and-server-units-linux:\n    env:\n          RUN_NATIVE: ${{ needs.changes.outputs.run_native }}\n    run: |\n            test "$RADIANCE_RESULT" = success\n            test "$RADIANCE_RESULT" = skipped\n          if [ "$RUN_CUDA" = true ]; then\n'
        cuda_jobs = b''.join(('  '+name+':\n    run: native-build\n').encode() for name in changes.CUDA_JOBS)
        old = old.replace(b'  host-and-server-units-linux:',cuda_jobs+b'  host-and-server-units-linux:')
        new = old.replace(b'      run_native: ${{ steps.classify.outputs.run_native }}\n',
            b'      run_native: ${{ steps.classify.outputs.run_native }}\n      run_radiance: ${{ steps.classify.outputs.run_radiance }}\n')
        new = new.replace(b'          RUN_NATIVE: ${{ needs.changes.outputs.run_native }}\n',
            b'          RUN_NATIVE: ${{ needs.changes.outputs.run_native }}\n          RUN_RADIANCE: ${{ needs.changes.outputs.run_radiance }}\n')
        new = new.replace(b'            test "$RADIANCE_RESULT" = success\n',b'').replace(b'            test "$RADIANCE_RESULT" = skipped\n',b'')
        new = new.replace(b'          if [ "$RUN_CUDA" = true ]; then',
            b'          if [ "$RUN_RADIANCE" = true ]; then\n            test "$RADIANCE_RESULT" = success\n          else\n            test "$RUN_RADIANCE" = false\n            test "$RADIANCE_RESULT" = skipped\n          fi\n          if [ "$RUN_CUDA" = true ]; then')
        attrs = b'*.md text\n'
        current_attrs = attrs + b'\n# These files retain their existing CRLF byte layout.\nbackends/radiance/upstream.json whitespace=cr-at-eol\nui/test-monitor.cjs whitespace=cr-at-eol\n.github/workflows/server-compat.yml whitespace=cr-at-eol\n'
        paths = ['backends/patches/radiance-vision-pipeline.patch', changes.WORKFLOW,
                 '.github/scripts/classify_changes.py', 'tests/test_ci_changes.py', '.gitattributes',
                 'tests/test_radiance_launcher.py',
                 'docs/benchmarks/radiance-vision-pipeline-20261104.json']
        def data(command, ref, **kwargs):
            self.assertEqual(command,'show')
            return {'old:'+changes.WORKFLOW:old,'new:'+changes.WORKFLOW:new,
                    'old:.gitattributes':attrs,'new:.gitattributes':current_attrs}[ref]
        with patch.object(changes,'git',side_effect=data):
            self.assertTrue(changes.radiance_only_change(paths,'old','new'))
            self.assertFalse(changes.requires_native(paths,'old','new'))
            self.assertFalse(changes.requires_cuda(paths,'old','new'))
            for extra in ['CMakeLists.txt','kvmem/src/host/kvmem_store.cpp',
                          'backends/patches/ninfer-new.patch','../backends/radiance/file.cpp']:
                self.assertTrue(changes.requires_native(paths+[extra],'old','new'))
                self.assertTrue(changes.requires_cuda(paths+[extra],'old','new'))
            changed = new.replace(b'cmake --build build --parallel 2',b'cmake --build other --parallel 2')
            self.assertNotEqual(new,changed)
            new = changed
            self.assertFalse(changes.radiance_only_change(paths,'old','new'))

    def test_only_exact_evidence_workflow_step_avoids_native_builds(self):
        from unittest.mock import patch
        old = 'name: Test\non: pull_request\njobs:\n  changes:\n    steps:\n'
        new = old + changes.EVIDENCE_TEST_STEP
        paths = [changes.WORKFLOW, *changes.EVIDENCE_PATHS]
        with patch.object(changes, 'git', side_effect=[old.encode(), new.encode()]):
            self.assertFalse(changes.requires_native(paths, 'a' * 40, 'b' * 40))
        for unsafe in (new + '      - run: cmake --build everything\n',
                       new.replace('python3 backends/', 'python3 other/'),
                       new + 'env:\n  CXXFLAGS: changed\n'):
            with patch.object(changes, 'git', side_effect=[old.encode(), unsafe.encode()]):
                self.assertTrue(changes.requires_native(paths, 'a' * 40, 'b' * 40))
        self.assertTrue(changes.requires_native(paths))

    def test_ninfer_manifest_changes_do_not_build_other_backends_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'backends/versions.json'
            manifest.parent.mkdir()
            base_data = {'layout': 'patched', 'ninfer': {'base_commit': 'old'},
                         'llamacpp': {'base_commit': 'llama'},
                         'polystrata': {'base_commit': 'poly'}}

            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root,
                                               stderr=subprocess.PIPE).decode().strip()

            def commit(data):
                manifest.write_text(json.dumps(data) if isinstance(data, dict) else data,
                                    encoding='utf-8')
                git('add', '.')
                git('-c', 'user.name=CI test', '-c', 'user.email=ci-test@example.invalid',
                    'commit', '-qm', 'manifest')
                return git('rev-parse', 'HEAD')

            git('init', '-q')
            before = commit(base_data)
            new_data = dict(base_data, ninfer={'base_commit': 'new'})
            patch = root / 'backends/patches/ninfer-feature.patch'
            patch.parent.mkdir()
            patch.write_text('native patch\n', encoding='utf-8')
            evidence = root / 'docs/benchmarks/ninfer-fp8-image-stores-20261103.json'
            evidence.parent.mkdir(parents=True)
            evidence.write_text('{"measurement": 1}\n', encoding='utf-8')
            after = commit(new_data)
            event = {'pull_request': {'base': {'sha': before}, 'head': {'sha': after}}}
            full, cuda, _, paths = changes.classify('pull_request', event, root)
            self.assertTrue(full)
            self.assertTrue(changes.requires_native(paths))
            self.assertFalse(cuda)
            self.assertFalse(changes.classify('push', {'before': before, 'after': after}, root)[1])
            self.assertFalse(changes.requires_cuda(paths + ['docs/guide.md', 'tests/test_ci_changes.py',
                                                         'tests/test_ninfer_dflash_conditioning.py',
                                                         'tests/query-pressure-regression.py'],
                                                  before, after, root))
            for extra in ('kvmem/src/host/memory_contract.cpp', 'CMakeLists.txt',
                          'backends/patches/llamacpp-kvmem.patch', 'backends/polystrata',
                          'docs/config.json', 'docs/benchmarks/polystrata-speed.json',
                          'docs/benchmarks/../ninfer-feature.json',
                          'backends/patches/../ninfer-feature.patch'):
                self.assertTrue(changes.requires_cuda(paths + [extra], before, after, root))
            self.assertTrue(changes.requires_cuda(paths, 'f' * 40, after, root))
            for key in ('llamacpp', 'polystrata', 'layout'):
                mixed = dict(new_data, **{key: 'changed'})
                changed = commit(mixed)
                self.assertTrue(changes.requires_cuda(paths, before, changed, root))
            for malformed in ('not json', '[]', '{}'):
                changed = commit(malformed)
                self.assertTrue(changes.requires_cuda(paths, before, changed, root))

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
                 'scripts/windows/package-unified.py', 'scripts/windows/native-process.ps1',
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

    def test_projector_picker_changes_run_console_checks_without_engine_builds(self):
        paths = ['.github/scripts/classify_changes.py', 'tests/test_ci_changes.py',
                 'scripts/windows/start-polystrata.ps1', 'scripts/windows/select-model.ps1',
                 'scripts/local-console.py', 'scripts/model_health.py', 'ui/monitor/launcher.js',
                 'ui/test-launcher-settings.cjs', 'tests/test_local_console.py',
                 'tests/test_model_health.py', 'ui/README.md']
        self.assertTrue(changes.requires_ci(paths))
        self.assertFalse(changes.requires_native(paths))
        self.assertFalse(changes.requires_cuda(paths, 'a', 'b'))
        for extra in ('backends/polystrata', 'backends/patches/polystrata-kvmem.patch',
                      'CMakeLists.txt', 'scripts/windows/build-backends.ps1'):
            with self.subTest(extra=extra):
                self.assertTrue(changes.requires_native(paths + [extra]))
                self.assertTrue(changes.requires_cuda(paths + [extra], 'a', 'b'))

    def test_workflow_eligibility_is_separate_from_cuda_recipe(self):
        original = ('name: Test\non: pull_request\njobs:\n  cuda-build-linux:\n'
                    "    needs: changes\n    if: needs.changes.outputs.run_ci == 'true'\n"
                    '    runs-on: ubuntu-latest\n    steps:\n      - run: cmake --build build\n'
                    '  polystrata-cuda-build-linux:\n'
                    "    needs: changes\n    if: needs.changes.outputs.run_ci == 'true'\n"
                    '    runs-on: ubuntu-latest\n    steps:\n      - run: cmake --build build-polystrata\n'
                    '  polystrata-hip-build-linux:\n'
                    "    needs: changes\n    if: needs.changes.outputs.run_ci == 'true'\n"
                    '    runs-on: ubuntu-latest\n    steps:\n      - run: cmake --build build-polystrata-hip\n'
                    '  other:\n    runs-on: ubuntu-latest\n')
        self.assertIsNotNone(changes.cuda_build_definition(original))
        eligibility = original.replace('outputs.run_ci', 'outputs.run_cuda')
        self.assertEqual(changes.cuda_build_definition(original), changes.cuda_build_definition(eligibility))
        recipe = original.replace('cmake --build build', 'cmake --build another-build')
        self.assertNotEqual(changes.cuda_build_definition(original), changes.cuda_build_definition(recipe))
        poly_recipe = original.replace('--build build-polystrata', '--build different-polystrata')
        self.assertNotEqual(changes.cuda_build_definition(original), changes.cuda_build_definition(poly_recipe))
        hip_recipe = original.replace('--build build-polystrata-hip', '--build different-hip')
        self.assertNotEqual(changes.cuda_build_definition(original), changes.cuda_build_definition(hip_recipe))
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
            hip_modified = modified.replace('-DSTRATA_ENABLE_HIP=ON', '-DSTRATA_ENABLE_HIP=OFF')
            self.assertNotEqual(modified, hip_modified)
            hip_after = commit(hip_modified)
            self.assertTrue(changes.requires_cuda([changes.WORKFLOW], after, hip_after, root))
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
