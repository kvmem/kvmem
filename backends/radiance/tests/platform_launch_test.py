"""CPU-only launcher contract tests: never load a model or GPU library."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build
import build_linux
import serve
from platform_dispatch import select_platform
from draft_control_test import container


class PlatformLaunchTest(unittest.TestCase):
    def args(self, root=Path('.')):
        return argparse.Namespace(exe=root/'probe', model=root/'model.rad', home=root/'home',
                                  rocm=root/'sdk', port=18380, wsl_runtime_dir=None,
                                  drafter='auto',speculative_tokens=None)

    def test_platform_selection_and_rejection(self):
        for actual, target in [('win32', 'windows'), ('linux', 'linux')]:
            self.assertEqual(select_platform('auto', actual), target)
            self.assertEqual(select_platform(target, actual), target)
            with self.assertRaises(ValueError):
                select_platform('linux' if target == 'windows' else 'windows', actual)
        with self.assertRaises(ValueError):
            select_platform('auto', 'darwin')

    def test_build_delegates_only_matching_host(self):
        with patch.object(build, 'select_platform', return_value='linux'), patch.object(build.subprocess, 'call', return_value=9) as run:
            self.assertEqual(build.main(['--platform', 'linux', '--source', 'native', '--jobs', '2']), 9)
            self.assertEqual(Path(run.call_args.args[0][1]).name, 'build_linux.py')
            self.assertEqual(run.call_args.args[0][2:], ['--source', 'native', '--jobs', '2'])

    def test_linux_build_retains_cached_kernel_profile_unless_explicit(self):
        revision=json.loads((Path(build_linux.__file__).parents[1]/'upstream.json').read_text())['revision']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            sdk=root/'sdk'
            for relative in ('bin/hipcc','llvm/bin/clang++','lib/libamdhip64.so','lib/libhsa-runtime64.so'):
                path=sdk/relative;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            base=['build_linux.py','--source',str(root/'native'),'--build',str(root/'build'),
                  '--rocm',str(sdk),'--arch','gfx1200','--target','radiance-kvmem-server']
            for profile in (None,'qwen35-dense-tp1','all'):
                argv=base+(['--kernel-profile',profile] if profile else [])
                with self.subTest(profile=profile), patch.object(build_linux,'os') as host, \
                        patch.object(sys,'argv',argv), \
                        patch.object(build_linux,'source_revision',return_value=revision), \
                        patch.object(build_linux.subprocess,'run') as execute:
                    host.name='posix';host.environ.copy.return_value={}
                    build_linux.main()
                    configure=execute.call_args_list[0].args[0]
                    selected=[flag for flag in configure if flag.startswith('-DR4D_PROFILE=')]
                    self.assertEqual(selected,[] if profile is None else ['-DR4D_PROFILE='+profile])
                    self.assertEqual(execute.call_args_list[1].args[0][-2:],['--target','radiance-kvmem-server'])

    def test_command_capacity_is_shared(self):
        config = serve.profile()
        cmd = serve.command_for(self.args(), config)
        self.assertEqual(cmd[-len(config['arguments']):], config['arguments'])
        for name, expected in [('kvmem-budget', '36864'), ('kvmem-replay-tokens', '16384'),
                               ('max-num-batched-tokens', '256'), ('max-model-len', '262144'),
                               ('num-speculative-tokens', '3'), ('kv-cache-dtype', 'fp8'),('prefix-cache-mib','0')]:
            self.assertEqual(cmd[cmd.index('--' + name) + 1], expected)

    def test_model_profile_preserves_mtp_and_selects_iqr_v4(self):
        with tempfile.TemporaryDirectory() as directory:
            a=self.args(Path(directory))
            container(a.model,['mtp.fc.weight'],[('mtp_num_hidden_layers',1)])
            historical=serve.profile()
            resolved,plugin=serve.model_profile(a,historical)
            self.assertEqual(plugin,'qwen35_fp8')
            self.assertEqual(resolved['arguments'][:-2],historical['arguments'])
            self.assertEqual(resolved['arguments'][-2:],['--kvmem-drafter','mtp'])
            container(a.model,['dflash.fc.weight'],[
                ('draft.architectures','DFlash2Model'),('draft.dflash_config.block_size',8)],
                quant='w8a8_fwht128')
            resolved,plugin=serve.model_profile(a,historical)
            self.assertEqual(plugin,'qwen35_w8a8')
            args=resolved['arguments']
            self.assertEqual(args[args.index('--num-speculative-tokens')+1],'7')
            self.assertEqual(args[-2:],['--kvmem-drafter','dflash2'])
            self.assertEqual(historical['arguments'][historical['arguments'].index('--num-speculative-tokens')+1],'3')
            for mode,depth in [('mtp',None),('dflash2',5),('off',7)]:
                a.drafter,a.speculative_tokens=mode,depth
                with self.subTest(mode=mode,depth=depth),self.assertRaises(ValueError):
                    serve.model_profile(a,historical)
            a.drafter,a.speculative_tokens='off',None
            resolved,_=serve.model_profile(a,historical)
            self.assertEqual(resolved['arguments'][-1],'off')
            self.assertEqual(resolved['arguments'][args.index('--num-speculative-tokens')+1],'0')

    def test_linux_environment_isolated(self):
        inherited = {'PATH':'system', 'LD_PRELOAD':'wrong.so', 'LD_LIBRARY_PATH':'wrong',
                     'RADIANCE_NO_DRAFT':'1', 'KVMEM_RAD_MTP_SHARED_TOPK':'1', 'KVMEM_RAD_HIP_SINGLE_STREAM':'0',
                     'KVMEM_DIAG_ROCDXG_DIR':'wrong', 'PAL_ALWAYS_RESIDENT':'1',
                     'RADIANCE_HIP_FIXED_VA':'2', 'KV_DIAG_FOO':'1', 'HSA_ENABLE_DXG_DETECTION':'1'}
        env = serve.child_environment(self.args(), 'linux', False, serve.profile(), inherited)
        for name in inherited:
            if name not in ('PATH', 'LD_LIBRARY_PATH'):
                self.assertNotIn(name, env)
        self.assertEqual(env['KVMEM_RAD_MTP_HEAD'], 'shared')
        self.assertEqual(env['KVMEM_CAPTURE_EARLY_SEAL'], '1')
        self.assertEqual(env['OMP_NUM_THREADS'], '2')
        self.assertNotIn('wrong', env['LD_LIBRARY_PATH'])

    def test_windows_single_stream_explicit_fallback(self):
        key = 'KVMEM_RAD_HIP_SINGLE_STREAM'
        for value in ('0', '1', 'invalid'):
            with self.subTest(value=value):
                env = serve.child_environment(self.args(), 'windows', False, serve.profile(),
                                              {key: value, 'KV_DIAG_FOO': '1'})
                self.assertEqual(env.get(key), '0' if value == '0' else None)
                self.assertNotIn('KV_DIAG_FOO', env)

    def test_wsl_environment_explicit_library_first(self):
        a = self.args()
        a.wsl_runtime_dir = Path('qualified-runtime')
        env = serve.child_environment(a, 'linux', True, serve.profile(), {})
        self.assertTrue(env['LD_LIBRARY_PATH'].startswith(str(a.wsl_runtime_dir.resolve()) + ':'))
        self.assertEqual(env['HSA_ENABLE_DXG_DETECTION'], '1')
        self.assertNotIn('LD_PRELOAD', env)

    def test_linux_inputs_and_wsl_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            a = self.args(Path(directory))
            paths = [a.exe, a.model, a.home/'kernels.rkb', a.home/'kernels/libr4d.so',
                     a.home/'architectures/qwen35_fp8.so', a.rocm/'lib/libamdhip64.so', a.rocm/'lib/libhsa-runtime64.so']
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'\x7fELFfixture')
            serve.validate_inputs(a, 'linux', False)
            with self.assertRaisesRegex(ValueError, 'WSL requires'):
                serve.validate_inputs(a, 'linux', True)
            a.wsl_runtime_dir = Path(directory)/'unqualified-runtime'
            with self.assertRaisesRegex(ValueError, 'qualified scratch'):
                serve.validate_inputs(a, 'linux', True)
            with self.assertRaisesRegex(ValueError, 'only valid'):
                serve.validate_inputs(a, 'linux', False)
            a.wsl_runtime_dir = None
            a.exe.write_bytes(b'MZwindows')
            with self.assertRaisesRegex(ValueError, 'not a Linux ELF'):
                serve.validate_inputs(a, 'linux', False)

    def test_runtime_all_aliases_checked_no_bypass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('librocdxg.so.1.2.0', 'librocdxg.so.1', 'librocdxg.so'):
                (root/name).write_bytes(b'fixture')
            with self.assertRaises(ValueError):
                serve.validate_runtime(root)
            with patch.object(serve.hashlib, 'sha256') as digest:
                digest.return_value.hexdigest.return_value = serve.WSL_RUNTIME_SHA256
                serve.validate_runtime(root)
                self.assertEqual(digest.call_count, 3)
                (root/'librocdxg.so').unlink()
                with self.assertRaises(ValueError):
                    serve.validate_runtime(root)

    def test_resource_counters_and_thresholds(self):
        self.assertEqual(serve.proc_kib('MemAvailable: 1024 kB\n', 'MemAvailable'), 1048576)
        for text in ('', 'VmSwap: -1 kB', 'VmSwap: 1 MB'):
            with self.assertRaises(RuntimeError):
                serve.proc_kib(text, 'VmSwap')
        with patch.object(serve, 'available_memory', return_value=(10*serve.GIB,)):
            serve.check_resources('linux', startup=True)
        with patch.object(serve, 'available_memory', return_value=(9*serve.GIB,)):
            with self.assertRaises(RuntimeError):
                serve.check_resources('linux', startup=True)
        with patch.object(serve, 'available_memory', return_value=(2*serve.GIB,)), patch.object(serve, 'child_swap', return_value=64*1024**2):
            serve.check_resources('linux', pid=123)
        with patch.object(serve, 'available_memory', return_value=(2*serve.GIB,)), patch.object(serve, 'child_swap', return_value=64*1024**2+1):
            with self.assertRaisesRegex(RuntimeError, 'swap'):
                serve.check_resources('linux', pid=123)

    def test_startup_failure_does_not_spawn(self):
        with patch.object(serve, 'check_resources', side_effect=RuntimeError('low')), patch.object(serve.subprocess, 'Popen') as spawn:
            with self.assertRaises(RuntimeError):
                serve.supervise(['probe'], {}, 'linux')
            spawn.assert_not_called()

    def test_child_exit_propagated_without_termination(self):
        child = Mock(pid=123, returncode=7)
        child.poll.return_value = 7
        with patch.object(serve, 'check_resources'), patch.object(serve.subprocess, 'Popen', return_value=child):
            self.assertEqual(serve.supervise(['probe'], {}, 'linux'), 7)
        child.terminate.assert_not_called()

    def test_resource_failure_terminates_only_spawned_child(self):
        child = Mock(pid=123)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('probe', 15), 0]
        with patch.object(serve, 'check_resources', side_effect=[None, RuntimeError('swap')]) as check, patch.object(serve.subprocess, 'Popen', return_value=child):
            with self.assertRaises(RuntimeError):
                serve.supervise(['probe'], {}, 'linux')
            self.assertEqual(check.call_args.kwargs, {'pid':123})
        child.terminate.assert_called_once()
        child.kill.assert_called_once()

    def test_zombie_resource_race_preserves_exit_code(self):
        child = Mock(pid=123, returncode=0)
        child.poll.side_effect = [None, 0, 0]
        with patch.object(serve, 'check_resources', side_effect=[None, RuntimeError('missing VmSwap')]), patch.object(serve.subprocess, 'Popen', return_value=child):
            self.assertEqual(serve.supervise(['probe'], {}, 'linux'), 0)
        child.terminate.assert_not_called()

    def test_dryrun_never_spawns(self):
        args = ['--exe','probe','--model','model','--home','home','--rocm','sdk','--dry-run']
        with patch.object(serve, 'model_profile',return_value=(serve.profile(),'qwen35_w8a8')), patch.object(serve, 'validate_inputs') as validate, patch.object(serve, 'supervise') as run:
            self.assertEqual(serve.main(args), 0)
            self.assertEqual(validate.call_args.args[-1],'qwen35_w8a8')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
