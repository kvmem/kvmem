"""Native startup contracts for the engine-specific Web UI templates (no GPU)."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

from test_windows_worker_args import WorkerArgsTests, SHELLS, ROOT

sys.path.insert(0, str(ROOT / 'scripts'))
from console_chat import context_payload


@unittest.skipUnless(sys.platform == 'win32' and SHELLS, 'Windows PowerShell launchers')
class EngineSettingsTests(unittest.TestCase):
    setUpClass = classmethod(WorkerArgsTests.setUpClass.__func__)
    setUp = WorkerArgsTests.setUp
    tearDown = WorkerArgsTests.tearDown
    run_launcher = WorkerArgsTests.run_launcher
    plan = WorkerArgsTests.plan
    def llama(self, shell, extra):
        model = self.folder / "model & ' spaces.gguf"
        model.touch()
        return subprocess.run([shell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', str(ROOT / 'scripts/windows/start-backend.ps1'), '-Backend', 'llamacpp',
            '-Worker', sys.executable, '-Model', str(model), '-Gpu', 'GPU-test', '-DryRun', *extra],
            capture_output=True, encoding='utf-8', errors='replace', timeout=45)

    def test_llama_thinking_vision_and_unlimited(self):
        projector = self.folder / "matching mmproj.gguf"
        projector.touch()
        for shell in SHELLS:
            for explicit in (False, True):
                with self.subTest(shell=shell, explicit=explicit):
                    extra = ['-Mmproj', str(projector)] if explicit else []
                    args = self.plan(self.llama(shell, ['-MaxTokens', '0', '-ThinkingMode', 'on',
                        '-ThinkingBudget', '16384', '-ReasoningEffort', 'xhigh', '-PreserveThinking',
                        '-Vision', '-VisionResidency', 'cpu', '-VisionTokens', '1024', *extra]))['argv']
                    for flag, value in {'-n':'-1', '--reasoning-budget':'16384', '--reasoning-effort':'xhigh',
                                        '--image-max-tokens':'1024', '--mmproj':str(projector)}.items():
                        self.assertEqual(args[args.index(flag)+1],value)
                    self.assertIn('--enable-thinking',args)
                    self.assertIn('--no-mmproj-offload',args)
                    self.assertEqual(json.loads(args[args.index('--chat-template-kwargs')+1]),{'preserve_thinking':True})
            args = self.plan(self.llama(shell,['-ThinkingBudget','0']))['argv']
            self.assertEqual(args[args.index('--reasoning-budget')+1],'-1')
            self.assertNotIn('--mmproj',args)
            other = self.folder / 'ambiguous-mmproj.gguf';other.touch()
            self.assertNotEqual(self.llama(shell,['-Vision']).returncode,0)
            other.unlink()
            for options in (['-ThinkingBudget','10','-WorkerArgs','--reasoning-budget','20'],
                            ['-MaxTokens','0','-WorkerArgs','-n','30'],
                            ['-Vision','-WorkerArgs','--mmproj',str(projector)]):
                self.assertNotEqual(self.llama(shell,options).returncode,0)

    def test_ninfer_unlimited_omits_only_default_output_cap(self):
        for shell in SHELLS:
            args = self.plan(self.run_launcher(shell,'start-backend.ps1',[
                '-MaxTokens','0','-ThinkingBudget','16384','-ReasoningEffort','xhigh','-PreserveThinking']))['argv']
            self.assertNotIn('--default-max-tokens',args)
            self.assertIn('--preserve-thinking',args)
            self.assertEqual(args[args.index('--default-thinking-budget')+1],'16384')
            limited = self.plan(self.run_launcher(shell,'start-backend.ps1',['-MaxTokens','123']))['argv']
            self.assertEqual(limited[limited.index('--default-max-tokens')+1],'123')


del WorkerArgsTests


class ChatDefaultsTests(unittest.TestCase):
    def state(self, backend, **params):
        return {'state':'ready','active_config':{'parameters':{'Backend':backend,**params}}}

    def test_native_preservation_takes_precedence(self):
        result=context_payload(self.state('llamacpp',PreserveThinking=False),{
            'kvmem':{'defaults':{'chat_template_kwargs':{'preserve_thinking':True,'reasoning_effort':'xhigh'}}}})
        self.assertTrue(result['effective']['preserve_thinking'])
        self.assertTrue(result['capabilities']['preserve'])
        self.assertIn('xhigh',result['capabilities']['efforts'])

    def test_ninfer_reasoning_uses_native_or_confirmed_launch_values(self):
        state = self.state('ninfer', Model='Qwen-unknown.ninfer', PreserveThinking=False)
        result = context_payload(state, {})
        self.assertNotIn('preserve_thinking',result['effective'])
        self.assertNotIn('enable_thinking',result['effective'])
        state['worker'] = {'explicit_thinking_switches': True}
        state['active_config']['parameters']['ThinkingMode'] = 'on'
        result = context_payload(state, {})
        self.assertFalse(result['effective']['preserve_thinking'])
        self.assertTrue(result['effective']['enable_thinking'])

    def test_poly_effort_alias_and_sampling_match(self):
        result=context_payload(self.state('polystrata'),{}, {'effective':{'reasoning_effort':'high'},
            'sampling_by_effort':{'high':{'temperature':.6}}})
        self.assertEqual(result['effective']['reasoning_effort'],'xhigh')
        self.assertEqual(result['sampling_by_effort']['xhigh'],{'temperature':.6})


if __name__ == '__main__':
    unittest.main()
