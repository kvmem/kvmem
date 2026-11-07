"""Windows recipe normalization and real recording-child argv; no inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from launcher.worker import require_launch_features
from test_layerwise_launcher import contract


@unittest.skipUnless(os.name == 'nt' and shutil.which('powershell.exe'), 'Windows recipe')
class NinferLayerwiseRecipeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='kvmem-ninfer-recipe-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.model = self.directory/'recording target.ninfer'
        self.model.write_text("import json,sys,os;print(json.dumps({'received':sys.argv[1:],'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}))",encoding='utf-8')

    def preview(self, *arguments):
        return subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
            '-File',str(ROOT/'scripts/windows/start-ninfer.ps1'),'-Model',str(self.model),
            '-Worker',sys.executable,'-Gpu','GPU-recipe-fixture','-DryRun',*arguments],
            capture_output=True,encoding='utf-8',errors='replace',timeout=20)

    def plan(self, *arguments):
        result=self.preview(*arguments)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout[result.stdout.index('{'):])

    def record(self, plan):
        # Execute the exact preview argv through the same production CRT
        # argument encoder. Python is a recording worker, not a model loader.
        data=self.directory/'argv.json';data.write_text(json.dumps(plan['argv']),encoding='utf-8')
        harness=self.directory/'record.ps1'
        harness.write_text("""param([string]$Helper,[string]$Data)
. $Helper
$argv=Get-Content -LiteralPath $Data -Raw | ConvertFrom-Json
$info=New-KVMemProcessInfo $argv[0] @($argv[1..($argv.Count-1)])
$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
$child=[Diagnostics.Process]::Start($info)
$output=$child.StandardOutput.ReadToEnd();$errors=$child.StandardError.ReadToEnd();$child.WaitForExit()
if($child.ExitCode){throw $errors};$output;$child.Dispose()
""",encoding='utf-8')
        run=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
            '-File',str(harness),'-Helper',str(ROOT/'scripts/windows/native-process.ps1'),'-Data',str(data)],
            capture_output=True,encoding='utf-8',errors='replace',timeout=15)
        self.assertEqual(run.returncode,0,run.stderr)
        received=json.loads(run.stdout)['received']
        self.assertEqual(received,plan['argv'][2:])
        return received

    def test_unbound_layerwise_starts_without_draft_or_vision(self):
        plan=self.plan('-RecallStrategy','layerwise')
        p=plan['effective_parameters'];argv=self.record(plan)
        self.assertEqual(p['SpeculativeMode'],'none')
        self.assertEqual(p['DraftTokens'],0)
        self.assertFalse(p['AdaptiveMtp']);self.assertEqual(p['NgramDrafts'],0)
        self.assertFalse(p['Vision'])
        for flag in ('--spec','--draft-tokens','--adaptive-mtp','--ngram-draft-tokens','--vision',
                     '--kvmem-layerwise-mtp-history','--kvmem-sink-tokens','--kvmem-recent-tokens'):
            self.assertNotIn(flag,argv)

    def test_explicit_options_survive_and_actual_profile_rejects_them(self):
        for extra, key, value in ((['-DraftTokens','5'],'DraftTokens',5),
                (['-AdaptiveMtp'],'AdaptiveMtp',True),(['-NgramDrafts','7'],'NgramDrafts',7),
                (['-Vision'],'Vision',True)):
            base=['-RecallStrategy','layerwise','-SpeculativeMode','mtp','-LayerwiseMtpHistory','first-layer']
            if key!='DraftTokens':base+=['-DraftTokens','3']
            plan=self.plan(*base,*extra)
            self.assertEqual(plan['effective_parameters'][key],value)
            self.record(plan)
            with self.assertRaises(ValueError):
                require_launch_features(contract(),plan['effective_parameters'])

    def test_checkpoint_target_and_manual_cap_reach_recording_worker(self):
        argv = self.record(self.plan('-RecallStrategy','layerwise','-LayerwiseCheckpointTarget','3',
                                     '-LayerwiseMetadataMiB','1024'))
        self.assertEqual(argv[argv.index('--kvmem-layerwise-checkpoint-target')+1], '3')
        self.assertEqual(argv[argv.index('--kvmem-layerwise-metadata-mib')+1], '1024')

    def test_history_omission_is_preserved_and_worker_owns_default(self):
        plan = self.plan('-RecallStrategy','layerwise','-SpeculativeMode','mtp','-DraftTokens','3')
        argv = self.record(plan)
        self.assertNotIn('--kvmem-layerwise-mtp-history', argv)
        self.assertFalse(plan['effective_parameters'].get('LayerwiseMtpHistory'))
        caps = contract()
        with self.assertRaisesRegex(ValueError, 'history policy'):
            require_launch_features(caps, plan['effective_parameters'])
        caps['recall_strategies']['layerwise']['speculative']['mtp']['default_history_policy'] = 'first-layer'
        require_launch_features(caps, plan['effective_parameters'])

    def test_width_must_be_explicit_and_positive(self):
        for extra, expected in (
                (['-SpeculativeMode','mtp','-LayerwiseMtpHistory','first-layer'],'explicit draft width'),
                (['-SpeculativeMode','mtp','-DraftTokens','0','-LayerwiseMtpHistory','first-layer'],'must be positive')):
            run=self.preview('-RecallStrategy','layerwise',*extra)
            self.assertNotEqual(run.returncode,0)
            self.assertIn(expected,run.stderr)

    def test_legacy_recipe_keeps_its_existing_literal_options(self):
        plan=self.plan();argv=self.record(plan)
        expected={'--kvmem-budget':'36864','--kvmem-gen-reserve':'16384','--kvmem-host-mib':'12288',
            '--draft-tokens':'4','--ngram-draft-tokens':'31','--vision-residency':'cpu',
            '--kvmem-sink-tokens':'12288','--kvmem-recent-tokens':'1024'}
        for flag,value in expected.items():self.assertEqual(argv[argv.index(flag)+1],value)
        self.assertIn('--adaptive-mtp',argv);self.assertIn('--vision',argv)
        self.assertNotIn('--kvmem-recall-strategy',argv)


if __name__=='__main__':unittest.main()
