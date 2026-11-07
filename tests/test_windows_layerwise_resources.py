"""Real PowerShell helper -> real recording child, without models or GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SHELLS=[found for name in ('powershell.exe','pwsh.exe') if (found:=shutil.which(name))]


@unittest.skipUnless(os.name=='nt' and SHELLS,'Windows PowerShell binding')
class WindowsLayerwiseResourcesTest(unittest.TestCase):
    def test_actual_child_argv_preserves_omission_explicit256_and_zero(self):
        with tempfile.TemporaryDirectory(prefix='kvmem-resource-argv-') as directory:
            folder=Path(directory)
            recorder=folder/'record.py'
            recorder.write_text('import json,sys\nprint(json.dumps(sys.argv[1:]))\n',encoding='utf-8')
            script=folder/'invoke.ps1'
            script.write_text('''param($Root,$Python,$Recorder,$CaseFile)
$ErrorActionPreference='Stop'
. (Join-Path $Root 'scripts/windows/recall-strategy.ps1')
. (Join-Path $Root 'scripts/windows/native-process.ps1')
$bound=@{}
$data=Get-Content -LiteralPath $CaseFile -Raw | ConvertFrom-Json
foreach($property in $data.PSObject.Properties){$bound[$property.Name]=$property.Value}
$arguments=@(Get-KVMemRecallArgs $bound 'ninfer' 'none')
$info=New-KVMemProcessInfo $Python (@($Recorder)+$arguments)
$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
$child=[Diagnostics.Process]::Start($info)
try{$output=$child.StandardOutput.ReadToEndAsync();$errorText=$child.StandardError.ReadToEndAsync()
if(!$child.WaitForExit(15000)){$child.Kill();throw 'recording child timed out'}
if($child.ExitCode){throw $errorText.Result};$output.Result}finally{$child.Dispose()}
''',encoding='utf-8')
            for shell in SHELLS:
                for explicit in ({},{'LayerwiseMetadataMiB':256},{'LayerwiseMetadataMiB':1024},
                                 {'LayerwiseActivationDiskMiB':0}):
                    with self.subTest(shell=shell,explicit=explicit):
                        case=folder/'case.json'
                        case.write_text(json.dumps(dict(RecallStrategy='layerwise',**explicit)),encoding='utf-8')
                        run=subprocess.run([shell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(script),
                            '-Root',str(ROOT),'-Python',sys.executable,'-Recorder',str(recorder),'-CaseFile',str(case)],
                            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=25)
                        self.assertEqual(run.returncode,0,run.stderr)
                        argv=json.loads(run.stdout)
                        self.assertEqual(argv[:2],['--kvmem-recall-strategy','layerwise'])
                        if 'LayerwiseMetadataMiB' in explicit:
                            self.assertEqual(argv[argv.index('--kvmem-layerwise-metadata-mib')+1],str(explicit['LayerwiseMetadataMiB']))
                        else:self.assertNotIn('--kvmem-layerwise-metadata-mib',argv)
                        if 'LayerwiseActivationDiskMiB' in explicit:self.assertEqual(argv[-2:],['--kvmem-layerwise-activation-disk-mib','0'])
                        else:self.assertNotIn('--kvmem-layerwise-activation-disk-mib',argv)


    def test_windows_newmode_wide_copy_keeps_physical_request_and_legacy_rules(self):
        with tempfile.TemporaryDirectory(prefix='kvmem-copy-argv-') as directory:
            folder=Path(directory)
            worker=folder/'recording-only.exe';worker.write_bytes(b'not executable: DryRun must never launch this file')
            model=folder/'recording-only.ninfer';model.write_bytes(b'synthetic: no model loading')
            common=['-Backend','ninfer','-Worker',str(worker),'-Model',str(model),
                '-Gpu','GPU-00000000-0000-0000-0000-000000000001','-Context','1024',
                '-Budget','192','-SpeculativeMode','mtp','-DraftTokens','15','-DraftPolicy','fixed',
                '-NgramDrafts','63','-KvType','bf16','-DryRun','-WarningAction','SilentlyContinue']
            for shell in SHELLS:
                for concurrent in (1,2):
                    command=[shell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
                        str(ROOT/'scripts/windows/start-backend.ps1'),*common,
                        '-RecallStrategy','layerwise','-LayerwiseMtpHistory','first-layer',
                        '-Reserve','64','-Prefill','64','-SinkTokens','0','-RecentTokens','0',
                        '-Concurrency',str(concurrent)]
                    run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=25)
                    self.assertEqual(run.returncode,0,run.stderr)
                    plan=json.loads(run.stdout)
                    self.assertEqual(plan['effective_parameters']['NgramDrafts'],63)
                    self.assertEqual(plan['argv'][plan['argv'].index('--ngram-draft-tokens')+1],'63')
                    self.assertEqual(plan['effective_parameters']['Reserve'],64)
                    self.assertEqual(plan['effective_parameters']['DraftTokens'],15)
                # Existing Legacy minimum Prefill and concurrent normalization stay intact.
                run=subprocess.run([shell,'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',
                    str(ROOT/'scripts/windows/start-backend.ps1'),*common,'-Reserve','128','-Prefill','128',
                    '-SinkTokens','64','-RecentTokens','64','-Concurrency','2'],
                    capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=25)
                self.assertEqual(run.returncode,0,run.stderr)
                plan=json.loads(run.stdout)
                self.assertEqual(plan['effective_parameters']['NgramDrafts'],15)
                self.assertEqual(plan['argv'][plan['argv'].index('--ngram-draft-tokens')+1],'15')


if __name__=='__main__':unittest.main()
