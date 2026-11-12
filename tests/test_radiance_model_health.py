"""RAD directory checks agree with launch preflight without reading weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'backends/radiance/tests'))
import model_health as health
from draft_control_test import container


class RadianceHealthTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root/'ordinary-name.RAD'

    def inspect(self, **settings):
        p = health.request_parameters(dict(schema_version=1, parameters=dict(
            Backend='radiance', Model=str(self.model), **settings)))
        return health.Inspection(p, self.root).run()

    def dflash(self, block=8, names=('dflash.fc.weight',)):
        container(self.model, names, [('draft.architectures','DFlash2Model'),
            ('draft.dflash_config.block_size',block),('mtp_num_hidden_layers',1)])

    def test_dflash_is_detected_even_when_disabled_and_nextn_is_not_mtp(self):
        for block in (8, '8'):
            self.dflash(block)
            for mode in ('none','dflash2'):
                with self.subTest(block=block, mode=mode):
                    report = self.inspect(SpeculativeMode=mode, DraftTokens=7)
                    self.assertTrue(report['can_start'])
                    self.assertEqual(report['features']['dflash2'], 'present')
                    self.assertEqual(report['features']['mtp'], 'absent')

    def test_selected_depth_uses_the_container_training_block(self):
        self.dflash()
        report = self.inspect(SpeculativeMode='dflash2', DraftTokens=3)
        self.assertFalse(report['can_start'])
        self.assertEqual(report['features']['dflash2'], 'present')
        issue = next(x for x in report['checks'] if x['code']=='rad_draft_config')
        self.assertIn('requires 7', issue['detail'])
        self.assertEqual(issue['fix'], 'disable_speculative')
        self.assertTrue(self.inspect(SpeculativeMode='dflash2', DraftTokens=7)['can_start'])
        self.dflash(4)
        self.assertTrue(self.inspect(SpeculativeMode='dflash2', DraftTokens=3)['can_start'])

    def test_metadata_or_filename_without_weights_does_not_claim_dflash(self):
        self.model = self.root/'model-dflash.rad'
        self.dflash(names=('blk.0.weight',))
        report = self.inspect(SpeculativeMode='dflash2', DraftTokens=7)
        self.assertEqual(report['features']['dflash2'], 'absent')
        self.assertFalse(report['can_start'])
        self.assertTrue(self.inspect(SpeculativeMode='none')['can_start'])

    def test_invalid_or_undeclared_draft_weights_are_unverified(self):
        for metadata in ([], [('draft.architectures','DFlash2Model'),
                              ('draft.dflash_config.block_size','8.0')]):
            with self.subTest(metadata=metadata):
                container(self.model, ['dflash.fc.weight'], metadata)
                self.assertEqual(self.inspect()['features']['dflash2'], 'unknown')
                self.assertFalse(self.inspect(SpeculativeMode='dflash2', DraftTokens=7)['can_start'])
                self.assertTrue(self.inspect(SpeculativeMode='none')['can_start'])

    def test_mtp_and_native_dflash_priority(self):
        container(self.model, ['mtp.fc.weight'], [('mtp_num_hidden_layers','1')])
        report = self.inspect(SpeculativeMode='mtp', DraftTokens=3)
        self.assertTrue(report['can_start'])
        self.assertEqual(report['features']['mtp'], 'present')
        self.assertEqual(report['features']['dflash2'], 'absent')
        self.assertTrue(self.inspect(MtpDrafts=2)['can_start'])
        self.dflash(names=('mtp.fc.weight','dflash.fc.weight'))
        report = self.inspect(SpeculativeMode='mtp', DraftTokens=3)
        self.assertFalse(report['can_start'])
        self.assertTrue(any('prioritizes DFlash' in x['detail'] for x in report['checks']))

    def test_corrupt_directory_and_isolated_cli(self):
        self.dflash()
        request = dict(schema_version=1, parameters=dict(Backend='radiance',
            Model=str(self.model),SpeculativeMode='dflash2',DraftTokens=7))
        run = subprocess.run([sys.executable,'-I',str(ROOT/'scripts/model_health.py')],
            input=json.dumps(request),capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(run.stdout)['features']['dflash2'], 'present')
        self.model.write_bytes(self.model.read_bytes()[:-1])
        self.assertFalse(self.inspect(SpeculativeMode='dflash2',DraftTokens=7)['can_start'])

    def test_draft_depth_is_an_integer_parameter(self):
        for bad in (True, -1, '7', 7.5):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                health.request_parameters(dict(schema_version=1,parameters=dict(
                    Backend='radiance', DraftTokens=bad)))

    def test_selected_drafter_reuses_the_bounded_directory_read(self):
        self.dflash()
        reads = []
        original = Path.open
        def opened(path, *args, **kwargs):
            if path == self.model:
                reads.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', opened):
            report = self.inspect(SpeculativeMode='dflash2', DraftTokens=7)
        self.assertTrue(report['can_start'])
        self.assertEqual(len(reads), 1)


if __name__ == '__main__':
    unittest.main()
