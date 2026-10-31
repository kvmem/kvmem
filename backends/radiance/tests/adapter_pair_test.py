import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from compare_adapter_pair import compare


class AdapterPairTests(unittest.TestCase):
    def pair(self):
        a = {'pass': True, 'model_sha256': 'm', 'kernel_sha256': 'k',
             'plugin_sha256': {'p': 'h'}, 'sdk': '/opt/rocm-7.2.3',
             'runtime_libraries': ['hip'], 'harness_sha256': 'h', 'exe_sha256': 'old',
             'command': ['/old', '--port', '1', '--kvmem-budget', '7680'],
             'rounds': [{'source_sha256': 'i', 'timings': {'predicted_n': 100, 'predicted_per_second': 40},
                         'content': 'x', 'reasoning_content': None, 'finish_reason': 'stop'}]}
        b = copy.deepcopy(a)
        b['exe_sha256'] = 'new'
        b['command'][0], b['command'][2] = '/new', '2'
        return a, b

    def test_only_adapter_and_ephemeral_port_may_change(self):
        self.assertTrue(compare(*self.pair())['pass'])

    def test_changed_inputs_or_stack_refused(self):
        for key, value in [('model_sha256', 'other'), ('kernel_sha256', 'other'),
                           ('plugin_sha256', {}), ('sdk', 'rocm10'), ('harness_sha256', ''),
                           ('runtime_libraries', []), ('exe_sha256', 'old')]:
            a, b = self.pair()
            b[key] = value
            self.assertFalse(compare(a, b)['pass'], key)
        a, b = self.pair()
        b['command'][-1] = '8192'
        self.assertFalse(compare(a, b)['pass'])
        a, b = self.pair()
        b['rounds'][0]['source_sha256'] = 'other'
        self.assertFalse(compare(a, b)['pass'])

    def test_each_round_speed_guard(self):
        a, b = self.pair()
        b['rounds'][0]['timings']['predicted_per_second'] = 37.9
        self.assertFalse(compare(a, b)['pass'])

    def test_output_difference_reported_not_accuracy_gate(self):
        a, b = self.pair()
        b['rounds'][0]['content'] = 'different'
        result = compare(a, b)
        self.assertTrue(result['pass'])
        self.assertEqual(result['round_outputs_identical'], [False])

    def test_insufficient_or_empty_output_rejected(self):
        a, b = self.pair()
        b['rounds'][0]['timings']['predicted_n'] = 1
        self.assertFalse(compare(a, b)['pass'])
        b['rounds'] = []
        self.assertFalse(compare(a, b)['pass'])


if __name__ == '__main__':
    unittest.main()
