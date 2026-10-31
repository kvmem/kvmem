import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from selection_checks import untrimmed_within_reserve


class SelectionChecksTest(unittest.TestCase):
    def test_dense_prefix_and_separate_suffix_reserve(self):
        self.assertTrue(untrimmed_within_reserve({'request': '1', 'tokens': '200'}, [], 8192, 3072))
        item = {'request': '5', 'tokens': '8400'}
        resumes = [{'request': '5', 'frontier': '6144'}]
        self.assertTrue(untrimmed_within_reserve(item, resumes, 8192, 3072))
        self.assertFalse(untrimmed_within_reserve(item, [], 8192, 3072))
        self.assertFalse(untrimmed_within_reserve(item, resumes, 4096, 9216))
        self.assertFalse(untrimmed_within_reserve(item, resumes, 8192, 2048))
        self.assertFalse(untrimmed_within_reserve(item, resumes * 2, 8192, 3072))
        self.assertFalse(untrimmed_within_reserve(item, [{'request': '4', 'frontier': '6144'}], 8192, 3072))

    def test_zero_frontier_replay_requires_evidence_and_whole_reserve(self):
        item = {'request': '2', 'session': '2', 'tokens': '1197'}
        replay = {'session': '2', 'begin': '0', 'end': '1095', 'before_sampler': '1'}
        self.assertTrue(untrimmed_within_reserve(item, [], 1024, 4096, [replay]))
        self.assertFalse(untrimmed_within_reserve(item, [], 1024, 4096))
        self.assertFalse(untrimmed_within_reserve(item, [], 1024, 1196, [replay]))
        self.assertFalse(untrimmed_within_reserve(item, [], 1024, 4096, [replay, replay]))
        for key, value in [('session', '1'), ('begin', '128'), ('end', '1198'),
                           ('before_sampler', '0')]:
            self.assertFalse(untrimmed_within_reserve(item, [], 1024, 4096, [{**replay, key: value}]))


if __name__ == '__main__':
    unittest.main()
