import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from statistics_checks import frontier_matches


class StatisticsChecksTest(unittest.TestCase):
    def row(self, finish='length'):
        return {'usage': {'prompt_tokens': 100, 'completion_tokens': 512},
                'finish_reason': finish, 'timings': {'draft_n': 112}}

    def test_ordinary_evaluated_frontier(self):
        for n in (611, 612):
            self.assertTrue(frontier_matches({'tokens': n}, self.row('stop')))
        for n in (610, 613):
            self.assertFalse(frontier_matches({'tokens': n}, self.row('stop')))

    def test_bounded_terminal_verify(self):
        for n in range(613, 619):
            self.assertTrue(frontier_matches({'tokens': n}, self.row(), draft_depth=7, output_limit=512))
        self.assertFalse(frontier_matches({'tokens': 619}, self.row(), draft_depth=7, output_limit=512))
        for kwargs in ({'draft_depth': 0, 'output_limit': 512},
                       {'draft_depth': 7}, {'draft_depth': 7, 'output_limit': 513}):
            self.assertFalse(frontier_matches({'tokens': 614}, self.row(), **kwargs))
        self.assertFalse(frontier_matches({'tokens': 614}, self.row('stop'), draft_depth=7, output_limit=512))
        row = self.row()
        row['timings']['draft_n'] = 0
        self.assertFalse(frontier_matches({'tokens': 614}, row, draft_depth=7, output_limit=512))

    def test_context_limit_and_missing_usage(self):
        self.assertTrue(frontier_matches({'tokens': 615}, self.row(), draft_depth=7, context_limit=612))
        self.assertFalse(frontier_matches({'tokens': 615}, {}, draft_depth=7, output_limit=512))


if __name__ == '__main__':
    unittest.main()
