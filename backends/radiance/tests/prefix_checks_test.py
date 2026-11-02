import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prefix_checks import input_prefix_continuations


class PrefixEvidenceTest(unittest.TestCase):
    def fixture(self):
        lines=[]
        for i in (1, 2, 3):
            lines += [f'I KVMEM_INPUT_RETAIN request={i} frontier={i*512} state_and_pages=1',
                      f'I KVMEM_INPUT_READY request={i} frontier={i*512}',
                      f'I KVMEM_QUERY_SPAN session={i} begin=-1 end=-1 proven=0']
            if i > 1:
                lines += [f'I KVMEM_CHECKPOINT_SELECT request={i} purpose=input frontier={(i-1)*512} lcp={(i-1)*512+10}',
                          f'I KVMEM_PREFIX_RESUME request={i} frontier={(i-1)*512} exact_prefix=1']
        rows=[{'usage': {'prompt_tokens': i*512+10, 'prompt_tokens_details': {'cached_tokens': (i-1)*512}}}
              for i in (1,2,3)]
        return lines,rows

    def test_complete_native_and_api_proof(self):
        self.assertTrue(input_prefix_continuations(*self.fixture())['pass'])

    def test_fabricated_cache_count_does_not_pass(self):
        lines,rows=self.fixture();rows[1]['usage']['prompt_tokens_details']['cached_tokens']+=1
        self.assertFalse(input_prefix_continuations(lines,rows)['pass'])

    def test_missing_native_state_does_not_pass(self):
        lines,rows=self.fixture();lines=[x for x in lines if 'KVMEM_INPUT_READY request=1 ' not in x]
        self.assertFalse(input_prefix_continuations(lines,rows)['pass'])

    def test_old_query_reselection_is_rejected(self):
        lines,rows=self.fixture();lines=[x.replace('begin=-1 end=-1 proven=0','begin=1 end=8 proven=1') for x in lines]
        self.assertFalse(input_prefix_continuations(lines,rows)['pass'])

    def test_restore_past_lcp_is_rejected(self):
        lines,rows=self.fixture();lines=[x.replace('lcp=522','lcp=511') for x in lines]
        self.assertFalse(input_prefix_continuations(lines,rows)['pass'])

if __name__=='__main__':unittest.main()
