"""The shared owner contract and protocol metadata must survive prompt edits."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'backends/polystrata'))
from serve.kvmem import Prompt,cache_owner

class CacheOwnerTest(unittest.TestCase):
    def test_priority_and_anonymous(self):
        self.assertEqual(cache_owner({}), '')
        self.assertEqual(cache_owner({'prompt_cache_key':'','promptCacheKey':None}), '')
        self.assertEqual(cache_owner({'prompt_cache_key':'sdk','promptCacheKey':'sdk'}),'sdk')
        self.assertEqual(cache_owner({'kvmem':{'conversation_id':'explicit'},'prompt_cache_key':'sdk'}),'explicit')

    def test_complete_identity_and_validation(self):
        key='same-prefix-'+'x'*130
        self.assertEqual(cache_owner({'prompt_cache_key':key+'A'}),key+'A')
        self.assertNotEqual(cache_owner({'prompt_cache_key':key+'A'}),cache_owner({'prompt_cache_key':key+'B'}))
        self.assertEqual(cache_owner({'prompt_cache_key':'界'*85}),'界'*85)
        for body in [{'prompt_cache_key':'界'*86}, {'promptCacheKey':42}, {'kvmem':{'conversation_id':''}},
                     {'kvmem':[]}, {'prompt_cache_key':'a','promptCacheKey':'b'},
                     {'kvmem':{'conversation_id':'override'},'prompt_cache_key':'a','promptCacheKey':'b'}]:
            with self.subTest(body=body),self.assertRaises(ValueError):cache_owner(body)

    def test_prompt_metadata_is_request_local(self):
        prompt=Prompt([1,2,3,4],(1,3),'owner')
        appended=prompt+[5]
        self.assertEqual((appended.kvmem_query,appended.kvmem_owner),((1,3),'owner'))
        sliced=appended[1:4]
        self.assertEqual((sliced.kvmem_query,sliced.kvmem_owner),((0,2),'owner'))
        continuation=Prompt([1,2],(0,0),'other')[1:]
        self.assertEqual((continuation.kvmem_query,continuation.kvmem_owner),((0,0),'other'))
        self.assertEqual(prompt.kvmem_owner,'owner')

if __name__=='__main__':unittest.main()
