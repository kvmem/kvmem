"""Core API/default/history contracts for the shared UI; mock engine, no GPU."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import urllib.request
import urllib.error
from unittest import mock

SOURCE = Path(os.environ.get('POLYSTRATA_SOURCE', Path(__file__).resolve().parents[2]/'backends/polystrata'))
sys.path.insert(0,str(SOURCE))
from serve.server import Service, MockEngine, ByteTokenizer, serve, CTX_SLACK
from serve.frontend import ChatTemplate

class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_limit=max_new
        yield from super().generate(ids,max_new,sampling,cancel,embeddings)

class UiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok=ByteTokenizer()
        cls.engine=RecordingEngine(tok,'</think>\n'+('xy'*40),max_context=4096)
        cls.svc=Service(cls.engine,tok,ChatTemplate(SOURCE/'serve/chat_template.jinja'))
        cls.svc.kvmem_default_max_tokens=17
        cls.svc.client_reasoning_history=True
        cls.svc.preserve_thinking=True
        cls.httpd=serve(cls.svc,port=0)
        cls.base='http://127.0.0.1:'+str(cls.httpd.server_address[1])
    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown();cls.httpd.server_close()
    def request(self,path,body=None):
        req=urllib.request.Request(self.base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=20) as response:return json.load(response)
    def test_three_api_limits_and_explicit_unlimited(self):
        for path,key,payload in [('/v1/chat/completions','max_tokens',{'messages':[{'role':'user','content':'hi'}]}),
                                 ('/v1/messages','max_tokens',{'messages':[{'role':'user','content':'hi'}]}),
                                 ('/v1/responses','max_output_tokens',{'input':'hi','store':False})]:
            for explicit,expected in [(None,17),(7,7),(0,None),(-1,None)]:
                with self.subTest(path=path,explicit=explicit):
                    body=dict(payload)
                    if explicit is not None:body[key]=explicit
                    if path == '/v1/responses' and explicit in (0,-1):
                        # Responses already requires a positive explicit limit; retain that contract.
                        with self.assertRaises(urllib.error.HTTPError) as error:self.request(path,body)
                        self.assertEqual(error.exception.code,400)
                        continue
                    self.request(path,body)
                    if expected is None:self.assertGreater(self.engine.last_limit,100)
                    else:self.assertEqual(self.engine.last_limit,expected)
        self.svc.shared={'max_tokens':11}
        try:
            for explicit,expected in [(None,11),(0,0),(-1,-1),(5,5)]:
                req={} if explicit is None else {'max_tokens':explicit}
                merged=self.svc.with_shared(req,'openai')
                self.assertEqual(self.svc.output_limit(merged,'max_tokens'),expected)
        finally:self.svc.shared={}
    def test_effective_settings_and_props(self):
        with urllib.request.urlopen(self.base+'/',timeout=10) as response:
            self.assertEqual(response.headers['Cache-Control'],'no-store')
        self.svc.config_effort='none'
        settings=self.request('/settings')
        self.assertEqual(settings['effective']['max_tokens'],17)
        self.assertEqual(settings['effective']['reasoning_effort'],'none')
        self.assertEqual(self.request('/props')['default_generation_settings']['params']['n_predict'],17)
        self.assertEqual(self.svc.with_shared({},'openai')['reasoning_effort'],'none')
        self.assertEqual(self.svc.with_shared({'reasoning_effort':'high'},'openai')['reasoning_effort'],'high')
    def test_unlimited_uses_logical_context_not_resident_window(self):
        # A 256-token working window must not cap a 4096-token logical conversation.
        before_info = getattr(self.engine, 'info', {})
        self.engine.info = {'kvmem':1, 'kvmem_budget':192, 'kvmem_gen_reserve':64}
        before = self.svc.kvmem_default_max_tokens
        try:
            self.svc.kvmem_default_max_tokens = None
            self.assertLessEqual(self.request('/props')['default_generation_settings']['params']['n_predict'], 0)
            for prompt in (66, 512, 4096 - CTX_SLACK - 1):
                for requested in (None, 0, -1):
                    with self.subTest(prompt=prompt, requested=requested), mock.patch.object(
                            self.svc, 'encode_prompt', return_value=[ord('x')] * prompt):
                        ids, _, maximum = self.svc.prepare([], None, {}, requested)
                        self.assertEqual(len(ids), prompt)
                        self.assertEqual(maximum, 4096 - CTX_SLACK - prompt)
            for path,payload in [('/v1/chat/completions',{'messages':[{'role':'user','content':'hi'}]}),
                                 ('/v1/messages',{'messages':[{'role':'user','content':'hi'}]}),
                                 ('/v1/responses',{'input':'hi','store':False})]:
                self.request(path,payload)
                self.assertGreater(self.engine.last_limit, 256)
        finally:
            self.svc.kvmem_default_max_tokens = before
            self.engine.info = before_info
    def test_history_is_client_owned_and_false_is_explicit(self):
        svc=Service.__new__(Service)
        svc.kvmem_default_max_tokens=17;svc.client_reasoning_history=True;svc.preserve_thinking=True
        svc._note_unreadable_tool_images=lambda *a:{}
        svc.restore_reasoning=lambda *a:self.fail('text-keyed recovery must not run')
        messages=[{'role':'user','content':'hi'}]
        svc.encode_prompt=lambda supplied,*a: [1,2] if supplied is messages else self.fail('history changed')
        svc.drop_oldest=False;svc.truncate_middle=False;svc.fit_max_tokens=False;svc.embeddings=SimpleNamespace();svc.engine=SimpleNamespace(max_context=4096)
        self.assertEqual(svc.prepare(messages,None,{})[2],17)
        supplied={}
        svc.template_tokens={};svc.call_format='qwen';svc.effort_end=False;svc.preserve_thinking=False
        svc.template=SimpleNamespace(render=lambda *a,**kw:supplied.update(kw) or 'prompt')
        svc.render_prompt(messages,None,{})
        self.assertIs(supplied['preserve_thinking'],False)

if __name__=='__main__':unittest.main()
