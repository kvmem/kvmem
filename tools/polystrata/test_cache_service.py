"""Exercise real HTTP Service request boundaries without a GPU or model."""
from pathlib import Path
import json
import sys
import unittest
import urllib.error
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'backends/polystrata'))
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, serve
from serve.kvmem_monitor import StorageLog, snapshot


class ObservedEngine(MockEngine):
    def __init__(self,tok):
        super().__init__(tok,'OK',max_context=4096)
        self.info={};self.gen=1;self.last={};self.cache_reuse={}
        self.observations=iter([('shared',80),('miss',0),('own',60),None])

    def generate(self,ids,max_new,sampling,cancel,embeddings=None):
        self.cache_reuse={};observation=next(self.observations)
        if observation is None:raise EngineDied('injected before native restore')
        source,actual=observation
        self.cache_reuse=dict(planned_cached_tokens=80,actual_cached_tokens=actual,cache_source=source)
        count=0
        try:
            for token in super().generate(ids,max_new,sampling,cancel,embeddings):
                count+=1;yield token
        finally:
            self.last=dict(prompt_tokens=len(ids),generated=count,reused=80,prompt_read=max(0,len(ids)-80),
                           prompt_ms=20.,decode_ms=10.,finish='stop',**self.cache_reuse)


class CacheServiceTest(unittest.TestCase):
    def test_c1_requests_and_error_have_independent_observations(self):
        tok=ByteTokenizer();engine=ObservedEngine(tok)
        template=Path(sys.modules['serve.server'].__file__).with_name('chat_template.jinja')
        service=Service(engine,tok,ChatTemplate(template))
        service.kvmem_monitor_gpu_uuid='GPU-test';service.kvmem_monitor_log=StorageLog()
        httpd=serve(service,port=0)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            for index,(source,actual) in enumerate([('shared',80),('miss',0),('own',60),(None,None)]):
                payload=dict(messages=[dict(role='user',content='Say OK. '+str(index))],max_tokens=16,temperature=0)
                request=urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                    data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                try:response=opener.open(request,timeout=10)
                except urllib.error.HTTPError as error:response=error
                with response:body=response.read()
                row=snapshot(service)['recent'][0]
                self.assertEqual(row['actual_cached_tokens'],actual)
                self.assertEqual(row['cache_source'],source)
                if actual is not None:
                    self.assertEqual(response.status,200,body)
                    self.assertEqual(row['cached_tokens'],actual)
                else:self.assertEqual(row['status'],'error')
            totals=snapshot(service)['kvmem']
            self.assertEqual(totals['own_history_hits'],1)
            self.assertEqual(totals['shared_history_hits'],1)
            self.assertEqual(totals['history_misses'],1)
        finally:
            httpd.shutdown();httpd.server_close()


if __name__=='__main__':unittest.main()
