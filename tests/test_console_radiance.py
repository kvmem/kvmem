from pathlib import Path
import sys,time,json,unittest
from unittest.mock import MagicMock
T=Path(__file__).resolve().parent
sys.path.insert(0,str(T.parent/'scripts'))
sys.path.insert(0,str(T))
from console_chat import context_payload,request_metrics,ChatBridge
from console_radiance import enrich_props,speculative_event,enrich_status,monitor_execution,effective_settings
import test_console_chat as fixture
Control=fixture.Control

class Contract(unittest.TestCase):
    def setUp(self):
        self.info={'request_defaults':{'temperature':1.0,'max_tokens':None,'thinking_token_budget':None,
            'chat_template_kwargs':{'enable_thinking':True}},'capabilities':{'reasoning':True,'reasoning_efforts':['none','low','medium','xhigh']},
            'engine':{'speculation':{'enabled':True,'drafter':'mtp','depth':3}}}
        self.props={'chat_template_caps':{'supports_preserve_reasoning':True}}
        self.state={'state':'ready','active_config':{'parameters':{'Backend':'radiance'}}}
    def test_defaults_are_native_and_unbounded(self):
        state=json.loads(json.dumps(self.state));state['active_config']['parameters'].update(ThinkingMode='off',ThinkingBudget=512,MaxTokens=100)
        result=context_payload(state,self.props,self.info)
        for key,value in {'enable_thinking':True,'reasoning_budget_tokens':-1,'max_tokens':-1}.items():
            self.assertEqual(result['effective'][key],value);self.assertEqual(result['sources'][key],'engine')
        self.assertEqual(result['capabilities']['efforts'],self.info['capabilities']['reasoning_efforts'])
        self.assertTrue(result['capabilities']['preserve']);self.assertIsNone(result['history_export_url'])
    def test_status_confirmation_requires_complete_native_report(self):
        state={'planned_scoring':{'scorer':'page-softmax','agg':'mean','lrfu':False}}
        actual={'scorer':'page-softmax','agg':'mean','lrfu':False}
        result=enrich_status(state,{'kvmem':{'scoring':actual}})
        self.assertTrue(result['scoring_confirmed']);self.assertFalse(result['scoring_mismatch'])
        self.assertFalse(enrich_status(state,{'kvmem':{'scoring':{'scorer':'page-softmax'}}})['scoring_confirmed'])
        self.assertTrue(enrich_status(state,{'kvmem':{'scoring':dict(actual,agg='max')}})['scoring_mismatch'])
        self.assertEqual(enrich_status(state,{}),state)

    def test_runtime_geometry_no_guessing(self):
        snapshot={'kvmem':{'kv_type':'k6_v4','state_cache_type':'f16','selected_tokens':36864}}
        result=enrich_props(self.props,self.info,snapshot)
        self.assertEqual(result['kvmem']['kv_type'],'k6_v4');self.assertEqual(result['speculative']['draft_tokens'],3)
        self.assertNotIn('adaptive',result['speculative']);self.assertEqual(result['kvmem']['defaults']['reasoning_budget_tokens'],-1)
        self.assertNotIn('kvmem',self.props)
    def test_mtp_and_missing_statistics(self):
        event={'id':'test','choices':[],'timings':{'draft_n':42,'draft_n_accepted':31}}
        result=speculative_event(event)
        self.assertEqual(result['kvmem']['speculative']['accepted_tokens'],31)
        self.assertNotIn('kvmem',speculative_event({'timings':{'draft_n':1,'draft_n_accepted':2}}))
        self.assertEqual(speculative_event({'usage':None}),{'usage':None})
    def test_template_effort_and_native_override(self):
        self.info['request_defaults']['reasoning_effort'] = None
        self.state['radiance_template_effort'] = 'xhigh'
        result = context_payload(self.state, self.props, self.info)
        self.assertEqual(result['effective']['reasoning_effort'], 'xhigh')
        self.assertTrue(result['uses_template_effort'])
        self.info['request_defaults']['reasoning_effort'] = 'none'
        result = context_payload(self.state, self.props, self.info)
        self.assertFalse(result['effective']['enable_thinking'])
        self.assertFalse(result['uses_template_effort'])
        self.info['capabilities']['reasoning_efforts'] = None
        self.assertEqual(context_payload(self.state,self.props,self.info)['capabilities']['efforts'], [])
        self.assertNotIn('kvmem', speculative_event({'timings':{'draft_n':0,'draft_n_accepted':0}}))

    def test_binding_requires_complete_native_rank_identity(self):
        device = dict(rank=0,device=2,name='AMD test GPU',arch='gfx1201',uuid='GPU-test',vram_bytes=16*1024**3,free_vram_bytes=123)
        engine = dict(tensor_parallel=1,devices=[device])
        result = monitor_execution(engine)
        self.assertEqual(result['mode'], 'single')
        bound = result['monitor_devices'][0]
        self.assertEqual(bound['hip_ordinal'], 2)
        self.assertEqual(bound['total_bytes'], 16*1024**3)
        self.assertNotIn('used_bytes',bound)
        self.assertNotIn('free_vram_bytes',bound)
        for invalid in ({},dict(engine,tensor_parallel=2),dict(engine,devices=[dict(device,rank=1)]),
                        dict(engine,devices=[dict(device,device=True)]),dict(engine,devices=[dict(device,name='')]),
                        dict(tensor_parallel=2,devices=[device,dict(device,rank=1)])):
            with self.subTest(engine=invalid):self.assertIsNone(monitor_execution(invalid))
        self.info['engine'].update(engine)
        props=enrich_props(self.props,self.info,{})
        self.assertEqual(props['execution'],result)
        self.assertNotIn('execution',self.props)

    def test_exact_identity_and_live_freshness(self):
        control=Control(1);control.backend='radiance';bridge=ChatBridge(control)
        bridge.records['browser']={'id':'browser','instance':'one','native_id':'chatcmpl-a','state':'answering'}
        row={'id':'2','native_request_id':'chatcmpl-a','status':'generating','decode_tps_mean':32,'decode_tps_live':42,
            'decode_tps_window_ms':2000,'decode_tps_sampled_at_ms':time.time()*1000}
        bridge.get_json=MagicMock(return_value={'active':[dict(row,native_request_id='other'),row]})
        self.assertEqual(bridge.record('browser')['metrics']['decode_tps'],42)
        row['decode_tps_sampled_at_ms']-=20000
        self.assertIsNone(bridge.record('browser')['metrics']['decode_tps'])
        bridge.get_json.return_value={'active':[row,dict(row,id='3')]}
        self.assertNotIn('metrics',bridge.record('browser'))

class RadianceTransport(fixture.TransportTests):
    def test_usage_request_does_not_override_thinking_or_output(self):
        self.control.backend='radiance'
        c,r=self.request();body=r.read();c.close()
        self.assertIn(b'data: [DONE]',body)
        payload=self.up.received[0][1]
        self.assertTrue(payload['stream_options']['include_usage'])
        for key in ('max_tokens','enable_thinking','reasoning_budget_tokens','_kvmem_request_id'):self.assertNotIn(key,payload)

if __name__=='__main__':unittest.main()
