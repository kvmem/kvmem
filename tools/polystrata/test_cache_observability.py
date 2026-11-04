"""Exercise the real pipe parser without loading tokenizer or model packages."""
import ast
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'backends/polystrata'))
import serve.kvmem_monitor as monitor

class CacheProtocolTest(unittest.TestCase):
    def test_materialized_restore_is_visible_before_more_native_progress(self):
        from serve.server import Service, StrataEngine

        class NativeLines:
            def __init__(self, line):
                self.line, self.reads = line, 0

            def get(self, *, timeout):
                self.reads += 1
                if self.reads > 1:
                    raise AssertionError('monitor update waited for a later PP/token')
                return self.line

        for actual, source, path in [(90, 'shared', 'snapshot'), (0, 'miss', 'miss')]:
            with self.subTest(actual=actual):
                engine = StrataEngine.__new__(StrataEngine)
                engine.batch, engine.gen = 2, 1
                engine.cache_reuse = {}
                previous = {'actual_cached_tokens': 77, 'cache_source': 'own'}
                engine.last = previous
                engine._ctl_mode = 'batch'
                engine.lines = NativeLines(
                    f'CACHE_REUSE planned_cached_tokens=90 actual_cached_tokens={actual} '
                    f'cache_source={source} cache_semantics_revision=2 restore_path={path}\n')
                observer = Service.__new__(Service)
                observer.engine, observer.status_lock = engine, threading.Lock()
                status = {'busy': True, 'phase': 'reading the prompt'}
                control = engine._control(threading.Event(), lambda _: self.fail('unexpected token'))
                try:
                    self.assertIsNone(next(control))
                    observer._observe_native_cache(status, previous)
                    self.assertEqual(status['actual_cached_tokens'], actual)
                    self.assertEqual(status['planned_cached_tokens'], 90)
                    self.assertEqual(status['restore_path'], path)
                    self.assertEqual(status['cache_semantics_revision'], 2)
                    self.assertEqual(engine.lines.reads, 1)
                    self.assertTrue(status['busy'])
                    observer._observe_native_cache(status, previous)
                    self.assertEqual(observer.kvmem_reuse_totals[source], 1)
                    self.assertEqual(sum(observer.kvmem_reuse_totals.values()), 1)
                finally:
                    control.close()

    def test_complete_canonical_boundary_and_internal_protocol(self):
        from serve import kvmem
        module=ast.parse(Path(kvmem.__file__).read_text(encoding='utf-8'))
        function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_end')
        function.body=[n for n in function.body if not isinstance(n,ast.ImportFrom)]
        namespace={'unmark_think_literals':lambda text,_:(text,[])}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(kvmem.__file__),'exec'),namespace)
        closed='system|document|short question|closed|'
        service=SimpleNamespace(template=SimpleNamespace(render=lambda *_a,**_k:closed),tok=SimpleNamespace(encode=lambda text,**_kw:list(text.encode())),literals={})
        complete=closed+'assistant|'
        self.assertEqual(namespace['canonical_end'](service,[],[],{},complete,list(complete.encode())),len(closed))
        service.tok.encode=lambda text,**_kw:list(text.encode())+[999]
        self.assertEqual(namespace['canonical_end'](service,[],[],{},complete,list(complete.encode())),-1,'boundary merges must opt out')
        prompt=kvmem.Prompt([1,2,3,4],(1,4),'display',3)
        self.assertEqual((prompt+[5]).kvmem_canonical_end,3)
        self.assertEqual(prompt[:].kvmem_canonical_end,3)
        self.assertEqual(prompt[1:].kvmem_canonical_end,-1)
        engine=self.engine_class()();engine.batch=2;engine.info={'kvmem':1};engine.turn_ids={3}
        self.assertIn('canonical_end=3',engine.turn_key(prompt))
        self.assertIn('canonical_end=-1',engine.turn_key(kvmem.Prompt([1,2,3],(1,2))))

    def engine_class(self):
        source=Path(monitor.__file__).with_name('server.py')
        module=ast.parse(source.read_text(encoding='utf-8'))
        engine=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='StrataEngine')
        methods=[n for n in engine.body if isinstance(n,ast.FunctionDef) and n.name in ('last','cache_reuse','_parse_cache_reuse','_parse_done','turn_key')]
        code=ast.Module(body=[ast.ClassDef(name='Engine',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
        namespace={'threading':threading};exec(compile(ast.fix_missing_locations(code),str(source),'exec'),namespace)
        return namespace['Engine']

    def test_mode_switch_retains_full_owner_without_recalling_old_query(self):
        from serve.kvmem import Prompt
        engine=self.engine_class()();engine.batch=2;engine.info={'kvmem':1};engine.turn_ids={3}
        owner='界'*60+'x'*20+'branch-B'
        prompt=Prompt([1,2,3,4],(1,4),owner)
        initial=engine.turn_key(prompt)
        continued=engine.turn_key(prompt,continuation=True)
        self.assertIn('query_begin=1 query_end=4',initial)
        self.assertIn('query_begin=0 query_end=0',continued)
        self.assertIn('cache_owner='+owner.encode().hex(),initial)
        self.assertIn('cache_owner='+owner.encode().hex(),continued)
        self.assertIn('turn_at=2',initial)
        self.assertIn('turn_at=2',continued)
        self.assertGreater(len(owner.encode()),128)

    def test_content_cache_revision_and_restore_path_survive_native_observation(self):
        engine=self.engine_class()();engine.batch=0;engine.cache_reuse={}
        engine._parse_cache_reuse('CACHE_REUSE planned_cached_tokens=90 actual_cached_tokens=0 cache_source=miss cache_semantics_revision=2 restore_path=miss')
        self.assertEqual(engine.cache_reuse['cache_semantics_revision'],2)
        self.assertEqual(engine.cache_reuse['restore_path'],'miss')
        self.assertEqual(engine.cache_reuse['actual_cached_tokens'],0)
        engine.cache_reuse={}
        engine._parse_done('DONE 2 100 10 20 stop 0 0 90 0 0 0 0 0 10 0 planned_cached_tokens=90 actual_cached_tokens=90 cache_source=shared cache_semantics_revision=2 restore_path=snapshot')
        self.assertEqual(engine.cache_reuse['restore_path'],'snapshot')
        engine.cache_reuse={}
        engine._parse_cache_reuse('CACHE_REUSE planned_cached_tokens=90 actual_cached_tokens=90 cache_source=unobserved cache_semantics_revision=2 restore_path=unobserved')
        self.assertIsNone(engine.cache_reuse['restore_path'])
        self.assertIsNone(engine.cache_reuse['cache_source'])

    def test_yield_keeps_first_restore_observation_separate_from_legacy_done(self):
        engine=self.engine_class()();engine.batch=2
        engine.cache_reuse={}
        engine._parse_cache_reuse('CACHE_REUSE planned_cached_tokens=90 actual_cached_tokens=90 cache_source=shared')
        # An internal resumed leg can include input computed by this request.
        engine._parse_cache_reuse('CACHE_REUSE planned_cached_tokens=95 actual_cached_tokens=95 cache_source=own')
        engine._parse_done('DONE 2 100 10 20 stop 0 0 95 0 0 0 0 0 5 0 planned_cached_tokens=95 actual_cached_tokens=95 cache_source=own')
        self.assertEqual(engine.last['reused'],95)
        self.assertEqual(engine.cache_reuse['actual_cached_tokens'],90)
        self.assertEqual(engine.cache_reuse['cache_source'],'shared')

    def test_live_observation_does_not_reuse_previous_request_or_overwrite_first_leg(self):
        from types import SimpleNamespace
        source=Path(monitor.__file__).with_name('server.py');tree=ast.parse(source.read_text(encoding='utf-8'))
        service=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Service')
        methods=[n for n in service.body if isinstance(n,ast.FunctionDef) and n.name in ('_observe_native_cache','_begin_native_cache_observation')]
        code=ast.Module(body=[ast.ClassDef(name='Observer',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
        namespace={};exec(compile(ast.fix_missing_locations(code),str(source),'exec'),namespace)
        observer=namespace['Observer']();observer.status_lock=threading.Lock()
        previous={'planned_cached_tokens':80,'actual_cached_tokens':80,'cache_source':'own'}
        observer.engine=SimpleNamespace(last=previous,cache_reuse={});status={}
        observer._observe_native_cache(status,previous);self.assertFalse(status)
        observer.engine.cache_reuse={'planned_cached_tokens':90,'actual_cached_tokens':0,'cache_source':'miss'}
        observer._observe_native_cache(status,previous)
        observer.engine.cache_reuse={'planned_cached_tokens':95,'actual_cached_tokens':95,'cache_source':'own'}
        observer._observe_native_cache(status,previous)
        self.assertEqual(status['actual_cached_tokens'],0)
        self.assertEqual(status['cache_source'],'miss')
        self.assertEqual(observer.kvmem_reuse_totals,{'own':0,'shared':0,'miss':1})
        status['busy']=True
        observer._begin_native_cache_observation(status)
        self.assertEqual(status,{'busy':True})
        observer._observe_native_cache(status,previous)
        self.assertEqual(status['actual_cached_tokens'],95)
        self.assertEqual(status['cache_source'],'own')
        self.assertEqual(observer.kvmem_reuse_totals,{'own':1,'shared':0,'miss':1})
        observer.engine.gen=1
        observer.engine.cache_reuse={'planned_cached_tokens':90,'actual_cached_tokens':90,'cache_source':'shared'}
        observer._observe_native_cache({},previous)
        self.assertEqual(observer.kvmem_reuse_totals,{'own':0,'shared':1,'miss':0})

    def test_old_engine_does_not_invent_actual_reuse(self):
        engine=self.engine_class()();engine.batch=0
        engine._parse_done('DONE 2 100 10 20 stop 0 0 90 0 0 0 0 0 10 0')
        self.assertEqual(engine.last['reused'],90)
        self.assertNotIn('actual_cached_tokens',engine.last)
        self.assertNotIn('cache_source',engine.last)

    def test_parallel_diagnostics_belong_to_each_request(self):
        engine=self.engine_class()();engine.batch=2
        barrier=threading.Barrier(2);results={}
        def parse(name,actual,source):
            engine._parse_done(f'DONE 2 100 10 20 stop 0 0 {actual} 0 0 0 0 0 10 0 planned_cached_tokens=90 actual_cached_tokens={actual} cache_source={source}')
            barrier.wait();results[name]=dict(engine.last)
        threads=[threading.Thread(target=parse,args=('a',90,'shared')),threading.Thread(target=parse,args=('b',0,'miss'))]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        self.assertEqual(results['a']['cache_source'],'shared')
        self.assertEqual(results['a']['actual_cached_tokens'],90)
        self.assertEqual(results['b']['cache_source'],'miss')
        self.assertEqual(results['b']['planned_cached_tokens'],90)
        self.assertEqual(results['b']['actual_cached_tokens'],0)

if __name__=='__main__':unittest.main()
