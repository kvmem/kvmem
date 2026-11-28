import copy
import json
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from test_console_memory import console, ROOT, MIB, KV_STORAGE
from console_memory import validate_budget_plan
from launcher.plan import token_settings


def budget_evidence(parameters):
    p=parameters
    effective={k:p[k] for k in ('Budget','Reserve','Prefill','Context','SinkTokens','RecentTokens')}
    report=dict(version=1,available=True,budget=49152,minimum_budget=13312,
        maximum_budget=(p['Context']-p['Reserve'])//64*64,reserve=p['Reserve'],prefill=p['Prefill'],
        context=p['Context'],concurrency=p.get('Concurrency',1),kv_storage=KV_STORAGE[p.get('KvType','int8')],
        available_bytes=1800*MIB,runtime_bytes=1500*MIB,safety_bytes=128*MIB,planning_us=50000)
    return {'effective':effective},report


class AutoBudget(unittest.TestCase):
    def setUp(self):
        supported=patch.object(console,'AUTO_BUDGET_SUPPORTED',True)
        supported.start();self.addCleanup(supported.stop)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.c=console.Controller(ROOT,self.tmp.name,18199)
        self.config=console.normalize({'schema_version':1,'parameters':dict(console.DEFAULTS,
            Model='test.ninfer',Gpu='GPU-12345678',Budget=16384,Reserve=8192,Prefill=256,KvType='rk8v4')},self.c.backends)
        self.p=self.config['parameters'];self.tokens,self.report=budget_evidence(self.p)
        self.plan={'argv':['test-worker'], 'token_settings':self.tokens,'worker':{'budget_planning_version':1}}
        self.output=lambda: 'KVMEM_BUDGET_PLAN '+json.dumps(self.report)

    def test_only_budget_changes_and_growth_is_allowed(self):
        before=self.c.status();original=copy.deepcopy(self.config)
        with patch.object(self.c,'inspect_model',return_value={'can_start':True}),patch.object(self.c,'plan',return_value=self.plan),patch.object(self.c,'_run_budget_plan',side_effect=lambda _:self.output()),patch.object(self.c,'start') as start:
            result=self.c.adjust_budget(self.config)
        self.assertEqual(result['changes'],{'Budget':49152})
        self.assertEqual(result['reserve'],8192)
        self.assertEqual(self.config,original)
        self.assertEqual(self.c.status(),before)
        self.assertFalse(self.c.draft_path.exists());start.assert_not_called()

    def test_unsupported_platform_is_rejected_before_worker_spawn(self):
        with patch.object(console,'AUTO_BUDGET_SUPPORTED',False),patch.object(self.c,'_run_budget_plan') as run,self.assertRaisesRegex(ValueError,'requires Windows'):
            self.c.adjust_budget(self.config)
        run.assert_not_called()

    def test_changed_report_cannot_adjust_settings(self):
        for field,value in [('reserve',4096),('prefill',128),('kv_storage',1),('context',131072),('concurrency',2),
                            ('budget',49153),('budget',True),('budget',262144),('minimum_budget',64),
                            ('safety_bytes',1),('runtime_bytes',1800*MIB),('version',2)]:
            report=dict(self.report);report[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                validate_budget_plan('KVMEM_BUDGET_PLAN '+json.dumps(report),self.p,self.tokens)
        with self.assertRaises(ValueError):validate_budget_plan(self.output()+'\n'+self.output(),self.p,self.tokens)

    def test_omitted_reserve_and_prefill_use_launcher_defaults(self):
        for omitted in (('Reserve',), ('Prefill',), ('Reserve', 'Prefill')):
            with self.subTest(omitted=omitted):
                config=copy.deepcopy(self.config)
                for key in omitted:config['parameters'].pop(key)
                original=copy.deepcopy(config)
                effective=dict(self.p,**{key:console.DEFAULTS[key] for key in omitted})
                tokens,report=budget_evidence(effective)
                plan=dict(self.plan,token_settings=tokens)
                with patch.object(self.c,'inspect_model',return_value={'can_start':True}),patch.object(self.c,'plan',return_value=plan),patch.object(self.c,'_run_budget_plan',return_value='KVMEM_BUDGET_PLAN '+json.dumps(report)) as run:
                    result=self.c.adjust_budget(config)
                self.assertEqual(result['changes'],{'Budget':49152})
                self.assertEqual(result['reserve'],effective['Reserve'])
                self.assertEqual(result['prefill'],effective['Prefill'])
                self.assertEqual(config,original)
                run.assert_called_once_with(plan)
                self.assertFalse(self.c.budget_busy)

    def test_zero_or_oversized_budget_does_not_constrain_planning(self):
        for budget in (0,2147483647):
            self.config['parameters']['Budget']=budget
            with patch.object(self.c,'inspect_model',return_value={'can_start':True}),patch.object(self.c,'plan',return_value=self.plan) as plan,patch.object(self.c,'_run_budget_plan',return_value=self.output()):
                result=self.c.adjust_budget(self.config)
            self.assertEqual(plan.call_args.args[0]['parameters']['Budget'],13312)
            self.assertEqual(result['changes'],{'Budget':49152})
            self.assertEqual(self.config['parameters']['Budget'],budget)

    def test_worker_failure_and_unavailable_keep_configuration(self):
        with patch.object(self.c,'inspect_model',return_value={'can_start':True}),patch.object(self.c,'plan',return_value=self.plan):
            with patch.object(self.c,'_run_budget_plan',side_effect=ValueError('worker failed')),self.assertRaisesRegex(ValueError,'worker failed'):
                self.c.adjust_budget(self.config)
            self.assertFalse(self.c.budget_busy)
            self.report.update(available=False,budget=0,runtime_bytes=0)
            with patch.object(self.c,'_run_budget_plan',return_value=self.output()),self.assertRaisesRegex(ValueError,'Minimum budget'):
                self.c.adjust_budget(self.config)
        self.assertFalse(self.c.budget_busy);self.assertFalse(self.c.draft_path.exists())

    def test_running_busy_multi_gpu_and_old_worker_are_rejected(self):
        self.c.budget_busy=True
        with self.assertRaises(ValueError):self.c.adjust_budget(self.config)
        self.c.budget_busy=False
        self.c.process=MagicMock();self.c.process.poll.return_value=None
        with self.assertRaises(ValueError):self.c.adjust_budget(self.config)
        self.c.process=None
        multi=copy.deepcopy(self.config);multi['parameters']['Gpu']+=' ,GPU-other'
        with self.assertRaises(ValueError):self.c.adjust_budget(multi)
        self.plan['worker']={}
        with patch.object(self.c,'inspect_model',return_value={'can_start':True}),patch.object(self.c,'plan',return_value=self.plan),patch.object(self.c,'_run_budget_plan') as run,self.assertRaisesRegex(ValueError,'Update worker'):
            self.c.adjust_budget(self.config)
        run.assert_not_called();self.assertFalse(self.c.budget_busy)

    def test_stop_cancels_planning_before_worker_spawn(self):
        def stopped_plan(_):
            self.c.stop()
            return self.plan
        with patch.object(self.c,'inspect_model',return_value={'can_start':True}),patch.object(self.c,'plan',side_effect=stopped_plan),patch.object(console,'WindowsJob') as job,patch.object(console.subprocess,'Popen') as spawn,self.assertRaisesRegex(ValueError,'cancelled'):
            self.c.adjust_budget(self.config)
        spawn.assert_not_called();job.return_value.close.assert_called_once();self.assertFalse(self.c.budget_busy)

    def test_timeout_closes_owned_job_and_waits(self):
        with patch.object(console,'WindowsJob') as job,patch.object(console.subprocess,'Popen') as spawn:
            process=spawn.return_value
            process.communicate.side_effect=subprocess.TimeoutExpired('plan',180)
            with self.assertRaisesRegex(ValueError,'timed out'):self.c._run_budget_plan(self.plan)
            job.return_value.close.assert_called_once();process.wait.assert_called_once()
            self.assertIsNone(self.c.budget_job)

    def test_budget_quote_uses_normalized_explicit_reserve_and_prefill(self):
        for prefill, effective_prefill in ((4095, 3968), (4352, 4096), (0, 1024)):
            with self.subTest(prefill=prefill):
                self.p.update(Prefill=prefill, Reserve=4097)
                original = copy.deepcopy(self.config)
                tokens = token_settings('ninfer', self.p)
                _, report = budget_evidence(tokens['effective'] | {'KvType': self.p['KvType']})
                plan = dict(self.plan, token_settings=tokens)
                output = 'KVMEM_BUDGET_PLAN ' + json.dumps(report)
                with patch.object(self.c, 'inspect_model', return_value={'can_start': True}), \
                        patch.object(self.c, 'plan', return_value=plan), \
                        patch.object(self.c, '_run_budget_plan', return_value=output) as run:
                    result = self.c.adjust_budget(self.config)
                run.assert_called_once_with(plan)
                self.assertEqual((result['prefill'], result['reserve']), (effective_prefill, 4096))
                self.assertEqual(result['changes'], {'Budget': 49152})
                self.assertEqual(self.config, original)
                self.assertFalse(self.c.budget_busy)


if __name__=='__main__':unittest.main()
