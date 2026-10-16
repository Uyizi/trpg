import json
import tempfile
import unittest
from pathlib import Path
from test_core import owner
from qwen_debugger.domain import CaseImport, RuntimeProfile, Budget
from qwen_debugger.tasks import compile_case
from qwen_debugger.storage import Store
from qwen_debugger.runner import create_job, execute_job, cancel_job, recover_jobs, claim
from qwen_debugger.runtime import MockRuntime
from qwen_debugger.report import export_report


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.s=Store(self.root/'db'); self.c=compile_case(CaseImport.model_validate(owner()))
        self.s.put_case(self.c)
    def tearDown(self): self.temp.cleanup()
    def job(self,**kwargs):
        p=RuntimeProfile(id='mock',backend='mock',max_new_tokens=16,**kwargs)
        return create_job(self.s,self.c,p,[7,7,7],Budget(requests=3,output_tokens=48,seconds=30))

    def test_budget_reservation_stops_before_next_call(self):
        p=RuntimeProfile(id='mock',backend='mock',max_new_tokens=16,mock_usage_unknown=True)
        job=create_job(self.s,self.c,p,[1,2,3],Budget(requests=3,output_tokens=16,seconds=30))
        execute_job(self.s,job,MockRuntime(p))
        obs=self.s.rows('observations')
        self.assertEqual([o['state'] for o in obs],['DONE','NOT_RUN','NOT_RUN'])
        self.assertEqual(self.s.rows('jobs')[0]['tokens'],16)
        self.assertEqual(self.s.rows('jobs')[0]['state'],'INCOMPLETE')

    def test_cancel_sends_no_requests(self):
        job=self.job(); cancel_job(self.s,job); execute_job(self.s,job,MockRuntime(RuntimeProfile(id='mock',backend='mock')))
        self.assertEqual(self.s.rows('attempts'),[])
        self.assertTrue(all(o['state']=='NOT_RUN' for o in self.s.rows('observations')))

    def test_claim_is_atomic_and_recovery_preserves_unknown_cost(self):
        job=self.job()
        a=claim(self.s,job); b=claim(self.s,job)
        self.assertIsNotNone(a); self.assertIsNone(b)
        recover_jobs(self.s)
        self.assertEqual(self.s.rows('observations')[0]['state'],'UNKNOWN_COMPLETION')
        self.assertEqual(self.s.rows('attempts')[0]['state'],'UNKNOWN_COMPLETION')
        self.assertEqual(self.s.rows('jobs')[0]['tokens'],16)

    def test_replay_exports_denominators_and_escaped_raw_evidence(self):
        job=self.job(mock_text='<script>alert(1)</script>')
        p=RuntimeProfile(id='mock',backend='mock',max_new_tokens=16,mock_text='<script>alert(1)</script>')
        execute_job(self.s,job,MockRuntime(p))
        dest=self.root/'report'; summary=export_report(self.s,job,dest)
        self.assertEqual(summary['counts']['FAIL'],3)
        self.assertEqual(summary['planned'],3)
        self.assertEqual(summary['independent_trials'],None)
        html=(dest/'report.html').read_text()
        self.assertNotIn('<script>alert(1)</script>',html)
        self.assertIn('&lt;script&gt;',html)
        self.assertEqual(len((dest/'observations.jsonl').read_text().splitlines()),3)
        self.assertTrue(list((dest/'artifacts').iterdir()))

    def test_timeout_does_not_become_model_failure(self):
        job=self.job(mock_finish='timeout')
        execute_job(self.s,job,MockRuntime(RuntimeProfile(id='mock',backend='mock',max_new_tokens=16,mock_finish='timeout')))
        self.assertEqual(self.s.rows('observations')[0]['state'],'UNKNOWN_COMPLETION')
        self.assertIsNone(self.s.rows('observations')[0]['outcome'])
        self.assertEqual(len(self.s.rows('attempts')),1)

if __name__=='__main__': unittest.main()

class ReviewRegressionTests(unittest.TestCase):
    def test_cancel_before_dispatch_preserves_not_run_and_releases_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)); c=compile_case(CaseImport.model_validate(owner()))
            p=RuntimeProfile(id='m',backend='mock',max_new_tokens=16)
            job=create_job(s,c,p,[1],Budget(requests=1,output_tokens=16,seconds=10))
            class Cancelling(MockRuntime):
                def prepare(self,*args,**kwargs):
                    request=super().prepare(*args,**kwargs); cancel_job(s,job); return request
                def generate(self,*args,**kwargs): raise AssertionError('must not generate')
            execute_job(s,job,Cancelling(p))
            self.assertEqual(s.rows('observations')[0]['state'],'NOT_RUN')
            self.assertEqual(s.rows('attempts')[0]['state'],'NOT_SENT')
            self.assertEqual(s.rows('jobs')[0]['tokens'],0)
            self.assertEqual(s.rows('jobs')[0]['requests'],0)

    def test_expired_deadline_never_starts_worker(self):
        from qwen_debugger.runtime import LocalRuntime,NotSent
        import time
        with tempfile.TemporaryDirectory() as d:
            r=LocalRuntime(RuntimeProfile(id='local',backend='local'),Path(d)/'log')
            r.deadline=time.monotonic()-1
            with self.assertRaises(NotSent): r.generate({})
            self.assertIsNone(r.proc)

    def test_frozen_identity_drift_prevents_dispatch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d)); c=compile_case(CaseImport.model_validate(owner()))
            p=RuntimeProfile(id='local',backend='local',model_path='/tmp/fake-model')
            with patch('qwen_debugger.runner.inspect_local',return_value={'model_path':'/tmp/fake-model','hash':'before'}):
                job=create_job(s,c,p,[1],Budget())
            with patch('qwen_debugger.runner.inspect_local',return_value={'model_path':'/tmp/fake-model','hash':'after'}):
                with self.assertRaisesRegex(ValueError,'身份'):
                    execute_job(s,job,MockRuntime(p))
            self.assertEqual(s.rows('attempts'),[])

    def test_output_decode_preserves_unrequested_whitespace(self):
        from qwen_debugger.worker_protocol import decode_output
        class Tokenizer:
            def decode(self,ids,skip_special_tokens,**kwargs):
                return ' 许遥\n' if skip_special_tokens else ' 许遥\n<eos>'
        self.assertEqual(decode_output(Tokenizer(),[1,2])['text'],' 许遥\n')

    def test_worker_inherits_device_lease(self):
        import os,subprocess,sys,fcntl
        from qwen_debugger.runner import device_lease
        with device_lease() as fd:
            self.assertIsInstance(fd,int)
            code='import os,sys; os.fstat(int(sys.argv[1])); print("inherited")'
            out=subprocess.check_output([sys.executable,'-c',code,str(fd)],pass_fds=(fd,),text=True)
            self.assertIn('inherited',out)

# CPU checks must never contend with the user's real diagnostic GPU lease.
def setUpModule():
    global _test_home, _home_patch
    from unittest.mock import patch
    _test_home=tempfile.TemporaryDirectory()
    _home_patch=patch('qwen_debugger.runner.Path.home',return_value=Path(_test_home.name))
    _home_patch.start()


def tearDownModule():
    _home_patch.stop()
    _test_home.cleanup()
