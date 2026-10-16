"""Guard against confounded transformations and misleading paired denominators."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_core import owner
from qwen_debugger.domain import CaseImport, RuntimeProfile, Budget, digest
from qwen_debugger.tasks import compile_case
from qwen_debugger.storage import Store
from qwen_debugger.runtime import MockRuntime
from qwen_debugger.runner import create_job, get_job, manifest_for, cancel_job
from qwen_debugger.report import export_report
from qwen_debugger.contrast_transforms import contrast_variant, validate_contrast
from qwen_debugger.contrast import contrast_plan, create_contrast, execute_contrast


def original(): return compile_case(CaseImport.model_validate(owner()))


class ContrastTransforms(unittest.TestCase):
    def test_rename_maps_entity_answer_and_preserves_events(self):
        a=original(); b=contrast_variant(a,'rename'); r=validate_contrast(a,b,'rename')
        self.assertEqual(r['status'],'rule_verified')
        self.assertNotEqual(a['gold']['expected'],b['gold']['expected'])
        self.assertEqual(r['label_mapping'][a['gold']['expected']],b['gold']['expected'])
        self.assertEqual(b['input']['payload']['events'],a['input']['payload']['events'])
        self.assertEqual(b['solver']['trace'],a['solver']['trace'])
        self.assertFalse(b['selected_on_outcome'])

    def test_noise_move_changes_only_position_and_can_reverse(self):
        a=original(); b=contrast_variant(a,'noise-position')
        self.assertEqual(validate_contrast(a,b,'noise-position')['status'],'rule_verified')
        x=a['messages'][0]['content'].splitlines(); y=b['messages'][0]['content'].splitlines()
        self.assertCountEqual(x,y)
        self.assertEqual(x[-1],y[-1]);self.assertNotEqual(x,y)
        self.assertEqual(contrast_variant(b,'noise-position')['messages'],a['messages'])
        self.assertEqual(a,original())  # Existing renderer stays byte-identical.

    def test_same_gold_does_not_allow_changing_facts(self):
        a=original(); b=contrast_variant(a,'rename')
        source={k:copy.deepcopy(b[k]) for k in CaseImport.model_fields if k in b}
        source['input']['payload']['events']=[{'kind':'transfer','from_id':'p1','to_id':'p3'}]
        wrong=compile_case(CaseImport.model_validate(source))
        self.assertEqual(validate_contrast(a,wrong,'rename')['status'],'invalid')
        self.assertEqual(validate_contrast(a,b,'noise-position')['status'],'invalid')

    def test_no_noise_or_unsupported_factor_rejected(self):
        a=owner();a['input']['payload']['independent_noise']=[]
        with self.assertRaises(ValueError): contrast_variant(compile_case(CaseImport.model_validate(a)),'noise-position')
        with self.assertRaises(ValueError): contrast_variant(original(),'anything')


class ContrastExecution(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.store=Store(self.root/'db')
        self.home=patch('qwen_debugger.runner.Path.home',return_value=self.root/'home');self.home.start()
        self.profile=RuntimeProfile(id='m',backend='mock',max_new_tokens=16)
        self.budget=Budget(requests=4,output_tokens=64,seconds=60)
    def tearDown(self): self.home.stop();self.tmp.cleanup()
    def create(self,factor='noise-position'):
        return create_contrast(self.store,original(),self.profile,self.budget,factor=factor,pairs=2,seed=42)
    def test_all_pairs_frozen_and_order_counterbalanced_before_dispatch(self):
        j=self.create();m=manifest_for(self.store,j)
        self.assertEqual([s['arm'] for s in m['schedule']],['original','candidate','candidate','original'])
        self.assertEqual(len(self.store.rows('observations')),4)
        self.assertEqual(len(self.store.rows('attempts')),0)
        execute_contrast(self.store,j,MockRuntime(self.profile))
        self.assertEqual(get_job(self.store,j)['requests'],4)
        summary=export_report(self.store,j,self.root/'report')
        self.assertEqual(summary['counts']['PASS'],4)
        self.assertEqual(summary['paired']['evaluable_pairs'],2)
        self.assertEqual(summary['paired']['delta'],0)
    def test_correct_renamed_labels_are_not_regressions(self):
        j=self.create('rename'); m=manifest_for(self.store,j)
        b=self.store.object(m['candidate_case_hash'])
        expected=b['gold']['expected'];marker=b['input']['payload']['entities'][0]['name']
        class Mapped(MockRuntime):
            def generate(self,request,heartbeat=None):
                r=super().generate(request,heartbeat)
                if marker in request['payload']['messages'][0]['content']:r.update(text=expected,raw_text=expected)
                return r
        execute_contrast(self.store,j,Mapped(self.profile))
        s=export_report(self.store,j,self.root/'report')
        self.assertEqual(s['counts']['PASS'],4)
        self.assertEqual(s['paired']['regressions'],0)
    def test_cancel_preserves_unrun_pairs_and_does_not_resume(self):
        j=self.create();store=self.store
        class Cancel(MockRuntime):
            def generate(self,request,heartbeat=None):
                cancel_job(store,j);return super().generate(request,heartbeat)
        execute_contrast(store,j,Cancel(self.profile))
        s=export_report(store,j,self.root/'report')
        self.assertEqual(get_job(store,j)['state'],'CANCELLED')
        self.assertEqual(s['planned'],4);self.assertEqual(s['unscored'],3)
        self.assertEqual(s['paired']['evaluable_pairs'],0)
        execute_contrast(store,j,MockRuntime(self.profile))
        self.assertEqual(get_job(store,j)['requests'],1)
    def test_budget_and_duplicate_schedule_are_rejected(self):
        with self.assertRaises(ValueError):
            contrast_plan(original(),self.profile,Budget(requests=3,output_tokens=64,seconds=60),factor='rename',pairs=2,seed=42)
        j=self.create();m=manifest_for(self.store,j)
        with self.assertRaises(ValueError):
            create_job(self.store,original(),self.profile,[42],self.budget,parent_id=j,phase='contrast',arm='original',receipt_hash=m['receipt_hash'])
        self.assertEqual(get_job(self.store,j)['requests'],0)
    def test_pending_report_keeps_denominator_and_receipt(self):
        j=self.create();s=export_report(self.store,j,self.root/'report')
        self.assertEqual(s['planned'],4);self.assertEqual(s['unscored'],4)
        self.assertIsNone(s['paired']['delta'])
        self.assertTrue((self.root/'report/receipt.json').exists())
    def test_unknown_completion_stops_without_retry(self):
        p=RuntimeProfile(id='timeout',backend='mock',max_new_tokens=16,mock_finish='timeout')
        j=create_contrast(self.store,original(),p,self.budget,factor='rename',pairs=2,seed=42)
        execute_contrast(self.store,j,MockRuntime(p))
        self.assertEqual(get_job(self.store,j)['requests'],1)
        self.assertEqual(get_job(self.store,j)['state'],'INCOMPLETE')
        s=export_report(self.store,j,self.root/'report')
        self.assertEqual(s['execution_states']['UNKNOWN_COMPLETION'],1)
        self.assertEqual(s['unscored'],4)

    def test_semantic_regression_is_counted_per_pair(self):
        j=self.create();p=self.profile
        class PositionSensitive(MockRuntime):
            def generate(self,request,heartbeat=None):
                r=super().generate(request,heartbeat)
                lines=request['payload']['messages'][0]['content'].splitlines()
                if '下雨' in lines[1]: r.update(text='顾川',raw_text='顾川')
                return r
        execute_contrast(self.store,j,PositionSensitive(p))
        s=export_report(self.store,j,self.root/'report')
        self.assertEqual(s['paired']['regressions'],2)
        self.assertEqual(s['paired']['candidate_target_failures'],2)
        self.assertEqual(s['paired']['delta'],1)
        self.assertEqual(s['arms']['original']['counts']['PASS'],2)
        self.assertEqual(s['arms']['candidate']['counts']['FAIL'],2)

    def test_cli_dry_run_has_no_jobs_and_queue_does_not_dispatch(self):
        import contextlib
        import io
        from qwen_debugger.cli import main
        base=['--data-dir',str(self.root/'cli')]
        args=['contrast','--case','owner','--profile','examples/debugger/mock.json',
              '--factor','rename','--pairs','2','--request-budget','4']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(base+['cases','import','examples/debugger/owner_transfer.json']),0)
            self.assertEqual(main(base+args+['--dry-run']),0)
            store=Store(self.root/'cli');self.assertEqual(store.rows('jobs'),[])
            self.assertEqual(main(base+args+['--queue']),0)
        self.assertEqual(len(store.rows('jobs')),5)
        self.assertEqual(len(store.rows('observations')),4)
        self.assertEqual(store.rows('attempts'),[])
