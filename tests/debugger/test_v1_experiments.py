import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_core import owner
from qwen_debugger.domain import CaseImport,RuntimeProfile,Budget
from qwen_debugger.tasks import compile_case
from qwen_debugger.storage import Store
from qwen_debugger.suites import create_suite,compare_plan,create_comparison
from qwen_debugger.fixed import execute_fixed
from qwen_debugger.fixed_report import summarize_fixed
from qwen_debugger.contrast import create_contrast
from qwen_debugger.confirmation import create_confirmation

class Experiments(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.s=Store(Path(self.t.name)/'db')
  self.home=patch('qwen_debugger.runner.Path.home',return_value=Path(self.t.name)/'home');self.home.start()
  self.a=compile_case(CaseImport.model_validate(owner()));self.s.put_case(self.a)
  self.p=RuntimeProfile(id='a',backend='mock',max_new_tokens=8)
  self.q=RuntimeProfile(id='b',backend='mock',max_new_tokens=8,mock_text='顾川')
 def tearDown(self):self.home.stop();self.t.cleanup()
 def test_regression_reports_neighbor_damage_and_macro_average(self):
  suite=create_suite(self.s,'example',['owner'])
  j=create_comparison(self.s,suite,self.p,self.q,Budget(requests=2,output_tokens=16,seconds=60))
  execute_fixed(self.s,j);r=summarize_fixed(self.s,j)
  self.assertEqual(r['paired']['regressions'],1)
  self.assertEqual(r['arms']['candidate']['end_to_end_pass_rate'],0)
  self.assertEqual(r['macro']['family_count'],1)
 def test_controlled_comparison_rejects_unlisted_changes(self):
  suite=create_suite(self.s,'example',['owner'])
  with self.assertRaises(ValueError):compare_plan(self.s,suite,self.p,self.q,Budget(requests=2,output_tokens=16),comparison_type='controlled_checkpoint_comparison')
 def test_confirm_is_fresh_64_observations_and_greedy_never_upgrades(self):
  j=create_contrast(self.s,self.a,self.q,Budget(requests=2,output_tokens=16,seconds=60),factor='noise-position',pairs=1)
  from qwen_debugger.contrast import execute_contrast
  from qwen_debugger.runtime import MockRuntime
  execute_contrast(self.s,j,MockRuntime(self.q))
  c=create_confirmation(self.s,j,Budget(requests=64,output_tokens=512,seconds=60))
  execute_fixed(self.s,c);r=summarize_fixed(self.s,c)
  self.assertEqual(r['planned'],64);self.assertEqual(r['confirmation']['complete'],True)
  self.assertEqual(r['confirmation']['conclusion_level'],'exploratory')
 def test_family_cannot_cross_splits(self):
  c=owner();c['case_id']='held';c['root_case_id']='held';c['split']='heldout'
  self.s.put_case(compile_case(CaseImport.model_validate(c)))
  with self.assertRaises(ValueError):create_suite(self.s,'leak',['owner','held'],purpose='heldout')
 def test_full_investigation_keeps_one_budget_and_runs_triggers_after_correct_baseline(self):
  from qwen_debugger.workflow import create_workflow
  from qwen_debugger.shrink import execute_investigation,children_for
  from qwen_debugger.runtime import MockRuntime
  from qwen_debugger.runner import get_job,manifest_for
  from qwen_debugger.report import export_report
  j=create_workflow(self.s,self.a,self.p,Budget(requests=19,output_tokens=152,seconds=60),pairs=1)
  execute_investigation(self.s,j,MockRuntime(self.p))
  self.assertEqual(get_job(self.s,j)['requests'],9)
  self.assertEqual(get_job(self.s,j)['state'],'COMPLETE')
  for link in children_for(self.s,j):self.assertEqual(manifest_for(self.s,link['child_id'])['budget_account_id'],j)
  r=export_report(self.s,j,Path(self.t.name)/'report')
  self.assertEqual(len(r['trigger_comparisons']),3)
  self.assertEqual(r['phases']['baseline']['planned'],3)
