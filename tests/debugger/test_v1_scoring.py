import unittest
from qwen_debugger.domain import CaseImport,Gold
from qwen_debugger.tasks import compile_case,grade
from test_core import owner


def raw(gold):
    c=owner();c['input']={'kind':'raw','messages':[{'role':'user','content':'按指定格式回答。'}]}
    c['gold']={**gold,'source':'user_provided'}
    return compile_case(CaseImport.model_validate(c))


class ExplicitScoring(unittest.TestCase):
    def test_json_fields_ignore_order_preserve_types_and_reject_duplicates(self):
        c=raw({'kind':'json_fields','expected':'{"value":2,"ready":true}'})
        for text,outcome,kind in [('{"ready":true,"value":2}','PASS',None),
                                 ('{"ready":1,"value":2}','FAIL','semantic_mismatch'),
                                 ('{"ready":true,"value":3}','FAIL','semantic_mismatch'),
                                 ('{"ready":true,"value":2,"other":0}','FAIL','required_format_missing'),
                                 ('{"ready":true,"value":2,"value":3}','FAIL','required_format_missing'),
                                 ('```json\n{"ready":true,"value":2}\n```','FAIL','required_format_missing')]:
            g=grade(c,{'text':text,'finish_reason':'stop'})
            self.assertEqual((g['outcome'],g['failure_kind']),(outcome,kind))
    def test_numeric_explicit_tolerance_boundary_and_nonfinite(self):
        c=raw({'kind':'numeric','expected':'0.3','absolute_tolerance':'0.01'})
        for text,outcome,kind in [('0.31','PASS',None),('3e-1','PASS',None),('0.310001','FAIL','semantic_mismatch'),
                                 ('NaN','FAIL','required_format_missing'),('Infinity','FAIL','required_format_missing'),
                                 ('答案是0.3','FAIL','required_format_missing')]:
            g=grade(c,{'text':text,'finish_reason':'stop'})
            self.assertEqual((g['outcome'],g['failure_kind']),(outcome,kind))
        self.assertEqual(grade(c,{'text':'0.3','finish_reason':'length'})['outcome'],'TRUNCATED')
    def test_invalid_gold_is_rejected_before_run(self):
        for gold in [{'kind':'numeric','expected':'1'}, {'kind':'numeric','expected':'1','absolute_tolerance':'-1'},
                     {'kind':'numeric','expected':'NaN','absolute_tolerance':'0'},
                     {'kind':'json_fields','expected':'{"x":NaN}'},
                     {'kind':'json_fields','expected':'{"x":1,"x":2}'},
                     {'kind':'json_fields','expected':'{"x":{"nested":1}}'},
                     {'kind':'exact_text','expected':'1','absolute_tolerance':'1'}]:
            with self.subTest(gold=gold),self.assertRaises(ValueError):raw(gold)
    def test_old_gold_snapshot_is_unchanged(self):
        old=owner()['gold'];self.assertEqual(Gold.model_validate(old).model_dump(),old)
    def test_structured_solver_cannot_change_scoring_kind(self):
        c=owner();c['gold'].update(kind='json_fields',expected='{"answer":"许遥"}')
        with self.assertRaises(ValueError):compile_case(CaseImport.model_validate(c))
