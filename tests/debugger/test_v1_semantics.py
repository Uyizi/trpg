import tempfile
import unittest
from pathlib import Path
from test_core import owner
from qwen_debugger.domain import CaseImport
from qwen_debugger.tasks import compile_case
from qwen_debugger.contrast_transforms import contrast_variant,validate_contrast
from qwen_debugger.storage import Store
from qwen_debugger.reviews import propose_review,decide_review,review_receipt

class SemanticTests(unittest.TestCase):
    def test_choice_permutation_recomputes_letter(self):
        c=owner();c['input']['renderer_id']='owner_choice_zh_v1'
        c['input']['payload']['query']['answer_format']='option_letter'
        c['input']['payload']['options']=['p1','p2','p3'];c['gold']['expected']='C'
        a=compile_case(CaseImport.model_validate(c));b=contrast_variant(a,'permute-options')
        self.assertEqual(b['gold']['expected'],'B')
        self.assertEqual(validate_contrast(a,b,'permute-options')['status'],'rule_verified')
        self.assertEqual(contrast_variant(a,'rename')['gold']['expected'],'C')
    def test_rule_reordering_and_counterfactual_are_solved(self):
        import json
        c=json.loads(Path('examples/debugger/scoped_rules.json').read_text())
        c['input']['payload']['predicates'].append({'id':'unproved','name':'未证明命题'})
        a=compile_case(CaseImport.model_validate(c))
        b=contrast_variant(a,'flip-query')
        self.assertEqual(validate_contrast(a,b,'flip-query')['status'],'rule_verified')
        self.assertNotEqual(a['input']['payload']['query'],b['input']['payload']['query'])
    def test_manual_review_starts_unknown_and_is_hash_bound(self):
        c=owner();c['input']={'kind':'raw','messages':[{'role':'user','content':'只输出2：1+1？'}]}
        c['gold'].update(expected='2',source='user_provided')
        a=compile_case(CaseImport.model_validate(c));c['case_id']='candidate';c['input']['messages'][0]['content']='1+1？只输出2'
        b=compile_case(CaseImport.model_validate(c))
        with tempfile.TemporaryDirectory() as d:
            s=Store(Path(d));r=propose_review(s,a,b,'人工改写')
            self.assertEqual(review_receipt(s,r)['status'],'unknown')
            with self.assertRaises(ValueError):
                decide_review(s,r,'accept','语义保持','test-human',parent_hash='wrong',candidate_hash='wrong')
            from qwen_debugger.domain import digest
            decide_review(s,r,'accept','语义保持','test-human',parent_hash=digest(a),candidate_hash=digest(b))
            self.assertEqual(review_receipt(s,r)['status'],'human_confirmed')
