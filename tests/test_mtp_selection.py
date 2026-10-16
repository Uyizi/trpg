"""Explicit MTP selection must fail before unrelated loading or tokenization."""
from pathlib import Path
import tempfile
import unittest

from models.pretrained import PretrainedChatModel


class MTPSelectionTests(unittest.TestCase):
    def test_legacy_speculation_and_resident_are_incompatible(self):
        for kwargs in ({'speculative':1},{'resident':True}):
            with self.assertRaisesRegex(ValueError,'MTP'):
                PretrainedChatModel(mtp=True,**kwargs)

    def test_missing_weights_fail_before_target_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'MTP'):
                PretrainedChatModel(mtp=True,mtp_checkpoint=Path(folder)/'missing.safetensors')

    def test_checkpoint_requires_explicit_mtp_choice(self):
        with self.assertRaisesRegex(ValueError,'mtp'):
            PretrainedChatModel(mtp_checkpoint='/unused')

    def test_answer_rejects_sampling_grammar_and_scoped_prefix_before_encoding(self):
        model=PretrainedChatModel.__new__(PretrainedChatModel)
        model.mtp=True
        def unexpected_encode(messages):raise AssertionError('Should reject before tokenization')
        model.encode=unexpected_encode
        for kwargs in ({'temperature':.7},{'temperature':0,'response_schema':{}},
                       {'temperature':0,'prefix_cache_key':'npc'}):
            with self.assertRaisesRegex(ValueError,'MTP'):
                model.answer([{'role':'user','content':'hello'}],max_new_tokens=4,**kwargs)


if __name__=='__main__':unittest.main()
