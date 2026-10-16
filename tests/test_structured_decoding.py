"""CPU native matcher integration; no model weights or downloads."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from models.structured_decoding import StructuredDecodingError, create_constraint


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    tokens = ['[UNK]', '[EOS]', 'WAIT', 'SPEAK', 'bad'] + [chr(i) for i in range(32, 127)]
    backend = Tokenizer(WordLevel({t: i for i, t in enumerate(tokens)}, unk_token='[UNK]'))
    backend.decoder = ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]')


SCHEMA = {'type': 'object', 'properties': {'a': {'enum': ['WAIT']}},
          'required': ['a'], 'additionalProperties': False}


class MissingDependencyTests(unittest.TestCase):
    def test_required_constraint_never_silently_becomes_unconstrained(self):
        with patch.dict(sys.modules, {'xgrammar': None}):
            with self.assertRaisesRegex(StructuredDecodingError, 'XGrammar'):
                create_constraint(tiny_tokenizer(), SCHEMA)


@unittest.skipUnless(importlib.util.find_spec('xgrammar'), 'XGrammar is not installed')
class NativeConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny_tokenizer()

    def scores(self):
        return torch.zeros(len(self.tokenizer), dtype=torch.float32)

    def test_deferred_mask_uses_validated_sampler_and_failed_request_stays_closed(self):
        from models.token_sampling import sample_token,TokenSamplingError
        constraint=create_constraint(self.tokenizer,SCHEMA)
        masked=constraint.mask_for_sampling(torch.full_like(self.scores(),-torch.inf),[])
        with self.assertRaises(TokenSamplingError) as failed:
            sample_token(masked,temperature=0)
        with self.assertRaises(StructuredDecodingError):
            constraint.reject_sampling(failed.exception)
        with self.assertRaisesRegex(StructuredDecodingError,'already failed'):
            constraint.apply(self.scores(),[])

    def test_direct_apply_still_rejects_all_masked_scores(self):
        constraint=create_constraint(self.tokenizer,SCHEMA)
        with self.assertRaises(StructuredDecodingError):
            constraint.apply(torch.full_like(self.scores(),-torch.inf),[])

    def test_mask_blocks_unknown_keys_and_incomplete_stop_but_allows_valid_json(self):
        constraint = create_constraint(self.tokenizer, SCHEMA)
        ids = []
        for token in ['{', '"', 'a', '"', ':', '"', 'WAIT', '"', '}']:
            logits = constraint.apply(self.scores(), ids)
            self.assertTrue(torch.isneginf(logits[self.tokenizer.eos_token_id]))
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.assertTrue(torch.isfinite(logits[token_id]), token)
            if token == 'a':
                self.assertTrue(torch.isneginf(logits[self.tokenizer.convert_tokens_to_ids('b')]))
            if token == 'WAIT':
                self.assertTrue(torch.isneginf(logits[self.tokenizer.convert_tokens_to_ids('SPEAK')]))
            ids.append(token_id)
        complete = constraint.apply(self.scores(), ids)
        self.assertTrue(torch.isfinite(complete[self.tokenizer.eos_token_id]))
        self.assertTrue(torch.isneginf(complete[self.tokenizer.convert_tokens_to_ids('bad')]))

    def test_request_matchers_do_not_share_state_and_repeated_apply_is_idempotent(self):
        first = create_constraint(self.tokenizer, SCHEMA)
        second = create_constraint(self.tokenizer, SCHEMA)
        first.apply(self.scores(), [self.tokenizer.convert_tokens_to_ids('{')])
        opening = self.tokenizer.convert_tokens_to_ids('{')
        self.assertTrue(torch.isfinite(second.apply(self.scores(), [])[opening]))
        left = first.apply(self.scores(), [opening])
        right = first.apply(self.scores(), [opening])
        self.assertTrue(torch.equal(left, right))

    def test_invalid_history_or_changed_prefix_fails_closed(self):
        constraint = create_constraint(self.tokenizer, SCHEMA)
        with self.assertRaises(StructuredDecodingError):
            constraint.apply(self.scores(), [self.tokenizer.convert_tokens_to_ids('bad')])
        constraint = create_constraint(self.tokenizer, SCHEMA)
        constraint.apply(self.scores(), [self.tokenizer.convert_tokens_to_ids('{')])
        with self.assertRaises(StructuredDecodingError):
            constraint.apply(self.scores(), [])

    def test_padded_model_vocabulary_is_supported_and_bad_shape_rejected(self):
        width = len(self.tokenizer) + 4
        constraint = create_constraint(self.tokenizer, SCHEMA, vocab_size=width)
        scores = constraint.apply(torch.zeros(width), [])
        self.assertTrue(torch.isneginf(scores[-4:]).all())
        with self.assertRaises(StructuredDecodingError):
            constraint.apply(torch.zeros(2, width), [])
        with self.assertRaises(StructuredDecodingError):
            constraint.apply(torch.zeros(width + 1), [])

    def test_every_actual_role_schema_compiles_and_masks_native_output(self):
        source = str(Path(__file__).resolve().parents[1] / 'local-trpg' / 'src')
        with patch.object(sys, 'path', [source, *sys.path]):
            from local_trpg.model.compact_npc import compact_npc_schema
            from local_trpg.schemas import PlayerIntent, SceneProposal, NarrationResult
        cases = [
            ('npc', compact_npc_schema(), '{"a":"WAIT"}'),
            ('interpreter', PlayerIntent.model_json_schema(), '{}'),
            ('director', SceneProposal.model_json_schema(), '{}'),
            ('narrator', NarrationResult.model_json_schema(), '{"text":"","source_event_ids":[]}'),
        ]
        for role, schema, wire in cases:
            with self.subTest(role=role):
                constraint = create_constraint(self.tokenizer, schema)
                history = []
                for char in wire:
                    scores = constraint.apply(self.scores(), history)
                    token_id = self.tokenizer.convert_tokens_to_ids(char)
                    self.assertTrue(torch.isfinite(scores[token_id]), (role, char, history))
                    self.assertTrue(torch.isneginf(scores[self.tokenizer.eos_token_id]))
                    history.append(token_id)
                scores = constraint.apply(self.scores(), history)
                self.assertTrue(torch.isfinite(scores[self.tokenizer.eos_token_id]))
                self.assertTrue(torch.isneginf(scores[self.tokenizer.convert_tokens_to_ids('{')]))


if __name__ == '__main__':
    unittest.main()
