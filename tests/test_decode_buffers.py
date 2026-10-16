"""Generated token positions and input views retain hybrid cache semantics."""
import inspect
import unittest
from unittest.mock import patch
import torch


class DecodeBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
            p=patch.object(m,name,inspect.unwrap(getattr(m,name)));p.start();cls.addClassCleanup(p.stop)
        torch.set_num_threads(1)

    def test_reused_position_views_match_reference_and_start_after_prompt(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        torch.manual_seed(714)
        model=TextOnlyQwen3_5ForCausalLM(tiny_config()).eval()
        model.generation_config.eos_token_id=[]
        ids=torch.tensor([[5,9,11,8,17,18]])
        expected,_=CachedGenerator(model,kv_mode='bf16',chunk_size=2,reuse_decode_buffers=False).generate(ids,4,temperature=0)
        seen=[];original=model.forward
        def tracked(*args,**kwargs):
            if kwargs['input_ids'].shape[-1]==1:
                positions=kwargs['position_ids']
                seen.append((positions,positions.clone(),kwargs['input_ids'].clone()))
            return original(*args,**kwargs)
        model.forward=tracked
        actual,_=CachedGenerator(model,kv_mode='bf16',chunk_size=2,reuse_decode_buffers=True).generate(ids,4,temperature=0)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(len(seen),3)
        self.assertEqual(len({v[0].untyped_storage().data_ptr() for v in seen}),1)
        for i,(_,positions,tokens) in enumerate(seen):
            self.assertEqual(positions.shape,(4,1,1))
            self.assertTrue((positions==len(ids[0])+i).all())
            self.assertEqual(tokens.item(),actual[0,len(ids[0])+i].item())

    def test_single_token_budget_does_not_need_a_decode_forward(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        model=TextOnlyQwen3_5ForCausalLM(tiny_config()).eval()
        ids=torch.tensor([[5,9,11,8]])
        result,stats=CachedGenerator(model,kv_mode='bf16',reuse_decode_buffers=True).generate(ids,1,temperature=0)
        self.assertEqual(result.shape,(1,5))
        self.assertEqual(stats['reused_tokens'],0)

    def test_seeded_sampling_and_rng_match_on_fixed_prefix_miss_and_hit(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        torch.manual_seed(91)
        model=TextOnlyQwen3_5ForCausalLM(tiny_config()).eval()
        model.generation_config.eos_token_id=[]
        reference=CachedGenerator(model,kv_mode='bf16',chunk_size=2,reuse_decode_buffers=False,optimized_sampling=False)
        optimized=CachedGenerator(model,kv_mode='bf16',chunk_size=2,reuse_decode_buffers=True,optimized_sampling=True)
        for last in (18,23):
            ids=torch.tensor([[5,9,11,8,17,last]])
            options=dict(temperature=.7,prefix_cache_key='campaign:npc',prefix_length=4)
            torch.manual_seed(155)
            expected,_=reference.generate(ids,8,**options);expected_rng=torch.get_rng_state()
            torch.manual_seed(155)
            actual,stats=optimized.generate(ids,8,**options)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
        self.assertTrue(stats['fixed_prefix_hit'])

    def test_rejected_sampling_invalidates_constraint_and_discards_live_state(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        from models.structured_decoding import StructuredDecodingError
        model=TextOnlyQwen3_5ForCausalLM(tiny_config()).eval()
        generator=CachedGenerator(model,kv_mode='bf16',chunk_size=2)
        class Rejected:
            failed=False
            def mask_for_sampling(self,logits,generated):return torch.full_like(logits,-torch.inf)
            def reject_sampling(self,error):
                self.failed=True
                raise StructuredDecodingError(str(error))
        constraint=Rejected()
        with self.assertRaises(StructuredDecodingError):
            generator.generate(torch.tensor([[5,9,11,8,17,18]]),3,temperature=.7,
                               prefix_cache_key='campaign:npc',prefix_length=4,constraint=constraint)
        self.assertTrue(constraint.failed)
        self.assertIsNone(generator.cache)
        self.assertEqual(generator.prefix,[])
        self.assertEqual([key[1] for key in generator.prefix_pool.entries],[(5,9,11,8)])


if __name__=='__main__':unittest.main()
