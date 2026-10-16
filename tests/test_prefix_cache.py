import copy
import inspect
import unittest
from unittest.mock import patch
import torch


class PrefixPoolTests(unittest.TestCase):
    def test_signature_invalidates_when_instance_normalization_computation_changes(self):
        from models.prefix_cache import model_cache_signature
        class Normalize(torch.nn.Module):
            def _norm(self,x):return x
            def forward(self,x):return self._norm(x)
        model=Normalize().eval()
        signature=model_cache_signature(model)
        model._norm=lambda x:x+1
        self.assertNotEqual(signature,model_cache_signature(model))

    def test_budget_counts_dictionary_held_recurrent_and_convolution_states(self):
        from types import SimpleNamespace
        from models.quantized_kv import cache_bytes
        recurrent=torch.zeros(64,dtype=torch.uint8)
        convolution=torch.zeros(16,dtype=torch.float32)
        layer=SimpleNamespace(recurrent_states={0:recurrent},conv_states={0:convolution},alias=recurrent[:8])
        cache=SimpleNamespace(layers=[layer])
        self.assertEqual(cache_bytes(cache),128)
        self.assertFalse(self.pool(max_bytes=100).put('c:npc:guard',[1],cache))

    def test_snapshot_discards_unused_packed_slabs_without_changing_source(self):
        from models.quantized_kv import cache_bytes
        cache=self.cache();layer=cache.layers[0]
        source_capacity=layer.value_bank.data.shape[2]
        expected=layer.value_bank.unpack(torch.float32).clone()
        pool=self.pool(max_bytes=128*1024)
        self.assertTrue(pool.put('c:npc:guard',[1,2],cache))
        snapshot=pool.get('c:npc:guard',[1,2])
        stored=snapshot.layers[0]
        self.assertLessEqual(cache_bytes(snapshot),128*1024)
        self.assertEqual(stored.value_bank.data.shape[2],stored.value_bank.units)
        torch.testing.assert_close(stored.value_bank.unpack(torch.float32),expected)
        self.assertEqual(layer.value_bank.data.shape[2],source_capacity)
        stored.update(torch.randn(1,1,256,64),torch.randn(1,1,256,64))
        self.assertEqual(layer.get_seq_length(),256)

    def pool(self,**kw):
        from models.prefix_cache import PrefixSnapshotPool
        return PrefixSnapshotPool(**kw)

    def cache(self):
        from transformers import Qwen3_5TextConfig
        from models.quantized_kv import make_cache
        c=make_cache(Qwen3_5TextConfig(num_hidden_layers=1,layer_types=['full_attention']),'q8')
        x=torch.randn(1,1,256,64)
        c.layers[0].update(x,x)
        return c

    def test_snapshots_are_exact_scoped_and_never_mutated_by_a_caller(self):
        pool=self.pool(max_entries=2,max_bytes=2**20)
        cache=self.cache(); original=cache.layers[0].keys.clone()
        self.assertTrue(pool.put('campaign:a:npc:guard',[1,2],cache))
        cache.layers[0].keys.zero_()
        self.assertIsNone(pool.get('campaign:b:npc:guard',[1,2]))
        self.assertIsNone(pool.get('campaign:a:npc:doctor',[1,2]))
        self.assertIsNone(pool.get('campaign:a:npc:guard',[1,3]))
        first=pool.get('campaign:a:npc:guard',[1,2])
        torch.testing.assert_close(first.layers[0].keys,original)
        first.layers[0].keys.zero_();first.layers[0].key_bank.data.zero_()
        second=pool.get('campaign:a:npc:guard',[1,2])
        torch.testing.assert_close(second.layers[0].keys,original)
        self.assertTrue(second.layers[0].key_bank.data.any())

    def test_pool_evicts_by_bytes_and_lru_and_rejects_oversized_entries(self):
        from models.quantized_kv import cache_bytes
        cache=self.cache()
        probe=self.pool();probe.put('probe',[1],cache)
        size=cache_bytes(probe.get('probe',[1]))
        pool=self.pool(max_entries=3,max_bytes=2*size)
        pool.put('a',[1],cache);pool.put('b',[1],cache)
        pool.get('a',[1]);pool.put('c',[1],cache)
        self.assertIsNone(pool.get('b',[1]))
        self.assertIsNotNone(pool.get('a',[1]))
        self.assertLessEqual(pool.info()['bytes'],2*size)
        small=self.pool(max_entries=2,max_bytes=size-1)
        self.assertFalse(small.put('a',[1],cache))
        self.assertEqual(small.info()['entries'],0)

    def test_signature_includes_quantization_metadata_and_non_tensor_routing(self):
        from types import SimpleNamespace
        from models.prefix_cache import model_cache_signature
        model=torch.nn.Linear(4,4).eval().requires_grad_(False)
        model.routing_strength=0.
        state=SimpleNamespace(absmax=torch.ones(4),code=torch.ones(4),blocksize=64,
                              quant_type='nf4',dtype=torch.bfloat16,shape=(4,4),offset=None,state2=None,nested=False)
        model.weight.quant_state=state
        first=model_cache_signature(model)
        state.absmax.add_(1)
        second=model_cache_signature(model)
        self.assertNotEqual(first,second)
        state.blocksize=128
        self.assertNotEqual(second,model_cache_signature(model))
        second=model_cache_signature(model);model.routing_strength=.5
        self.assertNotEqual(second,model_cache_signature(model))

    def test_actual_qwen_template_accepts_fixed_system_boundary_without_system_only_render(self):
        from transformers import AutoTokenizer
        from models.runtime_paths import DEFAULT_BUNDLE,read_bundle
        from models.pretrained import PretrainedChatModel
        from models.prefix_cache import exact_message_prefix
        model=object.__new__(PretrainedChatModel)
        model.tokenizer=AutoTokenizer.from_pretrained(read_bundle(DEFAULT_BUNDLE)['base'],local_files_only=True)
        model.inference_template=model.tokenizer.chat_template;model.thinking=False
        messages=[{'role':'system','content':'固定规则只回答当前问题。'},
                  {'role':'user','content':'PRIVATE_DYNAMIC_QUESTION_918273'}]
        ids=model.encode(messages)
        prefix=exact_message_prefix(model.encode,messages,1,ids)
        self.assertTrue(prefix)
        self.assertEqual(ids[:len(prefix)],prefix)
        self.assertNotIn('PRIVATE_DYNAMIC',model.tokenizer.decode(prefix))


class PrefixGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
            p=patch.object(m,name,inspect.unwrap(getattr(m,name)));p.start();cls.addClassCleanup(p.stop)
        torch.set_num_threads(1)

    def test_repeated_role_prefix_preserves_hybrid_state_and_different_questions(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        torch.manual_seed(42)
        model=TextOnlyQwen3_5ForCausalLM(tiny_config()).eval()
        generator=CachedGenerator(model,kv_mode='bf16',chunk_size=2)
        for question in ([5,9,11,8,17,18],[5,9,11,8,22,23]):
            ids=torch.tensor([question])
            reference=CachedGenerator(model,kv_mode='bf16',chunk_size=2)
            expected,_=reference.generate(ids,3,temperature=0)
            actual,stats=generator.generate(ids,3,temperature=0,prefix_cache_key='c:npc:guard',prefix_length=4)
            torch.testing.assert_close(actual,expected)
        self.assertEqual(stats['reused_tokens'],4)
        self.assertEqual(stats['prefill_tokens'],2)
        other_stats=generator.generate(ids,3,temperature=0,prefix_cache_key='c:npc:doctor',prefix_length=4)[1]
        self.assertEqual(other_stats['reused_tokens'],0)

    def test_generation_failure_clears_live_state_but_never_adds_question_to_snapshot(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        model=TextOnlyQwen3_5ForCausalLM(tiny_config()).eval()
        generator=CachedGenerator(model,kv_mode='bf16',chunk_size=2)
        ids=torch.tensor([[5,9,11,8,17,18]])
        class Rejected:
            def apply(self,logits,generated):raise RuntimeError('cancelled constraint')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            generator.generate(ids,3,temperature=0,prefix_cache_key='c:npc:g',prefix_length=4,constraint=Rejected())
        self.assertIsNone(generator.cache)
        self.assertEqual(generator.prefix,[])
        self.assertEqual([key[1] for key in generator.prefix_pool.entries],[(5,9,11,8)])
        actual,stats=generator.generate(ids,3,temperature=0,prefix_cache_key='c:npc:g',prefix_length=4)
        self.assertTrue(stats['fixed_prefix_hit'])
        with torch.no_grad():model.model.language_model.embed_tokens.weight.add_(.01)
        _,stats=generator.generate(ids,3,temperature=0,prefix_cache_key='c:npc:g',prefix_length=4)
        self.assertFalse(stats['fixed_prefix_hit'])

    def test_sampler_applies_constraint_before_sampling_and_stops_on_eos(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        model=TextOnlyQwen3_5ForCausalLM(tiny_config()).eval()
        generator=CachedGenerator(model,kv_mode='bf16',chunk_size=2)
        histories=[]
        class Forced:
            def apply(self,logits,generated):
                histories.append(list(generated))
                result=torch.full_like(logits,-float('inf'));result[[23,24,2][len(generated)]]=1
                return result
        ids=torch.tensor([[5,9,11,8]])
        actual,_=generator.generate(ids,5,temperature=.7,constraint=Forced())
        self.assertEqual(actual[0,4:].tolist(),[23,24,2])
        self.assertEqual(histories,[[],[23],[23,24]])

    def test_unversioned_model_weights_never_reuse_a_snapshot(self):
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        with torch.inference_mode():model=TextOnlyQwen3_5ForCausalLM(tiny_config()).eval()
        generator=CachedGenerator(model,kv_mode='bf16',chunk_size=2)
        ids=torch.tensor([[5,9,11,8,17,18]])
        for _ in range(2):
            _,stats=generator.generate(ids,3,temperature=0,prefix_cache_key='c:npc:g',prefix_length=4)
        self.assertFalse(stats['fixed_prefix_hit'])
        self.assertEqual(stats['prefix_snapshots']['entries'],0)

    def test_adapter_scaling_change_invalidates_prefix_and_matches_fresh_state(self):
        from peft import LoraConfig,get_peft_model
        from tests.test_text_only_qwen35 import tiny_config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        torch.manual_seed(61)
        model=get_peft_model(TextOnlyQwen3_5ForCausalLM(tiny_config()),LoraConfig(
            task_type='CAUSAL_LM',r=2,lora_alpha=4,target_modules=['gate_proj','up_proj','down_proj']))
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_' in name:p.normal_(std=.2)
        model.eval().requires_grad_(False)
        generator=CachedGenerator(model,kv_mode='bf16',chunk_size=2)
        ids=torch.tensor([[5,9,11,8,17,18]])
        generator.generate(ids,1,temperature=0,prefix_cache_key='c:npc:g',prefix_length=4)
        for module in model.modules():
            if isinstance(getattr(module,'scaling',None),dict):module.scaling['default']=0.
        actual,stats=generator.generate(ids,1,temperature=0,prefix_cache_key='c:npc:g',prefix_length=4)
        self.assertFalse(stats['fixed_prefix_hit'])
        fresh=CachedGenerator(model,kv_mode='bf16',chunk_size=2)
        expected,_=fresh.generate(ids,1,temperature=0)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(generator.cache.layers[1].keys,fresh.cache.layers[1].keys,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
