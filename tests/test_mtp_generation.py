"""CPU verification contracts; scripted draft is a test oracle, not real MTP."""
from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen3_5Config

from models.cached_generation import CachedGenerator
from models.qwen35_kernels import install_convolution_kernels
from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM


def tiny_target():
    install_convolution_kernels()
    torch.manual_seed(1907)
    config=Qwen3_5Config(text_config=dict(vocab_size=31,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=64,
        layer_types=['linear_attention','full_attention'],linear_num_key_heads=1,
        linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_conv_kernel_dim=4,attn_output_gate=True,tie_word_embeddings=True,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,
                         'mrope_section':[4,6,6]},max_position_embeddings=2048))
    model=TextOnlyQwen3_5ForCausalLM(config).eval().requires_grad_(False)
    model.generation_config.eos_token_id=[]
    return model


class ScriptedHead:
    """Finite golden draft to exercise accepting/rejecting real target forwards."""
    training=False
    def __init__(self,prompt_length,golden,*,reject=False):
        self.prompt_length=prompt_length;self.golden=golden;self.reject=reject;self.calls=[]
    def __call__(self,ids,hidden,positions,*,past_key_values=None,use_cache=False,logits_to_keep=1):
        self.calls.append((ids.clone(),positions.clone()))
        cache=past_key_values or SimpleNamespace(next_position=0)
        assert positions[0,0].item()==cache.next_position
        cache.next_position=positions[0,-1].item()+1
        index=int(positions[0,-1])-self.prompt_length+2
        token=self.golden[index] if 0<=index<len(self.golden) else 0
        if self.reject:token=(token+1)%31
        logits=torch.full((1,1,31),-10.)
        logits[0,0,token]=10
        return SimpleNamespace(logits=logits,hidden_states=hidden,past_key_values=cache)
    def prefill(self,ids,hidden,bonus,positions,**kwargs):
        shifted=torch.cat((ids[:,1:],bonus.reshape(1,1)),dim=1)
        return self(shifted,hidden,positions,**kwargs)


class Stream:
    def __init__(self,fail=False):self.values=[];self.ends=0;self.fail=fail
    def put(self,value):
        if self.fail and self.values:raise RuntimeError('cancelled')
        self.values.append(value.clone())
    def end(self):self.ends+=1


class MTPGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def generator(self,*,mode='bf16',reject=False,prompt=None,budget=7,chunk=1024):
        model=tiny_target()
        ids=torch.tensor([prompt or [3,5,8,11,2]])
        baseline,_=CachedGenerator(model,kv_mode=mode,chunk_size=chunk).generate(ids,budget,temperature=0)
        golden=baseline[0,ids.shape[1]:].tolist()
        head=ScriptedHead(ids.shape[1],golden,reject=reject)
        from models.mtp_generation import MTPGenerator
        generator=MTPGenerator(model,head,kv_mode=mode,chunk_size=chunk)
        return generator,ids,baseline,head

    def test_accepts_verified_drafts_with_fewer_target_calls(self):
        generator,ids,expected,_=self.generator()
        output,stats=generator.generate(ids,7,temperature=0)
        torch.testing.assert_close(output,expected)
        self.assertEqual(stats['accepted_tokens'],3)
        self.assertEqual(stats['verification_calls'],3)
        self.assertEqual(stats['replay_calls'],0)
        self.assertLess(stats['target_calls'],7)
        self.assertIsNone(generator.cache)

    def test_rejection_restores_hybrid_state_and_replays_target_correction(self):
        for mode in ('bf16','q8','q4'):
            with self.subTest(mode=mode):
                generator,ids,expected,_=self.generator(mode=mode,reject=True,budget=6)
                output,stats=generator.generate(ids,6,temperature=0)
                torch.testing.assert_close(output,expected)
                self.assertEqual(stats['accepted_tokens'],0)
                self.assertGreater(stats['replay_calls'],0)

    def test_chunked_head_prefill_shifts_across_chunk_boundaries(self):
        generator,ids,_,head=self.generator(chunk=2,budget=3)
        generator.generate(ids,3,temperature=0)
        shifted=torch.cat([row[0] for row in head.calls[:3]],dim=1)
        self.assertEqual(shifted[0,:-1].tolist(),ids[0,1:].tolist())
        positions=torch.cat([row[1] for row in head.calls[:3]],dim=1)
        self.assertEqual(positions.tolist(),[list(range(ids.shape[1]))])

    def test_quantized_flush_boundary_uses_single_target_step(self):
        for mode in ('q8','q4'):
            with self.subTest(mode=mode):
                generator,ids,expected,_=self.generator(mode=mode,prompt=[3,5,8]*85,budget=5,chunk=128)
                output,stats=generator.generate(ids,5,temperature=0)
                torch.testing.assert_close(output,expected)
                self.assertEqual(stats['quantized_boundary_steps'],1)

    def test_one_token_budget_does_not_run_head_or_verification(self):
        generator,ids,expected,head=self.generator(budget=1)
        output,stats=generator.generate(ids,1,temperature=0)
        torch.testing.assert_close(output,expected)
        self.assertEqual(head.calls,[])
        self.assertEqual(stats['verification_calls'],0)

    def test_eos_stops_inside_verified_pair(self):
        generator,ids,expected,_=self.generator()
        generator.model.generation_config.eos_token_id=[int(expected[0,ids.shape[1]+1])]
        output,_=generator.generate(ids,7,temperature=0)
        self.assertEqual(output.shape[1]-ids.shape[1],2)

    def test_budget_is_never_exceeded_and_stream_contains_only_output(self):
        for budget in (2,3,4,5):
            generator,ids,expected,_=self.generator(budget=budget)
            stream=Stream()
            output,_=generator.generate(ids,budget,temperature=0,streamer=stream)
            torch.testing.assert_close(output,expected)
            self.assertEqual(torch.cat([v.flatten() for v in stream.values[1:]]).tolist(),expected[0,ids.shape[1]:].tolist())
            self.assertEqual(stream.ends,1)

    def test_incompatible_sampling_and_constraints_fail_before_request(self):
        generator,ids,_,head=self.generator()
        for kwargs in ({'temperature':.7},{'temperature':float('nan')},
                       {'temperature':0,'constraint':object()},
                       {'temperature':0,'prefix_cache_key':'npc','prefix_length':2}):
            with self.assertRaises(ValueError):generator.generate(ids,3,**kwargs)
        self.assertEqual(head.calls,[])
        self.assertIsNone(generator.cache)

    def test_cancelled_stream_clears_request_state(self):
        generator,ids,_,_=self.generator()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            generator.generate(ids,5,temperature=0,streamer=Stream(fail=True))
        self.assertIsNone(generator.cache)
        self.assertEqual(generator.prefix,[])


if __name__=='__main__':unittest.main()
