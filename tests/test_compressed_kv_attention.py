"""Native packed cache decode and exact-current-residual boundary coverage."""
import copy
import importlib
import importlib.util
import unittest

import torch
from torch.nn import functional as F
from models.quantized_kv import QuantizedAttentionLayer


class QuantizedViewTests(unittest.TestCase):
    def test_current_attention_keeps_unpacked_residual_until_compaction(self):
        layer = QuantizedAttentionLayer(4)
        keys = torch.randn(1, 2, 256, 64)
        values = torch.randn_like(keys)
        layer.update(keys, values)
        self.assertTrue(hasattr(layer, 'append_for_attention'), 'A non-materializing cache update is required')
        new_keys, new_values = keys[..., :128, :].clone(), values[..., :128, :].clone()
        view = layer.append_for_attention(new_keys, new_values)
        self.assertEqual(view.packed_tokens, 128)
        self.assertEqual(view.length, 384)
        self.assertEqual(layer.key_bank.units, 1)
        self.assertTrue(torch.equal(view.keys, torch.cat((keys[..., 128:, :], new_keys), -2)))
        layer.compact_residual()
        self.assertEqual(layer.key_bank.units, 2)
        self.assertEqual(layer.keys.shape[-2], 128)
        self.assertEqual(view.keys.shape[-2], 256, 'Current attention view must survive later compaction')
        self.assertEqual(view.packed_tokens, 128)

    def test_cpu_and_arbitrary_masks_retain_reference_attention(self):
        self.assertIsNotNone(importlib.util.find_spec('models.compressed_kv_attention'))
        from models.compressed_kv_attention import compressed_kv_decode
        layer = QuantizedAttentionLayer(8)
        keys = torch.randn(1, 2, 256, 64)
        layer.update(keys, keys)
        view = layer.append_for_attention(keys[..., :1, :], keys[..., :1, :])
        query = torch.randn(1, 4, 1, 64)
        with torch.inference_mode():
            self.assertIsNone(compressed_kv_decode(query, view, .125))

    def test_only_the_last_query_causal_mask_is_eligible(self):
        from models.compressed_kv_attention import _supported_mask
        from torch.nn.attention.bias import causal_lower_right, causal_upper_left
        self.assertTrue(_supported_mask(None, 512))
        self.assertTrue(_supported_mask(causal_lower_right(1,512), 512))
        self.assertFalse(_supported_mask(causal_upper_left(1,512), 512))
        self.assertFalse(_supported_mask(causal_lower_right(2,512), 512))
        self.assertFalse(_supported_mask(causal_lower_right(1,511), 512))
        self.assertFalse(_supported_mask(torch.zeros(1,1,1,512), 512))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CompressedKVTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('models.compressed_kv_attention'))
        self.impl = importlib.import_module('models.compressed_kv_attention')

    def test_q4_q8_gqa_residual_and_split_boundaries_match_dequantized_sdpa(self):
        torch.manual_seed(31)
        for bits, length, dimension in ((4,256,64), (8,384,128), (4,4097,256), (8,8192,256)):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(bits=bits, length=length, dtype=dtype), torch.inference_mode():
                    layer = QuantizedAttentionLayer(bits)
                    key = torch.randn(1, 2, length, dimension, device='cuda', dtype=dtype)
                    value = torch.randn_like(key)
                    layer.update(key, value)
                    reference = copy.deepcopy(layer)
                    next_k, next_v = key[..., :1, :].contiguous(), value[..., :1, :].contiguous()
                    k, v = reference.update(next_k, next_v)
                    view = layer.append_for_attention(next_k, next_v)
                    q = torch.randn(1, 6, 1, dimension, device='cuda', dtype=dtype)
                    expected = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
                    actual = self.impl.compressed_kv_decode(q, view, dimension**-.5)
                    self.assertIsNotNone(actual)
                    torch.testing.assert_close(actual, expected, rtol=.025, atol=.004)
                    layer.compact_residual()
                    self.assertEqual(layer.get_seq_length(), length+1)

    def test_large_scores_remain_finite_and_unsupported_inputs_fall_back(self):
        with torch.inference_mode():
            layer = QuantizedAttentionLayer(4)
            keys = torch.randn(1, 2, 512, 64, device='cuda', dtype=torch.bfloat16)
            layer.update(keys, keys)
            ref = copy.deepcopy(layer)
            k, v = ref.update(keys[..., :1, :], keys[..., :1, :])
            view = layer.append_for_attention(keys[..., :1, :], keys[..., :1, :])
            q = keys[..., :1, :].repeat_interleave(2, dim=1) * 80
            actual = self.impl.compressed_kv_decode(q, view, .125)
            expected = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, rtol=.025, atol=.004)
            self.assertIsNone(self.impl.compressed_kv_decode(q.expand(2, -1, -1, -1), view, .125))
            self.assertIsNone(self.impl.compressed_kv_decode(q.expand(-1, -1, 2, -1), view, .125))
            self.assertIsNone(self.impl.compressed_kv_decode(q.float(), view, .125))
        with torch.enable_grad():
            self.assertIsNone(self.impl.compressed_kv_decode(q, view, .125))

    def test_successive_decode_positions_reuse_the_compiled_kernel(self):
        import triton
        compiles = []
        old = triton.knobs.runtime.jit_cache_hook
        def on_compile(**kwargs):
            fn = kwargs['fn']
            if fn.module == 'models.compressed_kv_kernels' and fn.name == '_partial':
                compiles.append(kwargs['key'])
            if old is not None:
                return old(**kwargs)
        triton.knobs.runtime.jit_cache_hook = on_compile
        try:
            with torch.inference_mode():
                layer = QuantizedAttentionLayer(8)
                x = torch.randn(1,1,1025,64,device='cuda',dtype=torch.bfloat16)
                layer.update(x,x)
                q = torch.randn(1,4,1,64,device='cuda',dtype=torch.bfloat16)
                for _ in range(5):
                    view = layer.append_for_attention(x[:,:,:1],x[:,:,:1])
                    self.assertIsNotNone(self.impl.compressed_kv_decode(q,view,.125))
                    layer.compact_residual()
            self.assertLessEqual(len(compiles), 1, 'Growing logical length must not compile a binary for each token')
        finally:
            triton.knobs.runtime.jit_cache_hook = old

    def test_cuda_affine_fma_rounding_matches_original_unpack(self):
        # Valid Q8 affine metadata where fused vs separate multiply-add lands
        # on opposite BF16 rounding sides: 2.109375 versus 2.125.
        for bank_name in ('value','key'):
            with self.subTest(bank=bank_name), torch.inference_mode():
                layer = QuantizedAttentionLayer(8)
                zero = torch.zeros(1,1,256,64,device='cuda',dtype=torch.bfloat16)
                layer.update(zero,zero)
                bank = getattr(layer,bank_name+'_bank')
                if bank_name == 'value':
                    bank.data[:,:,:128].fill_(204)
                    bank.data[:,:,:128,:,-2] = 0
                    bank.data[:,:,:128,:,-1] = 255
                    bank.meta[:,:,:128,:,0] = .020067401230335236
                    bank.meta[:,:,:128,:,1] = -1.9765625
                else:
                    bank.data[:,:,0,0].fill_(204)
                    bank.data[:,:,0,0,0] = 0
                    bank.data[:,:,0,0,1] = 255
                    bank.meta[:,:,0,0,0] = .020067401230335236
                    bank.meta[:,:,0,0,1] = -1.9765625
                    layer.keys[...,0] = 2.125
                    layer.value_bank.data[:,:,:128].zero_()
                    layer.value_bank.meta[:,:,:128,:,1] = 1.
                next_k = layer.keys[:,:,:1].clone()
                next_v = layer.values[:,:,:1].clone()
                view = layer.append_for_attention(next_k,next_v)
                keys = torch.cat((layer.key_bank.unpack(zero.dtype),view.keys),-2)
                values = torch.cat((layer.value_bank.unpack(zero.dtype),view.values),-2)
                q = torch.zeros(1,1,1,64,device='cuda',dtype=torch.bfloat16)
                if bank_name == 'key': q[...,0]=16
                expected = F.scaled_dot_product_attention(q,keys,values)
                actual = self.impl.compressed_kv_decode(q,view,.125)
                torch.testing.assert_close(actual[...,:1],expected[...,:1],rtol=0,atol=.002)

    def test_real_qwen_attention_preserves_cache_and_falls_back_for_custom_masks(self):
        from transformers import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
        from torch.nn.attention.bias import causal_lower_right
        from models.quantized_kv import make_cache
        config = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=64, layer_types=['full_attention'])
        config._attn_implementation = 'sdpa'
        module = Qwen3_5Attention(config, 0).cuda().bfloat16().eval().requires_grad_(False)
        self.impl.configure_compressed_kv_attention(module, False)
        cache = make_cache(config, 'q4')
        x = torch.randn(1,384,128,device='cuda',dtype=torch.bfloat16)
        pos = (torch.ones(1,384,64,device='cuda',dtype=torch.bfloat16),
               torch.zeros(1,384,64,device='cuda',dtype=torch.bfloat16))
        with torch.inference_mode():
            module(x, pos, causal_lower_right(384,384), past_key_values=cache)
            ref, direct = copy.deepcopy(cache), copy.deepcopy(cache)
            hidden = x[:,:1].contiguous()
            positions = tuple(t[:,:1].contiguous() for t in pos)
            expected, _ = module(hidden, positions, causal_lower_right(1,385), past_key_values=ref)
            self.impl.configure_compressed_kv_attention(module, True)
            actual, weights = module(hidden, positions, causal_lower_right(1,385), past_key_values=direct)
            self.assertIsNone(weights)
            torch.testing.assert_close(actual, expected, rtol=.03, atol=.004)
            self.assertEqual(self.impl.compressed_kv_info(module)['direct_decode_calls'], 1)

            self.assertEqual(direct.get_seq_length(), ref.get_seq_length())
            self.assertTrue(torch.equal(direct.layers[0].keys, ref.layers[0].keys))
            self.assertTrue(torch.equal(direct.layers[0].key_bank.data, ref.layers[0].key_bank.data))
            # A masked history must use upstream, not the all-visible last-query kernel.
            custom = torch.zeros(1,1,1,386,device='cuda',dtype=torch.bfloat16)
            custom[...,10:200] = -float('inf')
            self.impl.configure_compressed_kv_attention(module, False)
            expected, _ = module(hidden, positions, custom, past_key_values=ref)
            self.impl.configure_compressed_kv_attention(module, True)
            actual, _ = module(hidden, positions, custom, past_key_values=direct)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(self.impl.compressed_kv_info(module)['direct_decode_calls'], 1)
            bias = torch.zeros(1,1,1,387,device='cuda',dtype=torch.bfloat16)
            bias[...,100:250] = -30
            self.impl.configure_compressed_kv_attention(module, False)
            expected, _ = module(hidden, positions, None, past_key_values=ref,position_bias=bias)
            self.impl.configure_compressed_kv_attention(module, True)
            actual, _ = module(hidden, positions, None, past_key_values=direct,position_bias=bias)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(self.impl.compressed_kv_info(module)['direct_decode_calls'], 1)

    def test_actual_cached_generator_dispatches_direct_decode_through_decoder(self):
        from transformers import Qwen3_5Config
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        from models.cached_generation import CachedGenerator
        config = Qwen3_5Config(text_config=dict(
            vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            layer_types=['full_attention'], max_position_embeddings=1024,
            rope_parameters={'rope_type':'default','rope_theta':10000.,
                             'partial_rotary_factor':1.,'mrope_section':[16,8,8]}))
        config._attn_implementation = config.text_config._attn_implementation = 'sdpa'
        model = TextOnlyQwen3_5ForCausalLM(config).cuda().bfloat16().eval().requires_grad_(False)
        model.generation_config.eos_token_id = []
        ids = (torch.arange(384,device='cuda') % 64).unsqueeze(0)
        generator = CachedGenerator(model,kv_mode='q8',chunk_size=128)
        with torch.inference_mode():
            self.impl.configure_compressed_kv_attention(model,False)
            expected,_ = generator.generate(ids,4,temperature=0)
            generator.clear()
            self.impl.configure_compressed_kv_attention(model,True)
            actual,_ = generator.generate(ids,4,temperature=0)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(self.impl.compressed_kv_info(model)['direct_decode_calls'],3,
                         'Real decoder position_ids must not disable packed-cache decode')


if __name__ == '__main__':
    unittest.main()
