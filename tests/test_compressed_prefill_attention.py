"""Packed prefill numerical, causal-boundary, and public-fallback checks."""
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from torch.nn.attention.bias import causal_lower_right
from models.quantized_kv import QuantizedAttentionLayer
from models.compressed_prefill_attention import compressed_kv_prefill


class PrefillFallbackTests(unittest.TestCase):
    def test_cpu_never_imports_or_launches_the_kernel(self):
        layer = QuantizedAttentionLayer(4)
        x = torch.randn(1, 1, 256, 64)
        layer.update(x, x)
        view = layer.append_for_attention(x[:, :, :16], x[:, :, :16])
        with torch.inference_mode(), patch.dict('sys.modules', {'models.compressed_prefill_kernels': None}):
            self.assertIsNone(compressed_kv_prefill(x[:, :, :16], view, .125))


class PackedPrefillCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA required')

    def test_q4_q8_exact_residual_causal_and_noncontiguous_queries(self):
        torch.manual_seed(104)
        # Includes incomplete query tiles and residual/packed transition points.
        cases = [(4, 257, 16, 64, torch.bfloat16),
                 (8, 385, 17, 128, torch.float16),
                 (4, 1025, 63, 256, torch.bfloat16),
                 (8, 4096, 129, 256, torch.bfloat16)]
        for bits, history, qlen, dimension, dtype in cases:
            with self.subTest(bits=bits, qlen=qlen, dimension=dimension), torch.inference_mode():
                layer = QuantizedAttentionLayer(bits)
                key = torch.randn(1, 2, history, dimension, device='cuda', dtype=dtype)
                value = torch.randn_like(key)
                layer.update(key, value)
                next_k = torch.randn(1, 2, qlen, dimension, device='cuda', dtype=dtype)
                next_v = torch.randn_like(next_k)
                view = layer.append_for_attention(next_k, next_v)
                q = torch.randn(1, qlen, 6, dimension, device='cuda', dtype=dtype).transpose(1, 2)
                keys = torch.cat((layer.key_bank.unpack(dtype), view.keys), -2)
                values = torch.cat((layer.value_bank.unpack(dtype), view.values), -2)
                expected = F.scaled_dot_product_attention(q, keys, values,
                    attn_mask=causal_lower_right(qlen, view.length), enable_gqa=True)
                actual = compressed_kv_prefill(q, view, dimension**-.5)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, expected, rtol=.03, atol=.004)
                # Current attention must preserve the exact residual until caller compacts.
                self.assertEqual(layer.keys.shape[2], view.keys.shape[2])
                self.assertEqual(layer.key_bank.units * 128, view.packed_tokens)

    def test_large_logits_and_unsupported_inputs(self):
        with torch.inference_mode():
            layer = QuantizedAttentionLayer(8)
            x = torch.randn(1, 1, 256, 64, device='cuda', dtype=torch.bfloat16)
            layer.update(x, x)
            view = layer.append_for_attention(x[:, :, :32], x[:, :, :32])
            q = x[:, :, :32] * 40
            keys = torch.cat((layer.key_bank.unpack(q.dtype), view.keys), -2)
            values = torch.cat((layer.value_bank.unpack(q.dtype), view.values), -2)
            actual = compressed_kv_prefill(q, view, .125)
            expected = F.scaled_dot_product_attention(q, keys, values,
                attn_mask=causal_lower_right(32, view.length), enable_gqa=True)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, rtol=.03, atol=.004)
            for bad_q in (q[:, :, :15], q.float(), q.expand(2, -1, -1, -1)):
                self.assertIsNone(compressed_kv_prefill(bad_q, view, .125))
            self.assertIsNone(compressed_kv_prefill(q, view._replace(packed_tokens=0), .125))
            self.assertIsNone(compressed_kv_prefill(q, view, float('nan')))
            self.assertIsNone(compressed_kv_prefill(q, view._replace(keys=view.keys[:, :, :16]), .125))
        with torch.enable_grad():
            self.assertIsNone(compressed_kv_prefill(q, view, .125))

    def test_growing_lengths_and_slab_strides_reuse_compilation(self):
        import triton
        compiled = []
        old_hook = triton.knobs.runtime.jit_cache_hook
        def hook(**kwargs):
            fn = kwargs['fn']
            if fn.module == 'models.compressed_prefill_kernels' and fn.name == '_prefill':
                compiled.append(kwargs['key'])
            if old_hook is not None:
                return old_hook(**kwargs)
        triton.knobs.runtime.jit_cache_hook = hook
        try:
            with torch.inference_mode():
                for history in (4096, 4224, 8320):
                    layer = QuantizedAttentionLayer(8)
                    x = torch.randn(1, 2, history, 64, device='cuda', dtype=torch.bfloat16)
                    layer.update(x, x)
                    view = layer.append_for_attention(x[:, :, :33], x[:, :, :33])
                    q = torch.randn(1, 4, 33, 64, device='cuda', dtype=torch.bfloat16)
                    self.assertIsNotNone(compressed_kv_prefill(q, view, .125))
                torch.cuda.synchronize()
            self.assertLessEqual(len(compiled), 1)
        finally:
            triton.knobs.runtime.jit_cache_hook = old_hook


if __name__ == '__main__':
    unittest.main()
