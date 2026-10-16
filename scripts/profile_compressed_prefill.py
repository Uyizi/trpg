"""Offline packed prefill versus native unpack + SDPA on exactly the same view."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories', nargs='+', type=int, default=[8192, 32768, 65536])
    parser.add_argument('--queries', nargs='+', type=int, default=[1024, 4096])
    parser.add_argument('--bits', nargs='+', type=int, default=[4, 8])
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = Path.home()/'.cache/qwen-debugger/gpu.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from torch.nn import functional as F
        from torch.nn.attention.bias import causal_lower_right
        from models.quantized_kv import QuantizedAttentionLayer
        from models.compressed_prefill_attention import compressed_kv_prefill
        torch.manual_seed(104)
        rows = []
        with torch.inference_mode():
            for history in args.histories:
                for qlen in args.queries:
                    for bits in args.bits:
                        layer = QuantizedAttentionLayer(bits)
                        k = torch.randn(1, 4, history, 256, device='cuda', dtype=torch.bfloat16)
                        v = torch.randn_like(k)
                        layer.update(k, v)
                        next_k = torch.randn(1, 4, qlen, 256, device='cuda', dtype=torch.bfloat16)
                        next_v = torch.randn_like(next_k)
                        view = layer.append_for_attention(next_k, next_v)
                        del k, v, next_k, next_v
                        q = torch.randn(1, qlen, 16, 256, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
                        bias = causal_lower_right(qlen, view.length)
                        def reference():
                            keys = torch.cat((layer.key_bank.unpack(q.dtype), view.keys), -2)
                            values = torch.cat((layer.value_bank.unpack(q.dtype), view.values), -2)
                            return F.scaled_dot_product_attention(q, keys, values, attn_mask=bias,
                                                                  enable_gqa=True, scale=.0625)
                        def packed():
                            output = compressed_kv_prefill(q, view, .0625)
                            assert output is not None
                            return output
                        expected, actual = reference(), packed()
                        torch.testing.assert_close(actual, expected, rtol=.03, atol=.004)
                        row = dict(history=history, queries=qlen, length=view.length, bits=bits,
                                   max_abs_error=(actual.float()-expected.float()).abs().max().item())
                        del actual, expected
                        for name, call in (('unpack_sdpa', reference), ('packed', packed)):
                            for _ in range(2): call()
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            baseline = torch.cuda.memory_allocated()
                            samples = []
                            for _ in range(args.samples):
                                torch.cuda.synchronize()
                                start = time.perf_counter()
                                out = call()
                                torch.cuda.synchronize()
                                samples.append((time.perf_counter()-start)*1000)
                                del out
                            row[name] = dict(median_ms=round(statistics.median(samples), 4),
                                extra_peak_mib=round((torch.cuda.max_memory_allocated()-baseline)/2**20, 2))
                        row['speedup'] = round(row['unpack_sdpa']['median_ms']/row['packed']['median_ms'], 3)
                        rows.append(row)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(rows, indent=2)+'\n')
                        print(json.dumps(row), flush=True)
                        del layer, view, q
                        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
