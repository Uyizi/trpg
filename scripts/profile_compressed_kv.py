"""Offline native-cache decode comparison; no model loading or downloads."""
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
    parser.add_argument('--lengths', nargs='+', type=int, default=[8192,32768,65536,262144])
    parser.add_argument('--bits', nargs='+', type=int, default=[4,8])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = Path.home()/'.cache/qwen-debugger/gpu.lock'
    with lock.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from torch.nn import functional as F
        from models.quantized_kv import QuantizedAttentionLayer
        from models.compressed_kv_attention import compressed_kv_decode
        torch.manual_seed(41)
        results = []
        with torch.inference_mode():
            for target_length in args.lengths:
                length = target_length-1
                for bits in args.bits:
                    layer = QuantizedAttentionLayer(bits)
                    k = torch.randn(1,4,length,256,device='cuda',dtype=torch.bfloat16)
                    v = torch.randn_like(k)
                    layer.update(k,v)
                    next_k, next_v = k[:,:,:1].clone(), v[:,:,:1].clone()
                    view = layer.append_for_attention(next_k,next_v)
                    del k,v
                    q = torch.randn(1,16,1,256,device='cuda',dtype=torch.bfloat16)
                    def reference():
                        keys = torch.cat((layer.key_bank.unpack(q.dtype),view.keys),-2)
                        values = torch.cat((layer.value_bank.unpack(q.dtype),view.values),-2)
                        return F.scaled_dot_product_attention(q,keys,values,enable_gqa=True)
                    def fused():
                        output = compressed_kv_decode(q,view,.0625)
                        assert output is not None
                        return output
                    expected, actual = reference(), fused()
                    torch.testing.assert_close(actual,expected,rtol=.03,atol=.004)
                    measured = {}
                    for name, call in (('reference',reference),('direct',fused)):
                        call(); torch.cuda.synchronize()
                        samples = []
                        torch.cuda.reset_peak_memory_stats()
                        start_memory = torch.cuda.memory_allocated()
                        for _ in range(7):
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            result = call()
                            torch.cuda.synchronize()
                            samples.append((time.perf_counter()-start)*1000)
                            del result
                        measured[name] = dict(median_ms=round(statistics.median(samples),4),
                            extra_peak_mib=round((torch.cuda.max_memory_allocated()-start_memory)/2**20,2))
                    row = dict(length=length+1,bits=bits,cache_mib=round(layer.bytes()/2**20,2),
                               max_abs_error=round((actual.float()-expected.float()).abs().max().item(),7),**measured)
                    results.append(row)
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    args.output.write_text(json.dumps(results,indent=2)+'\n')
                    print(json.dumps(row),flush=True)
                    del layer,view,expected,actual,q,next_k,next_v
                    torch.cuda.empty_cache()


if __name__ == '__main__': main()
