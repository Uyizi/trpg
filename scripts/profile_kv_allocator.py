"""Bounded standalone replay of growing eight-layer Q8 prefill allocations."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with (Path.home()/'.cache/qwen-debugger/gpu.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from models.quantized_kv import QuantizedAttentionLayer
        base=torch.empty(3200*2**20,dtype=torch.uint8,device='cuda')
        layers=[QuantizedAttentionLayer(8) for _ in range(8)]
        x=torch.randn(1,4,1024,256,device='cuda',dtype=torch.bfloat16)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('w') as out,torch.inference_mode():
            for step in range(1,257):
                for layer in layers:
                    keys,values=layer.update(x,x)
                    del keys,values
                if step%8==0:
                    torch.cuda.synchronize()
                    allocated=torch.cuda.memory_allocated()/2**20
                    reserved=torch.cuda.memory_reserved()/2**20
                    row=dict(tokens=step*1024,allocated_mib=round(allocated),reserved_mib=round(reserved))
                    print(json.dumps(row),file=out,flush=True)
                    print(json.dumps(row),flush=True)
                    if reserved>12500:
                        print(json.dumps(dict(stopped='reserved memory pressure')),file=out,flush=True)
                        break


if __name__=='__main__':main()
