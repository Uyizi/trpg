#!/usr/bin/env python3
"""Opt-in isolated native SM120 FP4 projection benchmark; never a model loader."""
import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='Explicitly allow CUDA context and GEMM')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weight-dir', type=Path)
    p.add_argument('--weight-key', default='model.language_model.layers.0.mlp.gate_proj.weight')
    p.add_argument('--rows', default='1,16,128,512')
    p.add_argument('--warmup', type=int, default=12)
    p.add_argument('--repeats', type=int, default=40)
    p.add_argument('--api', choices=['public', 'legacy'], default='public')
    p.add_argument('--triton-packer', action='store_true', help='Also measure verified three-kernel online packing')
    args = p.parse_args()
    rows = [int(v) for v in args.rows.split(',')]
    if min(rows) < 1 or args.repeats < 1 or args.warmup < 1:
        p.error('rows, warmup and repeats must be positive')
    if not args.execute:
        p.error('--execute is required; run only during an exclusive GPU lease')
    lease_path=Path.home()/'.cache/qwen-debugger/gpu.lock'
    lease_path.parent.mkdir(parents=True,exist_ok=True)
    with lease_path.open('a+') as lease:
        try:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            report={'status':'blocked','blocker':'Shared GPU lease is busy; no CUDA was queried',
                    'scope':'standalone_single_projection','training':False,'model_integrated':False,
                    'cuda_queried':False,'native_fp4_executed':False}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report))
            return 2
        return run(args)


def run(args):
    import torch
    import torch.nn.functional as F
    from models.nvfp4_probe import quantize_nvfp4, native_scaled_mm, error_summary, dequantize_nvfp4
    report = {'scope': 'standalone_single_projection', 'training': False, 'model_integrated': False,
              'native_fp4_executed': False, 'api': args.api, 'seed': 20261015,
              'activation_source': 'synthetic standard-normal BF16 for every row count',
              'packing': 'E2M1 RNE low nibble first; E4M3 block16; FP32 global; SWIZZLE_32_4_4',
              'timing': 'CUDA events, median/p10/p90; eager launches; no CUDA graph; warmed; excludes weight packing',
              'warmup': args.warmup, 'repeats': args.repeats, 'cases': [], 'packages': {}}
    for name in ('torch','triton','bitsandbytes','flashinfer-python','torchao'):
        try: report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['packages'][name] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def bench(fn):
        for _ in range(args.warmup): fn()
        torch.cuda.synchronize()
        samples=[]
        for _ in range(args.repeats):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize(); samples.append(a.elapsed_time(b))
        ordered=sorted(samples)
        return {'median_ms': statistics.median(samples), 'p10_ms': ordered[int((len(ordered)-1)*.1)],
                'p90_ms': ordered[int((len(ordered)-1)*.9)], 'samples': len(samples)}

    try:
        torch.set_grad_enabled(False)
        torch.manual_seed(report['seed'])
        device=torch.device('cuda:0')
        props=torch.cuda.get_device_properties(device)
        report['gpu']={'name':props.name,'capability':list(torch.cuda.get_device_capability(device)),
                       'total_memory_bytes':props.total_memory,'cuda':torch.version.cuda}
        if tuple(report['gpu']['capability']) != (12,0):
            raise RuntimeError('This experiment requires the expected SM120 device')
        if args.weight_dir:
            from safetensors import safe_open
            index=json.loads((args.weight_dir/'model.safetensors.index.json').read_text())
            shard=args.weight_dir/index['weight_map'][args.weight_key]
            with safe_open(shard, framework='pt', device='cpu') as handle:
                weight=handle.get_tensor(args.weight_key)
            report['weight']={'source':'local dense BF16 tensor', 'directory':str(args.weight_dir),
                              'key':args.weight_key,'shard':shard.name,'dtype':str(weight.dtype)}
            if weight.dtype != torch.bfloat16:
                raise ValueError('Expected original BF16 tensor; no NF4 reinterpretation allowed')
        else:
            weight=torch.randn(9216,2560,dtype=torch.bfloat16)*.02
            report['weight']={'source':'synthetic BF16 matrix','dtype':str(weight.dtype)}
        if list(weight.shape) != [9216,2560]:
            raise ValueError('Expected this target projection shape [9216,2560]')
        weight=weight.to(device).contiguous()
        report['weight']['shape']=list(weight.shape)
        start=time.perf_counter()
        qw=quantize_nvfp4(weight)
        torch.cuda.synchronize()
        report['weight']['packing_seconds']=time.perf_counter()-start
        report['weight']['reconstruction_error']=error_summary(weight,dequantize_nvfp4(qw))
        report['weight']['bf16_bytes']=weight.numel()*weight.element_size()
        report['weight']['nvfp4_bytes']=qw.packed.numel()+qw.scales.numel()+4
        import bitsandbytes as bnb
        nf4_data,nf4_state=bnb.functional.quantize_4bit(weight,blocksize=64,compress_statistics=True,quant_type='nf4')
        nf4_fn=lambda x: bnb.matmul_4bit(x,nf4_data.t(),quant_state=nf4_state)
        for m in rows:
            x=torch.randn(m,2560,device=device,dtype=torch.bfloat16)
            qa=quantize_nvfp4(x)
            expected=F.linear(x,weight)
            result=native_scaled_mm(qa,qw,api=args.api)
            torch.cuda.synchronize()
            err=error_summary(expected,result)
            # Validate against the exactly decoded quantized inputs as well;
            # this separates FP4 quantization error from native layout mistakes.
            quantized_ref=F.linear(dequantize_nvfp4(qa),dequantize_nvfp4(qw))
            layout_error=error_summary(quantized_ref,result)
            if not err['finite'] or layout_error['relative_l2'] > .015:
                raise RuntimeError('Native result failed decoded-input accuracy/layout gate: '+str(layout_error))
            nf4_result=nf4_fn(x)
            if args.triton_packer:
                from models.nvfp4_quant_kernels import quantize_nvfp4_triton
                qt = quantize_nvfp4_triton(x)
                if not (torch.equal(qt.packed, qa.packed)
                        and torch.equal(qt.scales.view(torch.uint8), qa.scales.view(torch.uint8))
                        and torch.equal(qt.global_scale, qa.global_scale)):
                    raise RuntimeError('Triton packer differs from eager reference')
            record={'rows':m,'nvfp4_vs_bf16':err,'nvfp4_vs_decoded_fp32':layout_error,
                    'nf4_vs_bf16':error_summary(expected,nf4_result),
                    'bf16':bench(lambda:F.linear(x,weight)),
                    'nf4_weight_only':bench(lambda:nf4_fn(x)),
                    'nvfp4_prepacked_gemm':bench(lambda:native_scaled_mm(qa,qw,api=args.api)),
                    'nvfp4_dynamic_eager_pack_and_gemm':bench(lambda:native_scaled_mm(quantize_nvfp4(x,check_finite=False),qw,api=args.api))}
            if args.triton_packer:
                record['triton_pack_byte_equal_reference'] = True
                record['triton_pack_only'] = bench(lambda:quantize_nvfp4_triton(x))
                record['nvfp4_dynamic_triton_pack_and_gemm'] = bench(lambda:native_scaled_mm(quantize_nvfp4_triton(x),qw,api=args.api))
                record['triton_dynamic_speedup_vs_bf16'] = record['bf16']['median_ms']/record['nvfp4_dynamic_triton_pack_and_gemm']['median_ms']
            record['prepacked_speedup_vs_bf16']=record['bf16']['median_ms']/record['nvfp4_prepacked_gemm']['median_ms']
            record['dynamic_speedup_vs_bf16']=record['bf16']['median_ms']/record['nvfp4_dynamic_eager_pack_and_gemm']['median_ms']
            report['cases'].append(record)
            report['native_fp4_executed']=True
            persist()
            print(json.dumps({'rows':m,'weight_source':report['weight']['source'],'result':record}),flush=True)
        report['status']='ok'
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        persist()
        raise
    persist()
    print(str(args.output))


if __name__ == '__main__':
    raise SystemExit(main())
