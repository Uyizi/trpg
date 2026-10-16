"""Exclusive offline NF4/PEFT MLP probe, using the installed model's dimensions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=300)
    args = parser.parse_args()
    lock = Path.home() / '.cache/qwen-debugger/gpu.lock'
    with lock.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['HF_HUB_OFFLINE'] = '1'
        from models.inference_environment import configure_cuda_allocator
        configure_cuda_allocator()
        import bitsandbytes as bnb
        import torch
        from torch import nn
        from torch.nn import functional as F
        from peft import get_peft_model, LoraConfig
        from models.cuda_graph_decode import configure_cuda_graph_decode, cuda_graph_decode_info
        from models.qwen35_moe import Top1Experts
        from models.runtime_paths import DEFAULT_BUNDLE, read_bundle
        manifest = read_bundle(DEFAULT_BUNDLE)
        config = json.loads((Path(manifest['base']) / 'config.json').read_text())['text_config']
        adapter = json.loads((Path(manifest['moe_adapter']) / 'adapter_config.json').read_text())
        hidden, middle, rank = config['hidden_size'], config['intermediate_size'], adapter['r']
        torch.manual_seed(718)
        rows = []
        for dtype in (torch.bfloat16, torch.float16):
            class Dense(nn.Module):
                def __init__(self):
                    super().__init__()
                    for name, left, right in (('gate_proj', hidden, middle), ('up_proj', hidden, middle), ('down_proj', middle, hidden)):
                        setattr(self, name, bnb.nn.Linear4bit(left, right, bias=False, compute_dtype=dtype,
                                                            quant_type='nf4', compress_statistics=True))
                    self.act_fn = F.silu

            expert = Top1Experts(Dense().to('cuda'), share_frozen=True)
            expert.is_loaded_in_4bit = True
            model = get_peft_model(expert, LoraConfig(r=rank, lora_alpha=adapter['lora_alpha'],
                                                     lora_dropout=.05, modules_to_save=['router'],
                                                     target_modules=['gate_proj', 'up_proj', 'down_proj']))
            for name, parameter in model.named_parameters():
                if 'lora_' in name:
                    nn.init.normal_(parameter, std=.005)
            model.eval().requires_grad_(False)
            expert.fused_lora = True
            reference = expert.forward
            x = torch.randn(1, 1, hidden, device='cuda', dtype=dtype)
            with torch.inference_mode():
                for _ in range(5):
                    reference(x)
                configure_cuda_graph_decode(model)
                expected = reference(x)
                actual = model(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                info = cuda_graph_decode_info(model)
                if info['captured_layers'] != 1:
                    raise RuntimeError(info)
                layer = model._cuda_graph_decode.layers[0]
                def raw():
                    layer.input.copy_(x)
                    layer.graph.replay()
                    return layer.output.clone()
                measurements = {'eager_fused': [], 'graph_guarded': [], 'graph_raw_diagnostic': []}
                operations = {'eager_fused': lambda: reference(x), 'graph_guarded': lambda: model(x),
                              'graph_raw_diagnostic': raw}
                for repeat in range(5):
                    order = tuple(operations) if repeat % 2 == 0 else tuple(reversed(operations))
                    for name in order:
                        operation = operations[name]
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        for _ in range(args.iterations):
                            operation()
                        torch.cuda.synchronize()
                        measurements[name].append((time.perf_counter() - started) * 1e6 / args.iterations)
                row = dict(dtype=str(dtype), hidden=hidden, intermediate=middle, rank=rank,
                           exact_parity=True, microseconds={k:statistics.median(v) for k,v in measurements.items()},
                           samples=measurements, graph=cuda_graph_decode_info(model))
                row['guarded_speedup'] = row['microseconds']['eager_fused'] / row['microseconds']['graph_guarded']
                rows.append(row)
                print(json.dumps(row), flush=True)
            configure_cuda_graph_decode(model, False)
            del model, expert, reference, layer, x, actual, expected
            torch.cuda.empty_cache()
        report = dict(scope='synthetic frozen NF4 MLP; actual local model dimensions/adapters, not full-model speed',
                      torch=torch.__version__, gpu=torch.cuda.get_device_name(), rows=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
