"""Offline single-process decode profiling for the existing trained MoE bundle."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operators', action='store_true')
    parser.add_argument('--compare', action='store_true', help='Alternate reference/fused decode in one loaded model')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    lease_path = Path.home() / '.cache/qwen-debugger/gpu.lock'
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from torch.profiler import profile, ProfilerActivity, record_function
        from models.pretrained import PretrainedChatModel
        from models.qwen35_moe import expert_modules
        model = PretrainedChatModel(device='cuda', local_files_only=True)
        modules = expert_modules(model.model)
        messages = [{'role': 'system', 'content': '直接回答。'},
                    {'role': 'user', 'content': '从1开始逐个列出整数，直到100，中间用逗号隔开。'}]
        if args.compare:
            report = dict(kind='real_routed_lora_comparison', training=False, downloads=False,
                          runtime=model.description, runs=[])
            # Warm both modes so Triton compilation is not a claimed speed gain.
            for enabled in (False, True):
                for module in modules: module.fused_lora = enabled
                model.clear_cache()
                model.answer(messages, max_new_tokens=8, temperature=0)
            prompts = [messages,
                       [{'role': 'system', 'content': '用简短中文直接回答。'},
                        {'role': 'user', 'content': '用三句话解释KV缓存的作用。'}],
                       [{'role': 'system', 'content': '你是货舱保安韩松，现在身体健康。只报告你自己知道的情况。'},
                        {'role': 'user', 'content': '使用工作频道报告你的位置和身体状态。'}]]
            for repeat in range(2):
                for case, prompt in enumerate(prompts):
                    for enabled in ((False, True) if repeat == 0 else (True, False)):
                        for module in modules: module.fused_lora = enabled
                        model.clear_cache()
                        count = sum(m.fused_lora_projections for m in modules)
                        chunks = []
                        text = model.answer(prompt, max_new_tokens=64, temperature=0, on_text=chunks.append)
                        row = dict(repeat=repeat, case=case, fused=enabled, text=text,
                                   fused_projections=sum(m.fused_lora_projections for m in modules)-count,
                                   stats=dict(model.last_response_stats))
                        report['runs'].append(row)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                        print(json.dumps(row, ensure_ascii=False), flush=True)
            report['same_answers'] = all(
                pair[0]['text'] == pair[1]['text']
                for offset in range(0, len(report['runs']), 2)
                for pair in [report['runs'][offset:offset+2]])
            report['summary'] = {}
            for enabled in (False, True):
                rows = [r['stats'] for r in report['runs'] if r['fused'] == enabled]
                report['summary']['fused' if enabled else 'reference'] = dict(
                    output_tokens=sum(r['output_tokens'] for r in rows),
                    decode_seconds=round(sum(r['decode_seconds'] for r in rows), 3),
                    tokens_per_second=round(sum(r['output_tokens'] for r in rows)/sum(r['decode_seconds'] for r in rows), 3),
                    peak_torch_memory_mib=max(r['peak_torch_memory_mib'] for r in rows))
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps({'same_answers':report['same_answers'], 'summary':report['summary']}), flush=True)
            return
        model.answer(messages, max_new_tokens=24, temperature=0)
        model.clear_cache()
        hooks, contexts = [], {}
        def before(module, inputs):
            context = record_function('expert_mlp')
            context.__enter__()
            contexts[module] = context
        def after(module, inputs, output):
            contexts.pop(module).__exit__(None, None, None)
        if args.operators:
            for module in expert_modules(model.model):
                hooks += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=False, profile_memory=False) as prof:
                answer = model.answer(messages, max_new_tokens=16, temperature=0)
            events = prof.key_averages()
            def row(event):
                return dict(name=event.key, count=event.count,
                            self_cpu_us=round(event.self_cpu_time_total, 2),
                            cpu_us=round(event.cpu_time_total, 2),
                            self_device_us=round(event.self_device_time_total, 2),
                            device_us=round(event.device_time_total, 2))
            report = dict(kind='real_decode_operator_profile', downloads=False, training=False,
                          runtime=model.description, answer=answer, stats=model.last_response_stats,
                          top_cpu=[row(e) for e in sorted(events, key=lambda e:e.self_cpu_time_total, reverse=True)[:40]],
                          top_device=[row(e) for e in sorted(events, key=lambda e:e.self_device_time_total, reverse=True)[:40]],
                          expert_mlp=[row(e) for e in events if e.key == 'expert_mlp'])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps({'stats':model.last_response_stats, 'expert_mlp':report['expert_mlp'],
                              'top_cpu':report['top_cpu'][:8], 'top_device':report['top_device'][:8]}, ensure_ascii=False), flush=True)
        finally:
            for hook in hooks:
                hook.remove()


if __name__ == '__main__':
    main()
