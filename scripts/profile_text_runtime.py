"""Profile existing local text weights; never download or train a model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    lock = Path.home()/'.cache/qwen-debugger/gpu.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from models.pretrained import PretrainedChatModel
        from models.qwen35_moe import expert_modules

        start = time.monotonic()
        model = PretrainedChatModel(device='cuda', local_files_only=True, context_length=65536)
        torch.cuda.synchronize()
        report = {'kind': 'real_text_runtime_profile', 'training': False, 'downloads': False,
                  'cold_load_seconds': round(time.monotonic()-start, 3),
                  'gpu': torch.cuda.get_device_name(), 'runtime': model.description,
                  'resident_torch_mib': round(torch.cuda.memory_allocated()/2**20, 2),
                  'visual_parameters': [name for name, _ in model.model.named_parameters() if '.visual.' in name],
                  'expert_layers': len(expert_modules(model.model)), 'responses': []}
        assert not report['visual_parameters']
        assert report['expert_layers'] == 32
        print(json.dumps({'loaded': True, 'seconds': report['cold_load_seconds'],
                          'resident_torch_mib': report['resident_torch_mib']}, ensure_ascii=False), flush=True)
        messages = [{'role': 'system', 'content': '用简短中文直接回答问题。'},
                    {'role': 'user', 'content': '用三句话解释KV缓存的作用。'}]
        for label in ('first_shape', 'warm_repeat', 'continuation'):
            if label == 'warm_repeat':
                model.clear_cache()
            if label == 'continuation':
                messages += [{'role': 'assistant', 'content': text},
                             {'role': 'user', 'content': '再用一句话说明它为什么能加快逐字生成。'}]
            emitted = []
            text = model.answer(messages, max_new_tokens=128, temperature=0, on_text=emitted.append)
            assert text
            report['responses'].append({'case': label, 'text': text,
                                        'stream_chunks': len(emitted),
                                        'truncated': model.last_response_truncated,
                                        'stats': dict(model.last_response_stats)})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(report['responses'][-1], ensure_ascii=False), flush=True)
        report['success'] = True
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
