"""Offline official-MTP agreement/overhead on the actual trained target.

No training/downloads/service changes. The missing-weight/check-only paths
do not import torch or inspect CUDA. Real execution requires the shared GPU
lease and loads the custom target exactly once. This does not enable serving
speculation, target cache rollback, JSON grammar or speculative sampling.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path('/mnt/d/ai_project_data/qwen35_cloud')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mtp', type=Path, default=ROOT / 'mtp/Qwen3.5-4B-official/model.safetensors')
    parser.add_argument('--config', type=Path, default=ROOT / 'base/Qwen3.5-4B/config.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--tokens', type=int, default=16, help='Verified draft/target pairs per prompt')
    parser.add_argument('--prompt', action='append', help='Repeat for custom short plain-text user prompts')
    args = parser.parse_args(argv)
    if args.tokens < 1:
        parser.error('--tokens must be positive')
    report = {'kind': 'official-qwen35-mtp-standalone-profile', 'training': False,
              'downloads': False, 'target_loaded': False, 'cuda_queried': False,
              'speculative_decoding_usable': False, 'status': 'blocked', 'rows': [],
              'mtp_path': str(args.mtp), 'hidden_tap': 'target-final-normalized',
              'production_wiring_changed': False, 'cache_rollback_implemented': False,
              'json_grammar_implemented': False, 'sampling_implemented': False}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    def emit(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)
    # Fail before importing torch or trying to reserve the target/GPU.
    try:
        from models.mtp_probe import expected_mtp_shapes, read_safetensors_header
        expected = expected_mtp_shapes(json.loads(args.config.read_text()))
        header = read_safetensors_header(args.mtp)
        if set(header) != set(expected):
            raise ValueError('Need exactly all 15 official MTP tensors')
        for name, shape in expected.items():
            if header[name]['shape'] != shape or header[name]['dtype'] != 'BF16':
                raise ValueError(f'Unsupported MTP tensor: {name}')
        report['verified_header_tensor_count'] = len(header)
        report['checkpoint_bytes'] = args.mtp.stat().st_size
        provenance = args.mtp.parent / 'provenance.json'
        if provenance.is_file():
            report['provenance'] = json.loads(provenance.read_text())
    except (OSError, ValueError, KeyError) as error:
        report['blocker'] = str(error)
        save(); emit(report)
        return 2
    if args.check_only:
        report['status'] = 'headers_validated_only'
        save(); emit(report)
        return 0
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    lease_path = Path.home() / '.cache/qwen-debugger/gpu.lock'
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    with lease_path.open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report['blocker'] = 'Shared GPU lease is busy; no model was loaded'
            save(); emit(report)
            return 2
        from models.inference_environment import configure_cuda_allocator
        configure_cuda_allocator()
        import torch
        from models.pretrained import PretrainedChatModel
        from models.qwen35_mtp import Qwen35MTPHead, FinalHiddenCapture
        torch.set_num_threads(1)
        started = time.monotonic()
        report['cuda_queried'] = True
        chat = PretrainedChatModel(device='cuda', local_files_only=True, context_length=65536,
                                   kv_cache='bf16', cuda_graphs=False)
        target = chat.model
        report['target_loaded'] = True
        report['target_load_seconds'] = time.monotonic() - started
        report['target_runtime'] = chat.description
        embed, lm_head = target.get_input_embeddings(), target.get_output_embeddings()
        mtp = Qwen35MTPHead.from_safetensors(args.mtp, target.config, embed, lm_head,
                                            device=embed.weight.device, dtype=embed.weight.dtype)
        report['mtp_parameters'] = sum(p.numel() for p in mtp.parameters())
        report['shared_embedding_identity'] = mtp.embed_tokens is embed and mtp.lm_head is lm_head
        norms = [module for name, module in target.named_modules()
                 if name.endswith('language_model.norm')]
        if len(norms) != 1:
            raise RuntimeError('Cannot identify exactly one target text final norm')
        emit({'target_loaded': True, 'mtp_loaded': True, 'mtp_parameters': report['mtp_parameters']})
        def synchronize():
            torch.cuda.synchronize(embed.weight.device)
        def timed(call):
            synchronize(); tick = time.perf_counter()
            result = call()
            synchronize()
            return result, (time.perf_counter() - tick) * 1000
        prompts = args.prompt or ['用三句话解释KV缓存的作用。', '计算 17 加 25，只返回答案。',
                                  '写一个JSON对象，包含answer字段，值是中国的首都。']
        stop_ids = target.generation_config.eos_token_id
        stops = {stop_ids} if isinstance(stop_ids, int) else set(stop_ids or [])
        with torch.inference_mode(), FinalHiddenCapture(norms[0]) as capture:
            # All target tokens below are real greedy target continuations.
            # A rejected proposal never enters the target cache or final output.
            for case_index, prompt in enumerate([prompts[0]] + prompts):
                warmup = case_index == 0
                token_limit = min(args.tokens, 4) if warmup else args.tokens
                messages = [{'role': 'system', 'content': '用简短中文直接回答。'},
                            {'role': 'user', 'content': prompt}]
                encoded = chat.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,
                    return_tensors='pt', return_dict=True)
                ids = encoded['input_ids'].to(embed.weight.device)
                length = ids.shape[1]
                positions = torch.arange(length, device=ids.device)[None]
                output, target_prefill_ms = timed(lambda: target(ids, use_cache=True, logits_to_keep=1))
                bonus = output.logits[:, -1].argmax(-1)
                target_cache = output.past_key_values
                drafted, mtp_prefill_ms = timed(lambda: mtp.prefill(ids, capture.hidden_states,
                    bonus, positions, use_cache=True, logits_to_keep=1))
                draft_cache = drafted.past_key_values
                generated = [int(bonus.item())]
                row = {'prompt': prompt, 'prompt_tokens': length, 'pairs': [],
                       'target_prefill_ms': target_prefill_ms, 'mtp_prefill_ms': mtp_prefill_ms,
                       'target_decode_ms': [], 'mtp_decode_ms': []}
                for step in range(token_limit):
                    if generated[-1] in stops:
                        break
                    top = drafted.logits[:, -1].float().topk(min(10, lm_head.weight.shape[0])).indices[0].tolist()
                    output, elapsed = timed(lambda: target(bonus.reshape(1, 1),
                        past_key_values=target_cache, use_cache=True, logits_to_keep=1))
                    row['target_decode_ms'].append(elapsed)
                    bonus = output.logits[:, -1].argmax(-1)
                    actual = int(bonus.item())
                    agreed = top[0] == actual
                    verified = top[0] if agreed else actual
                    assert verified == actual  # Only verified target token is committed.
                    generated.append(verified)
                    row['pairs'].append({'target': actual, 'draft': top[0], 'top1': agreed,
                                         'top5': actual in top[:5], 'top10': actual in top})
                    if step + 1 < token_limit and actual not in stops:
                        slot = torch.tensor([[length + step]], device=ids.device)
                        drafted, elapsed = timed(lambda: mtp(bonus.reshape(1, 1),
                            capture.hidden_states, slot, past_key_values=draft_cache, logits_to_keep=1))
                        row['mtp_decode_ms'].append(elapsed)
                row['generated_ids'] = generated
                row['text'] = chat.tokenizer.decode(generated, skip_special_tokens=True)
                if warmup:
                    report['warmup_pairs'] = len(row['pairs'])
                    emit({'warmup_completed': True, 'pairs': report['warmup_pairs']})
                else:
                    report['rows'].append(row)
                    save(); emit(row)
        pairs = [pair for row in report['rows'] for pair in row['pairs']]
        target_ms = [ms for row in report['rows'] for ms in row['target_decode_ms']]
        draft_ms = [ms for row in report['rows'] for ms in row['mtp_decode_ms']]
        report['summary'] = {'verified_pairs': len(pairs),
            **{key + '_agreement': sum(p[key] for p in pairs) / len(pairs) if pairs else None
               for key in ('top1', 'top5', 'top10')},
            'mean_target_decode_ms': sum(target_ms) / len(target_ms) if target_ms else None,
            'mean_mtp_decode_ms': sum(draft_ms) / len(draft_ms) if draft_ms else None,
            'peak_torch_mib': torch.cuda.max_memory_allocated() / 2**20}
        report['timing_scope'] = 'Warmed synchronized standalone head/target calls, native BF16 cache, final norm output capture; not serving throughput or speculative speedup'
        report['status'] = 'standalone_head_profiled'
        save(); emit(report['summary'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
