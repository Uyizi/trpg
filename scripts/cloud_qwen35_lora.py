"""Qwen3.5-4B text-only BF16 LoRA: fresh abstracts + chat replay.

Loads only the language decoder from the official checkpoint and saves a separate adapter.
"""
import argparse
import fcntl
import hashlib
import json
import math
import signal
import sys
from pathlib import Path
import random
import time

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed

# Cloud source lives under app/models; local source lives at the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM

ROOT = Path('/mnt/workspace/qwen35_update')
SEED = 20260921


class Blocks(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = torch.tensor(self.rows[i]['input_ids'], dtype=torch.long)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': ids.clone()}


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def pack(source, tokenizer, block_size, count, replay=False):
    rows = [json.loads(line) for line in source.open()]
    random.Random(SEED).shuffle(rows)
    ids, blocks, documents = [], [], []
    for row in rows:
        # Preserve the actual Qwen3.5 conversation format in replay samples.
        text = (tokenizer.apply_chat_template(row['messages'], tokenize=False,
                add_generation_prompt=False, enable_thinking=False) if replay else row['text'])
        tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        documents.append(row.get('paper_id', row.get('id')))
        ids.extend(tokens)
        while len(ids) >= block_size:
            blocks.append({'input_ids': ids[:block_size]})
            del ids[:block_size]
            if len(blocks) == count:
                return blocks, documents
    raise ValueError(f'Insufficient unique data: {source.name} supplied {len(blocks)}/{count} blocks')


def prepare(args, tokenizer):
    destination = ROOT / 'prepared' / 'qwen35_cpt_v1'
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('tokenizer') != str(tokenizer.name_or_path) or
                manifest['block_size'] != args.block_size or manifest['requested_tokens'] != args.tokens):
            raise ValueError('Existing prepared dataset has a different budget. Use its original configuration.')
        for split, digest in manifest['packed_sha256'].items():
            if sha(destination / (split + '.jsonl')) != digest:
                raise ValueError('Prepared dataset hash changed')
        return destination
    fresh = ROOT / 'data' / 'arxiv_snapshot'
    status = json.loads((fresh / 'manifest.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('Fresh corpus collection is incomplete')
    replay = Path('/mnt/workspace/qwen_update/prepared')
    total = args.tokens // args.block_size
    replay_blocks = total // 5
    if total < 32:
        raise ValueError('Token budget too small for this baseline')
    train_fresh, fresh_ids = pack(fresh / 'train.jsonl', tokenizer, args.block_size, total - replay_blocks)
    train_replay, replay_ids = pack(replay / 'train.jsonl', tokenizer, args.block_size, replay_blocks, True)
    validation_fresh, _ = pack(fresh / 'validation.jsonl', tokenizer, args.block_size, 48)
    validation_replay, _ = pack(replay / 'validation.jsonl', tokenizer, args.block_size, 16, True)
    train = train_fresh + train_replay
    random.Random(SEED).shuffle(train)
    destination.mkdir(parents=True, exist_ok=True)
    splits = {'train': train, 'validation_fresh': validation_fresh, 'validation_replay': validation_replay}
    for name, rows in splits.items():
        with (destination / (name + '.jsonl')).open('w') as out:
            for row in rows:
                out.write(json.dumps(row) + '\n')
    manifest = {'tokenizer': str(tokenizer.name_or_path), 'seed': SEED, 'block_size': args.block_size, 'requested_tokens': args.tokens,
        'train_tokens': len(train) * args.block_size, 'fresh_tokens': len(train_fresh) * args.block_size,
        'replay_tokens': len(train_replay) * args.block_size, 'fresh_document_ids': fresh_ids,
        'replay_document_ids': replay_ids, 'corpus_manifest': status,
        'source_sha256': {str(p): sha(p) for p in [fresh / 'train.jsonl', fresh / 'validation.jsonl', replay / 'train.jsonl', replay / 'validation.jsonl']},
        'packed_sha256': {name: sha(destination / (name + '.jsonl')) for name in splits},
        'packing': 'Complete EOS-separated documents concatenated into fixed blocks; final unused tail discarded.',
        'loss': 'All token next-token CPT loss, not assistant-only SFT loss.'}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print('QWEN35_DATA_READY', {k: manifest[k] for k in ('train_tokens', 'fresh_tokens', 'replay_tokens')}, flush=True)
    return destination


class Progress(TrainerCallback):
    def __init__(self, folder, max_hours, tokens_per_step):
        self.folder, self.deadline, self.start = folder, time.monotonic() + max_hours * 3600, time.monotonic()
        self.tokens_per_step = tokens_per_step
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def request_stop(self, *_):
        self.stop_requested = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = {k: float(v) if isinstance(v, (float, int)) else v for k, v in (logs or {}).items()}
        if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
            raise FloatingPointError('Non-finite training metric')
        values.update(step=state.global_step, total_steps=state.max_steps,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
            elapsed_seconds=time.monotonic() - self.start)
        with (self.folder / 'metrics.jsonl').open('a') as out:
            out.write(json.dumps(values) + '\n')

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_requested or time.monotonic() >= self.deadline:
            control.should_training_stop = True
            control.should_save = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=int, default=2_097_152)
    p.add_argument('--block-size', type=int, default=1024)
    p.add_argument('--max-hours', type=float, default=3)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not Path('/etc/dsw').exists() or not torch.cuda.is_bf16_supported():
        raise ValueError('This experiment requires BF16 CUDA on DSW')
    if args.block_size < 128 or args.tokens < 1 or args.max_hours <= 0:
        raise ValueError('Invalid training budget')
    output = ROOT / 'runs' / 'qwen35_cpt_v1'
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoints = sorted((p for p in output.glob('checkpoint-*') if p.name.split('-')[-1].isdigit()
            and all((p / name).is_file() for name in ('trainer_state.json', 'optimizer.pt', 'adapter_model.safetensors'))),
            key=lambda p: int(p.name.split('-')[-1]))
        if checkpoints and not args.resume and not args.prepare_only:
            raise ValueError('Checkpoints already exist; use --resume for this unfinished run')
        resume_checkpoint = checkpoints[-1] if args.resume and checkpoints else None
        step_start = json.loads((resume_checkpoint / 'trainer_state.json').read_text())['global_step'] if resume_checkpoint else 0
        if (output / 'completed.json').exists():
            raise ValueError('Baseline already completed; do not overwrite it')
        base = ROOT / 'base' / 'Qwen3.5-4B'
        set_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        folder = prepare(args, tokenizer)
        if args.prepare_only:
            return
        def dataset(name):
            return Blocks([json.loads(line) for line in (folder / (name + '.jsonl')).open()])
        free, _ = torch.cuda.mem_get_info()
        if free < 18 * 1024 ** 3:
            raise RuntimeError('Need at least 18 GiB free GPU memory; another task may be using this A10')
        model = TextOnlyQwen3_5ForCausalLM.from_pretrained(base, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa')
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
            # Restrict adapters to language layers, including hybrid DeltaNet projections.
            target_modules=r'model\.language_model\.layers\.\d+\.(self_attn|linear_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)'))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0
        assert not any('.visual.' in n for n, _ in model.named_parameters())
        print('QWEN35_LORA_READY', {'base': str(base), 'trainable_parameters': trainable}, flush=True)
        train_args = TrainingArguments(output_dir=str(output), num_train_epochs=1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=16,
            learning_rate=2e-5, lr_scheduler_type='cosine', warmup_steps=0.03,
            weight_decay=0.01, max_grad_norm=1.0, bf16=True, tf32=True,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            optim='adamw_torch', logging_steps=1, logging_nan_inf_filter=False,
            save_steps=32, save_total_limit=2, eval_strategy='no',
            report_to='none', prediction_loss_only=True, dataloader_num_workers=0,
            seed=SEED, data_seed=SEED, disable_tqdm=True)
        trainer = Trainer(model=model, args=train_args, train_dataset=dataset('train'),
            processing_class=tokenizer, callbacks=[Progress(output, args.max_hours, 16 * args.block_size)])
        metadata = {'state': 'running', 'base': str(base), 'trainable_parameters': trainable,
            'resumed_from': str(resume_checkpoint) if resume_checkpoint else None, 'step_start': step_start,
            'precision': 'bf16', 'lora_rank': 16, 'lora_alpha': 32, 'optimizer': 'adamw_torch',
            'attention': 'PyTorch SDPA automatic kernel dispatch', 'seed': SEED,
            'base_receipt_sha256': sha(base / 'download-complete.json'),
            'data_manifest_sha256': sha(folder / 'manifest.json'),
            'training_args': train_args.to_dict()}
        (output / 'status.json').write_text(json.dumps(metadata, indent=2, default=str))
        try:
            if resume_checkpoint is None:
                before = {split: trainer.evaluate(dataset('validation_' + split), metric_key_prefix='before_' + split)
                          for split in ('fresh', 'replay')}
                (output / 'before.json').write_text(json.dumps(before, indent=2))
            result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
            trainer.save_model(str(output / 'adapter'))
            tokenizer.save_pretrained(output / 'adapter')
            trainer.save_state()
            trainer.save_metrics('train', result.metrics)
            complete = trainer.state.global_step >= trainer.state.max_steps
            metadata.update(state='completed' if complete else 'stopped_saved', step=trainer.state.global_step,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
            samples_before = min(step_start * 16, len(trainer.train_dataset))
            samples_after = min(trainer.state.global_step * 16, len(trainer.train_dataset))
            processed_tokens = (samples_after - samples_before) * args.block_size
            metadata['tokens_per_second'] = processed_tokens / result.metrics['train_runtime']
            if complete:
                after = {split: trainer.evaluate(dataset('validation_' + split), metric_key_prefix='after_' + split)
                         for split in ('fresh', 'replay')}
                metadata['after'] = after
                (output / 'completed.json').write_text(json.dumps(metadata, indent=2, default=str))
        except Exception as exc:
            metadata.update(state='failed', error=type(exc).__name__)
            raise
        finally:
            (output / 'status.json').write_text(json.dumps(metadata, indent=2, default=str))


if __name__ == '__main__':
    main()
