"""Continue the Qwen3.5 CPT adapter through style SFT, then Top-1 MoE SFT."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
from peft import LoraConfig, TaskType, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
# Local workspace fallback; cloud source lives under app/models.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.qwen35_moe import load_dense, build_moe, expert_modules
from cloud_qwen35_lora import Progress, ROOT, SEED, sha

BASE = ROOT / 'base/Qwen3.5-4B'
STYLE = ('直接回答，覆盖所有子问题。省略寒暄、致谢、复述和重复总结。'
         '除非要求，不用比喻、表情和“不是……而是……”句式。'
         '数学和代码保留必要步骤；不为简短而漏掉条件。'
         '区分事实、假设和未知信息；没有实际检索结果时，不声称已经联网核实。')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


class Rows(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(line)['encoded'] for line in path.open()]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]


def encode_last_answer(tokenizer, messages, max_length):
    # Qwen3.5 renders earlier assistant turns differently from the final answer.
    # Supervise only the final complete answer, using offsets from ONE tokenization.
    if messages[-1]['role'] != 'assistant':
        return None
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prefix):
        raise ValueError('Chat template prefix changed; refusing invalid assistant masking')
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    if len(ids) > max_length:
        return None  # Do not teach a cut-off answer.
    labels = [token if start >= len(prefix) and end > start else -100
              for token, (start, end) in zip(ids, encoded['offset_mapping'])]
    if not any(v != -100 for v in labels):
        return None
    return dict(input_ids=ids, attention_mask=[1]*len(ids), labels=labels)


def prepare():
    destination = ROOT / 'prepared/qwen35_style_v1'
    source = Path('/mnt/workspace/qwen_update/prepared')
    fingerprint = {n: sha(source / (n + '.jsonl')) for n in ('train','validation')}
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if m['source_sha256'] != fingerprint:
            raise ValueError('Source data changed')
        for n, digest in m['packed_sha256'].items():
            if sha(destination / (n + '.jsonl')) != digest:
                raise ValueError('Prepared SFT data changed')
        return destination
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    destination.mkdir(parents=True, exist_ok=True)
    counts, seen = {}, set()
    for split, cap in [('validation',128),('train',2048)]:
        candidates = [json.loads(line) for line in (source / (split + '.jsonl')).open()]
        random.Random(SEED).shuffle(candidates)
        selected = []
        for row in candidates:
            messages = [dict(m) for m in row['messages']]
            if any(not isinstance(m.get('content'),str) or m.get('role') not in ('system','user','assistant') for m in messages):
                continue
            # Preserve the published target verbatim; filtering is not teacher distillation.
            if messages[0]['role'] != 'system':
                messages.insert(0, {'role':'system','content':STYLE})
            encoded = encode_last_answer(tokenizer, messages, 1024)
            if encoded is None:
                continue
            key = hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            selected.append({'id':row['id'], 'source':row['source'], 'encoded':encoded})
            if len(selected) >= cap:
                break
        minimum = 24 if split == 'validation' else 512
        if len(selected) < minimum:
            raise ValueError(f'Too few complete style examples: {split}={len(selected)}')
        with (destination / (split+'.jsonl')).open('w') as out:
            for row in selected:
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
        counts[split] = {'examples':len(selected), 'tokens':sum(len(r['encoded']['input_ids']) for r in selected),
            'supervised_tokens':sum(sum(t!=-100 for t in r['encoded']['labels']) for r in selected),
            'sources':dict(Counter(r['source'] for r in selected))}
    atomic_json(manifest_path, {'counts':counts,'source_sha256':fingerprint,
        'packed_sha256':{n:sha(destination/(n+'.jsonl')) for n in counts},
        'base':str(BASE),'max_length':1024,'seed':SEED,
        'policy':'Reuse previously filtered published SmolTalk examples; last complete assistant answer only; original conversation split retained.'})
    print('STYLE_DATA_READY',json.dumps(counts,ensure_ascii=False),flush=True)
    return destination


class MoETrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.moe = expert_modules(self.model)
        # This override, like the base Trainer loss, normalizes one microbatch.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        for module in self.moe:
            module.token_mask = inputs.get('attention_mask')
        outputs = model(**inputs)
        loss = outputs.loss
        if model.training:
            aux = [m.balance_loss for m in self.moe if m.balance_loss is not None]
            if aux:
                loss = loss + 0.01 * torch.stack(aux).mean()
        for module in self.moe:
            module.balance_loss = None
        return (loss,outputs) if return_outputs else loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['style','moe'],default='style')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    folder=prepare()
    if args.prepare_only: return
    stage=args.stage
    out=ROOT/'runs'/f'qwen35_{stage}_v1'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'completed.json').exists():
            print('ALREADY_COMPLETED',stage,flush=True); return
        checkpoints=sorted((p for p in out.glob('checkpoint-*') if p.name.split('-')[-1].isdigit()
            and all((p/n).exists() for n in ('trainer_state.json','optimizer.pt','adapter_model.safetensors'))),
            key=lambda p:int(p.name.split('-')[-1]))
        if checkpoints and not args.resume: raise ValueError('Use --resume for existing checkpoints')
        checkpoint=checkpoints[-1] if args.resume and checkpoints else None
        source=ROOT/'runs'/('qwen35_cpt_v1' if stage=='style' else 'qwen35_style_v1')
        parent=json.loads((source/'status.json').read_text())
        if parent['state']!='completed' or not (source/'completed.json').exists():
            raise ValueError('Parent training is not complete')
        adapter=source/'adapter'
        meta={'state':'loading','stage':stage,'base':str(BASE),'parent_adapter':str(adapter),
              'parent_adapter_sha256':sha(adapter/'adapter_model.safetensors'),
              'data_manifest_sha256':sha(folder/'manifest.json'),'pid':__import__('os').getpid()}
        atomic_json(out/'status.json',meta)
        try:
            set_seed(SEED)
            if torch.cuda.mem_get_info()[0] < 19*1024**3:
                raise RuntimeError('Need 19 GiB free GPU memory; do not run cloud chat concurrently')
            tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
            tokenizer.padding_side='right'
            if stage=='style':
                model=load_dense(BASE,adapter,trainable=True)
            else:
                model=build_moe(BASE,adapter)
                model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=16,lora_alpha=32,
                    lora_dropout=0.05,bias='none',modules_to_save=['router'],
                    target_modules=r'model\.language_model\.layers\.\d+\.(?:(?:self_attn|linear_attn)\.(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj))'))
            model.config.use_cache=False
            model.config.text_config.use_cache=False
            meta['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
            meta['total_stored_parameters']=sum(p.numel() for p in model.parameters())
            meta['experts_per_layer']=2 if stage=='moe' else 1
            print('TRAINING_READY',json.dumps(meta),flush=True)
            training_args=TrainingArguments(output_dir=str(out),num_train_epochs=1,
                per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=16,
                learning_rate=1e-5,lr_scheduler_type='cosine',warmup_steps=0.03,weight_decay=0.01,
                max_grad_norm=1.0,bf16=True,tf32=True,optim='adamw_torch',
                gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                logging_steps=1,logging_nan_inf_filter=False,save_steps=32,save_total_limit=2,
                eval_strategy='no',report_to='none',prediction_loss_only=True,
                dataloader_num_workers=0,seed=SEED,data_seed=SEED,disable_tqdm=True)
            # Fixed one-epoch budgets, no automatic indefinite restart or time-based truncation.
            callback=Progress(out,float('inf'),0)
            trainer_cls=MoETrainer if stage=='moe' else Trainer
            trainer=trainer_cls(model=model,args=training_args,train_dataset=Rows(folder/'train.jsonl'),
                processing_class=tokenizer,callbacks=[callback],
                data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True,pad_to_multiple_of=8,label_pad_token_id=-100))
            meta['state']='running';atomic_json(out/'status.json',meta)
            result=trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
            trainer.save_model(str(out/'adapter'))
            tokenizer.save_pretrained(out/'adapter')
            trainer.save_state()
            complete=trainer.state.global_step>=trainer.state.max_steps
            meta.update(state='completed' if complete else 'stopped_saved',step=trainer.state.global_step,
                total_steps=trainer.state.max_steps,metrics=result.metrics,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            if stage=='moe':
                meta['last_expert_fractions']=[m.last_counts.tolist() if m.last_counts is not None else None for m in expert_modules(model)]
            atomic_json(out/'status.json',meta)
            if complete: atomic_json(out/'completed.json',meta)
            print('STAGE_SAVED',stage,meta['state'],flush=True)
        except BaseException as exc:
            meta.update(state='failed',error=type(exc).__name__,detail=str(exc)[:500])
            atomic_json(out/'status.json',meta)
            raise


if __name__=='__main__': main()
