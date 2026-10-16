"""Continue trained Qwen3.5 MoE with NF4 QLoRA, 8-bit AdamW and dynamic routing.

Routing bias follows arXiv:2408.15664. Bias only changes expert selection,
never the differentiable combination weight. Warm in over 16 optimizer steps.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time
import os

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
from cloud_qwen35_followup import ROOT,BASE,Rows,prepare,atomic_json,sha,SEED
from cloud_qwen35_lora import Progress
from models.qwen35_moe import build_moe,expert_modules,routing_state,restore_routing

OUT=ROOT/'runs/qwen35_qlora_v2'
MERGED=ROOT/'exports/Qwen3.5-4B-Dense-style-v1'
PARENT=ROOT/'runs/qwen35_moe_v1/adapter'


class RoutingProgress(Progress):
    def __init__(self,folder,modules):
        super().__init__(folder,float('inf'),0)
        self.modules=modules
        self.last_balance={}

    def on_step_begin(self,args,state,control,**kwargs):
        strength=min(state.global_step/16,1.)
        for module in self.modules:
            module.routing_strength=strength
            module.routing_counts.zero_()

    def on_step_end(self,args,state,control,**kwargs):
        fractions=[]
        with torch.no_grad():
            for module in self.modules:
                counts=module.routing_counts
                total=counts.sum()
                if total>0:
                    frequency=counts/total
                    # One update per accumulated optimizer batch; no gradient for this bias.
                    module.routing_bias.add_(0.001*(0.5-frequency))
                    module.routing_bias.sub_(module.routing_bias.mean()).clamp_(-0.05,0.05)
                    fractions.append(frequency.cpu().tolist())
                module.routing_strength=min(state.global_step/16,1.)
        self.last_balance={'expert_fractions':fractions,'routing_strength':min(state.global_step/16,1.)}
        super().on_step_end(args,state,control,**kwargs)

    def on_log(self,args,state,control,logs=None,**kwargs):
        super().on_log(args,state,control,logs=logs,**kwargs)
        if state.global_step:
            with (self.folder/'routing_metrics.jsonl').open('a') as out:
                out.write(json.dumps({'step':state.global_step,**self.last_balance})+'\n')

    def on_save(self,args,state,control,**kwargs):
        atomic_json(self.folder/f'checkpoint-{state.global_step}'/'routing_state.json',routing_state(kwargs['model']))


class BalancedTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.moe=expert_modules(self.model)
        self.model_accepts_loss_kwargs=False

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        for module in self.moe:
            module.token_mask=inputs.get('attention_mask')
            module.collect_routing=model.training
        try:
            outputs=model(**inputs)
        finally:
            # Non-reentrant checkpoint recomputation must NOT double-count token loads.
            for module in self.moe:module.collect_routing=False
        loss=outputs.loss
        if model.training:
            aux=[m.balance_loss for m in self.moe if m.balance_loss is not None]
            coefficient=0.01*max(0.,1-self.state.global_step/16)
            if aux and coefficient:loss=loss+coefficient*torch.stack(aux).mean()
        for module in self.moe:module.balance_loss=None
        return (loss,outputs) if return_outputs else loss


def publish():
    target=ROOT/'exports/Qwen3.5-MoE-NF4-v2'
    parent=ROOT/'exports/Qwen3.5-4B-MoE-style-v1/moe_manifest.json'
    manifest=json.loads(parent.read_text())
    manifest.update(format='qwen35-top1-nf4-v2',quantization='nf4',merged_base=str(MERGED),
        moe_adapter=str(OUT/'adapter'),moe_adapter_sha256=sha(OUT/'adapter/adapter_model.safetensors'),
        routing_state=str(OUT/'adapter/routing_state.json'),routing_sha256=sha(OUT/'adapter/routing_state.json'),
        state='completed',optimizer='paged_adamw_8bit',
        note='NF4 + double quantization, BF16 compute. Frozen experts; trainable LoRAs/routers; dynamic bias balancing.')
    atomic_json(target/'moe_manifest.json',manifest)
    pending=ROOT/'active-model.next'
    if pending.is_symlink():pending.unlink()
    pending.symlink_to(target,target_is_directory=True)
    pending.replace(ROOT/'active-model')
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'completed.json').exists():
            print('ALREADY_COMPLETED',publish(),flush=True);return
        checkpoints=sorted((p for p in OUT.glob('checkpoint-*') if p.name.split('-')[-1].isdigit()
            and all((p/n).exists() for n in ['trainer_state.json','optimizer.pt','adapter_model.safetensors','routing_state.json'])),
            key=lambda p:int(p.name.split('-')[-1]))
        if checkpoints and not args.resume:raise ValueError('Existing checkpoint; use --resume')
        checkpoint=checkpoints[-1] if checkpoints else None
        source=checkpoint if checkpoint else PARENT
        folder=prepare()
        metadata={'state':'loading','pid':os.getpid(),'parent_adapter':str(PARENT),
            'parent_sha256':sha(PARENT/'adapter_model.safetensors'),'quantization':'NF4 double quantization',
            'compute_dtype':'bfloat16','optimizer':'paged_adamw_8bit','resume':str(checkpoint) if checkpoint else None,
            'data_manifest_sha256':sha(folder/'manifest.json'),'bias_warmup_steps':16}
        atomic_json(OUT/'status.json',metadata)
        try:
            set_seed(SEED)
            if torch.cuda.mem_get_info()[0]<12*1024**3:raise RuntimeError('Need 12 GiB free GPU for QLoRA')
            model=build_moe(BASE,ROOT/'runs/qwen35_style_v1/adapter',merged_base=MERGED,quantized=True)
            model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant':False})
            model=PeftModel.from_pretrained(model,source,is_trainable=True,local_files_only=True)
            if checkpoint:restore_routing(model,json.loads((checkpoint/'routing_state.json').read_text()))
            model.config.use_cache=False;model.config.text_config.use_cache=False
            import bitsandbytes as bnb
            metadata['linear4bit_modules']=sum(isinstance(m,bnb.nn.Linear4bit) for m in model.modules())
            if metadata['linear4bit_modules']==0:raise RuntimeError('NF4 quantization was not applied')
            metadata['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('NF4_MODEL_READY',json.dumps(metadata),flush=True)
            tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True);tokenizer.padding_side='right'
            train_args=TrainingArguments(output_dir=str(OUT),num_train_epochs=1,per_device_train_batch_size=1,
                per_device_eval_batch_size=1,gradient_accumulation_steps=16,learning_rate=5e-6,
                lr_scheduler_type='cosine',warmup_steps=0.03,weight_decay=0.01,max_grad_norm=1.,
                bf16=True,tf32=True,optim='paged_adamw_8bit',gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant':False},logging_steps=1,logging_nan_inf_filter=False,
                save_steps=16,save_total_limit=2,eval_strategy='no',report_to='none',prediction_loss_only=True,
                dataloader_num_workers=0,seed=SEED,data_seed=SEED,disable_tqdm=True)
            trainer=BalancedTrainer(model=model,args=train_args,train_dataset=Rows(folder/'train.jsonl'),
                processing_class=tokenizer,callbacks=[RoutingProgress(OUT,expert_modules(model))],
                data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True,pad_to_multiple_of=8,label_pad_token_id=-100))
            metadata['state']='running';atomic_json(OUT/'status.json',metadata)
            result=trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
            trainer.save_model(str(OUT/'adapter'));tokenizer.save_pretrained(OUT/'adapter');trainer.save_state()
            atomic_json(OUT/'adapter/routing_state.json',routing_state(model))
            complete=trainer.state.global_step>=trainer.state.max_steps
            metadata.update(state='completed' if complete else 'stopped_saved',step=trainer.state.global_step,
                total_steps=trainer.state.max_steps,metrics=result.metrics,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
            atomic_json(OUT/'status.json',metadata)
            if complete:
                atomic_json(OUT/'completed.json',metadata)
                print('PUBLISHED',publish(),flush=True)
        except BaseException as exc:
            metadata.update(state='failed',error=type(exc).__name__,detail=str(exc)[:600])
            atomic_json(OUT/'status.json',metadata);raise

if __name__=='__main__':main()
