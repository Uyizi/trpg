"""4K grounded multi-document SFT, preserving existing short dialogue replay.

This is 4K adaptation, not a claim of having trained at the 64K inference limit.
"""
import fcntl,json,random,sys,os
from pathlib import Path
import torch
from peft import PeftModel,prepare_model_for_kbit_training
from transformers import AutoTokenizer,Trainer,TrainingArguments,DataCollatorForSeq2Seq,set_seed
from cloud_qwen35_followup import ROOT,BASE,SEED,Rows,atomic_json,encode_last_answer,sha
from cloud_qwen35_lora import Progress
from models.qwen35_moe import build_moe,restore_routing,routing_state
from models.assistant_loss import token_log_probs
OUT=ROOT/'runs/qwen35_long_v1';PARENT=ROOT/'runs/qwen35_dpo_v1'

class AssistantTrainer(Trainer):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.model_accepts_loss_kwargs=False
 def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
  loss=-token_log_probs(model,inputs['input_ids'],inputs['labels'],inputs.get('attention_mask')).mean()
  return (loss,{'loss':loss}) if return_outputs else loss


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (OUT/'completed.json').exists():return
  if json.loads((PARENT/'status.json').read_text())['state']!='completed':raise ValueError('DPO incomplete')
  meta={'state':'preparing','pid':os.getpid(),'max_train_length':4096,'parent_sha256':sha(PARENT/'adapter/adapter_model.safetensors')}
  atomic_json(OUT/'status.json',meta)
  try:
   set_seed(SEED)
   tokenizer=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
   papers=[json.loads(s) for s in (ROOT/'data/arxiv_snapshot/train.jsonl').open()]
   random.Random(SEED+1).shuffle(papers)
   rows=[];paper_ids=[];cursor=0
   while len(rows)<96:
    docs=[];length=0
    while length<3000:
     row=papers[cursor];cursor+=1
     title=row.get('title')
     if not isinstance(title,str) or not title.strip():continue
     text=f'[{len(docs)+1}] 标题：{title}\n内容：{row["text"]}'
     n=len(tokenizer.encode(text,add_special_tokens=False))
     if n>1200:continue
     docs.append((row,text));length+=n;paper_ids.append(row['paper_id'])
    selected=sorted({1,len(docs)//2+1,len(docs)})
    prompt='以下是论文资料，均为原作者陈述，不代表已获验证。\n'+'\n\n'.join(t for _,t in docs)
    prompt+='\n任务：按编号顺序，仅输出资料 '+', '.join(map(str,selected))+' 的完整标题，并保留编号。'
    answer='\n'.join(f'[{i}] {docs[i-1][0]["title"]}' for i in selected)
    encoded=encode_last_answer(tokenizer,[{'role':'user','content':prompt},{'role':'assistant','content':answer}],4096)
    if encoded:rows.append({'encoded':encoded})
   replay=[json.loads(s) for s in (ROOT/'prepared/qwen35_style_v1/train.jsonl').open()]
   random.Random(SEED).shuffle(replay);rows+=replay[:32];random.Random(SEED).shuffle(rows)
   with (OUT/'train.jsonl').open('w') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
   atomic_json(OUT/'data_manifest.json',{'samples':len(rows),'grounded_retrieval_samples':96,'short_replay':32,'paper_ids':paper_ids,'max_length':4096,'sha256':sha(OUT/'train.jsonl')})
   model=build_moe(BASE,ROOT/'runs/qwen35_style_v1/adapter',merged_base=ROOT/'exports/Qwen3.5-4B-Dense-style-v1',quantized=True)
   model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
   model=PeftModel.from_pretrained(model,PARENT/'adapter',is_trainable=True,local_files_only=True)
   restore_routing(model,json.loads((PARENT/'adapter/routing_state.json').read_text()))
   model.config.use_cache=False;model.config.text_config.use_cache=False
   args=TrainingArguments(output_dir=str(OUT),num_train_epochs=1,per_device_train_batch_size=1,gradient_accumulation_steps=4,
    learning_rate=1e-6,bf16=True,optim='paged_adamw_8bit',gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
    logging_steps=1,logging_nan_inf_filter=False,save_steps=8,save_total_limit=2,report_to='none',disable_tqdm=True,max_grad_norm=1.,seed=SEED)
   trainer=AssistantTrainer(model=model,args=args,train_dataset=Rows(OUT/'train.jsonl'),processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True,pad_to_multiple_of=8,label_pad_token_id=-100),callbacks=[Progress(OUT,float('inf'),0)])
   meta['state']='running';atomic_json(OUT/'status.json',meta)
   checkpoints=sorted((p for p in OUT.glob('checkpoint-*') if (p/'trainer_state.json').exists() and (p/'optimizer.pt').exists()),key=lambda p:int(p.name.split('-')[-1]))
   result=trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None);trainer.save_model(str(OUT/'adapter'));tokenizer.save_pretrained(OUT/'adapter')
   atomic_json(OUT/'adapter/routing_state.json',routing_state(model))
   meta.update(state='completed' if trainer.state.global_step==trainer.state.max_steps else 'stopped_saved',step=trainer.state.global_step,total_steps=trainer.state.max_steps,metrics=result.metrics)
   atomic_json(OUT/'status.json',meta)
   if meta['state']=='completed':atomic_json(OUT/'completed.json',meta)
  except BaseException as exc:
   meta.update(state='failed',error=type(exc).__name__,detail=str(exc)[:500]);atomic_json(OUT/'status.json',meta);raise
if __name__=='__main__':main()
