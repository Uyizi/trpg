"""Publish the final adapter only after context, decoding and retrieval are usable.

Long prompts here measure runtime capacity, not long-document answer quality.
"""
import gc,json,sys,time
from pathlib import Path
import torch
from cloud_qwen35_followup import ROOT,atomic_json,sha
from models.qwen35_moe import load_bundle
from models.stateful_generation import generate
OUT=ROOT/'exports/Qwen3.5-MoE-NF4-DPO-long-v3'
PARENT=ROOT/'runs/qwen35_long_v1'


def main():
    if not (PARENT/'completed.json').exists():raise ValueError('Long-context adaptation incomplete')
    if not (ROOT/'data/semantic_arxiv/manifest.json').exists():raise ValueError('Semantic index incomplete')
    manifest=json.loads((ROOT/'exports/Qwen3.5-MoE-NF4-v2/moe_manifest.json').read_text())
    manifest.update(moe_adapter=str(PARENT/'adapter'),moe_adapter_sha256=sha(PARENT/'adapter/adapter_model.safetensors'),
        routing_state=str(PARENT/'adapter/routing_state.json'),routing_sha256=sha(PARENT/'adapter/routing_state.json'),
        stages=['NF4 QLoRA','DPO','4096-token grounded retrieval SFT'],
        note='Two-expert Top-1 NF4 MoE; DPO and 4K adaptation. Inference capacity recorded separately.')
    atomic_json(OUT/'moe_manifest.json',manifest)
    model,_=load_bundle(OUT)
    model.requires_grad_(False)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(manifest['base'],local_files_only=True)
    profile={'state':'profiling','adapter_sha256':manifest['moe_adapter_sha256'],
             'training_context_length':4096,'max_new_tokens':8192,'prefill_chunk_size':2048,'probes':[]}
    # Compare the enabled custom decoder to the native decoder on the final model.
    ids=tokenizer.apply_chat_template([{'role':'user','content':'仅重复：学习模型需要数据，学习模型需要数据。'}],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False).cuda()
    with torch.inference_mode():
        native=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=24,do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
        custom,stats=generate(model,ids,24,temperature=0,chunk_size=2048,speculative=4)
    if not torch.equal(native,custom):raise RuntimeError('Speculative greedy output differs from native output')
    profile['speculative']={'method':'prompt lookup, full hybrid-cache snapshot and replay','greedy_only':True,
                            'default_candidates':0,'parity_passed':True,**stats}
    del native,custom,ids
    unit=tokenizer.encode('这是用于测量上下文容量的普通文本。\n',add_special_tokens=False)
    for context in (32768,65536):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        count=context-128
        ids=torch.tensor([(unit*((count+len(unit)-1)//len(unit)))[:count]],device='cuda')
        start=time.monotonic()
        try:
            result,stats=generate(model,ids,2,temperature=0,chunk_size=2048)
            torch.cuda.synchronize()
            probe={'context_length':context,'input_tokens':count,'state':'passed',
                   'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
            del result
        except torch.cuda.OutOfMemoryError:
            probe={'context_length':context,'input_tokens':count,'state':'out_of_memory'}
        finally:del ids
        profile['probes'].append(probe);atomic_json(OUT/'runtime_profile.json',profile)
        print('CONTEXT_PROBE',json.dumps(probe),flush=True)
        if probe['state']!='passed':break
    passed=[p['context_length'] for p in profile['probes'] if p['state']=='passed']
    if not passed:raise RuntimeError('32K context could not fit; preserve previous active model')
    profile.update(state='completed',context_length=max(passed),
                   limitation='Capacity probe only; does not establish long-context reasoning/retrieval accuracy or universal speculative speedup.')
    del model;gc.collect();torch.cuda.empty_cache()
    from models.semantic_retrieval import SemanticIndex
    hits=SemanticIndex().search('大语言模型的推理和训练',2)
    if not hits:raise RuntimeError('Semantic retrieval returned no document')
    profile['semantic_documents']=json.loads((ROOT/'data/semantic_arxiv/manifest.json').read_text())['documents']
    atomic_json(OUT/'runtime_profile.json',profile)
    pending=ROOT/'active-model.next'
    if pending.is_symlink():pending.unlink()
    pending.symlink_to(OUT,target_is_directory=True);pending.replace(ROOT/'active-model')
    atomic_json(OUT/'completed.json',{'state':'completed','adapter_sha256':manifest['moe_adapter_sha256']})
    print('FINAL_MODEL_READY',OUT,flush=True)
if __name__=='__main__':main()
