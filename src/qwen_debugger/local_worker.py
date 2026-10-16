"""Model-environment worker. Standard-library protocol; no Pydantic dependency.

Only this process owns the diagnostic model. stdout is reserved for JSON lines.
"""
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

PROJECT=Path(os.environ.get('QDEBUG_PROJECT_ROOT',Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(PROJECT))
from models.inference_environment import configure_cuda_allocator
configure_cuda_allocator()
protocol=sys.stdout
sys.stdout=sys.stderr
from worker_protocol import decode_output, exit_with_parent
exit_with_parent()
backend=None
tokenizer=None
settings=None


def prepare(command):
    global tokenizer,settings
    from transformers import AutoTokenizer,AutoConfig
    from models.runtime_paths import DEFAULT_BUNDLE,read_bundle
    p=command['profile']
    if settings is not None and p!=settings: raise ValueError('一个 worker 只能使用一份冻结 profile')
    settings=p
    folder=Path(p['model_path'] or DEFAULT_BUNDLE)
    base=read_bundle(folder)['base']
    if tokenizer is None: tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    template=tokenizer.chat_template
    marker='if loop.index0 > ns.last_query_index'
    if not p['thinking'] and template.count(marker)==1:
        template=template.replace(marker,marker+' or (enable_thinking is defined and enable_thinking is false)')
    messages=command['messages']
    kwargs=dict(chat_template=template,add_generation_prompt=True,enable_thinking=p['thinking'])
    ids=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,**kwargs)
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,**kwargs)
    if len(ids)>p['max_input_tokens']: raise ValueError('输入超过 profile.max_input_tokens；没有截短输入')
    if len(ids)+p['max_new_tokens']>p['context_length']: raise ValueError('输入加输出超过窗口；没有降低输出预算')
    payload={'messages':messages,'seed':command['seed'],'profile':p}
    canonical=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    return {'payload':payload,'fingerprint':hashlib.sha256(canonical.encode()).hexdigest(),
            'rendered_prompt':prompt,'input_ids':ids,'input_tokens':len(ids),'count_source':'actual_tokenizer',
            'chat_template':template,'tokenizer_eos':tokenizer.eos_token_id,'tokenizer_pad':tokenizer.pad_token_id,
            'sampling':{'temperature':p['temperature'],'top_p':0.8 if p['temperature']>0 else None,
                        'top_k':20 if p['temperature']>0 else None},
            'thinking':p['thinking'],'stop_policy':'runtime generation EOS union tokenizer EOS'}


def generate(command):
    global backend
    import torch
    import importlib.metadata as metadata
    from models.pretrained import PretrainedChatModel
    from models.runtime_paths import DEFAULT_BUNDLE
    request=command['request']; p=request['payload']['profile']
    if p!=settings: raise ValueError('请求与已准备 profile 不一致')
    if backend is None:
        backend=PretrainedChatModel(model_id=p['model_path'] or str(DEFAULT_BUNDLE),device='cuda',
            local_files_only=True,context_length=p['context_length'],thinking=p['thinking'],
            kv_cache=p['kv_cache'],fast_kernels=p['fast_kernels'],prefill_chunk_size=p['prefill_chunk_size'])
    messages=request['payload']['messages']
    if backend.encode(messages)!=request['input_ids'] or backend.inference_template!=request['chat_template']:
        raise ValueError('实际推理模板或编码与冻结请求不一致')
    backend.clear_cache()
    captured={}
    original=backend.generator.generate
    def capture(*args,**kwargs):
        output,stats=original(*args,**kwargs)
        captured['ids']=output[0,len(request['input_ids']):].tolist()
        return output,stats
    backend.generator.generate=capture
    try:
        with torch.random.fork_rng(devices=[backend.device.index or 0]):
            torch.manual_seed(request['payload']['seed'])
            text=backend.answer(messages,max_new_tokens=p['max_new_tokens'],temperature=p['temperature'])
    finally:
        backend.generator.generate=original
        backend.clear_cache()
    ids=captured['ids']; eos=backend.model.generation_config.eos_token_id
    ended=bool(ids) and ids[-1] in eos
    finish='stop' if ended else ('length' if backend.last_response_truncated else 'unknown')
    decoded=decode_output(backend.tokenizer,ids)
    return {**decoded,'reasoning':None,'reasoning_extraction':'not_separated; retained in text',
            'finish_reason':finish,'output_ids':ids,'usage':{'input_tokens':len(request['input_ids']),'output_tokens':len(ids)},
            'stats':backend.last_response_stats,'backend':'local','runtime':backend.description,
            'effective_generation_config':backend.model.generation_config.to_dict(),
            'seed_control':'torch.manual_seed; kernel determinism not established',
            'dependencies':{n:metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes']}}


for line in sys.stdin:
    try:
        command=json.loads(line)
        result=prepare(command) if command['op']=='prepare' else generate(command)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result={'error':type(exc).__name__+': '+str(exc)}
    protocol.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n'); protocol.flush()
