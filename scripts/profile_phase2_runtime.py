"""Offline trained-model speed comparison and staged context fact retrieval."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FACTS = [('北岑','K7R4-D2P9'),('南阁','F8M2-P7C6'),('西浦','B5V3-N6X8')]
FILLER = '普通航行日志：设备维持原有状态，本条未记录任何档案核对码。\n'
SYSTEM = '只根据用户提供的档案回答，找不到的项目写未找到。按所问顺序回答。'


def context_messages(model, window, reserve):
    target = window-reserve
    header = '以下是按时间排列的航行档案。记录中的核对码须逐字保留。\n'
    question = '\n请按北岑、南阁、西浦顺序，只写三个档案核对码，不作解释。'
    def make(repeats):
        counts = [int(repeats*.1),int(repeats*.4),int(repeats*.4)]
        counts.append(repeats-sum(counts))
        content = header
        for i, (name,code) in enumerate(FACTS):
            content += FILLER*counts[i]+f'特殊档案：{name}。核对码：{code}。\n'
        content += FILLER*counts[3]+question
        return [{'role':'system','content':SYSTEM},{'role':'user','content':content}]
    step = max(1,len(model.tokenizer.encode(FILLER,add_special_tokens=False)))
    overhead = len(model.encode(make(0)))
    repeats = max(0,(target-overhead)//step)
    messages = make(repeats)
    actual = len(model.encode(messages))
    while actual > target:
        repeats = max(0,repeats-max(1,math.ceil((actual-target)/step)))
        messages = make(repeats)
        actual = len(model.encode(messages))
    return messages, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', nargs='*', type=int, default=[8192,32768,65536,262144])
    parser.add_argument('--skip-comparison', action='store_true')
    parser.add_argument('--kv', choices=['q4','q8'], default='q8')
    parser.add_argument('--trace-memory', action='store_true')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    lock = Path.home()/'.cache/qwen-debugger/gpu.lock'
    with lock.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from models.inference_environment import configure_cuda_allocator
        configure_cuda_allocator()
        import torch
        from models.pretrained import PretrainedChatModel
        from models.attention_lora_decode import configure_attention_lora_decode
        from models.compressed_kv_attention import configure_compressed_kv_attention
        from models.compressed_kv_attention import compressed_kv_info
        started = time.monotonic()
        model = PretrainedChatModel(device='cuda',local_files_only=True,
                    context_length=max([65536,*args.lengths]),kv_cache=args.kv)
        report = dict(kind='real_phase2_runtime',training=False,downloads=False,
                      allocator_config=os.environ.get('PYTORCH_ALLOC_CONF') or os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                      loaded_seconds=round(time.monotonic()-started,3),runtime=model.description,
                      comparison=[],contexts=[])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        def save(): args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'loaded':True,'seconds':report['loaded_seconds']}),flush=True)
        def configure(attention,direct):
            configure_attention_lora_decode(model.model,attention)
            configure_compressed_kv_attention(model.model,direct)
        if not args.skip_comparison:
            short = [
                [{'role':'system','content':'用简短中文直接回答。'},
                 {'role':'user','content':'用三句话解释KV缓存的作用。'}],
                [{'role':'system','content':'直接回答。'},
                 {'role':'user','content':'从1开始逐个列出整数，直到100，中间用逗号隔开。'}],
                context_messages(model,1536,96)[0]]
            for attention,direct in ((False,False),(True,True)):
                configure(attention,direct)
                model.clear_cache()
                model.answer(short[2],max_new_tokens=8,temperature=0)
                if direct and not compressed_kv_info(model.model)['direct_decode_calls']:
                    raise RuntimeError('Real trained model did not dispatch compressed-cache decode')
            for repeat in range(2):
                for case,messages in enumerate(short):
                    for enabled in ((False,True) if repeat == 0 else (True,False)):
                        configure(enabled,enabled)
                        model.clear_cache()
                        chunks=[]
                        text=model.answer(messages,max_new_tokens=64,temperature=0,on_text=chunks.append)
                        row=dict(repeat=repeat,case=case,new_paths=enabled,text=text,
                                 stats=dict(model.last_response_stats))
                        report['comparison'].append(row);save()
                        print(json.dumps(row,ensure_ascii=False),flush=True)
            report['same_answers']=all(report['comparison'][i]['text']==report['comparison'][i+1]['text']
                                      for i in range(0,len(report['comparison']),2))
            report['comparison_summary']={}
            for enabled in (False,True):
                stats=[r['stats'] for r in report['comparison'] if r['new_paths']==enabled]
                tokens=sum(s['output_tokens'] for s in stats)
                seconds=sum(s['decode_seconds'] for s in stats)
                report['comparison_summary']['new' if enabled else 'expert_only']=dict(
                    tokens=tokens,decode_seconds=round(seconds,3),tokens_per_second=round(tokens/seconds,3),
                    peak_torch_mib=max(s['peak_torch_memory_mib'] for s in stats))
            save();print(json.dumps({'comparison_summary':report['comparison_summary'],
                                    'same_answers':report['same_answers']}),flush=True)
        configure(True,True)
        original=model.model.forward
        progress={'tokens':0,'next':8192,'start':0.}
        def tracked_forward(*a,**kw):
            if args.trace_memory and progress['tokens'] >= 65536:
                print(json.dumps({'starting_chunk':progress['tokens'],
                                  'allocated_mib':round(torch.cuda.memory_allocated()/2**20),
                                  'reserved_mib':round(torch.cuda.memory_reserved()/2**20)}),flush=True)
            output=original(*a,**kw)
            ids=kw.get('input_ids')
            if ids is not None and ids.shape[-1]>1:
                progress['tokens']+=ids.shape[-1]
                if progress['tokens']>=progress['next']:
                    print(json.dumps({'prefill_tokens':progress['tokens'],
                                      'elapsed_seconds':round(time.monotonic()-progress['start'],1),
                                      'allocated_mib':round(torch.cuda.memory_allocated()/2**20),
                                      'reserved_mib':round(torch.cuda.memory_reserved()/2**20)}),flush=True)
                    progress['next']+=8192
            return output
        model.model.forward=tracked_forward
        for window in args.lengths:
            model.clear_cache()
            torch.cuda.empty_cache()
            messages,actual=context_messages(model,window,96)
            progress.update(tokens=0,next=8192,start=time.monotonic())
            print(json.dumps({'starting_context':window,'actual_input_tokens':actual,'kv':args.kv}),flush=True)
            row=dict(window=window,actual_input_tokens=actual,kv=args.kv,expected=dict(FACTS))
            try:
                before = compressed_kv_info(model.model)['direct_decode_calls']
                text=model.answer(messages,max_new_tokens=96,temperature=0,on_text=lambda _:None)
                found={name:code in text.upper() for name,code in FACTS}
                calls = compressed_kv_info(model.model)['direct_decode_calls']-before
                if model.last_response_stats['output_tokens'] > 1 and not calls:
                    raise RuntimeError('Long context did not dispatch compressed-cache decode')
                row.update(text=text,found=found,all_found=all(found.values()),
                           stats=dict(model.last_response_stats),direct_decode_calls=calls,completed=True)
            except torch.OutOfMemoryError as exc:
                row.update(completed=False,error='CUDA out of memory',detail=str(exc))
                model.clear_cache()
                torch.cuda.empty_cache()
            report['contexts'].append(row);save();print(json.dumps(row,ensure_ascii=False),flush=True)
            if not row['completed']: break
        model.model.forward=original
        report['completed']=True
        save()


if __name__=='__main__': main()
