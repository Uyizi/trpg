"""Offline warmed A/B for sampler/input reuse/norm fusion on the trained model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--skip-thread-probe',action='store_true')
    parser.add_argument('--auto-sampling',action='store_true',help='Use validated sampling only for constrained requests')
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from models.inference_environment import configure_cuda_allocator
    configure_cuda_allocator()
    with (Path.home()/'.cache/qwen-debugger/gpu.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from models.pretrained import PretrainedChatModel
        from models.text_norm_fusion import configure_text_norm_fusion,text_norm_fusion_info
        from scripts.profile_phase2_runtime import context_messages
        started=time.monotonic()
        model=PretrainedChatModel(device='cuda',local_files_only=True,context_length=65536)
        report={'kind':'phase4-real-trained-model','training':False,'downloads':False,
                'load_seconds':round(time.monotonic()-started,3),'comparison':[],
                'cpu_threads':torch.get_num_threads(),'runtime':model.description}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        def save():args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)
        def configure(enabled):
            configure_text_norm_fusion(model.model,enabled)
            model.generator.optimized_sampling=None if enabled and args.auto_sampling else enabled
            model.generator.reuse_decode_buffers=enabled
            model.clear_cache();model.generator.prefix_pool.clear()
        emit({'loaded':True,'seconds':report['load_seconds']})
        schema={'type':'object','properties':{'answer':{'type':'string'}},
                'required':['answer'],'additionalProperties':False}
        cases=[
            {'name':'short_greedy','temperature':0.,'limit':64,
             'messages':[{'role':'system','content':'用简短中文直接回答。'},
                         {'role':'user','content':'用三句话解释KV缓存的作用。'}]},
            {'name':'constrained_sampled','temperature':.1,'limit':96,'schema':schema,
             'messages':[{'role':'system','content':'只返回JSON对象，answer字段放答案。不要输出其他字段、思考或Markdown。'},
                         {'role':'user','content':'用三句话解释上下文长度如何影响计算开销，每句15到25个汉字。'}]},
            {'name':'context_greedy','temperature':0.,'limit':64,
             'messages':context_messages(model,1536,96)[0]},
        ]
        for enabled in (False,True):
            configure(enabled)
            model.answer(cases[2]['messages'],max_new_tokens=8,temperature=0)
            model.answer(cases[1]['messages'],max_new_tokens=8,temperature=.1,response_schema=schema)
        for repeat in range(2):
            for case in cases:
                for enabled in ((False,True) if repeat==0 else (True,False)):
                    configure(enabled);torch.manual_seed(713+repeat)
                    chunks=[]
                    text=model.answer(case['messages'],max_new_tokens=case['limit'],temperature=case['temperature'],
                                      response_schema=case.get('schema'),on_text=chunks.append)
                    row={'repeat':repeat,'case':case['name'],'optimized':enabled,'text':text,
                         'stats':dict(model.last_response_stats)}
                    if 'schema' in case:row['parsed']=json.loads(text)
                    report['comparison'].append(row);save();emit(row)
        report['summary']={}
        for enabled in (False,True):
            selected=[r['stats'] for r in report['comparison'] if r['optimized']==enabled]
            seconds=sum(r['decode_seconds'] for r in selected);tokens=sum(r['output_tokens'] for r in selected)
            report['summary']['optimized' if enabled else 'phase3']={
                'tokens':tokens,'decode_seconds':round(seconds,3),'tokens_per_second':round(tokens/seconds,3),
                'peak_torch_mib':max(r['peak_torch_memory_mib'] for r in selected)}
        report['paired_text_matches']=[{'repeat':a['repeat'],'case':a['case'],'same':a['text']==b['text']}
                                     for a,b in zip(report['comparison'][::2],report['comparison'][1::2])]
        emit({'summary':report['summary'],'paired_text_matches':report['paired_text_matches']});save()
        # One CPU-thread change at a time, after all GPU paths are warmed. These
        # pairs do not measure AVX/CPU offloading; they measure dispatch overhead.
        configure(True);report['thread_pairs']=[]
        original_threads=torch.get_num_threads()
        for threads in (() if args.skip_thread_probe else dict.fromkeys((original_threads,1,4))):
            torch.set_num_threads(threads)
            model.answer(cases[0]['messages'],max_new_tokens=4,temperature=0)
            for repeat in range(2):
                model.clear_cache()
                text=model.answer(cases[0]['messages'],max_new_tokens=64,temperature=0)
                row={'threads':threads,'repeat':repeat,'text':text,'stats':dict(model.last_response_stats)}
                report['thread_pairs'].append(row);save();emit({'thread_probe':row})
        torch.set_num_threads(original_threads)
        report['norm_dispatch']=text_norm_fusion_info(model.model)
        report['runtime']=model.description;report['completed']=True;save()


if __name__=='__main__':main()
