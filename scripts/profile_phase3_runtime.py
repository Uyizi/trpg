"""One offline model load: graph pairs, native grammar/prefix reuse and prefill sizing."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.profile_phase2_runtime import context_messages,FACTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from models.inference_environment import configure_cuda_allocator
    configure_cuda_allocator()
    with (Path.home()/'.cache/qwen-debugger/gpu.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from models.pretrained import PretrainedChatModel
        from models.cuda_graph_decode import cuda_graph_decode_info
        started=time.monotonic()
        model=PretrainedChatModel(device='cuda',local_files_only=True,context_length=65536,prefill_chunk_size=1024)
        report=dict(load_seconds=round(time.monotonic()-started,3),runtime=model.description,graph_pairs=[],prefix=[],prefill=[])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        def save():args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        def emit(value):print(json.dumps(value,ensure_ascii=False),flush=True)
        emit(dict(loaded=True,seconds=report['load_seconds']))
        manager=model.model._cuda_graph_decode
        # Probe-only switch keeps captured pools alive for fair warmed pairs.
        # Restore the exact bound wrapper so its live binding guards stay valid.
        def graphs(enabled):
            for layer in manager.layers:layer.module.forward=layer.wrapper if enabled else layer.original
        short=[
            [{'role':'system','content':'用简短中文直接回答。'},{'role':'user','content':'用三句话解释KV缓存的作用。'}],
            [{'role':'system','content':'直接回答。'},{'role':'user','content':'从1开始逐个列出整数，直到100，中间用逗号隔开。'}],
            context_messages(model,1536,96)[0]]
        for enabled in (False,True):
            graphs(enabled);model.clear_cache();model.answer(short[2],max_new_tokens=8,temperature=0)
        graph_info=cuda_graph_decode_info(model.model)
        emit(dict(graph_warmup=graph_info))
        if not graph_info['captured_layers']:
            raise RuntimeError('Real trained model captured no CUDA Graph layers: '+json.dumps(graph_info))
        for repeat in range(2):
            for case,messages in enumerate(short):
                for enabled in ((False,True) if repeat==0 else (True,False)):
                    graphs(enabled);model.clear_cache()
                    before=cuda_graph_decode_info(model.model)['replays']
                    text=model.answer(messages,max_new_tokens=64,temperature=0,on_text=lambda _:None)
                    row=dict(repeat=repeat,case=case,graphs=enabled,text=text,stats=dict(model.last_response_stats),
                             replay_delta=cuda_graph_decode_info(model.model)['replays']-before)
                    report['graph_pairs'].append(row);save();emit(row)
        report['same_paired_texts']=all(report['graph_pairs'][i]['text']==report['graph_pairs'][i+1]['text']
                                        for i in range(0,len(report['graph_pairs']),2))
        report['graph_summary']={}
        for enabled in (False,True):
            stats=[r['stats'] for r in report['graph_pairs'] if r['graphs']==enabled]
            tokens=sum(s['output_tokens'] for s in stats);seconds=sum(s['decode_seconds'] for s in stats)
            report['graph_summary']['graph' if enabled else 'eager']=dict(tokens=tokens,decode_seconds=round(seconds,3),
                tokens_per_second=round(tokens/seconds,3),peak_mib=max(s['peak_torch_memory_mib'] for s in stats))
        emit(dict(graph_summary=report['graph_summary'],same_paired_texts=report['same_paired_texts']));save()
        graphs(True)
        schema={'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}
        system='根据当前问题直接回答。仅返回JSON对象，answer字段放答案。不要添加分析、解释或其他字段。'
        for i,question in enumerate(('1+1等于几？answer只写一个数字。','2+2等于几？answer只写一个数字。')):
            messages=[{'role':'system','content':system},{'role':'user','content':question}]
            model.clear_cache()
            text=model.answer(messages,max_new_tokens=32,temperature=0,response_schema=schema,
                              prefix_cache_key='isolated-phase3:math:role',prefix_messages=1)
            parsed=json.loads(text)
            row=dict(index=i,question=question,text=text,parsed=parsed,stats=dict(model.last_response_stats))
            report['prefix'].append(row);save();emit(dict(prefix=row))
        if not report['prefix'][1]['stats']['fixed_prefix_hit']:
            raise RuntimeError('Real model did not reuse the exact fixed prefix')
        model.generator.prefix_pool.clear()
        for window,chunk in ((32768,1024),(32768,2048),(32768,4096),(65536,0)):
            model.clear_cache();torch.cuda.empty_cache();model.generator.chunk_size=chunk
            messages,actual=context_messages(model,window,96)
            emit(dict(starting_prefill=window,chunk=chunk,actual_input_tokens=actual))
            text=model.answer(messages,max_new_tokens=96,temperature=0,on_text=lambda _:None)
            row=dict(window=window,requested_chunk=chunk,actual_input_tokens=actual,text=text,
                     found={name:code in text.upper() for name,code in FACTS},stats=dict(model.last_response_stats))
            report['prefill'].append(row);save();emit(dict(prefill=row))
        report['completed']=True;report['runtime']=model.description;save()


if __name__=='__main__':main()
