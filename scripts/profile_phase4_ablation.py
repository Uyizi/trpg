"""Separate Phase4 candidates; fixed budgets and reversed measurement order."""
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
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from models.inference_environment import configure_cuda_allocator
    configure_cuda_allocator()
    with (Path.home()/'.cache/qwen-debugger/gpu.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import torch
        from models.pretrained import PretrainedChatModel
        from models.text_norm_fusion import configure_text_norm_fusion
        started=time.monotonic()
        model=PretrainedChatModel(device='cuda',local_files_only=True,context_length=65536)
        report={'load_seconds':round(time.monotonic()-started,3),'rows':[],'profiles':[],'threads':torch.get_num_threads()}
        def save():args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
        variants=[('phase3',False,False,False),('buffers',False,True,False),
                  ('sampling',False,False,True),('norm',True,False,False),
                  ('sampling_buffers',False,True,True),('all',True,True,True)]
        def configure(variant):
            _,norm,buffers,sampling=variant
            configure_text_norm_fusion(model.model,norm)
            model.generator.reuse_decode_buffers=buffers;model.generator.optimized_sampling=sampling
            model.clear_cache();model.generator.prefix_pool.clear()
        cases=[('greedy',0.,None,[{'role':'system','content':'直接回答，不作解释。'},
                                  {'role':'user','content':'从1开始按顺序写出所有正整数，直到100，用逗号隔开。'}]),
               ('json',.1,{'type':'object','properties':{'answer':{'type':'string','minLength':160}},'required':['answer'],'additionalProperties':False},
                [{'role':'system','content':'只返回JSON对象，answer字段写正文。不要写其他字段。'},
                 {'role':'user','content':'详细解释自回归生成和KV缓存，正文至少200个汉字。'}])]
        for variant in (variants[0],variants[-1]):
            configure(variant)
            for _,temp,schema,messages in cases:
                model.answer(messages,max_new_tokens=8,temperature=temp,response_schema=schema)
        for repeat in range(2):
            for kind,temp,schema,messages in cases:
                order=variants if repeat==0 else variants[::-1]
                for variant in order:
                    configure(variant);torch.manual_seed(213+repeat)
                    text=model.answer(messages,max_new_tokens=64,temperature=temp,response_schema=schema,on_text=lambda _:None)
                    row={'repeat':repeat,'case':kind,'mode':variant[0],'text':text,'truncated':model.last_response_truncated,'stats':dict(model.last_response_stats)}
                    report['rows'].append(row);save()
                    emit({'row':len(report['rows']),'case':kind,'mode':variant[0],'tokens':row['stats']['output_tokens'],'decode_tps':row['stats']['decode_tokens_per_second']})
        report['summary']={}
        for variant in variants:
            selected=[r['stats'] for r in report['rows'] if r['mode']==variant[0]]
            tokens=sum(s['output_tokens'] for s in selected);seconds=sum(s['decode_seconds'] for s in selected)
            report['summary'][variant[0]]={'tokens':tokens,'decode_seconds':round(seconds,3),'tps':round(tokens/seconds,3)}
        # Trace is collected separately from timings. One measured workload per
        # mode suffices to distinguish launch count from device compute time.
        for variant in (variants[0],variants[-1]):
            configure(variant)
            from torch.profiler import profile,ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as traced:
                model.answer(cases[0][3],max_new_tokens=8,temperature=0)
                torch.cuda.synchronize()
            kernels=[e for e in traced.events() if e.device_type==torch.autograd.DeviceType.CUDA]
            cpu=sorted(traced.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:12]
            report['profiles'].append({'mode':variant[0],'cuda_kernels':len(kernels),
                'kernel_total_us':sum(e.device_time_total for e in kernels),
                'top_cpu_self':[{'name':e.key,'count':e.count,'us':e.self_cpu_time_total} for e in cpu]})
            save()
        report['completed']=True;save();emit({'summary':report['summary']})


if __name__=='__main__':main()
