"""Offline same-process target/MTP greedy comparison; owns exclusive GPU lease."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=48)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--modes',default='bf16,q8,q4')
    p.add_argument('--prompt',action='append')
    p.add_argument('--long-prompt-tokens',type=int,default=0)
    p.add_argument('--reference-verification',action='store_true',help='Disable MTP-only tiny expert/recurrent paths')
    p.add_argument('--mtp',type=Path,default=Path('/mnt/d/ai_project_data/qwen35_cloud/mtp/Qwen3.5-4B-official/model.safetensors'))
    args=p.parse_args()
    if args.tokens<1 or args.repeats<1 or args.long_prompt_tokens<0:p.error('Invalid profile budget')
    modes=args.modes.split(',')
    if any(m not in ('bf16','q8','q4') for m in modes):p.error('Unknown KV mode')
    report={'status':'blocked','training':False,'downloads':False,'model_loaded':False,
            'cuda_queried':False,'production_defaults_changed':False,'rows':[]}
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not args.mtp.is_file():
        report['blocker']='Official MTP weights are missing';save();return 2
    lock=Path.home()/'.cache/qwen-debugger/gpu.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a+') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            report['blocker']='Shared GPU lease is busy';save();return 2
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        from models.inference_environment import configure_cuda_allocator
        configure_cuda_allocator()
        import torch
        from models.pretrained import PretrainedChatModel
        from models.cached_generation import CachedGenerator
        from models.mtp_generation import MTPGenerator
        from models.compressed_kv_attention import configure_compressed_kv_attention
        torch.set_num_threads(1)
        try:
            report['cuda_queried']=True
            tick=time.monotonic()
            chat=PretrainedChatModel(device='cuda',local_files_only=True,context_length=65536,
                                     mtp=True,mtp_checkpoint=args.mtp)
            model=chat.model;report['model_loaded']=True
            report['status']='running'
            report['target_load_seconds']=time.monotonic()-tick
            head=chat.generator.head
            report['head_loaded_via_pretrained_opt_in']=chat.description['native_mtp']['enabled']
            report['target']=chat.description
            report['head_parameters']=sum(t.numel() for t in head.parameters())
            prompts=args.prompt or ['用三句话解释KV缓存的作用。','写一个Python函数，返回列表中不同元素的数量。']
            cases=[]
            for prompt in prompts:
                ids=torch.tensor([chat.encode([{'role':'system','content':'用简短中文直接回答。'},
                                               {'role':'user','content':prompt}])],device=chat.device)
                cases.append((prompt,ids))
            if args.long_prompt_tokens:
                sentence=chat.tokenizer.encode('档案记录：设备编号、人员分工和行动进度按时间更新。')
                pad=(sentence*((args.long_prompt_tokens//len(sentence))+1))[:args.long_prompt_tokens]
                tail=chat.encode([{'role':'user','content':'用三句话解释KV缓存的作用。'}])
                cases.append(('long-context',torch.tensor([pad+tail],device=chat.device)))
            with torch.inference_mode():
                for mode in modes:
                    configure_compressed_kv_attention(model,mode in ('q4','q8'))
                    normal=CachedGenerator(model,kv_mode=mode,chunk_size=1024)
                    mtp=MTPGenerator(model,head,kv_mode=mode,chunk_size=1024,
                                     fast_verification=not args.reference_verification)
                    # Warm both algorithms before collecting matched pairs.
                    for generator in (normal,mtp):
                        generator.clear();generator.generate(cases[0][1],8,temperature=0)
                    for repeat in range(args.repeats):
                        for label,ids in cases:
                            pair={'mode':mode,'repeat':repeat,'prompt':label,'input_tokens':ids.shape[1]}
                            order=(('normal',normal),('mtp',mtp)) if repeat%2==0 else (('mtp',mtp),('normal',normal))
                            outputs={}
                            for variant,generator in order:
                                generator.clear();torch.cuda.reset_peak_memory_stats(chat.device)
                                torch.cuda.synchronize(chat.device);tick=time.monotonic()
                                output,stats=generator.generate(ids,args.tokens,temperature=0)
                                torch.cuda.synchronize(chat.device)
                                elapsed=time.monotonic()-tick
                                generated=output[0,ids.shape[1]:].tolist();outputs[variant]=generated
                                pair[variant]={'seconds':elapsed,'tokens':len(generated),
                                    'tokens_per_second_including_prefill':len(generated)/elapsed,
                                    'stats':stats,'ids':generated,'text':chat.tokenizer.decode(generated,skip_special_tokens=True),
                                    'peak_torch_mib':torch.cuda.max_memory_allocated(chat.device)/2**20}
                            pair['identical_tokens']=outputs['normal']==outputs['mtp']
                            pair['decode_speed_ratio']=pair['mtp']['stats']['decode_tokens_per_second']/pair['normal']['stats']['decode_tokens_per_second']
                            report['rows'].append(pair);save()
                            print(json.dumps({k:v for k,v in pair.items() if k not in ('normal','mtp')},ensure_ascii=False),flush=True)
                    normal.clear();mtp.clear()
            report['status']='profiled'
            report['all_identical']=all(r['identical_tokens'] for r in report['rows'])
            report['timing_scope']='Same target/head resident, matched order alternation, wall-clock ordinary versus two-token verification; includes snapshot/replay/head costs'
        except Exception as exc:
            report.update(status='failed',error=type(exc).__name__,message=str(exc))
            save();raise
        save()
    return 0


if __name__=='__main__':raise SystemExit(main())
