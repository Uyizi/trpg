"""CLI for the same core that a later web UI will call."""
import argparse
import json
import os
import sys
from pathlib import Path
from .domain import CaseImport,RuntimeProfile,Budget,canonical
from .inventory import doctor,PROJECT
from .storage import Store
from .tasks import compile_case
from .runner import create_job,execute_job,cancel_job,recover_jobs,device_lease,get_job,manifest_for
from .runtime import MockRuntime,LocalRuntime
from .report import export_report
from .shrink import shrink_plan,create_investigation,execute_investigation,children_for,result_for,steps_for
from .contrast import contrast_plan,create_contrast,execute_contrast,contrast_result


def read_json(path):
    p=Path(path)
    if p.stat().st_size>4*1024*1024: raise ValueError('导入文件超过 4 MiB')
    return json.loads(p.read_text(encoding='utf-8'))


def show(obj): print(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


def parser():
    p=argparse.ArgumentParser(prog='qdebug',description='本地模型行为调试器：重放、有效缩减与证据')
    p.add_argument('--data-dir',type=Path,default=Path(os.environ.get('QDEBUG_DATA_DIR',PROJECT/'data/debugger')))
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='只读环境盘点，不加载模型')
    cases=sub.add_parser('cases',help='案例库').add_subparsers(dest='action',required=True)
    cases.add_parser('import').add_argument('file',type=Path)
    cases.add_parser('list')
    c=cases.add_parser('show'); c.add_argument('id'); c.add_argument('--revision',type=int)
    replay=sub.add_parser('replay',help='有预算的重放；默认同种子三次')
    replay.add_argument('--case',required=True)
    replay.add_argument('--revision',type=int)
    replay.add_argument('--profile',type=Path,required=True)
    replay.add_argument('--repeats',type=int,default=3)
    replay.add_argument('--seed',type=int,default=42)
    replay.add_argument('--request-budget',type=int,default=3)
    replay.add_argument('--token-budget',type=int,default=3072)
    replay.add_argument('--seconds',type=int,default=600)
    replay.add_argument('--dry-run',action='store_true')
    replay.add_argument('--queue',action='store_true',help='只创建持久化任务，稍后 jobs run')
    shrink=sub.add_parser('shrink',help='保持题意的背景句缩减与新种子对照')
    shrink.add_argument('--case',required=True);shrink.add_argument('--profile',type=Path,required=True)
    shrink.add_argument('--revision',type=int);shrink.add_argument('--preset',choices=['quick'],default='quick')
    shrink.add_argument('--request-budget',type=int,default=48);shrink.add_argument('--token-budget',type=int,default=24576)
    shrink.add_argument('--seconds',type=int,default=1200);shrink.add_argument('--dry-run',action='store_true')
    shrink.add_argument('--queue',action='store_true')
    contrast=sub.add_parser('contrast',help='单因素配对：名字替换或背景位置')
    contrast.add_argument('--case',required=True);contrast.add_argument('--profile',type=Path,required=True)
    contrast.add_argument('--revision',type=int)
    from .semantic_transforms import FACTORS
    contrast.add_argument('--factor',choices=FACTORS,required=True)
    contrast.add_argument('--pairs',type=int,default=5);contrast.add_argument('--seed',type=int,default=42)
    contrast.add_argument('--request-budget',type=int,default=10);contrast.add_argument('--token-budget',type=int,default=10240)
    contrast.add_argument('--seconds',type=int,default=600);contrast.add_argument('--dry-run',action='store_true')
    contrast.add_argument('--queue',action='store_true')
    jobs=sub.add_parser('jobs',help='查看、取消及运行任务').add_subparsers(dest='action',required=True)
    jobs.add_parser('list'); jobs.add_parser('recover',help='崩溃恢复为完成未知，不自动重发')
    for action in ['show','cancel','run','resume','start']: jobs.add_parser(action).add_argument('id')
    exp=sub.add_parser('export',help='导出离线 HTML 和原始证据')
    exp.add_argument('--experiment',required=True); exp.add_argument('--output',required=True,type=Path)
    from .cli_extra import register
    register(sub)
    return p


def run_existing(store,id):
    from .service import run_job
    run_job(store,id)
    show(get_job(store,id))


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='doctor': show(doctor()); return 0
        store=Store(args.data_dir)
        from .cli_extra import handle
        if handle(store,args,show,read_json):return 0
        if args.command=='cases':
            if args.action=='import':
                case=compile_case(CaseImport.model_validate(read_json(args.file)))
                h=store.put_case(case); show({'case':case['case_id'],'revision':case['revision'],'hash':h,'solver':case['solver']})
            elif args.action=='show': show(store.case(args.id,args.revision))
            else:
                show([{'id':r['id'],'revision':r['revision'],'hash':r['hash']} for r in store.rows('objects') if r['kind']=='case'])
        elif args.command=='replay':
            case=store.case(args.case,args.revision); profile=RuntimeProfile.model_validate(read_json(args.profile))
            budget=Budget(requests=args.request_budget,output_tokens=args.token_budget,seconds=args.seconds)
            if not 1<=args.repeats<=48 or not 0<=args.seed<2**63: raise ValueError('repeats 必须为 1—48；seed 为合法非负整数')
            plan={'case':case['case_id'],'case_revision':case['revision'],'backend':profile.backend,
                  'profile':profile.id,'planned_observations':args.repeats,'seed':args.seed,
                  'budget':budget.model_dump(),'per_request_output_limit':profile.max_new_tokens,
                  'conservative_output_tokens_required':args.repeats*profile.max_new_tokens,
                  'can_fit_full_plan':args.repeats<=budget.requests and args.repeats*profile.max_new_tokens<=budget.output_tokens,
                  'wall_time_includes_model_loading':True,'input_tokens':'checked by actual tokenizer before generation',
                  'statistics':'相同种子重复只统计观察次数，不作为独立样本'}
            show(plan)
            if args.dry_run: return 0
            job=create_job(store,case,profile,[args.seed]*args.repeats,budget)
            print('job_id: '+job,flush=True)
            if not args.queue: run_existing(store,job)
        elif args.command=='shrink':
            case=store.case(args.case,args.revision);profile=RuntimeProfile.model_validate(read_json(args.profile))
            budget=Budget(requests=args.request_budget,output_tokens=args.token_budget,seconds=args.seconds)
            show(shrink_plan(case,profile,budget))
            if args.dry_run: return 0
            job=create_investigation(store,case,profile,budget)
            print('job_id: '+job,flush=True)
            if not args.queue:
                run_existing(store,job)
                show(result_for(store,job))
        elif args.command=='contrast':
            case=store.case(args.case,args.revision);profile=RuntimeProfile.model_validate(read_json(args.profile))
            budget=Budget(requests=args.request_budget,output_tokens=args.token_budget,seconds=args.seconds)
            options={'factor':args.factor,'pairs':args.pairs,'seed':args.seed}
            show(contrast_plan(case,profile,budget,**options))
            if args.dry_run: return 0
            job=create_contrast(store,case,profile,budget,**options)
            print('job_id: '+job,flush=True)
            if not args.queue:
                run_existing(store,job);show(contrast_result(store,job))
        elif args.command=='jobs':
            if args.action=='list': show(store.rows('jobs'))
            elif args.action=='show':
                from .background import status
                show({'job':status(store,args.id),'observations':[o for o in store.rows('observations') if o['job_id']==args.id],
                      'children':children_for(store,args.id),'shrink_result':result_for(store,args.id),
                      'shrink_steps':steps_for(store,args.id),'contrast_result':contrast_result(store,args.id)})
            elif args.action=='cancel': cancel_job(store,args.id); show(get_job(store,args.id))
            elif args.action=='start':
                from .background import start_worker
                m=manifest_for(store,args.id)
                if m.get('parent_manifest_id') or get_job(store,args.id)['state']!='PENDING':raise ValueError('仅启动待运行的父任务')
                show(start_worker(store))
            elif args.action=='recover':
                with device_lease(): recover_jobs(store)
                show(store.rows('jobs'))
            elif args.action=='resume':
                # Only unsent observations can resume, using the original budget and deadline.
                with device_lease():
                    manifest=manifest_for(store,args.id)
                    if manifest['kind']!='observation_batch' or manifest.get('parent_manifest_id'):
                        raise ValueError('调查、对照及其子任务不能自动续跑；可导出轨迹并新建实验')
                    job=get_job(store,args.id)
                    if job['state'] not in ('INCOMPLETE','CANCELLED'): raise ValueError('只有未完成或取消的任务可以续跑')
                    with store.transaction() as db:
                        db.execute("UPDATE jobs SET state='PENDING',cancel=0 WHERE id=?",(args.id,))
                run_existing(store,args.id)
            else: run_existing(store,args.id)
        elif args.command=='export':
            summary=export_report(store,args.experiment,args.output)
            show({'report':str(args.output.resolve()/'report.html'),'counts':summary['counts'],'planned':summary['planned']})
        return 0
    except (ValueError,OSError,RuntimeError) as exc:
        print('qdebug: '+str(exc),file=sys.stderr); return 2


if __name__=='__main__': raise SystemExit(main())
