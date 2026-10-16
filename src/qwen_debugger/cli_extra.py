"""CLI extensions kept separate from basic replay commands."""
from pathlib import Path
from .domain import RuntimeProfile,Budget


def register(sub):
    for name in ('compare','confirm','investigate'):
        p=sub.add_parser(name,help={'compare':'版本回归','confirm':'新数据独立确认','investigate':'固定因素调查'}[name])
        p.add_argument('--request-budget',type=int,default=64 if name=='confirm' else 48)
        p.add_argument('--token-budget',type=int,default=65536 if name=='confirm' else 24576)
        p.add_argument('--seconds',type=int,default=1800 if name=='confirm' else 1200)
        p.add_argument('--dry-run',action='store_true');p.add_argument('--queue',action='store_true');p.add_argument('--background',action='store_true')
        if name=='confirm':p.add_argument('--experiment',required=True);p.add_argument('--preset',choices=['quick','confirm32'],default='confirm32');p.add_argument('--factor')
        elif name=='compare':
            p.add_argument('--suite',required=True);p.add_argument('--profile-a',type=Path,required=True);p.add_argument('--profile-b',type=Path,required=True)
            p.add_argument('--comparison-type',choices=['complete_pipeline_comparison','controlled_checkpoint_comparison'],default='complete_pipeline_comparison')
            p.add_argument('--factors',nargs='*',default=[])
        else:
            p.add_argument('--case',required=True);p.add_argument('--profile',type=Path,required=True)
            p.add_argument('--factors',nargs='+',default=['rename','noise-position','insert-noise']);p.add_argument('--pairs',type=int,default=1)
    profiles=sub.add_parser('profiles',help='运行配置').add_subparsers(dest='action',required=True)
    profiles.add_parser('list');profiles.add_parser('import').add_argument('file',type=Path)
    suite=sub.add_parser('suite',help='固定回归套件').add_subparsers(dest='action',required=True)
    suite.add_parser('list');suite.add_parser('show').add_argument('name')
    p=suite.add_parser('create');p.add_argument('--name',required=True);p.add_argument('--cases',nargs='*',default=[])
    p.add_argument('--from-experiment');p.add_argument('--revision',type=int,default=1);p.add_argument('--purpose',choices=['development','heldout'],default='development')
    reviews=sub.add_parser('reviews',help='原始候选人工审核').add_subparsers(dest='action',required=True)
    reviews.add_parser('list');reviews.add_parser('show').add_argument('id')
    p=reviews.add_parser('propose');p.add_argument('--original',required=True);p.add_argument('--candidate',required=True);p.add_argument('--note',default='')
    p=reviews.add_parser('decide');p.add_argument('id');p.add_argument('--decision',choices=['accept','reject'],required=True)
    p.add_argument('--parent-hash',required=True);p.add_argument('--candidate-hash',required=True);p.add_argument('--actor',required=True);p.add_argument('--note',required=True)
    p=reviews.add_parser('run');p.add_argument('id');p.add_argument('--profile',required=True,type=Path);p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('consume',help='标记确认数据已用于开发选择');p.add_argument('--experiment',required=True);p.add_argument('--reason',required=True)
    p=sub.add_parser('restore',help='导入证据档案及案例/套件，不重启旧任务');p.add_argument('directory',type=Path)
    sub.add_parser('archives',help='已导入证据档案')
    p=sub.add_parser('worker',help='串行处理持久化任务队列');p.add_argument('--once',action='store_true')
    p=sub.add_parser('serve',help='本地网页工作台');p.add_argument('--host',choices=['127.0.0.1','localhost','::1'],default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
    p=sub.add_parser('fixtures',help='导入24个明确标记的合成案例');p.add_argument('--prefix',default='fixture')


def handle(store,args,show,read_json):
    from .service import create_experiment,run_job
    if args.command in ('compare','confirm','investigate'):
        data={'kind':args.command,'budget':{'requests':args.request_budget,'output_tokens':args.token_budget,'seconds':args.seconds}}
        if args.command=='confirm':data.update(experiment=args.experiment,preset=args.preset,factor=args.factor)
        elif args.command=='compare':data.update(suite=args.suite,profile_a=read_json(args.profile_a),profile_b=read_json(args.profile_b),comparison_type=args.comparison_type,factors=args.factors)
        else:data.update(case=args.case,profile=read_json(args.profile),factors=args.factors,pairs=args.pairs)
        show(create_experiment(store,data,dry_run=True))
        if args.dry_run:return True
        id=create_experiment(store,data);print('job_id: '+id,flush=True)
        if args.background:
            from .background import start_worker
            show(start_worker(store))
        elif not args.queue:run_job(store,id)
        from .background import status
        show(status(store,id))
    elif args.command=='profiles':
        if args.action=='import':
            from .fixed import freeze_profile
            p=RuntimeProfile.model_validate(read_json(args.file));h,identity=freeze_profile(store,p);show({'profile_hash':h,'identity':identity})
        else:show([{'hash':r['hash'],'profile':store.object(r['hash'])} for r in store.rows('objects') if r['kind']=='profile'])
    elif args.command=='suite':
        from .suites import create_suite,suite_get
        if args.action=='create':show(create_suite(store,args.name,args.cases,revision=args.revision,purpose=args.purpose,from_experiment=args.from_experiment))
        elif args.action=='show':show(suite_get(store,args.name))
        else:show([store.object(r['hash']) for r in store.rows('objects') if r['kind']=='suite'])
    elif args.command=='reviews':
        from .reviews import propose_review,decide_review,review_receipt,proposal
        if args.action=='propose':show({'review_id':propose_review(store,store.case(args.original),store.case(args.candidate),args.note)})
        elif args.action=='decide':show({'decision_hash':decide_review(store,args.id,args.decision,args.note,args.actor,parent_hash=args.parent_hash,candidate_hash=args.candidate_hash)})
        elif args.action=='show':show({'proposal':proposal(store,args.id),'receipt':review_receipt(store,args.id)})
        elif args.action=='run':
            data={'kind':'review','review':args.id,'profile':read_json(args.profile),'pairs':2,'budget':{'requests':4,'output_tokens':4096,'seconds':600}}
            show(create_experiment(store,data,dry_run=True))
            if not args.dry_run:id=create_experiment(store,data);print('job_id: '+id,flush=True);run_job(store,id)
        else:show([review_receipt(store,r['id']) for r in store.rows('objects') if r['kind']=='review_proposal'])
    elif args.command=='consume':
        from .confirmation import mark_consumed
        mark_consumed(store,args.experiment,args.reason);show({'experiment':args.experiment,'consumed':True})
    elif args.command=='restore':
        from .bundle import import_bundle
        show(import_bundle(store,args.directory))
    elif args.command=='archives':show([store.object(r['hash']) for r in store.rows('objects') if r['kind']=='archive'])
    elif args.command=='worker':
        from .background import work
        work(store,args.once)
    elif args.command=='serve':
        from .web import serve
        serve(store,args.host,args.port)
    elif args.command=='fixtures':
        from .fixtures import install_fixtures
        show(install_fixtures(store,args.prefix))
    else:return False
    return True
