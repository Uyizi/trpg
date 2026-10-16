"""Application services shared by CLI and web. No rendering or model oracle here."""
from .domain import CaseImport,RuntimeProfile,Budget
from .tasks import compile_case
from .runner import manifest_for,execute_job,create_job
from .shrink import create_investigation,execute_investigation,shrink_plan
from .contrast import create_contrast,execute_contrast,contrast_plan
from .fixed import KINDS,execute_fixed,adapter,freeze_profile


def run_job(store,id):
    m=manifest_for(store,id)
    if m['kind'] in KINDS:return execute_fixed(store,id)
    p=RuntimeProfile.model_validate(store.object(m['profile_hash']));runtime=adapter(store,id,p)
    if m['kind']=='investigation':return execute_investigation(store,id,runtime)
    if m['kind']=='contrast':return execute_contrast(store,id,runtime)
    return execute_job(store,id,runtime)


def get_profile(store,ref):
    if isinstance(ref,dict):return RuntimeProfile.model_validate(ref)
    rows=[r for r in store.rows('objects') if r['kind']=='profile' and (r['hash']==ref or r['id']==ref)]
    if not rows:raise ValueError('请先导入运行配置')
    row=max(rows,key=lambda r:r['revision']);return RuntimeProfile.model_validate(store.object(row['hash']))


def create_experiment(store,data,*,dry_run=False):
    kind=data['kind'];budget=Budget.model_validate(data.get('budget',{}))
    if kind=='confirm':
        from .confirmation import confirmation_plan,create_confirmation
        fn=confirmation_plan if dry_run else create_confirmation
        return fn(store,data['experiment'],budget,preset=data.get('preset','confirm32'),factor=data.get('factor') or None)
    if kind=='compare':
        from .suites import suite_get,compare_plan,create_comparison
        suite=suite_get(store,data['suite'],data.get('suite_revision'))
        a=get_profile(store,data['profile_a']);b=get_profile(store,data['profile_b'])
        fn=compare_plan if dry_run else create_comparison
        return fn(store,suite,a,b,budget,comparison_type=data.get('comparison_type','complete_pipeline_comparison'),factors=data.get('factors',[]))
    p=get_profile(store,data['profile'])
    if kind=='probe':
        case=compile_case(CaseImport.model_validate({'case_id':'termination_probe_v1','revision':1,
            'root_case_id':'termination_probe_v1','family_id':'runtime_probe','provenance':{'kind':'synthetic'},
            'selected_on_outcome':False,'input':{'kind':'raw','messages':[{'role':'user','content':'只输出 OK，不要其他内容。'}]},
            'gold':{'kind':'exact_text','expected':'OK','normalizers':['strip_outer_whitespace'],'source':'unknown'}}))
        if budget.requests<1 or budget.output_tokens<p.max_new_tokens:raise ValueError('预算不能容纳原 profile 的输出上限')
        if dry_run:return {'probe':'short termination','budget':budget.model_dump(),'profile':p.model_dump(mode='json'),
                           'seed_verification':'not performed; one response cannot verify independent sampling'}
        return create_job(store,case,p,[42],budget)
    if kind=='review':
        from .reviews import review_receipt
        from .validity import check_receipt
        from .investigate import create_review_comparison
        r=review_receipt(store,data['review']);check_receipt(store,r)
        pairs=int(data.get('pairs',2))
        if budget.requests<2*pairs or budget.output_tokens<2*pairs*p.max_new_tokens:raise ValueError('预算不足')
        if dry_run:return {'budget':budget.model_dump(),'pairs':pairs,'receipt':r}
        return create_review_comparison(store,data['review'],p,budget,pairs=pairs)
    case=store.case(data['case'],data.get('revision'))
    if kind=='replay':
        repeats=int(data.get('repeats',3));seed=int(data.get('seed',42))
        if not 1<=repeats<=48 or not 0<=seed<2**63:raise ValueError('无效重复数或种子')
        if repeats>budget.requests or repeats*p.max_new_tokens>budget.output_tokens:raise ValueError('预算不能容纳重放计划')
        if dry_run:return {'budget':budget.model_dump(),'repeats':repeats,'seed':seed,'case_hash':case['content_hash'],'profile':p.model_dump(mode='json')}
        return create_job(store,case,p,[seed]*repeats,budget)
    if kind=='shrink':return (shrink_plan(case,p,budget) if dry_run else create_investigation(store,case,p,budget))
    if kind=='contrast':
        kwargs={'factor':data['factor'],'pairs':int(data.get('pairs',5)),'seed':int(data.get('seed',42))}
        return (contrast_plan if dry_run else create_contrast)(*(([case,p,budget]) if dry_run else [store,case,p,budget]),**kwargs)
    if kind=='investigate':
        if case['input'].get('task_type')=='closed_world_owner_v1':
            from .workflow import full_plan,create_workflow
            kwargs={'factors':data.get('factors'),'pairs':int(data.get('pairs',1)),'seed':int(data.get('seed',100))}
            return (full_plan if dry_run else create_workflow)(*(([case,p,budget]) if dry_run else [store,case,p,budget]),**kwargs)
        from .investigate import investigation_plan,create_matrix
        kwargs={'factors':data.get('factors'),'pairs':int(data.get('pairs',2)),'seed':int(data.get('seed',100))}
        return (investigation_plan if dry_run else create_matrix)(*(([case,p,budget]) if dry_run else [store,case,p,budget]),**kwargs)
    raise ValueError('未知实验类型')
