"""Immutable suite revisions with explicit root/family grouping and profile differences."""
from .domain import digest,utcnow
from .fixed import create_fixed


def suite_get(store,name,revision=None):
    rows=[r for r in store.rows('objects') if r['kind']=='suite' and r['id']==name and (revision is None or r['revision']==revision)]
    if not rows:raise ValueError('套件不存在')
    return store.object(max(rows,key=lambda r:r['revision'])['hash'])


def create_suite(store,name,case_ids,*,purpose='development',revision=1,from_experiment=None):
    from .domain import Name
    from pydantic import TypeAdapter
    TypeAdapter(Name).validate_python(name)
    if purpose not in ('development','heldout'):raise ValueError('无效数据用途')
    cases=[store.case(i) if isinstance(i,str) else store.case(i['id'],i.get('revision')) for i in case_ids]
    if from_experiment:
        from .runner import manifest_for
        m=manifest_for(store,from_experiment)
        hashes={s['case_hash'] for s in m.get('schedule',[]) if 'case_hash' in s}
        hashes.update(m[k] for k in ('case_hash','original_case_hash','candidate_case_hash') if m.get(k))
        from .shrink import result_for
        r=result_for(store,from_experiment)
        if r:hashes.add(r['final_case_hash'])
        cases.extend(store.object(h) for h in sorted(hashes))
    cases=list({digest(c):c for c in cases}.values())
    if not cases:raise ValueError('套件至少包含一个案例')
    if len(cases)>32:raise ValueError('首版套件最多32个案例，保证双臂不超过64次')
    families={};roots={}
    for c in cases:
        families.setdefault(c['family_id'],set()).add(c['split']);roots.setdefault(c['root_case_id'],set()).add(c['split'])
    if any(len(s)>1 for s in [*families.values(),*roots.values()]):raise ValueError('同根或同 family 不能跨 development/heldout')
    if purpose=='heldout':
        if any(c['split']!='heldout' or c['selected_on_outcome'] for c in cases):raise ValueError('独立评估不得包含开发案例或按失败挑选案例')
        used={}
        for row in store.rows('objects'):
            if row['kind']=='case':
                c=store.object(row['hash'])
                if c['split']=='development':used[c['family_id']]=True
        if set(families)&set(used):raise ValueError('这些 family 已用于开发，不能标为独立家族评估')
    parent=None
    if revision>1:
        old=suite_get(store,name,revision-1);parent=digest(old)
    suite={'id':name,'revision':revision,'parent_hash':parent,'purpose':purpose,'created':utcnow(),
           'case_hashes':[digest(c) for c in cases],'family_ids':sorted(families),'root_case_ids':sorted(roots),
           'grouping_evidence':'user-declared source/template families; not statistical independence proof',
           'source_experiment':from_experiment}
    from .integrity import pool_from_rows,validate_suite
    validate_suite(suite,pool_from_rows(store.rows('objects')))
    store.put_object('suite',name,revision,suite)
    if from_experiment:
        from .confirmation import mark_consumed
        mark_consumed(store,from_experiment,'suite:'+name)
    return suite


def profile_diff(a,b):
    x=a.model_dump(mode='json');y=b.model_dump(mode='json')
    return {k:{'a':x.get(k),'b':y.get(k)} for k in sorted(x.keys()|y.keys()) if k not in ('id','revision') and x.get(k)!=y.get(k)}


def compare_plan(store,suite,a,b,budget,*,comparison_type='complete_pipeline_comparison',factors=None,seed=42):
    from .integrity import pool_from_rows,validate_suite
    validate_suite(suite,pool_from_rows(store.rows('objects')))
    if comparison_type not in ('complete_pipeline_comparison','controlled_checkpoint_comparison'):raise ValueError('未知比较类型')
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('无效种子')
    diffs=profile_diff(a,b);factors=factors or []
    if comparison_type=='controlled_checkpoint_comparison' and set(diffs)!=set(factors):
        raise ValueError('受控比较的声明因素必须恰好覆盖全部 profile 差异')
    if comparison_type=='controlled_checkpoint_comparison' and ('api' in (a.backend,b.backend)):
        raise ValueError('API 权重/模板身份部分未知，使用整套方案比较')
    n=len(suite['case_hashes']);tokens=n*(a.max_new_tokens+b.max_new_tokens)
    if 2*n>budget.requests or tokens>budget.output_tokens:raise ValueError('预算不能容纳全部套件双臂')
    return {'suite_hash':digest(suite),'comparison_type':comparison_type,'declared_factors':factors,
            'profile_differences':diffs,'planned_observations':2*n,'output_tokens_reserved':tokens,
            'budget':budget.model_dump(),'seed':seed,'purpose':suite['purpose'],
            'execution_order':'profile A block, then profile B block; avoids simultaneous model residency; time/order is a limitation',
            'conclusion_level':'exploratory'}


def create_comparison(store,suite,a,b,budget,**options):
    plan=compare_plan(store,suite,a,b,budget,**options)
    cases={h:store.object(h) for h in suite['case_hashes']};slots=[]
    for arm in ('original','candidate'):
        for h in cases:slots.append({'case':h,'profile':arm,'arm':arm,'seed':plan['seed'],'pair_id':h})
    return create_fixed(store,kind='regression',profiles={'original':a,'candidate':b},cases=cases,slots=slots,budget=budget,metadata=plan)
