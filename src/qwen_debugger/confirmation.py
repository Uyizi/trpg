"""New confirmation data are never recycled search observations."""
import secrets
from .domain import RuntimeProfile,digest,utcnow
from .runner import manifest_for
from .shrink import result_for,children_for
from .transforms import validate_deletion
from .fixed import create_fixed
from .validity import check_receipt


def mark_consumed(store,experiment,reason):
    manifest_for(store,experiment)
    if not reason.strip():raise ValueError('需说明数据用于哪次开发选择')
    record={'experiment':experiment,'reason':reason,'created':utcnow()}
    store.put_object('data_consumption',digest(record),1,record)


def confirmation_source(store,experiment,factor=None):
    m=manifest_for(store,experiment)
    if m['kind']=='contrast':
        rh=m['receipt_hash'];a,b=check_receipt(store,store.object(rh));p=m['profile_hash']
    elif m['kind']=='investigation':
        r=result_for(store,experiment)
        if factor:
            selected=[t for t in m.get('trigger_plans',[]) if t['factor']==factor]
            if len(selected)!=1:raise ValueError('该调查没有这个触发因素')
            rh=selected[0]['receipt_hash'];a,b=check_receipt(store,store.object(rh));p=m['profile_hash']
        else:
            if not r or r['final_case_hash']==r['original_case_hash']:raise ValueError('尚未选定有效缩减候选；可用 --factor 选择已运行的触发候选')
            a=store.object(r['original_case_hash']);b=store.object(r['final_case_hash'])
            receipt=validate_deletion(a,b)
            if receipt['status']!='rule_verified':raise ValueError('最终候选未获认证')
            rh=store.put_object('validity_receipt',digest(receipt),1,receipt);p=m['profile_hash']
    elif m['kind']=='matrix':
        pairs=m['metadata'].get('pairs',[])
        if len(pairs)!=1:raise ValueError('请先通过单因素 contrast 或单候选审核对照选定唯一候选')
        rh=pairs[0]['receipt_hash'];a,b=check_receipt(store,store.object(rh));p=m['profile_hash']
    else:raise ValueError('confirm 仅接受已选定候选的探索实验')
    observed=set()
    for link in children_for(store,experiment):
        if any(o['job_id']==link['child_id'] and o['state']=='DONE' for o in store.rows('observations')):
            observed.add(manifest_for(store,link['child_id'])['case_hash'])
    if not {digest(a),digest(b)}<=observed:raise ValueError('原题和选定候选尚未都取得探索响应')
    return m,a,b,p,rh


def confirmation_plan(store,experiment,budget,*,preset='confirm32',factor=None):
    m,a,b,ph,rh=confirmation_source(store,experiment,factor)
    pairs=32 if preset=='confirm32' else 5 if preset=='quick' else 0
    if not pairs:raise ValueError('确认预设必须 quick 或 confirm32')
    profile=RuntimeProfile.model_validate(store.object(ph));calls=2*pairs
    if budget.requests<calls or budget.output_tokens<calls*profile.max_new_tokens:raise ValueError('预算无法容纳确认计划')
    # No backend has a verified independent seed capability yet. Do not take declarations as proof.
    rule={'mode':'observed_recurrence' if pairs==32 else 'descriptive','pairs':pairs,
          'all_completed_evaluable':True,'each_arm_target_failures_required':pairs,
          'seed_prerequisite':'verified independent sampling and non-greedy decoding',
          'seed_capability':'unverified','eligible_for_upgrade':False,'allowed_conclusion':'exploratory'}
    return {'source_experiment':experiment,'original_case_hash':digest(a),'candidate_case_hash':digest(b),
            'profile_hash':ph,'receipt_hash':rh,'pairs':pairs,'preset':preset,'budget':budget.model_dump(),
            'planned_observations':calls,'conservative_output_tokens_required':calls*profile.max_new_tokens,
            'confirmation_rule':rule,'failure_predicate':{'outcome':'FAIL','failure_kind':'semantic_mismatch'},
            'purpose':'fresh_within_case_confirmation','warning':'当前种子能力未独立验证；只报告实际复现次数，不自动晋级。'}


def create_confirmation(store,experiment,budget,*,preset='confirm32',factor=None):
    plan=confirmation_plan(store,experiment,budget,preset=preset,factor=factor)
    previous={o['seed'] for o in store.rows('observations')}
    seeds=[]
    while len(seeds)<plan['pairs']:
        s=secrets.randbelow(2**31)
        if s not in previous and s not in seeds:seeds.append(s)
    plan['seeds']=seeds
    p=RuntimeProfile.model_validate(store.object(plan['profile_hash']));slots=[]
    for i,s in enumerate(seeds):
        for arm in (('original','candidate') if i%2==0 else ('candidate','original')):
            slots.append({'case':arm,'profile':'shared','arm':arm,'seed':s,'pair_id':str(s),'receipt_hash':plan['receipt_hash']})
    cases={arm:store.object(plan[arm+'_case_hash']) for arm in ('original','candidate')}
    return create_fixed(store,kind='confirmation',profiles={'shared':p},cases=cases,slots=slots,budget=budget,metadata=plan)
