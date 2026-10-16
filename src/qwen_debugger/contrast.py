"""Predeclared paired comparisons; one parent budget and one model worker."""
import time
from .domain import Budget,RuntimeProfile,digest,utcnow
from .inventory import inspect_local
from .runner import uid,get_job,manifest_for,create_job,execute_job,device_lease,recover_jobs
from .shrink import children_for,observations_for
from .contrast_transforms import contrast_variant,validate_contrast


def contrast_plan(case,profile,budget,*,factor,pairs=5,seed=42):
    candidate=contrast_variant(case,factor)
    if type(pairs) is not int or not 1<=pairs<=24: raise ValueError('pairs 必须为 1—24')
    if type(seed) is not int or not 0<=seed<2**63-pairs+1: raise ValueError('seed 超出合法范围')
    calls=pairs*2
    if budget.requests<calls or budget.output_tokens<calls*profile.max_new_tokens:
        raise ValueError('预算无法容纳全部配对；请明确调整 pairs 或预算')
    seeds=list(range(seed,seed+pairs))
    schedule=[{'seed':s,'arm':arm} for i,s in enumerate(seeds)
              for arm in (('original','candidate') if i%2==0 else ('candidate','original'))]
    return {'factor':factor,'pairs':pairs,'planned_observations':calls,'seeds':seeds,'schedule':schedule,
            'budget':budget.model_dump(),'per_request_output_limit':profile.max_new_tokens,
            'conservative_output_tokens_required':calls*profile.max_new_tokens,
            'original_case_hash':digest(case),'candidate_case_hash':digest(candidate),
            'case_hash':digest(case),'receipt':validate_contrast(case,candidate,factor),
            'conclusion_level':'exploratory','statistical_unit':'within_case_paired_observation',
            'independent_trials':None,'automatic_retries':0,'wall_time_includes_model_loading':True,
            'interpretation':'单题对照仅描述变化；不同种子在贪心解码下不保证独立样本。'}


def create_contrast(store,case,profile,budget,*,factor,pairs=5,seed=42):
    plan=contrast_plan(case,profile,budget,factor=factor,pairs=pairs,seed=seed)
    candidate=contrast_variant(case,factor)
    from .fixed import runtime_identity
    identity=runtime_identity(profile)
    if profile.backend=='local':
        profile=RuntimeProfile.model_validate({**profile.model_dump(mode='json'),'model_path':identity['model_path']})
    store.put_case(case);store.put_case(candidate)
    ph=store.put_object('profile',profile.id,profile.revision,profile.model_dump(mode='json'))
    receipt=plan.pop('receipt');rh=store.put_object('validity_receipt',digest(receipt),1,receipt)
    job=uid('contrast')
    manifest={**plan,'id':job,'kind':'contrast','schema_version':'1.0','state':'FROZEN','profile_hash':ph,
              'receipt_hash':rh,'identity':identity,'budget_account_id':job,'created':utcnow(),
              'selected_on_outcome':case['selected_on_outcome'],'purpose':'development_single_factor_contrast'}
    mh=store.put_object('manifest',job,1,manifest)
    with store.transaction() as db:
        db.execute('INSERT INTO jobs(id,manifest_hash,state,created) VALUES(?,?,?,?)',(job,mh,'PENDING',utcnow()))
    # Persist every planned observation before dispatch. A partial enqueue is never run.
    for slot in plan['schedule']:
        create_job(store,case if slot['arm']=='original' else candidate,profile,[slot['seed']],
                   Budget(requests=1,output_tokens=profile.max_new_tokens,seconds=budget.seconds),
                   parent_id=job,phase='contrast',arm=slot['arm'],receipt_hash=rh)
    return job


def contrast_result(store,job):
    rows=[r for r in store.rows('objects') if r['kind']=='contrast_result' and r['id']==job]
    return store.object(rows[0]['hash']) if rows else None


def execute_contrast(store,job,runtime):
    with device_lease() as fd:
        recover_jobs(store)
        m=manifest_for(store,job)
        if m['kind']!='contrast': raise ValueError('不是触发条件对照')
        prior=contrast_result(store,job)
        if prior is not None: return prior
        if get_job(store,job)['state']!='PENDING': raise ValueError('中断对照不自动重发；请导出已有结果并新建对照')
        if runtime.profile.model_dump(mode='json')!=store.object(m['profile_hash']): raise ValueError('profile 不匹配')
        if runtime.profile.backend=='local' and inspect_local(runtime.profile.model_path)!=m['identity']:
            raise ValueError('运行身份变化；请新建对照')
        links=children_for(store,job)
        actual=[{'seed':manifest_for(store,l['child_id'])['seeds'][0],'arm':l['arm']} for l in links]
        if actual!=m['schedule']: raise ValueError('配对计划未完整创建；请导出已有轨迹并新建对照')
        state='INCOMPLETE';reason=None
        with store.transaction() as db:
            db.execute("UPDATE jobs SET state='RUNNING',started=?,heartbeat=? WHERE id=?",(time.time(),utcnow(),job))
        try:
            for link in links:
                if get_job(store,job)['cancel']: state='CANCELLED';reason='用户取消';break
                child=link['child_id']
                execute_job(store,child,runtime,lease_fd=fd,close_runtime=False)
                obs=observations_for(store,child)
                if get_job(store,child)['state']!='COMPLETE' or any(o['state']!='DONE' for o in obs):
                    reason='子请求未正常完成；停止后续调用，保留未运行观察';break
            else: state='COMPLETE'
        except (Exception,KeyboardInterrupt) as exc:
            reason=type(exc).__name__+': '+str(exc)
        finally: runtime.close()
        if get_job(store,job)['cancel']:state='CANCELLED'
        result={'id':job,'state':state,'reason':reason,'factor':m['factor'],
                'conclusion_level':'exploratory','created':utcnow()}
        store.put_object('contrast_result',job,1,result)
        with store.transaction() as db:
            db.execute('UPDATE jobs SET state=?,heartbeat=? WHERE id=?',(state,utcnow(),job))
        return result
