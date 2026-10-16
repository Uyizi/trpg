"""Fixed multi-case/profile experiments, all slots frozen before any inference."""
import time
from .domain import Budget,RuntimeProfile,digest,utcnow
from .inventory import inspect_local
from .runner import uid,manifest_for,get_job,create_job,execute_job,device_lease,recover_jobs
from .shrink import children_for,observations_for
from .runtime import MockRuntime,LocalRuntime

KINDS=('regression','confirmation','matrix')


def runtime_identity(profile):
    if profile.backend=='local':return inspect_local(profile.model_path)
    if profile.backend=='api':return {'backend':'api','endpoint':profile.api_base,'model':profile.api_model,'observability':'partial; service declarations only'}
    return {'backend':'mock','identity':'synthetic'}


def freeze_profile(store,profile):
    identity=runtime_identity(profile)
    if profile.backend=='local':profile=RuntimeProfile.model_validate({**profile.model_dump(mode='json'),'model_path':identity['model_path']})
    return store.put_object('profile',profile.id,profile.revision,profile.model_dump(mode='json')),identity


def adapter(store,job,profile):
    if profile.backend=='mock':return MockRuntime(profile)
    if profile.backend=='api':
        from .api_runtime import ApiRuntime
        return ApiRuntime(profile)
    return LocalRuntime(profile,store.root/'logs'/f'{job}.log')


def validate_slot(store,parent,case,seeds,phase,arm,receipt_hash,confirmation_id,profile_hash=None):
    slots=[s for s in parent['schedule'] if s['case_hash']==digest(case) and s['arm']==arm and seeds==[s['seed']]]
    if profile_hash is not None:slots=[s for s in slots if s['profile_hash']==profile_hash]
    if phase!='fixed' or len(slots)!=1 or confirmation_id is not None or slots[0].get('receipt_hash')!=receipt_hash:
        raise ValueError('子观察与冻结计划不匹配')
    if receipt_hash:
        receipt=store.object(receipt_hash)
        if receipt['status'] not in ('rule_verified','human_confirmed'):raise ValueError('变换尚未认证')
        from .validity import check_receipt
        check_receipt(store,receipt)


def create_fixed(store,*,kind,profiles,cases,slots,budget,metadata):
    if kind not in KINDS:raise ValueError('未知固定实验类型')
    if not slots or len(slots)>budget.requests:raise ValueError('请求预算不足')
    required=sum(profiles[s['profile']].max_new_tokens for s in slots)
    if required>budget.output_tokens:raise ValueError('输出 token 预算不足以容纳完整计划')
    pairs=[(s['pair_id'],s['arm']) for s in slots]
    if len(set(pairs))!=len(pairs):raise ValueError('重复配对处理臂')
    frozen={k:freeze_profile(store,p) for k,p in profiles.items()}
    ph={k:v[0] for k,v in frozen.items()};identities={h:ident for h,ident in frozen.values()}
    ch={k:store.put_case(c) for k,c in cases.items()}
    schedule=[{'pair_id':s['pair_id'],'arm':s['arm'],'seed':s['seed'],'case_hash':ch[s['case']],
               'profile_hash':ph[s['profile']],'receipt_hash':s.get('receipt_hash')} for s in slots]
    # Ambiguous arm/case/seed would weaken the child authorization contract.
    if len({(s['arm'],s['case_hash'],s['seed']) for s in schedule})!=len(schedule):raise ValueError('配对槽位存在歧义')
    job=uid(kind);m={'schema_version':'1.0','id':job,'kind':kind,'state':'FROZEN','budget_account_id':job,
                    'budget':budget.model_dump(),'profile_hash':next(iter(ph.values())),'profile_hashes':ph,
                    'identities':identities,'schedule':schedule,'metadata':metadata,'created':utcnow(),
                    'planned_observations':len(schedule),'purpose':metadata.get('purpose','development'),
                    'conclusion_level':'exploratory','automatic_retries':0}
    mh=store.put_object('manifest',job,1,m)
    with store.transaction() as db:db.execute('INSERT INTO jobs(id,manifest_hash,state,created) VALUES(?,?,?,?)',(job,mh,'PENDING',utcnow()))
    for s in schedule:
        p=RuntimeProfile.model_validate(store.object(s['profile_hash']))
        create_job(store,store.object(s['case_hash']),p,[s['seed']],
                   Budget(requests=1,output_tokens=p.max_new_tokens,seconds=budget.seconds),parent_id=job,
                   phase='fixed',arm=s['arm'],receipt_hash=s.get('receipt_hash'))
    return job


def fixed_result(store,job):
    rows=[r for r in store.rows('objects') if r['kind']=='fixed_result' and r['id']==job]
    return store.object(rows[0]['hash']) if rows else None


def execute_fixed(store,job,*,factory=adapter):
    with device_lease() as fd:
        recover_jobs(store);m=manifest_for(store,job)
        if m['kind'] not in KINDS:raise ValueError('不是固定实验')
        if fixed_result(store,job):return fixed_result(store,job)
        if get_job(store,job)['state']!='PENDING':raise ValueError('中断实验不自动重发；可导出证据并新建实验')
        links=children_for(store,job)
        if len(links)!=len(m['schedule']):raise ValueError('计划未完整创建，不执行部分计划')
        runtime=None;active=None;state='INCOMPLETE';error=None
        with store.transaction() as db:db.execute("UPDATE jobs SET state='RUNNING',started=?,heartbeat=? WHERE id=?",(time.time(),utcnow(),job))
        try:
            for link,slot in zip(links,m['schedule']):
                cm=manifest_for(store,link['child_id'])
                if (cm['case_hash'],cm['profile_hash'],cm['arms'][0],cm['seeds'])!=(slot['case_hash'],slot['profile_hash'],slot['arm'],[slot['seed']]):raise ValueError('冻结顺序不匹配')
                if get_job(store,job)['cancel']:state='CANCELLED';break
                if active!=slot['profile_hash']:
                    if runtime:runtime.close()
                    runtime=factory(store,job,RuntimeProfile.model_validate(store.object(slot['profile_hash'])))
                    active=slot['profile_hash']
                execute_job(store,link['child_id'],runtime,lease_fd=fd,close_runtime=False)
                if get_job(store,link['child_id'])['state']!='COMPLETE' or any(o['state']!='DONE' for o in observations_for(store,link['child_id'])):
                    error='运行异常或未完成，后续观察保留未运行';break
            else:state='COMPLETE'
        except (Exception,KeyboardInterrupt) as exc:error=type(exc).__name__+': '+str(exc)
        finally:
            if runtime:runtime.close()
        if get_job(store,job)['cancel']:state='CANCELLED'
        result={'id':job,'state':state,'error':error,'created':utcnow()}
        store.put_object('fixed_result',job,1,result)
        with store.transaction() as db:db.execute('UPDATE jobs SET state=?,heartbeat=? WHERE id=?',(state,utcnow(),job))
        return result
