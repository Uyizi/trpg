"""Deterministic, validity-gated deletion with a single bounded budget account."""
import json
import secrets
import time
from .domain import Budget,RuntimeProfile,canonical,digest,utcnow
from .inventory import inspect_local
from .runner import (uid,get_job,manifest_for,create_job,execute_job,device_lease,recover_jobs)
from .transforms import require_supported,deletion_candidates,validate_deletion
from .runtime import NotSent,UnknownCompletion

QUICK_BUDGET=Budget(requests=48,output_tokens=24576,seconds=1200)


def shrink_plan(case,profile,budget=None):
    require_supported(case)
    budget=budget or QUICK_BUDGET
    reserve={'requests':10,'output_tokens':10*profile.max_new_tokens}
    required={'requests':13,'output_tokens':13*profile.max_new_tokens}
    if budget.requests<required['requests'] or budget.output_tokens<required['output_tokens']:
        raise ValueError('预算无法容纳 3 次原题重放和 5 对对照；不能静默降低原输出上限')
    return {'algorithm':'safe_background_deletion_v1','preset':'quick','budget':budget.model_dump(),
            'confirmation_reserve':reserve,'minimum_baseline_and_confirmation':required,
            'max_candidates':12,'screening_repeats':2,'baseline_repeats':3,'confirmation_pairs':5,
            'failure_predicate':{'oracle':'closed_exact_v1','outcome':'FAIL','failure_kind':'semantic_mismatch'},
            'baseline_rule':'all_3_target_failures','screening_rule':'both_target_failures',
            'size_policy':'rendered_tokens_if_observable_else_message_json_characters',
            'confirmation_rule':{'mode':'descriptive','allowed_conclusion':'exploratory'},
            'minimality':'best_found','automatic_retries':0}


def create_investigation(store,case,profile,budget=None,*,workflow_plan=None):
    plan=workflow_plan or shrink_plan(case,profile,budget)
    from .fixed import runtime_identity
    identity=runtime_identity(profile)
    if profile.backend=='local': profile=RuntimeProfile.model_validate({**profile.model_dump(mode='json'),'model_path':identity['model_path']})
    ch=store.put_case(case);ph=store.put_object('profile',profile.id,profile.revision,profile.model_dump(mode='json'))
    if workflow_plan:
        plans=[]
        for t in plan['trigger_plans']:
            rh=store.put_object('validity_receipt',digest(t['receipt']),1,t['receipt'])
            plans.append({'factor':t['factor'],'candidate_hash':store.put_case(t['candidate']),
                          'receipt_hash':rh,'seeds':t['seeds']})
        plan={**plan,'trigger_plans':plans}
    job=uid('shrink'); confirmation=[]
    while len(confirmation)<5:
        seed=secrets.randbelow(2**31)
        if seed not in [42,43,*confirmation]: confirmation.append(seed)
    manifest={**plan,'schema_version':'1.0','id':job,'kind':'investigation','state':'FROZEN',
              'case_hash':ch,'profile_hash':ph,'identity':identity,'budget_account_id':job,
              'development_seeds':[42,43],'confirmation_seeds':confirmation,
              'allowed_transforms':['delete_safe_background_v1'],'created':utcnow(),
              'purpose':'development_investigation','selected_on_outcome':case['selected_on_outcome']}
    mh=store.put_object('manifest',job,1,manifest)
    with store.transaction() as db:
        db.execute('INSERT INTO jobs(id,manifest_hash,state,created) VALUES(?,?,?,?)',(job,mh,'PENDING',utcnow()))
    for t in plan.get('trigger_plans',[]):
        for i,seed in enumerate(t['seeds']):
            for arm in (('original','candidate') if i%2==0 else ('candidate','original')):
                create_job(store,case if arm=='original' else store.object(t['candidate_hash']),profile,[seed],
                           Budget(requests=1,output_tokens=profile.max_new_tokens,seconds=plan['budget']['seconds']),
                           parent_id=job,phase='trigger:'+t['factor'],arm=arm,receipt_hash=t['receipt_hash'])
    return job


def children_for(store,job):
    return sorted([r for r in store.rows('manifest_links') if r['parent_id']==job],key=lambda r:r['position'])


def result_for(store,job):
    rows=[r for r in store.rows('objects') if r['kind']=='shrink_result' and r['id']==job]
    return store.object(rows[0]['hash']) if rows else None


def steps_for(store,job):
    return sorted([store.object(r['hash']) for r in store.rows('objects')
                   if r['kind']=='shrink_step' and r['id'].startswith(job+'_')],key=lambda s:s['position'])


def observations_for(store,child):
    return sorted([o for o in store.rows('observations') if o['job_id']==child],key=lambda o:o['position'])


def target_failure(obs):
    g=json.loads(obs['grade']) if obs['grade'] else {}
    return obs['state']=='DONE' and obs['outcome']=='FAIL' and g.get('failure_kind')=='semantic_mismatch'


def batch_target(store,child):
    obs=observations_for(store,child)
    return bool(obs) and all(target_failure(o) for o in obs)


def execute_investigation(store,job,runtime):
    with device_lease() as lease_fd:
        recover_jobs(store)
        m=manifest_for(store,job)
        if m['kind']!='investigation': raise ValueError('不是缩减调查')
        if result_for(store,job) is not None: return result_for(store,job)
        if get_job(store,job)['state']!='PENDING':
            raise ValueError('中断调查请先导出已有轨迹，再新建调查；不自动续搜或重发')
        if runtime.profile.model_dump(mode='json')!=store.object(m['profile_hash']): raise ValueError('profile 不匹配')
        if runtime.profile.backend=='local' and inspect_local(runtime.profile.model_path)!=m['identity']:
            raise ValueError('运行身份变化；请新建调查')
        runtime.lease_fd=lease_fd
        with store.transaction() as db:
            db.execute("UPDATE jobs SET state='RUNNING',started=?,heartbeat=? WHERE id=?",(time.time(),utcnow(),job))
        context=Investigation(store,job,m,runtime,lease_fd)
        try:
            context.run()
            if context.status in ('shrunk','no_smaller_failure','baseline_not_reproduced'):
                for link in children_for(store,job):
                    if link['phase'].startswith('trigger:'):context.execute(link['child_id'])
        except (NotSent,UnknownCompletion,KeyboardInterrupt) as exc:
            context.status='cancelled' if get_job(store,job)['cancel'] else 'incomplete'
            context.reason=type(exc).__name__+': '+str(exc)
        except Exception as exc:
            context.status='error';context.reason=type(exc).__name__+': '+str(exc)
        finally: runtime.close()
        return context.finish()


class Investigation:
    def __init__(self,store,job,manifest,runtime,lease_fd):
        self.store,self.job,self.m,self.runtime,self.lease_fd=store,job,manifest,runtime,lease_fd
        self.original=store.object(manifest['case_hash']);self.best=self.original
        self.original_size=None;self.best_size=None
        self.status='incomplete';self.reason=None;self.confirmation_id=None
        self.confirmation={'planned_pairs':5,'complete':False,'mode':'descriptive','conclusion_level':'exploratory',
                           'reason':'not_started','pairs':[]}
        self.visited=set();self.attempted=0

    def heartbeat(self):
        with self.store.transaction() as db: db.execute('UPDATE jobs SET heartbeat=? WHERE id=?',(utcnow(),self.job))

    def guard(self):
        j=get_job(self.store,self.job)
        if j['cancel']: raise NotSent('父调查已取消')
        remaining=self.m['budget']['seconds']-(time.time()-j['started'])
        if remaining<=0: raise NotSent('父调查时间预算已用尽')
        self.runtime.deadline=time.monotonic()+remaining
        self.runtime.cancelled=lambda:bool(get_job(self.store,self.job)['cancel'])

    def measure(self,case):
        self.guard()
        # Tokenizer-only preparation is not a model generation. Persist its actual evidence.
        prepared=self.runtime.prepare(case,self.m['development_seeds'][0],self.heartbeat)
        h=self.store.artifact(canonical(prepared).encode())
        tokens=prepared.get('input_tokens')
        metric='rendered_tokens' if type(tokens) is int else 'message_json_characters'
        value=tokens if metric=='rendered_tokens' else len(canonical(case['messages']))
        return {'metric':metric,'value':value,'prepared_request_hash':h,
                'count_source':prepared.get('count_source','unknown')}

    def can_explore(self,calls=2):
        j=get_job(self.store,self.job);b=self.m['budget'];r=self.m['confirmation_reserve']
        return (j['requests']+calls+r['requests']<=b['requests']
                and j['tokens']+calls*self.runtime.profile.max_new_tokens+r['output_tokens']<=b['output_tokens'])

    def new_batch(self,case,seeds,phase,arm='original',receipt_hash=None,confirmation_id=None):
        return create_job(self.store,case,self.runtime.profile,seeds,
                          Budget(requests=len(seeds),output_tokens=len(seeds)*self.runtime.profile.max_new_tokens,
                                 seconds=self.m['budget']['seconds']),
                          parent_id=self.job,phase=phase,arm=arm,receipt_hash=receipt_hash,confirmation_id=confirmation_id)

    def execute(self,child):
        self.guard()
        execute_job(self.store,child,self.runtime,lease_fd=self.lease_fd,close_runtime=False)
        if get_job(self.store,child)['state']!='COMPLETE':
            raise NotSent('子批次未完整运行：'+child)

    def step(self,entry):
        index=len(steps_for(self.store,self.job))
        entry={**entry,'position':index,'created':utcnow()}
        self.store.put_object('shrink_step',self.job+'_'+str(index),1,entry)

    def run(self):
        self.guard()
        self.original_size=self.best_size=self.measure(self.original)
        baseline=self.new_batch(self.original,[self.m['development_seeds'][0]]*3,'baseline')
        self.execute(baseline)
        self.step({'phase':'baseline','child_id':baseline,'case_hash':digest(self.original),'size':self.original_size,
                   'target_failure':batch_target(self.store,baseline)})
        if any(o['outcome'] not in ('PASS','FAIL') for o in observations_for(self.store,baseline)):
            self.status='baseline_inconclusive';self.reason='原题包含不可评价输出，未启动搜索';return
        if not batch_target(self.store,baseline):
            self.status='baseline_not_reproduced'
            self.reason='原题三次未全部复现目标语义错误，未开始筛选'
            return
        while self.attempted<self.m['max_candidates']:
            accepted=False
            for candidate in deletion_candidates(self.best):
                key=digest({'input':candidate['input'],'gold':candidate['gold']})
                if key in self.visited: continue
                self.guard()
                if self.attempted>=self.m['max_candidates'] or not self.can_explore(): break
                self.visited.add(key);self.attempted+=1
                self.store.put_case(candidate)
                receipt=validate_deletion(self.best,candidate)
                receipt_hash=self.store.put_object('validity_receipt',digest(receipt),1,receipt)
                entry={'phase':'search','parent_case_hash':digest(self.best),'candidate_hash':digest(candidate),
                       'receipt_hash':receipt_hash,'accepted':False}
                if receipt['status']!='rule_verified':
                    self.step({**entry,'reason':'invalid_transformation'});continue
                self.step({**entry,'event':'candidate_proposed'})
                size=self.measure(candidate);entry['size']=size
                if size['metric']!=self.best_size['metric']: raise ValueError('候选与原题尺寸计量不一致')
                if size['value']>=self.best_size['value']:
                    self.step({**entry,'reason':'not_shorter'});continue
                child=self.new_batch(candidate,self.m['development_seeds'],'search','candidate',receipt_hash)
                # Persist the candidate before dispatch so interrupted searches remain reviewable.
                self.step({**entry,'child_id':child,'event':'screening_planned'})
                self.execute(child)
                accepted=batch_target(self.store,child)
                self.step({**entry,'child_id':child,'event':'screening_result','accepted':accepted,
                           'reason':'both_target_failures' if accepted else 'target_failure_not_preserved'})
                if accepted:
                    self.best=candidate;self.best_size=size;break
            if not accepted: break
        self.guard()
        if digest(self.best)==digest(self.original):
            self.status='no_smaller_failure';self.reason='本预算内未找到更短且满足目标失败的有效候选'
            return
        self.confirm()
        self.status='shrunk';self.reason='探索阶段找到更短的有效失败候选；新对照仅作描述，不升级确认等级'

    def confirm(self):
        self.confirmation_id=uid('confirmation')
        cm={'id':self.confirmation_id,'kind':'confirmation','parent_manifest_id':self.job,
            'budget_account_id':self.job,'original_case_hash':digest(self.original),'candidate_case_hash':digest(self.best),
            'profile_hash':self.m['profile_hash'],'seeds':self.m['confirmation_seeds'],
            'failure_predicate':self.m['failure_predicate'],'rule':self.m['confirmation_rule'],
            'purpose':'fresh_within_case_observations','created':utcnow()}
        cmh=self.store.put_object('confirmation',self.confirmation_id,1,cm)
        pairs=[]
        for seed in cm['seeds']:
            a=self.new_batch(self.original,[seed],'confirmation','original',confirmation_id=cmh)
            b=self.new_batch(self.best,[seed],'confirmation','candidate',confirmation_id=cmh)
            pairs.append({'seed':seed,'original_job':a,'candidate_job':b})
        self.confirmation.update(manifest_hash=cmh,reason=None,pairs=pairs)
        for pair in pairs:
            self.execute(pair['original_job']);self.execute(pair['candidate_job'])
        self.confirmation['complete']=True
        self.confirmation['original_target_failures']=sum(batch_target(self.store,p['original_job']) for p in pairs)
        self.confirmation['candidate_target_failures']=sum(batch_target(self.store,p['candidate_job']) for p in pairs)

    def finish(self):
        if get_job(self.store,self.job)['cancel']: self.status='cancelled'
        result={'investigation_id':self.job,'status':self.status,'reason':self.reason,
                'original_case_hash':digest(self.original),'final_case_hash':digest(self.best),
                'original_size':self.original_size,'final_size':self.best_size,'candidate_count':self.attempted,
                'minimality':'best_found','confirmation':self.confirmation,'conclusion_level':'exploratory',
                'created':utcnow(),'steps':steps_for(self.store,self.job)}
        self.store.put_object('shrink_result',self.job,1,result)
        state='COMPLETE' if self.status in ('shrunk','no_smaller_failure','baseline_not_reproduced') else 'INCOMPLETE'
        if self.status=='cancelled': state='CANCELLED'
        with self.store.transaction() as db: db.execute('UPDATE jobs SET state=?,heartbeat=? WHERE id=?',(state,utcnow(),self.job))
        return result
