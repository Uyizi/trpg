"""Durable bounded replay. Unknown completion is never silently retried."""
import fcntl
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from .domain import canonical,digest,utcnow,RuntimeProfile
from .tasks import grade,oracle_id
from .runtime import UnknownCompletion,NotSent
from .inventory import inspect_local


def uid(prefix): return prefix+'_'+uuid.uuid4().hex[:16]


@contextmanager
def device_lease():
    # One lock across all debugger data directories, released by the OS on exit.
    path=Path.home()/'.cache/qwen-debugger/gpu.lock'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise ValueError('另一个调试器任务持有设备锁；请等待或查看 jobs') from exc
        try: yield f.fileno()
        finally: fcntl.flock(f,fcntl.LOCK_UN)


def validate_child(store,parent,case,seeds,phase,arm,receipt_hash,confirmation_id,profile_hash=None):
    from .transforms import validate_deletion
    ch=digest(case)
    if parent['kind'] in ('regression','confirmation','matrix'):
        from .fixed import validate_slot
        return validate_slot(store,parent,case,seeds,phase,arm,receipt_hash,confirmation_id,profile_hash)
    if parent['kind']=='contrast':
        from .contrast_transforms import validate_contrast
        if (phase!='contrast' or arm not in ('original','candidate') or len(seeds)!=1
            or {'seed':seeds[0],'arm':arm} not in parent['schedule']
            or ch!=parent[arm+'_case_hash'] or receipt_hash!=parent['receipt_hash']
            or confirmation_id is not None):
            raise ValueError('对照子任务与冻结的案例、处理臂、种子或收据不匹配')
        receipt=validate_contrast(store.object(parent['original_case_hash']),
                                  store.object(parent['candidate_case_hash']),parent['factor'])
        if receipt['status']!='rule_verified' or store.object(receipt_hash)!=receipt:
            raise ValueError('对照收据无效')
        return
    if phase and phase.startswith('trigger:'):
        from .contrast_transforms import validate_contrast
        plans=[t for t in parent.get('trigger_plans',[]) if t['factor']==phase.split(':',1)[1]]
        if len(plans)!=1:raise ValueError('未声明的触发因素')
        t=plans[0];a=store.object(parent['case_hash']);b=store.object(t['candidate_hash'])
        if (arm not in ('original','candidate') or ch!=(digest(a) if arm=='original' else digest(b))
            or len(seeds)!=1 or seeds[0] not in t['seeds'] or receipt_hash!=t['receipt_hash']
            or store.object(receipt_hash)!=validate_contrast(a,b,t['factor'])):
            raise ValueError('触发观察与冻结条件不一致')
    elif phase=='baseline':
        if ch!=parent['case_hash'] or arm!='original' or seeds!=[parent['development_seeds'][0]]*3:
            raise ValueError('原题重放必须沿用父清单的案例和种子')
    elif phase=='search':
        if not receipt_hash or arm!='candidate' or seeds!=parent['development_seeds']:
            raise ValueError('筛选子任务必须引用有效性收据和探索种子')
        receipt=store.object(receipt_hash)
        prior=store.object(receipt['parent_hash'])
        if (receipt!=validate_deletion(prior,case) or receipt['status']!='rule_verified'
            or validate_deletion(store.object(parent['case_hash']),case)['status']!='rule_verified'):
            raise ValueError('候选有效性收据与父题不匹配')
    elif phase=='confirmation':
        if not confirmation_id: raise ValueError('确认子任务必须引用冻结确认清单')
        cm=store.object(confirmation_id)
        expected=cm.get(arm+'_case_hash') if arm in ('original','candidate') else None
        if (cm.get('kind')!='confirmation' or cm.get('parent_manifest_id')!=parent['id']
            or cm.get('budget_account_id')!=parent['id'] or cm.get('profile_hash')!=parent['profile_hash']
            or cm.get('original_case_hash')!=parent['case_hash'] or cm.get('seeds')!=parent['confirmation_seeds']
            or cm.get('rule')!=parent['confirmation_rule'] or cm.get('failure_predicate')!=parent['failure_predicate']
            or expected!=ch or len(seeds)!=1 or seeds[0] not in parent['confirmation_seeds']):
            raise ValueError('确认子任务与冻结两臂、种子或预算账户不匹配')
    else: raise ValueError('无效调查阶段')


def create_job(store,case,profile,seeds,budget,*,parent_id=None,phase=None,arm="original",receipt_hash=None,confirmation_id=None):
    if not 1<=len(seeds)<=48 or any(type(s) is not int or s<0 or s>=2**63 for s in seeds):
        raise ValueError('必须提供 1—48 个合法非负种子')
    parent=manifest_for(store,parent_id) if parent_id else None
    if parent and parent['kind'] not in ('investigation','contrast','regression','confirmation','matrix'): raise ValueError('子任务必须引用调查父清单')
    from .fixed import runtime_identity
    identity=parent.get('identity') if parent else (inspect_local(profile.model_path) if profile.backend=='local' else runtime_identity(profile))
    if parent and identity is None:
        # Profiles are already normalized/frozen by fixed-experiment creation.
        identity=parent['identities'].get(digest(profile.model_dump(mode='json')))
        if identity is None:raise ValueError('profile 不属于父实验')
    if profile.backend=='local':
        profile=RuntimeProfile.model_validate({**profile.model_dump(mode='json'),'model_path':identity['model_path']})
    ch=store.put_case(case)
    ph=store.put_object('profile',profile.id,profile.revision,profile.model_dump(mode='json'))
    if parent and ph not in parent.get('identities',{parent['profile_hash']:None}): raise ValueError('子任务 profile 必须与父清单一致')
    if parent: validate_child(store,parent,case,seeds,phase,arm,receipt_hash,confirmation_id,ph)
    job=uid('replay')
    manifest={'schema_version':'1.0','id':job,'kind':'observation_batch','state':'FROZEN',
              'case_hash':ch,'profile_hash':ph,'identity':identity,'budget':budget.model_dump(),
              'budget_account_id':parent_id or job,'arms':[arm],'seeds':seeds,'oracle':oracle_id(case),
              'extractor':'exact_v1' if case['gold']['kind']=='exact_text' else case['gold']['kind']+'_v1','purpose':parent.get('purpose','development') if parent else 'development_replay','statistical_unit':'repeated_observation',
              'conclusion_level':'exploratory','stop_rules':['budget','cancel','unknown_completion'],
              'automatic_retries':0,'created':utcnow()}
    if parent:
        manifest.update(parent_manifest_id=parent_id,phase=phase,receipt_hash=receipt_hash,confirmation_manifest_id=confirmation_id)
    mh=store.put_object('manifest',job,1,manifest)
    with store.transaction() as db:
        if parent:
            if phase in ('confirmation','contrast') or (phase and phase.startswith('trigger:')):
                existing=db.execute('SELECT o.body FROM manifest_links l JOIN jobs j ON j.id=l.child_id JOIN objects o ON o.hash=j.manifest_hash WHERE l.parent_id=? AND l.phase=?',(parent_id,phase)).fetchall()
                if any(json.loads(r['body'])['arms']==[arm] and json.loads(r['body'])['seeds']==seeds for r in existing):
                    raise ValueError('确认种子/处理臂已安排，不能重复消费')
            position=db.execute('SELECT COUNT(*) FROM manifest_links WHERE parent_id=?',(parent_id,)).fetchone()[0]
            if parent['kind'] in ('regression','confirmation','matrix'):
                expected={'case_hash':ch,'profile_hash':ph,'arm':arm,'seed':seeds[0],'receipt_hash':receipt_hash}
                if position>=len(parent['schedule']) or any(parent['schedule'][position].get(k)!=v for k,v in expected.items()):
                    raise ValueError('固定实验槽位重复或顺序错误')
            if parent['kind']=='contrast' and (position>=len(parent['schedule']) or
                parent['schedule'][position]!={'seed':seeds[0],'arm':arm}):
                raise ValueError('子任务必须按冻结对照顺序安排')
            db.execute('INSERT INTO manifest_links VALUES(?,?,?,?,?)',(parent_id,job,phase,arm,position))
        db.execute('INSERT INTO jobs(id,manifest_hash,state,created) VALUES(?,?,?,?)',(job,mh,'PENDING',utcnow()))
        for i,seed in enumerate(seeds):
            db.execute('INSERT INTO observations(id,job_id,position,seed) VALUES(?,?,?,?)',(uid('obs'),job,i,seed))
    return job


def get_job(store,job):
    with store.connect() as db: r=db.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone()
    if not r: raise ValueError('任务不存在：'+job)
    return dict(r)


def manifest_for(store,job): return store.object(get_job(store,job)['manifest_hash'])


def claim(store,job):
    manifest=manifest_for(store,job); p=store.object(manifest['profile_hash']); b=manifest['budget']
    # The planned wire input is already evidence, even if prepare fails later.
    case=store.object(manifest['case_hash'])
    account_id=manifest['budget_account_id']
    account_manifest=manifest_for(store,account_id) if account_id!=job else manifest
    if account_id!=job:
        validate_child(store,account_manifest,case,manifest['seeds'],manifest['phase'],manifest['arms'][0],
                       manifest.get('receipt_hash'),manifest.get('confirmation_manifest_id'),manifest['profile_hash'])
    reserve=account_manifest.get('confirmation_reserve',{}) if manifest.get('phase') in ('baseline','search') else {}
    with store.transaction() as db:
        j=db.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone()
        if j['cancel'] or j['state'] in ('COMPLETE','CANCELLED','INCOMPLETE'): return None
        if db.execute("SELECT 1 FROM observations WHERE job_id=? AND state='RUNNING'",(job,)).fetchone(): return None
        if account_id!=job and db.execute("SELECT 1 FROM observations o JOIN manifest_links l ON l.child_id=o.job_id WHERE l.parent_id=? AND o.state='RUNNING'",(account_id,)).fetchone(): return None
        account=db.execute('SELECT * FROM jobs WHERE id=?',(account_id,)).fetchone()
        if account['cancel'] or account['state'] in ('COMPLETE','CANCELLED','INCOMPLETE'): return None
        started=account['started'] or time.time()
        limits=account_manifest['budget']
        if (account['requests']+1+reserve.get('requests',0)>limits['requests']
            or account['tokens']+p['max_new_tokens']+reserve.get('output_tokens',0)>limits['output_tokens']
            or time.time()-started>=limits['seconds']): return None
        if j['requests']>=b['requests'] or j['tokens']+p['max_new_tokens']>b['output_tokens'] or time.time()-started>=b['seconds']:
            return None
        obs=db.execute("SELECT * FROM observations WHERE job_id=? AND state='NOT_RUN' ORDER BY position LIMIT 1",(job,)).fetchone()
        if not obs: return None
        aid=uid('attempt')
        # Save planned payload before claiming (artifact transaction runs separately).
        planned={'messages':case['messages'],'profile':p,'seed':obs['seed'],'stage':'planned_not_prepared'}
        raw=canonical(planned).encode(); import hashlib,os,tempfile
        h=hashlib.sha256(raw).hexdigest(); target=store.artifact_path(h)
        fd,tmp=tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd,'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        db.execute('INSERT OR IGNORE INTO artifacts VALUES(?,?)',(h,len(raw)))
        db.execute("INSERT INTO attempts(id,observation_id,started,state,request_hash,reserved_tokens) VALUES(?,?,?,'RUNNING',?,?)",
                   (aid,obs['id'],utcnow(),h,p['max_new_tokens']))
        db.execute("UPDATE observations SET state='RUNNING',attempt_id=? WHERE id=?",(aid,obs['id']))
        db.execute("UPDATE jobs SET state='RUNNING',started=?,heartbeat=?,requests=requests+1,tokens=tokens+? WHERE id=?",
                   (started,utcnow(),p['max_new_tokens'],job))
        if account_id!=job:
            db.execute("UPDATE jobs SET state='RUNNING',started=?,heartbeat=?,requests=requests+1,tokens=tokens+? WHERE id=?",
                       (started,utcnow(),p['max_new_tokens'],account_id))
        return {**dict(obs),'attempt_id':aid,'reserved_tokens':p['max_new_tokens']}


def cancel_job(store,job):
    get_job(store,job)
    with store.transaction() as db: db.execute('UPDATE jobs SET cancel=1 WHERE id=?',(job,))


def recover_jobs(store):
    """Call only while holding device_lease; cannot distinguish sent vs lost response."""
    with store.transaction() as db:
        db.execute("UPDATE attempts SET state='UNKNOWN_COMPLETION',ended=? WHERE state='RUNNING'",(utcnow(),))
        db.execute("UPDATE observations SET state='UNKNOWN_COMPLETION' WHERE state='RUNNING'")
        db.execute("UPDATE jobs SET state='INCOMPLETE' WHERE state='RUNNING'")


def execute_job(store,job,runtime,*,lease_fd=None,close_runtime=True):
    if lease_fd is None:
        if manifest_for(store,job).get('parent_manifest_id'):
            raise ValueError('调查子任务只能由父调查调度，不能单独补跑')
        with device_lease() as fd:
            recover_jobs(store)
            return execute_job(store,job,runtime,lease_fd=fd)
    runtime.lease_fd=lease_fd
    return _execute_job(store,job,runtime,close_runtime=close_runtime)


def _execute_job(store,job,runtime,*,close_runtime):
    manifest=manifest_for(store,job); case=store.object(manifest['case_hash'])
    p=store.object(manifest['profile_hash'])
    if get_job(store,job)['cancel']:
        with store.transaction() as db: db.execute("UPDATE jobs SET state='CANCELLED' WHERE id=?",(job,))
        return
    if runtime.profile.model_dump(mode='json')!=p: raise ValueError('runtime 与冻结 profile 不一致')
    if p['backend']=='local' and inspect_local(p['model_path'])!=manifest['identity']:
        raise ValueError('运行身份已变化；请新建实验，不能沿用冻结清单')
    account_id=manifest['budget_account_id']
    account_manifest=manifest_for(store,account_id) if account_id!=job else manifest
    runtime.cancelled=lambda:bool(get_job(store,job)['cancel'] or get_job(store,account_id)['cancel'])
    last_heartbeat=[0.0]
    def heartbeat():
        if time.monotonic()-last_heartbeat[0]>1:
            with store.transaction() as db: db.execute('UPDATE jobs SET heartbeat=? WHERE id IN (?,?)',(utcnow(),job,account_id))
            last_heartbeat[0]=time.monotonic()
    try:
        while (obs:=claim(store,job)) is not None:
            aid=obs['attempt_id']; started=time.monotonic(); rh=None
            outcome=None; g=None; error=None; charged=obs['reserved_tokens']; state='DONE'
            remaining=account_manifest['budget']['seconds']-(time.time()-get_job(store,account_id)['started'])
            runtime.deadline=time.monotonic()+max(0,remaining)
            sent=False
            try:
                request=runtime.prepare(case,obs['seed'],heartbeat)
                reqh=store.artifact(canonical(request).encode())
                with store.transaction() as db: db.execute('UPDATE attempts SET request_hash=? WHERE id=?',(reqh,aid))
                if runtime.cancelled():
                    # No generation sent: preserve this interrupted observation without a fabricated outcome.
                    raise NotSent('取消发生在输入准备后，未发送生成')
                if time.monotonic()>=runtime.deadline: raise NotSent('预算已到，未发送生成')
                sent=True
                result=runtime.generate(request,heartbeat)
                rh=store.artifact(canonical(result).encode())
                used=result.get('usage',{}).get('output_tokens')
                if used is not None:
                    if type(used) is not int or not 0<=used<=charged: raise ValueError('后端 usage 超出冻结预算或格式无效')
                    charged=used
                g=grade(case,result); outcome=g['outcome']
            except NotSent as exc:
                state='NOT_SENT'; charged=0; error=str(exc)
            except (UnknownCompletion,KeyboardInterrupt) as exc:
                state='UNKNOWN_COMPLETION' if sent else 'NOT_SENT'
                if not sent: charged=0
                error=type(exc).__name__+': '+str(exc)
            except Exception as exc:
                state='ERROR'; outcome='ERROR'; error=type(exc).__name__+': '+str(exc)
                if hasattr(exc,'evidence'):
                    rh=store.artifact(canonical(exc.evidence).encode())
            with store.transaction() as db:
                db.execute('UPDATE attempts SET state=?,ended=?,response_hash=?,error=?,charged_tokens=?,elapsed=? WHERE id=?',
                           (state,utcnow(),rh,error,charged,time.monotonic()-started,aid))
                db.execute('UPDATE observations SET state=?,outcome=?,grade=? WHERE id=?',
                           ('NOT_RUN' if state=='NOT_SENT' else state,outcome,canonical(g) if g else None,obs['id']))
                db.execute('UPDATE jobs SET tokens=tokens-?+?,heartbeat=? WHERE id=?',
                           (obs['reserved_tokens'],charged,utcnow(),job))
                if state=='NOT_SENT': db.execute('UPDATE jobs SET requests=requests-1 WHERE id=?',(job,))
                if account_id!=job:
                    db.execute('UPDATE jobs SET tokens=tokens-?+?,heartbeat=? WHERE id=?',
                               (obs['reserved_tokens'],charged,utcnow(),account_id))
                    if state=='NOT_SENT': db.execute('UPDATE jobs SET requests=requests-1 WHERE id=?',(account_id,))
            if state in ('UNKNOWN_COMPLETION','NOT_SENT'): break
    finally:
        if close_runtime: runtime.close()
    with store.transaction() as db:
        j=db.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone()
        unfinished=db.execute("SELECT COUNT(*) FROM observations WHERE job_id=? AND state NOT IN ('DONE','ERROR')",(job,)).fetchone()[0]
        state='CANCELLED' if j['cancel'] else ('INCOMPLETE' if unfinished else 'COMPLETE')
        db.execute('UPDATE jobs SET state=?,heartbeat=? WHERE id=?',(state,utcnow(),job))
