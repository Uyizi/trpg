"""Detached FIFO worker; closing the browser never owns model lifetime."""
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from .domain import utcnow
from .runner import manifest_for,device_lease,recover_jobs


def status(store,job):
    from .runner import get_job
    j=get_job(store,job);m=manifest_for(store,job)
    children={r['child_id'] for r in store.rows('manifest_links') if r['parent_id']==job}
    obs=[o for o in store.rows('observations') if o['job_id'] in children|{job}]
    ids={o['id'] for o in obs};attempts=[a for a in store.rows('attempts') if a['observation_id'] in ids]
    planned=m.get('planned_observations',len(obs))
    return {**j,'kind':m['kind'],'planned':planned,'scheduled':len(obs),
            'completed':sum(o['state'] in ('DONE','ERROR') for o in obs),'not_run':sum(o['state']=='NOT_RUN' for o in obs),
            'budget':m['budget'],'remaining_requests':max(0,m['budget']['requests']-j['requests']),
            'remaining_output_tokens':max(0,m['budget']['output_tokens']-j['tokens']),
            'last_error':next((a['error'] for a in reversed(attempts) if a['error']),None)}


def start_worker(store):
    logs=store.root/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    with (logs/'background.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-m','qwen_debugger.cli','--data-dir',str(store.root),'worker','--once'],
                           stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env=env)
    return {'pid':p.pid,'log':str(logs/'background.log'),'mode':'detached_queue_drain'}


def work(store,once=False):
    from .service import run_job
    # Serialize queue workers per data directory, independently of the shared GPU lock.
    with (store.root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        recovered=False
        while True:
            try:
                with device_lease():
                    if not recovered:recover_jobs(store);recovered=True
            except ValueError:
                time.sleep(1);continue
            pending=[j for j in store.rows('jobs') if j['state']=='PENDING' and not manifest_for(store,j['id']).get('parent_manifest_id')]
            if not pending:
                if once:return
                time.sleep(1);continue
            j=min(pending,key=lambda j:j['created'])
            if j['cancel']:
                with store.transaction() as db:db.execute("UPDATE jobs SET state='CANCELLED' WHERE id=?",(j['id'],))
                continue
            try:run_job(store,j['id'])
            except Exception as exc:
                # Do not leave invalid configuration tasks endlessly pending.
                if '持有设备锁' in str(exc):time.sleep(1);continue
                with store.transaction() as db:db.execute("UPDATE jobs SET state='INCOMPLETE',heartbeat=? WHERE id=?",(utcnow(),j['id']))
                print(json.dumps({'job':j['id'],'error':type(exc).__name__+': '+str(exc)},ensure_ascii=False),flush=True)
