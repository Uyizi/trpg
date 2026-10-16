"""Self-contained evidence bundles; import restores assets, never schedules old jobs."""
import hashlib
import json
import re
from pathlib import Path
from .domain import canonical,digest,utcnow,CaseImport,RuntimeProfile
from .tasks import compile_case
from .transforms import source_of


def export_bundle(store,job,destination):
    dest=Path(destination);all_objects={r['hash']:r for r in store.rows('objects')}
    ids={job}|{r['child_id'] for r in store.rows('manifest_links') if r['parent_id']==job}
    jobs=[j for j in store.rows('jobs') if j['id'] in ids]
    obs=[o for o in store.rows('observations') if o['job_id'] in ids];oid={o['id'] for o in obs}
    attempts=[a for a in store.rows('attempts') if a['observation_id'] in oid]
    selected={j['manifest_hash'] for j in jobs}
    selected.update(r['hash'] for r in all_objects.values() if r['id'] in ids or r['id'].startswith(job+'_'))
    blobs={a[k] for a in attempts for k in ('request_hash','response_hash') if a[k]}
    def refs(x):
        if isinstance(x,str):yield x
        elif isinstance(x,dict):
            for v in x.values():yield from refs(v)
        elif isinstance(x,list):
            for v in x:yield from refs(v)
    done=set()
    while selected-done:
        h=next(iter(selected-done));body=store.object(h);done.add(h)
        for v in refs(body):
            if v in all_objects:selected.add(v)
        for key in ('review_id',):
            if body.get(key):selected.update(r['hash'] for r in all_objects.values() if r['id']==body[key])
        for v in refs(body):
            if isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) and store.artifact_path(v).is_file():blobs.add(v)
    (dest/'artifacts').mkdir(exist_ok=True)
    artifacts=[]
    for h in sorted(blobs):
        data=store.read_artifact(h);(dest/'artifacts'/h).write_bytes(data);artifacts.append({'hash':h,'size':len(data)})
    payload={'schema':'qdebug-evidence-1','source_experiment':job,'objects':[all_objects[h] for h in sorted(selected)],
             'jobs':jobs,'observations':obs,'attempts':attempts,'artifacts':artifacts,
             'manifest_links':[l for l in store.rows('manifest_links') if l['parent_id']==job]}
    document={'digest':digest(payload),'payload':payload}
    (dest/'bundle.json').write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return document['digest']


def import_bundle(store,directory):
    root=Path(directory).resolve();p=root/'bundle.json'
    if p.stat().st_size>64*1024*1024:raise ValueError('bundle 清单超过64MiB')
    doc=json.loads(p.read_text(encoding='utf-8'));payload=doc['payload']
    if payload.get('schema')!='qdebug-evidence-1' or digest(payload)!=doc.get('digest'):raise ValueError('导出包 schema 或散列无效')
    records=payload['objects'];bodies={r['hash']:json.loads(r['body']) for r in records}
    identities={}
    for r in records:
        b=bodies[r['hash']]
        key=(r['kind'],r['id'],r['revision'])
        if key in identities and identities[key]!=r['hash']:raise ValueError('导入包内部存在冲突修订')
        identities[key]=r['hash']
        if digest(b)!=r['hash']:raise ValueError('快照散列损坏')
        if r['kind']=='case' and (r['id'],r['revision'])!=(b['case_id'],b['revision']):raise ValueError('案例记录身份不匹配')
        if r['kind'] in ('profile','suite') and (r['id'],r['revision'])!=(b['id'],b['revision']):raise ValueError('资产记录身份不匹配')
        if r['kind']=='case' and compile_case(CaseImport.model_validate(source_of(b)))!=b:raise ValueError('案例求解快照无效')
        if r['kind']=='profile':RuntimeProfile.model_validate(b)
    evidence={}
    for a in payload['artifacts']:
        h=a['hash']
        if not re.fullmatch('[0-9a-f]{64}',h):raise ValueError('非法证据路径')
        file=root/'artifacts'/h
        if file.is_symlink() or file.resolve().parent!=root/'artifacts':raise ValueError('证据路径逃逸')
        if a['size']>64*1024*1024:raise ValueError('单项证据超过64MiB')
        data=file.read_bytes()
        if len(data)!=a['size'] or hashlib.sha256(data).hexdigest()!=h:raise ValueError('证据缺失或损坏')
        evidence[h]=data
    for a in payload['attempts']:
        if a['request_hash'] not in evidence or (a['response_hash'] and a['response_hash'] not in evidence):raise ValueError('尝试记录缺少原始证据')
    from .integrity import pool_from_rows,validate_case,validate_suite
    live=[r for r in records if r['kind'] in ('case','profile','suite')]
    def validate_assets(existing_rows):
        existing={(r['kind'],r['id'],r['revision']):r['hash'] for r in existing_rows}
        pool=pool_from_rows(existing_rows)
        for r in live:
            key=(r['kind'],r['id'],r['revision'])
            if key in existing and existing[key]!=r['hash']:raise ValueError('同名资产修订冲突，使用新的 --data-dir 导入')
            pool[r['hash']]={'kind':r['kind'],'id':r['id'],'revision':r['revision'],'value':bodies[r['hash']]}
        for r in live:
            b=bodies[r['hash']]
            if r['kind']=='case':validate_case(b,pool)
            elif r['kind']=='suite':validate_suite(b,pool)
    validate_assets(store.rows('objects'))
    for h,data in evidence.items():store.artifact(data)
    # Check again under the same write transaction that registers every live asset.
    with store.transaction() as db:
        validate_assets(db.execute('SELECT * FROM objects').fetchall())
        for r in live:
            db.execute('INSERT OR IGNORE INTO objects VALUES(?,?,?,?,?)',
                       (r['kind'],r['id'],r['revision'],r['hash'],canonical(bodies[r['hash']])))
    archive=store.artifact(canonical(payload).encode())
    result={'id':'archive_'+doc['digest'][:20],'source_digest':doc['digest'],'source_experiment':payload['source_experiment'],
            'payload_artifact':archive,'created':utcnow(),'restored_assets':sum(r['kind'] in ('case','profile','suite') for r in records),
            'jobs_restarted':False,'provenance':'imported evidence; reviews are historical, not new local user decisions'}
    old=[r for r in store.rows('objects') if r['kind']=='archive' and r['id']==result['id']]
    if old:return store.object(old[0]['hash'])
    store.put_object('archive',result['id'],1,result);return result
