"""Loopback-only workbench using the same services as the CLI."""
import hmac
import json
import mimetypes
import secrets
import socket
import zipfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from .domain import CaseImport,RuntimeProfile
from .tasks import compile_case
from .background import status,start_worker
from .runner import manifest_for,cancel_job,device_lease,recover_jobs,get_job
from .service import create_experiment
from .fixed import freeze_profile

STATIC=Path(__file__).with_name('static')


def state(store):
    objects=store.rows('objects');cases=[];profiles=[];suites=[];reviews=[]
    for r in objects:
        if r['kind']=='case':
            c=store.object(r['hash']);cases.append({'hash':r['hash'],'id':r['id'],'revision':r['revision'],'family':c['family_id'],
                   'root':c['root_case_id'],'source':c['provenance'].get('kind'),'gold':c['gold']['source'],'split':c['split']})
        elif r['kind']=='profile':profiles.append({'hash':r['hash'],**store.object(r['hash'])})
        elif r['kind']=='suite':suites.append(store.object(r['hash']))
        elif r['kind']=='review_proposal':
            from .reviews import review_receipt
            reviews.append(review_receipt(store,r['id']))
    jobs=[status(store,j['id']) for j in store.rows('jobs') if not manifest_for(store,j['id']).get('parent_manifest_id')]
    return {'cases':cases,'profiles':profiles,'suites':suites,'reviews':reviews,'jobs':list(reversed(jobs)),
            'data_directory':str(store.root),'whitebox':'未启用（原方案 v1.1 可选扩展）'}


def job_detail(store,id):
    from .fixed_report import summarize_fixed
    from .contrast import contrast_result
    from .shrink import result_for,steps_for,children_for
    m=manifest_for(store,id);links=children_for(store,id);ids={id}|{l['child_id'] for l in links}
    obs=[o for o in store.rows('observations') if o['job_id'] in ids];oids={o['id'] for o in obs}
    attempts=[a for a in store.rows('attempts') if a['observation_id'] in oids]
    cases=set(s['case_hash'] for s in m.get('schedule',[]) if s.get('case_hash'))
    cases.update(m[k] for k in ('case_hash','original_case_hash','candidate_case_hash') if m.get(k))
    for l in links:cases.add(manifest_for(store,l['child_id'])['case_hash'])
    return {'status':status(store,id),'manifest':m,'cases':[store.object(h) for h in sorted(cases)],'observations':obs,
            'attempts':attempts,'children':links,'steps':steps_for(store,id),
            'summary':summarize_fixed(store,id) if m['kind'] in ('confirmation','regression','matrix') else None,
            'result':result_for(store,id) or contrast_result(store,id)}


def action(store,name,p):
    if name=='case.import':
        c=compile_case(CaseImport.model_validate(p['case']));return {'hash':store.put_case(c),'case':c}
    if name=='profile.import':
        profile=RuntimeProfile.model_validate(p['profile']);h,identity=freeze_profile(store,profile);return {'hash':h,'identity':identity}
    if name=='experiment.preview':return create_experiment(store,p,dry_run=True)
    if name=='experiment.create':
        # Idempotency key prevents double clicks from creating duplicate generations.
        key=p.get('request_id')
        if not isinstance(key,str) or not 16<=len(key)<=100:raise ValueError('缺少有效操作 ID')
        # Serialize creation across concurrent HTTP requests using a dedicated lock file.
        import fcntl
        with (store.root/'web-create.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            prior=[r for r in store.rows('objects') if r['kind']=='web_request' and r['id']==key]
            from .domain import digest
            if prior:
                record=store.object(prior[0]['hash'])
                if record['request_hash']!=digest(p):raise ValueError('操作 ID 已用于不同条件，请重新提交')
                return record
            create_experiment(store,p,dry_run=True);id=create_experiment(store,p)
            result={'job_id':id,'request_hash':digest(p)};store.put_object('web_request',key,1,result)
        start_worker(store);return result
    if name=='jobs.start':
        m=manifest_for(store,p['id'])
        if m.get('parent_manifest_id') or get_job(store,p['id'])['state']!='PENDING':raise ValueError('只能启动待运行父任务')
        return start_worker(store)
    if name=='jobs.cancel':cancel_job(store,p['id']);return status(store,p['id'])
    if name=='jobs.recover':
        with device_lease():recover_jobs(store)
        return {'status':'已恢复记录；未知完成未重发'}
    if name=='suite.create':
        from .suites import create_suite
        return create_suite(store,p['name'],p.get('cases',[]),purpose=p.get('purpose','development'),revision=int(p.get('revision',1)),from_experiment=p.get('from_experiment') or None)
    if name=='review.propose':
        from .reviews import propose_review
        return {'review_id':propose_review(store,store.case(p['original']),store.case(p['candidate']),p.get('note',''))}
    if name=='review.decide':
        from .reviews import decide_review
        return {'hash':decide_review(store,p['id'],p['decision'],p['note'],p['actor'],parent_hash=p['parent_hash'],candidate_hash=p['candidate_hash'])}
    if name=='fixtures':
        from .fixtures import install_fixtures
        return install_fixtures(store)
    if name=='export':
        from .report import export_report
        id=p['id'];manifest_for(store,id)
        dest=store.root/'exports'/(id+'_'+secrets.token_hex(4))
        export_report(store,id,dest)
        archive=dest.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(dest.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(dest))
        return {'report':'/exports/'+dest.name+'/report.html','download':'/download/'+archive.name}
    raise ValueError('未知操作')


def make_server(store,host='127.0.0.1',port=8765):
    if host not in ('127.0.0.1','localhost','::1'):raise ValueError('工作台仅允许本机访问')
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def allowed_host(self):
            incoming=self.headers.get('Host','')
            # Port must match; reject DNS-rebinding hostnames before reading any data.
            return incoming in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}',f'[::1]:{self.server.server_port}'}
        def reply(self,data,status_code=200,content_type='application/json; charset=utf-8'):
            raw=json.dumps(data,ensure_ascii=False).encode() if not isinstance(data,bytes) else data
            self.send_response(status_code);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if not self.allowed_host():return self.reply({'error':'Host rejected'},403)
            u=urlsplit(self.path);q=parse_qs(u.query)
            try:
                if u.path=='/':return self.reply((STATIC/'index.html').read_bytes().replace(b'__CSRF__',token.encode()),content_type='text/html; charset=utf-8')
                if u.path in ('/app.js','/style.css'):
                    file=STATIC/u.path[1:];return self.reply(file.read_bytes(),content_type='text/javascript; charset=utf-8' if file.suffix=='.js' else 'text/css; charset=utf-8')
                if u.path=='/api/state':return self.reply(state(store))
                if u.path=='/api/job':return self.reply(job_detail(store,q['id'][0]))
                if u.path=='/api/object':return self.reply(store.object(q['hash'][0]))
                if u.path=='/api/artifact':return self.reply(json.loads(store.read_artifact(q['hash'][0])))
                if u.path=='/api/doctor':
                    from .inventory import doctor
                    return self.reply(doctor())
                if u.path.startswith(('/download/','/exports/')):
                    relative=u.path.removeprefix('/download/').removeprefix('/exports/')
                    root=(store.root/'exports').resolve();file=(root/relative).resolve()
                    if not file.is_relative_to(root) or file.is_symlink() or not file.is_file():raise ValueError('文件不存在')
                    if not (file.name=='report.html' or file.suffix=='.zip'):raise ValueError('不支持该导出文件')
                    return self.reply(file.read_bytes(),content_type='application/zip' if file.suffix=='.zip' else 'text/html; charset=utf-8')
                return self.reply({'error':'Not found'},404)
            except (ValueError,KeyError,OSError) as exc:return self.reply({'error':str(exc)},400)
        def do_POST(self):
            if not self.allowed_host():return self.reply({'error':'Host rejected'},403)
            origin=self.headers.get('Origin')
            origins={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}',f'http://[::1]:{self.server.server_port}'}
            if origin and origin not in origins:return self.reply({'error':'Origin rejected'},403)
            if not hmac.compare_digest(self.headers.get('X-Qdebug-Token',''),token):return self.reply({'error':'操作令牌无效，请刷新页面'},403)
            if urlsplit(self.path).path!='/api/action':return self.reply({'error':'Not found'},404)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=4*1024*1024:raise ValueError('请求大小必须在4MiB以内')
                body=json.loads(self.rfile.read(size));return self.reply(action(store,body['action'],body.get('params',{})))
            except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:return self.reply({'error':str(exc)},400)
    class Server(ThreadingHTTPServer):
        daemon_threads=True
        address_family=socket.AF_INET6 if host=='::1' else socket.AF_INET
    return Server((host,port),Handler)


def serve(store,host='127.0.0.1',port=8765):
    server=make_server(store,host,port)
    print(f'行为调试器：http://{host}:{server.server_port} （Ctrl+C 关闭网页服务；后台实验独立运行）',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
