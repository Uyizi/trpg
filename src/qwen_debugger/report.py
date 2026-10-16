"""Offline evidence export. All planned observations remain in denominators."""
import html
import json
from pathlib import Path
from .domain import canonical
from .runner import get_job,manifest_for


def export_report(store,job,destination):
    summary=_export_report(store,job,destination)
    from .bundle import export_bundle
    export_bundle(store,job,destination)
    return summary


def _export_report(store,job,destination):
    if manifest_for(store,job)['kind'] in ('regression','confirmation','matrix'):
        from .fixed_report import export_fixed
        return export_fixed(store,job,destination)
    if manifest_for(store,job)['kind']=='contrast':
        from .contrast_report import export_contrast
        return export_contrast(store,job,destination)
    if manifest_for(store,job)['kind']=='investigation':
        from .investigation_report import export_investigation
        return export_investigation(store,job,destination)
    dest=Path(destination)
    if dest.exists() and any(dest.iterdir()): raise ValueError('导出目录非空；请选择新目录，避免覆盖旧证据')
    manifest=manifest_for(store,job); case=store.object(manifest['case_hash']); profile=store.object(manifest['profile_hash'])
    obs=sorted([o for o in store.rows('observations') if o['job_id']==job],key=lambda o:o['position'])
    ids={o['id'] for o in obs}; attempts=[a for a in store.rows('attempts') if a['observation_id'] in ids]
    hashes={a[k] for a in attempts for k in ('request_hash','response_hash') if a[k]}
    evidence={h:store.read_artifact(h) for h in hashes}  # Verify everything before writing report.
    counts={name:sum(o['outcome']==name for o in obs) for name in ['PASS','FAIL','ERROR','TRUNCATED','INCONCLUSIVE']}
    n=len(obs); passed=counts['PASS']; failed=counts['FAIL']; u=n-passed-failed
    texts=[json.loads(evidence[a['response_hash']])['text'] for a in attempts if a['response_hash']]
    summary={'planned':n,'counts':counts,'job':get_job(store,job),'unscored':u,
             'evaluable_fraction':(passed+failed)/n,'failure_rate_evaluable':failed/(passed+failed) if passed+failed else None,
             'failure_sensitivity_range':[failed/n,(failed+u)/n], 'independent_trials':None,
             'conclusion_level':'exploratory','same_output_observations':len(texts) if texts and len(set(texts))==1 else None,
             'limitations':['相同种子重放不代表独立抽样；不计算概率置信区间。',
                            '仅重放明确导入的消息；不自动加入检索、记忆或其他聊天上下文。',
                            '本报告是单条件重放，不包含缩减或对照结论。']}
    dest.mkdir(parents=True,exist_ok=True); (dest/'artifacts').mkdir(exist_ok=True)
    for h,data in evidence.items(): (dest/'artifacts'/h).write_bytes(data)
    for name,obj in [('manifest.json',manifest),('profiles.json',[profile]),('summary.json',summary)]:
        (dest/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('cases.jsonl',[case]),('observations.jsonl',obs),('attempts.jsonl',attempts),('receipts.jsonl',[])]:
        (dest/name).write_text(''.join(canonical(row)+'\n' for row in rows))
    e=html.escape
    def block(obj): return '<pre>'+e(json.dumps(obj,ensure_ascii=False,indent=2))+'</pre>'
    rows=[]
    for o in obs:
        a=next((a for a in attempts if a['id']==o['attempt_id']),None)
        result=json.loads(evidence[a['response_hash']]) if a and a['response_hash'] else None
        request=json.loads(evidence[a['request_hash']]) if a else None
        rows.append('<section><h2>观察 '+str(o['position']+1)+' · '+e(o['outcome'] or o['state'])+'</h2>'+block(o)+
                    '<h3>请求</h3>'+block(request)+'<h3>原始响应</h3>'+block(result)+'<h3>尝试</h3>'+block(a)+'</section>')
    title='模型行为重放 · '+('MOCK 合成输出' if profile['backend']=='mock' else '本地真实模型')
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>行为重放报告</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.6 system-ui;color:#17212b;background:#fafafa}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f4;padding:16px}section{border-top:1px solid #ccd3db;margin-top:24px}h1{font-size:26px}summary{cursor:pointer}</style>'''
    document+='<h1>'+e(title)+'</h1><p>结论等级：探索性观察。运行完成不等于模型答对。</p><h2>计数和限制</h2>'+block(summary)
    document+='<h2>案例与答案来源</h2>'+block(case)+'<details><summary>冻结运行清单与 profile</summary>'+block(manifest)+block(profile)+'</details>'+''.join(rows)+'</html>'
    (dest/'report.html').write_text(document)
    (dest/'README.txt').write_text('离线打开 report.html。原始请求/响应位于 artifacts，以 SHA256 命名。\n'
        'manifest.json 冻结条件，cases.jsonl 保存原输入与答案来源。未复制模型参数。\n'
        '本阶段支持 replay；不把重复运行当成独立确认。使用 qdebug replay --case CASE --profile PROFILE.json 重放会创建新任务。\n')
    return summary
