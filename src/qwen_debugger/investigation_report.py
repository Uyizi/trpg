"""Offline shrink evidence, with exploration and fresh paired observations separate."""
import difflib
import html
import json
from pathlib import Path
from .domain import canonical
from .runner import manifest_for,get_job
from .shrink import children_for,result_for,steps_for,target_failure


def counts_for(observations,expected=None):
    counts={k:sum(o['outcome']==k for o in observations) for k in ('PASS','FAIL','ERROR','TRUNCATED','INCONCLUSIVE')}
    n=len(observations) if expected is None else expected
    p,f=counts['PASS'],counts['FAIL'];u=n-p-f
    return {'planned':n,'scheduled':len(observations),'counts':counts,
            'execution_states':{s:sum(o['state']==s for o in observations) for s in sorted({o['state'] for o in observations})},
            'not_scheduled':n-len(observations),'unscored':u,
            'evaluable_fraction':(p+f)/n if n else None,'failure_rate_evaluable':f/(p+f) if p+f else None,
            'failure_sensitivity_range':[f/n,(f+u)/n] if n else None}


def paired_summary(cm,children,manifests,observations,*,phase='confirmation'):
    if cm is None: return {'status':'not_started','planned_pairs':5,'complete_pairs':0,'delta':None}
    byjob={o['job_id']:o for o in observations if o['job_id'] in manifests}
    pairs=[];b=c=n=0
    for seed in cm['seeds']:
        arms={}
        for link in children:
            m=manifests[link['child_id']]
            if link['phase']==phase and m['seeds']==[seed]: arms[link['arm']]=byjob.get(link['child_id'])
        a,z=arms.get('original'),arms.get('candidate')
        valid=bool(a and z and a['outcome'] in ('PASS','FAIL') and z['outcome'] in ('PASS','FAIL'))
        if valid:
            n+=1;b+=a['outcome']=='PASS' and z['outcome']=='FAIL';c+=a['outcome']=='FAIL' and z['outcome']=='PASS'
        pairs.append({'seed':seed,'original':a,'candidate':z,'evaluable':valid})
    missing=len(pairs)-n
    return {'status':'descriptive','planned_pairs':len(pairs),'complete_pairs':sum(bool(p['original'] and p['candidate'] and
            p['original']['state'] in ('DONE','ERROR') and p['candidate']['state'] in ('DONE','ERROR')) for p in pairs),
            'evaluable_pairs':n,'missing_or_unevaluable_pairs':missing,'regressions':b,'improvements':c,
            'delta':(b-c)/n if n else None,'delta_missing_sensitivity':[(b-c-missing)/len(pairs),(b-c+missing)/len(pairs)],
            'original_target_failures':sum(bool(p['original']) and target_failure(p['original']) for p in pairs),
            'candidate_target_failures':sum(bool(p['candidate']) and target_failure(p['candidate']) for p in pairs),
            'independent_trials':None,'conclusion_level':'exploratory','pairs':pairs}


def export_investigation(store,job,destination):
    dest=Path(destination)
    if dest.exists() and any(dest.iterdir()): raise ValueError('导出目录非空，请使用新目录')
    parent=manifest_for(store,job);links=children_for(store,job);steps=steps_for(store,job);result=result_for(store,job)
    manifests={l['child_id']:manifest_for(store,l['child_id']) for l in links}
    cm_rows=[r for r in store.rows('objects') if r['kind']=='confirmation' and store.object(r['hash'])['parent_manifest_id']==job]
    cm=store.object(cm_rows[-1]['hash']) if cm_rows else None
    rows_byjob={l['child_id']:[o for o in store.rows('observations') if o['job_id']==l['child_id']] for l in links}
    observations=[o for link in links for o in sorted(rows_byjob[link['child_id']],key=lambda o:o['position'])]
    phases={phase:counts_for([o for link in links if link['phase']==phase for o in rows_byjob[link['child_id']]],
                           2*len(cm['seeds']) if phase=='confirmation' and cm else None)
            for phase in ('baseline','search','confirmation',*[l['phase'] for l in links if l['phase'].startswith('trigger:')])}
    summary=counts_for(observations,sum(v['planned'] for v in phases.values()))
    summary.update(job=get_job(store,job),phases=phases,result=result,
                   confirmation=paired_summary(cm,links,manifests,observations),conclusion_level='exploratory',
                   limitations=['quick 对照仅为描述；不升级 confirmed_within_case，不推断总体失败概率。',
                                '只删除固定审核背景句，未实现任意文本缩减、confirm32 或白盒机制分析。',
                                '预算耗尽、中断、格式错误和截断保留在分母中。'])
    summary['trigger_comparisons']={t['factor']:paired_summary({'seeds':t['seeds']},links,manifests,observations,phase='trigger:'+t['factor'])
                                     for t in parent.get('trigger_plans',[])}
    summary['hypotheses']=[{'factor':f,'experiment_id':job,'regressions':v['regressions'],'improvements':v['improvements'],
                           'evaluable_pairs':v['evaluable_pairs'],'untested':'其他根案例与运行配置',
                           'next_experiment':'固定同一因素，在新根案例上进行配对；当前不能解释内部机制'}
                          for f,v in summary['trigger_comparisons'].items()]
    ids={o['id'] for o in observations}
    attempts=[a for a in store.rows('attempts') if a['observation_id'] in ids]
    hashes={a[k] for a in attempts for k in ('request_hash','response_hash') if a[k]}
    sizes=[s.get('size') for s in steps]
    if result: sizes.extend([result.get('original_size'),result.get('final_size')])
    hashes.update(s['prepared_request_hash'] for s in sizes if s)
    evidence={h:store.read_artifact(h) for h in hashes}
    case_hashes={parent['case_hash']}|{m['case_hash'] for m in manifests.values()}
    case_hashes.update(s['candidate_hash'] for s in steps if 'candidate_hash' in s)
    cases={h:store.object(h) for h in case_hashes}
    receipts={s['receipt_hash']:store.object(s['receipt_hash']) for s in steps if s.get('receipt_hash')}
    receipts.update({t['receipt_hash']:store.object(t['receipt_hash']) for t in parent.get('trigger_plans',[])})
    profile=store.object(parent['profile_hash'])
    best=result['final_case_hash'] if result else next((s['candidate_hash'] for s in reversed(steps) if s.get('accepted')),parent['case_hash'])
    original=cases[parent['case_hash']];candidate=cases[best]
    diff='\n'.join(difflib.unified_diff(canonical(original['messages']).splitlines(),canonical(candidate['messages']).splitlines(),
                                      fromfile='original',tofile='candidate',lineterm=''))
    dest.mkdir(parents=True,exist_ok=True);(dest/'artifacts').mkdir(exist_ok=True)
    for h,data in evidence.items(): (dest/'artifacts'/h).write_bytes(data)
    objects=[('manifest.json',parent),('profiles.json',[profile]),('summary.json',summary),
             ('result.json',result),('confirmation.json',cm),('children.json',links)]
    for name,obj in objects: (dest/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('cases.jsonl',cases.values()),('observations.jsonl',observations),('attempts.jsonl',attempts),
                      ('receipts.jsonl',receipts.values()),('child_manifests.jsonl',manifests.values()),('steps.jsonl',steps)]:
        (dest/name).write_text(''.join(canonical(r)+'\n' for r in rows))
    def block(obj): return '<pre>'+html.escape(json.dumps(obj,ensure_ascii=False,indent=2))+'</pre>'
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>有效输入缩减报告</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#182431}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f5;padding:14px}section{border-top:1px solid #bbb}summary{cursor:pointer}h1{font-size:26px}</style>'''
    doc+='<h1>行为调查 · '+('MOCK 合成输出' if profile['backend']=='mock' else '本地实际后端')+'</h1>'
    doc+='<p>缩减结果最多标为 best_found；未找到更短失败时保留原题。触发对照单独计数，未认证全局最短或内部机制。</p><h2>结果与全部分母</h2>'+block(summary)
    doc+='<h2>原题</h2>'+block(original['messages'])+'<h2>最终候选</h2>'+block(candidate['messages'])
    doc+='<h2>输入差异</h2><pre>'+html.escape(diff or '未接受更短候选')+'</pre>'
    doc+='<h2>变换有效性</h2>'+block(list(receipts.values()))+'<h2>完整轨迹</h2>'+block(steps)
    for link in links:
        doc+='<section><h2>'+html.escape(link['phase']+' / '+link['arm'])+'</h2>'+block(manifests[link['child_id']])
        for obs in rows_byjob[link['child_id']]:
            doc+=block(obs)
            for attempt in (a for a in attempts if a['observation_id']==obs['id']):
                doc+='<details><summary>尝试 '+html.escape(attempt['id'])+'</summary>'+block(attempt)
                for key in ('request_hash','response_hash'):
                    if attempt[key]: doc+='<h3>'+key+'</h3>'+block(json.loads(evidence[attempt[key]]))
                doc+='</details>'
        doc+='</section>'
    (dest/'report.html').write_text(doc+'</html>')
    (dest/'README.txt').write_text('离线打开 report.html。父清单共享唯一预算，child_manifests.jsonl 冻结每次观察条件。\n'
        '探索与新对照分别统计。artifact 文件名为 SHA256，包含实际请求与原始响应。\n'
        '若确认清单已冻结但调度中断，报告另列尚未创建观察的数量，不能视为已运行。\n'
        'MOCK 只验证软件流程；不代表真实模型失败。没有随包复制模型权重。\n')
    return summary
