"""Paired, root-weighted and family-weighted evidence for fixed experiments."""
import html
import json
from collections import Counter,defaultdict
from pathlib import Path
from .domain import canonical
from .runner import manifest_for,get_job
from .shrink import children_for,observations_for,target_failure
from .investigation_report import counts_for
from .fixed import fixed_result


def metrics(obs,n):
    r=counts_for(obs,n)
    grades=[json.loads(o['grade']) for o in obs if o.get('grade')]
    r.update(end_to_end_pass_rate=r['counts']['PASS']/n if n else None,
             semantic_failures=sum(g.get('failure_kind')=='semantic_mismatch' for g in grades),
             format_failures=sum(g.get('failure_kind') in ('required_format_missing','unclassified_response') for g in grades))
    r['semantic_failure_rate_planned']=r['semantic_failures']/n if n else None
    r['format_failure_rate_planned']=r['format_failures']/n if n else None
    return r


def summarize_fixed(store,job):
    m=manifest_for(store,job);links=children_for(store,job)
    obs=[];rows=[];by_pair=defaultdict(dict);byroot=defaultdict(lambda:defaultdict(list));families={}
    attempts={a['id']:a for a in store.rows('attempts')}
    for index,slot in enumerate(m['schedule']):
        observations=observations_for(store,links[index]['child_id']) if index<len(links) else []
        o=observations[0] if observations else None
        if o:obs.append(o)
        a=attempts.get(o['attempt_id']) if o else None
        req=json.loads(store.read_artifact(a['request_hash'])) if a else None
        res=json.loads(store.read_artifact(a['response_hash'])) if a and a['response_hash'] else None
        case=store.object(slot['case_hash']);root=case['root_case_id'];families[root]=case['family_id']
        row={**slot,'observation':o,'attempt':a,'expected':case['gold']['expected'],'root_case_id':root,'family_id':case['family_id'],
             'task_type':case['input'].get('task_type','raw'),'raw_text':res.get('text') if res else None,
             'input_tokens':req.get('input_tokens') if req else None,'input_fingerprint':req.get('fingerprint') if req else None,
             'input_ids_hash':None,'rendered_prompt_hash':None,'output_tokens':res.get('usage',{}).get('output_tokens') if res else None,
             'finish_reason':res.get('finish_reason') if res else None,'elapsed':a['elapsed'] if a else None}
        from .domain import digest
        if req and req.get('input_ids') is not None:row['input_ids_hash']=digest(req['input_ids'])
        if req and req.get('rendered_prompt') is not None:row['rendered_prompt_hash']=digest(req['rendered_prompt'])
        rows.append(row);by_pair[slot['pair_id']][slot['arm']]=row;byroot[root][slot['arm']].append(o)
    pairs=[];b=c=n=0
    for id,arms in by_pair.items():
        x,y=arms.get('original'),arms.get('candidate');a=x['observation'] if x else None;z=y['observation'] if y else None
        valid=bool(a and z and a['outcome'] in ('PASS','FAIL') and z['outcome'] in ('PASS','FAIL'))
        regression=bool(valid and a['outcome']=='PASS' and z['outcome']=='FAIL')
        improvement=bool(valid and a['outcome']=='FAIL' and z['outcome']=='PASS')
        if valid:n+=1;b+=regression;c+=improvement
        tokens_known=bool(x and y and x['input_ids_hash'] and y['input_ids_hash'])
        pairs.append({'pair_id':id,'evaluable':valid,'regression':regression,'improvement':improvement,
                      'token_ids_differ':x['input_ids_hash']!=y['input_ids_hash'] if tokens_known else None,
                      'input_token_counts':[x['input_tokens'] if x else None,y['input_tokens'] if y else None],
                      'original':x,'candidate':y})
    missing=len(pairs)-n
    paired={'planned_pairs':len(pairs),'evaluable_pairs':n,'missing_or_unevaluable_pairs':missing,
            'regressions':b,'improvements':c,'delta':(b-c)/n if n else None,
            'delta_missing_sensitivity':[(b-c-missing)/len(pairs),(b-c+missing)/len(pairs)],'pairs':pairs}
    arms={arm:metrics([r['observation'] for r in rows if r['arm']==arm and r['observation']],sum(r['arm']==arm for r in rows)) for arm in ('original','candidate')}
    roots={root:{arm:metrics([o for o in items if o],len(items)) for arm,items in arms_.items()} for root,arms_ in byroot.items()}
    family={}
    for f in sorted(set(families.values())):
        rr=[r for r in roots if families[r]==f]
        family[f]={'root_count':len(rr),'roots':rr,'end_to_end_pass_rate':{arm:sum(roots[r][arm]['end_to_end_pass_rate'] for r in rr)/len(rr) for arm in arms}}
    macro={'root_count':len(roots),'family_count':len(family),'roots':roots,'families':family,
           'family_macro_pass_rate':{arm:sum(v['end_to_end_pass_rate'][arm] for v in family.values())/len(family) for arm in arms},
           'weighting':'variant/repeat mean within root, equal roots within family, equal families overall'}
    result=metrics(obs,m['planned_observations'])
    result.update(job=get_job(store,job),kind=m['kind'],metadata=m['metadata'],arms=arms,paired=paired,macro=macro,rows=rows,
                  result=fixed_result(store,job),conclusion_level='exploratory',
                  limitations=['缺失敏感性范围不是置信区间。','无独立种子能力证明，不报告强统计结论。',
                    'family 来自声明的来源分组；少数合成模板不能当成大量独立任务。',
                    'profile 分块顺序运行以限制显存；耗时包含首请求加载，不能直接当作纯生成速度。',
                    '输入 token 不可观察时标 null；实际 token 不同不作单模块因果归因。'])
    if m['kind']=='confirmation':
        target={arm:sum(target_failure(r['observation']) for r in rows if r['arm']==arm and r['observation']) for arm in arms}
        complete=all(r['observation'] and r['observation']['state']=='DONE' for r in rows)
        recurrence=complete and all(target[arm]==m['metadata']['pairs'] for arm in arms)
        rule=m['metadata']['confirmation_rule'];passed=bool(recurrence and rule['mode']=='observed_recurrence' and rule['eligible_for_upgrade'])
        consumed=[store.object(r['hash']) for r in store.rows('objects') if r['kind']=='data_consumption' and store.object(r['hash'])['experiment']==job]
        result['confirmation']={'complete':complete,'all_target_recurred':recurrence,'rule_passed':passed,
                                'target_failures':target,'rule':rule,'consumed_for_development':bool(consumed),
                                'conclusion_level':'confirmed_within_case' if passed and not consumed else 'exploratory'}
    if m['kind']=='matrix':
        result['hypotheses']=[{'factor':p['factor'],'experiment_id':job,
                              'claim':'观察此因素是否伴随更多精确判分失败；尚非内部机制解释',
                              'support_pairs':[x['pair_id'] for x in pairs if x['pair_id'].startswith(p['factor']+':') and x['regression']],
                              'counterexample_pairs':[x['pair_id'] for x in pairs if x['pair_id'].startswith(p['factor']+':') and x['evaluable'] and not x['regression']],
                              'untested':'其他案例、长度和模型配置', 'next_experiment':'在新的同类根案例上固定同一因素重做配对'} for p in m['metadata'].get('pairs',[])[:3]]
    return result


def export_fixed(store,job,destination):
    dest=Path(destination)
    if dest.exists() and any(dest.iterdir()):raise ValueError('导出目录非空')
    summary=summarize_fixed(store,job);m=manifest_for(store,job);links=children_for(store,job)
    attempts=[r['attempt'] for r in summary['rows'] if r['attempt']]
    hashes={a[k] for a in attempts for k in ('request_hash','response_hash') if a[k]}
    evidence={h:store.read_artifact(h) for h in hashes}
    cases={s['case_hash']:store.object(s['case_hash']) for s in m['schedule']}
    profiles={s['profile_hash']:store.object(s['profile_hash']) for s in m['schedule']}
    receipts={s['receipt_hash']:store.object(s['receipt_hash']) for s in m['schedule'] if s.get('receipt_hash')}
    dest.mkdir(parents=True,exist_ok=True);(dest/'artifacts').mkdir(exist_ok=True)
    for h,data in evidence.items():(dest/'artifacts'/h).write_bytes(data)
    for name,obj in [('manifest.json',m),('summary.json',summary),('profiles.json',list(profiles.values())),('children.json',links)]:
        (dest/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,items in [('cases.jsonl',cases.values()),('receipts.jsonl',receipts.values()),('attempts.jsonl',attempts),
                       ('observations.jsonl',[r['observation'] for r in summary['rows'] if r['observation']]),
                       ('child_manifests.jsonl',[manifest_for(store,l['child_id']) for l in links])]:
        (dest/name).write_text(''.join(canonical(x)+'\n' for x in items),encoding='utf-8')
    def block(x):return '<pre>'+html.escape(json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
    title={'regression':'版本回归','confirmation':'独立确认','matrix':'触发调查'}[m['kind']]
    label='MOCK 合成结果' if all(p['backend']=='mock' for p in profiles.values()) else '实际运行（各 profile 详见清单）'
    doc='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+title+'</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f4;padding:14px}td,th{border:1px solid #ccc;padding:8px;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}</style>'
    doc+='<h1>'+title+' · '+label+'</h1><p>全部计划分母保留；精确判分失败包含格式问题。声明的运行差异不自动证明内部原因。</p>'
    doc+='<h2>两臂计数</h2>'+block(summary['arms'])+'<h2>逐对结果</h2><table><tr><th>案例</th><th>A 原始回答</th><th>B 原始回答</th><th>变化</th><th>输入 token</th></tr>'
    for p in summary['paired']['pairs']:
        values=[p['pair_id'],p['original']['raw_text'],p['candidate']['raw_text'],
                '退步' if p['regression'] else '改善' if p['improvement'] else '未改变判分' if p['evaluable'] else '不可评价',p['input_token_counts']]
        doc+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in values)+'</tr>'
    doc+='</table><h2>家族宏平均</h2>'+block(summary['macro'])+'<h2>条件差异</h2>'+block(m['metadata'])
    for name,obj in [('完整统计',summary),('原始案例',list(cases.values())),('变换与审核收据',list(receipts.values())),('冻结清单',m)]:
        doc+='<details><summary>'+name+'</summary>'+block(obj)+'</details>'
    for h,data in evidence.items():doc+='<details><summary>原始证据 '+h+'</summary>'+block(json.loads(data))+'</details>'
    (dest/'report.html').write_text(doc+'</html>',encoding='utf-8')
    (dest/'README.txt').write_text('离线打开 report.html。cases.jsonl 可再次导入；完整迁移使用 bundle.json。\n预算未完成与判分未知均保留；权重不随报告分发。\n',encoding='utf-8')
    return summary
