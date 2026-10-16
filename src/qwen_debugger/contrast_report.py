"""Offline paired evidence. Changed names are scored against mapped labels."""
import difflib
import html
import json
from pathlib import Path
from .domain import canonical
from .runner import manifest_for,get_job
from .shrink import children_for,observations_for
from .contrast import contrast_result
from .investigation_report import counts_for,paired_summary


def export_contrast(store,job,destination):
    dest=Path(destination)
    if dest.exists() and any(dest.iterdir()): raise ValueError('导出目录非空，请使用新目录')
    m=manifest_for(store,job);links=children_for(store,job)
    manifests={l['child_id']:manifest_for(store,l['child_id']) for l in links}
    observations=[o for l in links for o in observations_for(store,l['child_id'])]
    ids={o['id'] for o in observations}
    attempts=[a for a in store.rows('attempts') if a['observation_id'] in ids]
    hashes={a[k] for a in attempts for k in ('request_hash','response_hash') if a[k]}
    evidence={h:store.read_artifact(h) for h in hashes}
    profile=store.object(m['profile_hash']);receipt=store.object(m['receipt_hash'])
    cases={arm:store.object(m[arm+'_case_hash']) for arm in ('original','candidate')}
    rows=[]
    for link in links:
        for obs in observations_for(store,link['child_id']):
            attempt=next((a for a in attempts if a['id']==obs['attempt_id']),None)
            response=json.loads(evidence[attempt['response_hash']]) if attempt and attempt['response_hash'] else None
            request=json.loads(evidence[attempt['request_hash']]) if attempt else None
            raw=response.get('text') if response else None
            normalized=raw.strip() if raw is not None and 'strip_outer_whitespace' in cases[link['arm']]['gold']['normalizers'] else raw
            mapping=receipt['inverse_label_mapping'] if link['arm']=='candidate' else {k:k for k in receipt['label_mapping']}
            g=json.loads(obs['grade']) if obs['grade'] else None
            rows.append({'position':link['position'],'seed':obs['seed'],'arm':link['arm'],
                         'expected':cases[link['arm']]['gold']['expected'],'raw_text':raw,
                         'mapped_exact_label':mapping.get(normalized),'outcome':obs['outcome'],'state':obs['state'],
                         'failure_kind':g.get('failure_kind') if g else None,
                         'input_tokens':request.get('input_tokens') if request else None,
                         'output_tokens':response.get('usage',{}).get('output_tokens') if response else None,
                         'observation_id':obs['id'],'attempt':attempt})
    summary=counts_for(observations,m['planned_observations'])
    arms={arm:counts_for([o for l in links if l['arm']==arm for o in observations_for(store,l['child_id'])],m['pairs'])
          for arm in ('original','candidate')}
    summary.update(job=get_job(store,job),factor=m['factor'],arms=arms,rows=rows,
                   paired=paired_summary(m,links,manifests,observations,phase='contrast'),
                   result=contrast_result(store,job),selected_on_outcome=m['selected_on_outcome'],conclusion_level='exploratory',
                   limitations=['每个因素、案例单独报告，不以最差结果代替全部观察。',
                                '语义保持只适用于结构化实体题；改名可能改变分词长度，报告记录实际 token 数。',
                                '配对差值是变体相对原题的精确判分失败率差；含格式错误，语义错误另列。',
                                '不同种子不保证独立，尤其是贪心解码；未提供置信区间、总体结论或内部机制结论。',
                                '未运行、异常、截断及不可评价结果均保留在全部计划分母中。'])
    dest.mkdir(parents=True,exist_ok=True);(dest/'artifacts').mkdir(exist_ok=True)
    for h,data in evidence.items(): (dest/'artifacts'/h).write_bytes(data)
    for name,obj in [('manifest.json',m),('profiles.json',[profile]),('receipt.json',receipt),('summary.json',summary),
                     ('result.json',summary['result']),('children.json',links)]:
        (dest/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,items in [('cases.jsonl',cases.values()),('observations.jsonl',observations),('attempts.jsonl',attempts),
                       ('child_manifests.jsonl',manifests.values())]:
        (dest/name).write_text(''.join(canonical(v)+'\n' for v in items),encoding='utf-8')
    def block(obj):return '<pre>'+html.escape(json.dumps(obj,ensure_ascii=False,indent=2))+'</pre>'
    text={arm:'\n'.join(v['content'] for v in case['messages']) for arm,case in cases.items()}
    diff='\n'.join(difflib.unified_diff(text['original'].splitlines(),text['candidate'].splitlines(),
                                      fromfile='原题',tofile='变体',lineterm=''))
    label={'rename':'实体改名','noise-position':'背景位置','permute-options':'选项置换','insert-noise':'独立背景','reorder-facts':'无序事实','flip-query':'反事实查询'}[m['factor']]
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>触发条件对照</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#182431}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f5;padding:14px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd3db;padding:8px;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style>'''
    doc+='<h1>触发条件对照 · '+label+' · '+('MOCK 合成输出' if profile['backend']=='mock' else '本地真实模型')+'</h1>'
    doc+='<p>原题与变体分别对照各自的标准答案。此报告描述当前案例的观察，不证明内部原因或总体可靠性。</p>'
    doc+='<h2>逐次结果（执行顺序）</h2><table><tr><th>种子</th><th>条件</th><th>标准答案</th><th>原始回答</th><th>判分 / 分类</th><th>输入 token</th></tr>'
    for row in rows:
        values=[row['seed'],row['arm'],row['expected'],row['raw_text'],
                (row['outcome'] or row['state'])+' / '+(row['failure_kind'] or '—'),row['input_tokens']]
        doc+='<tr>'+''.join('<td>'+html.escape(str(v) if v is not None else '未取得')+'</td>' for v in values)+'</tr>'
    doc+='</table><h2>输入差异</h2><pre>'+html.escape(diff)+'</pre><h2>配对统计</h2>'+block(summary['paired'])
    doc+='<details><summary>原题、变体与有效性规则</summary>'+block(cases)+block(receipt)+'</details>'
    doc+='<details><summary>全部分母与限制</summary>'+block(summary)+'</details>'
    doc+='<details><summary>冻结运行计划</summary>'+block(m)+block(profile)+'</details>'
    for row in rows:
        doc+='<details><summary>'+html.escape(str(row['seed'])+' / '+row['arm'])+' · 原始证据</summary>'+block(row)
        if row['attempt']:
            for key in ('request_hash','response_hash'):
                h=row['attempt'][key]
                if h:doc+='<h3>'+key+'</h3>'+block(json.loads(evidence[h]))
        doc+='</details>'
    (dest/'report.html').write_text(doc+'</html>',encoding='utf-8')
    (dest/'README.txt').write_text('离线打开 report.html。每个处理臂按自己的 gold 判分，receipt.json 记录双向名称映射。\n'
        'manifest.json 冻结全部配对及顺序；子任务共用父预算。输入 token 不可取得时记为 null。\n'
        'MOCK 是软件演示。结果仅探索性，不是训练收益或模型内部机制的证明。\n',encoding='utf-8')
    return summary
