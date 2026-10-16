"""Derive reports from real HTTP artifacts, including every failed attempt."""
import argparse,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def percentile(values,p):
 ordered=sorted(values);return ordered[max(0,math.ceil(p*len(ordered))-1)] if ordered else None

def main():
 p=argparse.ArgumentParser();p.add_argument('--complete-campaign');args=p.parse_args()
 rows=[]
 for path in sorted((ROOT/'artifacts').glob('live-*.jsonl')):
  for line in path.read_text().splitlines():
   row=json.loads(line)
   if 'model_runs' in row:row['artifact']=path.name;rows.append(row)
 live=[r for r in rows if any(json.loads(m['usage']).get('output_tokens',0)>0 for m in r['model_runs'])]
 elapsed=[(r.get('server_elapsed_ms') or r['elapsed_ms'])/1000 for r in live if not r.get('harness_paused')]
 calls=[len(r['model_runs']) for r in live]
 usage=[json.loads(m['usage']) for r in live for m in r['model_runs']]
 statuses={s:sum(r['status']['status']==s for r in live) for s in sorted({r['status']['status'] for r in live})}
 gpu_file=ROOT/'artifacts/gpu-samples.jsonl';gpu=[json.loads(x)['device_memory_mib'] for x in gpu_file.read_text().splitlines()] if gpu_file.exists() else []
 result={'real_model_turns':len(live),'harness_paused_turns':sum(bool(r.get('harness_paused')) for r in live),'distinct_player_texts':len({r['request']['text'] for r in live}),'zero_model_navigation_turns':len(rows)-len(live),'statuses':statuses,'degraded_commits':sum(bool((r['status'].get('result') or {}).get('degraded')) for r in live),'p50_seconds':round(statistics.median(elapsed),2) if live else None,'p95_seconds':round(percentile(elapsed,.95),2) if live else None,'model_calls':sum(calls),'max_calls_per_turn':max(calls,default=0),'input_tokens':sum(u.get('input_tokens',0) for u in usage),'output_tokens':sum(u.get('output_tokens',0) for u in usage),'max_torch_memory_mib':max((u.get('peak_torch_memory_mib',0) for u in usage),default=0),'max_device_memory_mib':max(gpu,default=0),'all_attempts_retained':True,'human_playtest':'not_supplied'}
 result['by_source_version']={}
 # Zero-generation clarification/rejection is not a navigation action.
 result['zero_model_turns']=len(rows)-len(live)
 live_ids={r['turn_id'] for r in live}
 result['zero_model_navigation_turns']=sum(r['turn_id'] not in live_ids and r['status']['status']=='committed' and r['request']['text'].startswith('前往') for r in rows)
 for version in sorted({r.get('source_version','unversioned_development') for r in live}):
  group=[r for r in live if r.get('source_version','unversioned_development')==version]
  times=[(r.get('server_elapsed_ms') or r['elapsed_ms'])/1000 for r in group if not r.get('harness_paused')]
  result['by_source_version'][version]={'turns':len(group),'statuses':{s:sum(r['status']['status']==s for r in group) for s in sorted({r['status']['status'] for r in group})},'degraded_commits':sum(bool((r['status'].get('result') or {}).get('degraded')) for r in group),'p50_seconds':round(statistics.median(times),2) if times else None,'p95_seconds':round(percentile(times,.95),2) if times else None}
 completions={r['campaign_id']:(r['status'].get('result') or {}).get('view',{}).get('outcome') for r in live if (r['status'].get('result') or {}).get('view',{}).get('outcome')}
 result['completed_campaigns']=completions
 (ROOT/'artifacts/live-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 lines=['# 真实模型回合记录','','由`evals/summarize_live.py`从原始JSONL生成；包括失败和重述，排除零模型移动。自动代理通过玩家HTTP接口操作，不是真人主观试玩。','','| 序号 | 战役 | 玩家输入 | tick | 状态 | 生成次数 | 秒 | 降级 |','|---|---|---|---:|---|---:|---:|---|']
 for i,r in enumerate(live,1):
  status=r['status'];res=status.get('result') or {};view=res.get('view',{});text=r['request']['text'].replace('|','／').replace('\n',' ')[:95]
  lines.append(f"| {i} | {r['campaign_id'][:8]} | {text} | {view.get('tick','未提交')} | {status['status']} | {len(r['model_runs'])} | {r['elapsed_ms']/1000:.1f} | {res.get('degraded','—')} |")
 lines+=['','完整请求/回应见对应`artifacts/live-<campaign_id>.jsonl`。其中可能含评测私下说明，保持在本地；`live-<campaign_id>.md`为玩家可见记录。']
 (ROOT/'docs/live-turns.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if args.complete_campaign:
  assert args.complete_campaign in completions,'Requested adventure has not completed'
  assert len(live)>=20,'Fewer than 20 real model turns'
if __name__=='__main__':main()
