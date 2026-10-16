"""Round-trip an actual completed campaign through public save/export APIs."""
import argparse,json,hashlib
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('campaign');args=parser.parse_args()
 with httpx.Client(base_url='http://127.0.0.1:8768',trust_env=False,timeout=15) as client:
  def get(path):
   response=client.get(path);response.raise_for_status();return response.json()
  def post(path,body=None):
   response=client.post(path,json=body);response.raise_for_status();return response.json()
  before=get(f'/api/campaigns/{args.campaign}')
  assert before['outcome'],'The real adventure must finish before this check.'
  bundle=get(f'/api/campaigns/{args.campaign}/export')
  saved=post(f'/api/campaigns/{args.campaign}/saves',{'name':'实机完整结局'})
  save_id=saved.get('save_id',saved.get('id'))
  fork=post(f'/api/saves/{save_id}/fork')
  restored=post('/api/import',bundle)
  # Compare all authoritative snapshot fields; only fork identity/revision differ.
  restored_bundle=get(f'/api/campaigns/{restored["campaign_id"]}/export')
  fork_bundle=get(f'/api/campaigns/{fork["campaign_id"]}/export')
  def meaningful(state):return {k:v for k,v in state.items() if k not in ('campaign_id','revision')}
  assert meaningful(bundle['state'])==meaningful(restored_bundle['state'])==meaningful(fork_bundle['state'])
  assert before==get(f'/api/campaigns/{args.campaign}')
  transcript=client.get(f'/api/campaigns/{args.campaign}/transcript');transcript.raise_for_status()
  assert '/home/example' not in transcript.text and 'private_aside' not in transcript.text
  report={'campaign_id':args.campaign,'save_id':save_id,'fork_campaign_id':fork['campaign_id'],'import_campaign_id':restored['campaign_id'],'outcome':before['outcome'],'all_snapshot_fields_preserved_except_identity_revision':True,'original_unchanged':True,'share_excludes_local_paths_and_private_aside_field':True,'backup_sha256':hashlib.sha256(json.dumps(bundle,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
  path=ROOT/'artifacts'/f'live-save-{args.campaign}.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
  print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
