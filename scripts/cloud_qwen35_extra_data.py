"""Download preference data and multilingual retrieval encoder directly on DSW."""
import json,urllib.request,urllib.parse,concurrent.futures
from pathlib import Path
from cloud_download import download
ROOT=Path('/mnt/workspace/qwen35_update')
def main():
 if not Path('/etc/dsw').exists():raise RuntimeError('Cloud only')
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 repo='intfloat/multilingual-e5-small'
 with op.open(f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true',timeout=30) as r:fs=json.load(r)['Data']['Files']
 names={'config.json','model.safetensors','tokenizer.json','tokenizer_config.json','sentencepiece.bpe.model','special_tokens_map.json','README.md'}
 selected=[f for f in fs if f['Path'] in names]
 folder=ROOT/'base/multilingual-e5-small';folder.mkdir(parents=True,exist_ok=True)
 (folder/'download-manifest.json').write_text(json.dumps({'model':repo,'files':selected},indent=2))
 def fetch(f):
  q=urllib.parse.urlencode({'Revision':f['Revision'],'FilePath':f['Path']})
  return download(f'https://modelscope.cn/api/v1/models/{repo}/repo?{q}',folder/f['Path'],f['Size'],f['Sha256'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(fetch,selected))
 (folder/'download-complete.json').write_text(json.dumps({'files':receipts,'cloud_direct':True},indent=2))
 repo='HuggingFaceH4/ultrafeedback_binarized';rev='40d965eb4ca093fd105933016ca5cc8623607fa3'
 with op.open(f'https://modelscope.cn/api/v1/datasets/{repo}/repo/tree?Revision={rev}&Root=&Recursive=true',timeout=30) as r:fs=json.load(r)['Data']['Files']
 files=[f for f in fs if f['Path'] in ['README.md','data/train_prefs-00000-of-00001.parquet','data/test_prefs-00000-of-00001.parquet']]
 target=ROOT/'data/ultrafeedback';target.mkdir(parents=True,exist_ok=True)
 receipts=[download(f'https://modelscope.cn/datasets/{repo}/resolve/{rev}/{f["Path"]}',target/Path(f['Path']).name,f['Size'],f['Sha256']) for f in files]
 (target/'manifest.json').write_text(json.dumps({'source':repo,'revision':rev,'license':'MIT','language':'en','files':receipts},indent=2))
 print('EXTRA_DATA_READY',flush=True)
if __name__=='__main__':main()
