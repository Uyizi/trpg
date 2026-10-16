"""Merge completed style adapter in BF16 before NF4 quantization."""
import sys,json,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from models.qwen35_moe import load_dense
from cloud_qwen35_followup import atomic_json,BASE,ROOT,sha
from transformers import AutoTokenizer

def main():
 out=ROOT/'exports/Qwen3.5-4B-Dense-style-v1'
 adapter=ROOT/'runs/qwen35_style_v1/adapter'
 with (ROOT/'logs/export_dense.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  digest=sha(adapter/'adapter_model.safetensors')
  if (out/'merged_provenance.json').exists():
   if json.loads((out/'merged_provenance.json').read_text())['adapter_sha256']!=digest:raise ValueError('Dense export changed')
   print('DENSE_EXPORT_READY',out,flush=True);return
  if out.exists():raise ValueError('Incomplete destination exists')
  tmp=out.with_name(out.name+'.incomplete');tmp.mkdir(parents=True,exist_ok=True)
  model=load_dense(BASE,adapter).merge_and_unload(safe_merge=True)
  model.config.use_cache=True;model.config.text_config.use_cache=True
  model.save_pretrained(tmp,max_shard_size='2GB',safe_serialization=True)
  AutoTokenizer.from_pretrained(BASE,local_files_only=True).save_pretrained(tmp)
  atomic_json(tmp/'merged_provenance.json',{'base':str(BASE),'adapter':str(adapter),'adapter_sha256':digest})
  tmp.rename(out)
  print('DENSE_EXPORT_READY',out,flush=True)
if __name__=='__main__':main()
