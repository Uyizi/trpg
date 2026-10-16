"""Optional local telemetry: device-wide memory, not an app-only allocation."""
import json,subprocess,time
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'artifacts/gpu-samples.jsonl'
try:
 while True:
  p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
  if p.returncode==0:
   memory,load=map(int,p.stdout.splitlines()[0].split(','))
   with out.open('a') as f:f.write(json.dumps({'time':time.time(),'device_memory_mib':memory,'gpu_percent':load})+'\n')
  time.sleep(3)
except KeyboardInterrupt:pass
