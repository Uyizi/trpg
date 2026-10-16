import json
import tempfile
import threading
import unittest
import os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from qwen_debugger.domain import RuntimeProfile,Budget
from qwen_debugger.fixtures import fixture_cases
from qwen_debugger.storage import Store
from qwen_debugger.runner import create_job,execute_job
from qwen_debugger.api_runtime import ApiRuntime

class ApiTests(unittest.TestCase):
 def test_api_payload_unknown_identity_and_error_evidence(self):
  captured=[];failure=[False]
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    captured.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
    self.send_response(500 if failure[0] else 200);self.send_header('Content-Type','application/json');self.end_headers()
    self.wfile.write(json.dumps({'error':'synthetic error'} if failure[0] else {'choices':[{'message':{'content':'顾川'},'finish_reason':'stop'}],'usage':{'prompt_tokens':20,'completion_tokens':2}}).encode())
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with tempfile.TemporaryDirectory() as d,patch('qwen_debugger.runner.Path.home',return_value=Path(d)/'home'),patch.dict(os.environ,{'HTTP_PROXY':'http://127.0.0.1:1'}):
    store=Store(Path(d)/'db');p=RuntimeProfile(id='api-test',backend='api',api_base=f'http://127.0.0.1:{server.server_port}/v1',api_model='test',max_new_tokens=8)
    for bad in (False,True):
     failure[0]=bad;j=create_job(store,fixture_cases()[0],p,[42],Budget(requests=1,output_tokens=8))
     execute_job(store,j,ApiRuntime(p))
     o=[o for o in store.rows('observations') if o['job_id']==j][0]
     self.assertEqual(o['outcome'],'ERROR' if bad else 'PASS')
     a=next(a for a in store.rows('attempts') if a['id']==o['attempt_id'])
     self.assertIsNotNone(a['response_hash'])
    self.assertEqual(captured[0]['seed'],42)
    self.assertEqual(captured[0]['chat_template_kwargs'],{'enable_thinking':False})
  finally:server.shutdown();server.server_close();thread.join()
 def test_remote_address_or_secret_in_url_rejected(self):
  for url in ['https://example.com/v1','http://user:secret@localhost:8080/v1','http://localhost:8080/v1?key=secret']:
   with self.assertRaises(ValueError):RuntimeProfile(id='bad',backend='api',api_base=url,api_model='x')
