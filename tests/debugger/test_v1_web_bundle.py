import json
import re
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from unittest.mock import patch
from qwen_debugger.web import make_server,action
from qwen_debugger.storage import Store
from qwen_debugger.fixtures import fixture_cases
from qwen_debugger.domain import RuntimeProfile,Budget
from qwen_debugger.runner import create_job,execute_job
from qwen_debugger.runtime import MockRuntime
from qwen_debugger.report import export_report
from qwen_debugger.bundle import import_bundle

class WebAndBundle(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.store=Store(self.root/'db')
  self.home=patch('qwen_debugger.runner.Path.home',return_value=self.root/'home');self.home.start()
 def tearDown(self):self.home.stop();self.t.cleanup()
 def test_bundle_restores_cases_and_archive_without_dispatch(self):
  c=fixture_cases()[0];p=RuntimeProfile(id='m',backend='mock');j=create_job(self.store,c,p,[42],Budget())
  execute_job(self.store,j,MockRuntime(p));export_report(self.store,j,self.root/'out')
  new=Store(self.root/'new');r=import_bundle(new,self.root/'out')
  self.assertEqual(new.case(c['case_id']),c);self.assertFalse(r['jobs_restarted']);self.assertEqual(new.rows('jobs'),[])
  a=next((self.root/'out/artifacts').iterdir());a.write_bytes(b'tampered')
  with self.assertRaises(ValueError):import_bundle(Store(self.root/'bad'),self.root/'out')
 def test_http_is_local_csrf_protected_and_import_to_export_works(self):
  server=make_server(self.store,port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  base='http://127.0.0.1:'+str(server.server_port);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  try:
   html=opener.open(base).read().decode();token=re.search(r'name="qdebug-token" content="([^"]+)"',html)[1]
   def call(action_,params,csrf=token,origin=base):
    req=urllib.request.Request(base+'/api/action',data=json.dumps({'action':action_,'params':params}).encode(),headers={'Content-Type':'application/json','X-Qdebug-Token':csrf,'Origin':origin})
    return json.load(opener.open(req))
   with self.assertRaises(urllib.error.HTTPError):call('fixtures',{},csrf='bad')
   with self.assertRaises(urllib.error.HTTPError):call('fixtures',{},origin='https://evil.example')
   self.assertEqual(call('fixtures',{})['count'],24)
   ref=call('profile.import',{'profile':{'id':'web-mock','backend':'mock','max_new_tokens':8}})['hash']
   p={'kind':'replay','case':'fixture_owner_1','profile':ref,'repeats':1,'budget':{'requests':1,'output_tokens':8,'seconds':60},'request_id':'unique-browser-action-1'}
   with patch('qwen_debugger.web.start_worker'):
    a=call('experiment.create',p);b=call('experiment.create',p)
   self.assertEqual(a,b);self.assertEqual(len(self.store.rows('jobs')),1)
   from qwen_debugger.service import run_job
   run_job(self.store,a['job_id']);r=call('export',{'id':a['job_id']})
   self.assertIn(b'PK',opener.open(base+r['download']).read(2))
   self.assertIn('PASS',opener.open(base+r['report']).read().decode())
   self.assertEqual(len(json.load(opener.open(base+'/api/state'))['jobs']),1)
  finally:server.shutdown();server.server_close();thread.join()
