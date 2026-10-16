import copy,json,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from test_core import owner
from qwen_debugger.domain import CaseImport,RuntimeProfile,Budget,digest
from qwen_debugger.tasks import compile_case
from qwen_debugger.storage import Store
from qwen_debugger.suites import create_suite,compare_plan
from qwen_debugger.bundle import import_bundle
from qwen_debugger.api_runtime import ApiRuntime
from qwen_debugger.runtime import UnknownCompletion

class ReviewFixes(unittest.TestCase):
 def setUp(self):self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);self.s=Store(self.root/'db')
 def tearDown(self):self.t.cleanup()
 def make_case(self,id='owner',root='owner',family='family',split='development'):
  c=owner();c.update(case_id=id,root_case_id=root,family_id=family,split=split);return compile_case(CaseImport.model_validate(c))
 def test_existing_root_cannot_change_family(self):
  self.s.put_case(self.make_case())
  with self.assertRaises(ValueError):self.s.put_case(self.make_case('held','owner','fake','heldout'))
 def test_heldout_rechecked_after_development_suite_consumes_it(self):
  c=self.make_case(split='heldout');self.s.put_case(c);suite=create_suite(self.s,'held',['owner'],purpose='heldout')
  create_suite(self.s,'dev',['owner'])
  p=RuntimeProfile(id='m',backend='mock',max_new_tokens=8)
  with self.assertRaises(ValueError):compare_plan(self.s,suite,p,p,Budget(requests=2,output_tokens=16))
 def bundle(self,records):
  out=self.root/'bundle';out.mkdir();(out/'artifacts').mkdir()
  payload={'schema':'qdebug-evidence-1','source_experiment':'external','objects':records,'jobs':[],
           'observations':[],'attempts':[],'artifacts':[],'manifest_links':[]}
  (out/'bundle.json').write_text(json.dumps({'digest':digest(payload),'payload':payload}));return out
 def record(self,kind,id,body):return {'kind':kind,'id':id,'revision':1,'hash':digest(body),'body':json.dumps(body)}
 def test_import_rejects_forged_suite_and_missing_case_parent(self):
  a=self.make_case();h=self.s.put_case(a)
  suite={'id':'forged','revision':1,'parent_hash':None,'purpose':'heldout','created':'external','case_hashes':[h],
         'family_ids':['invented'],'root_case_ids':['other'],'grouping_evidence':'invented','source_experiment':None}
  with self.assertRaises(ValueError):import_bundle(self.s,self.bundle([self.record('suite','forged',suite)]))
  self.assertFalse(any(r['kind']=='suite' for r in self.s.rows('objects')))
 def test_rejected_duplicate_identity_import_is_atomic(self):
  p=RuntimeProfile(id='same',backend='mock').model_dump(mode='json');q={**p,'max_new_tokens':64}
  with self.assertRaises(ValueError):import_bundle(self.s,self.bundle([self.record('profile','same',p),self.record('profile','same',q)]))
  self.assertEqual(self.s.rows('objects'),[])
 def test_api_trickle_cannot_extend_absolute_deadline(self):
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_POST(self):
    self.rfile.read(int(self.headers['Content-Length']));self.send_response(200);self.end_headers()
    try:
     for _ in range(25):self.wfile.write(b' ');self.wfile.flush();time.sleep(.1)
     self.wfile.write(b'{}')
    except (BrokenPipeError,ConnectionResetError):pass
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  p=RuntimeProfile(id='api',backend='api',api_base=f'http://127.0.0.1:{server.server_port}/v1',api_model='test')
  r=ApiRuntime(p);r.deadline=time.monotonic()+.5;start=time.monotonic()
  try:
   with self.assertRaises(UnknownCompletion):r.generate(r.prepare(self.make_case(),42))
   self.assertLess(time.monotonic()-start,1.2)
  finally:r.close();server.shutdown();server.server_close();thread.join()
 def test_import_missing_case_revision_parent_is_rejected_before_writes(self):
  c=owner();c.update(revision=2,parent_hash='0'*64)
  compiled=compile_case(CaseImport.model_validate(c));r=self.record('case',c['case_id'],compiled);r['revision']=2
  with self.assertRaises(ValueError):import_bundle(self.s,self.bundle([r]))
  self.assertEqual(self.s.rows('objects'),[])
