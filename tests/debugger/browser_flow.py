"""Manual headless-browser harness, isolated MOCK database; never production user review."""
import os
import sys
from pathlib import Path
from qwen_debugger.storage import Store
from qwen_debugger.fixtures import install_fixtures
from qwen_debugger.fixed import freeze_profile
from qwen_debugger.domain import RuntimeProfile
from qwen_debugger.web import make_server

DRIVER=r'''
(async()=>{
 const out=document.createElement('pre');out.id='browser-acceptance';document.body.prepend(out);
 async function wait(fn){for(let i=0;i<100;i++){if(fn())return;await fetch('/browser-tick');}throw Error('timeout');}
 try{
  await wait(()=>document.querySelector('[name=profile]').options.length);
  document.querySelector('#experiment-form [name=repeats]').value='1';
  document.querySelector('#experiment-form').requestSubmit();
  await wait(()=>document.querySelector('#job-list')?.textContent.includes('COMPLETE'));
  document.querySelector('#job-list button').click();
  await wait(()=>!document.querySelector('#job-panel').hidden);
  document.querySelector('#export').click();
  await wait(()=>document.querySelector('#download-links a'));
  if(document.querySelector('#notice').classList.contains('error'))throw Error(document.querySelector('#notice').textContent);
  out.textContent='BROWSER_FLOW_PASS: create -> detached worker -> completed -> inspect -> export';
 }catch(e){out.textContent='BROWSER_FLOW_FAIL: '+e.message;}
})();
'''

def main():
 root=Path(sys.argv[1]);store=Store(root);install_fixtures(store)
 freeze_profile(store,RuntimeProfile(id='browser-mock',backend='mock',max_new_tokens=8))
 server=make_server(store,port=8767);Handler=server.RequestHandlerClass;original=Handler.do_GET;reply=Handler.reply
 def get(self):
  if self.path=='/browser-tick':
   import time
   time.sleep(.1)
   return self.reply({'tick':True})
  if self.path=='/driver.js':return self.reply(DRIVER.encode(),content_type='text/javascript; charset=utf-8')
  return original(self)
 def wrapped_reply(self,data,*args,**kwargs):
  if self.path=='/' and isinstance(data,bytes):data=data.replace(b'</head>',b'<script defer src="/driver.js"></script></head>')
  return reply(self,data,*args,**kwargs)
 Handler.do_GET=get;Handler.reply=wrapped_reply
 server.serve_forever()
if __name__=='__main__':main()
