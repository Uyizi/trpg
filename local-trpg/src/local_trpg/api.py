"""Player-only HTTP DTOs; private module files are never static assets."""
import asyncio,json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import Response,FileResponse,JSONResponse
from pydantic import Field
from .config import Settings,PROJECT_ROOT
from .schemas import Strict,PlayerCommand
from .storage import Store,Conflict
from .modules import available_modules,module_by_id
from .engine.rules import initial_state,RuleError
from .engine.turns import TurnService
from .model.queue import GenerationQueue
from .model.runner_backend import RunnerBackend
class PlayerSetup(Strict):
 name:str=Field(default='检修员',min_length=1,max_length=30)
 character_id:str|None=Field(default=None,max_length=40)
class NewCampaign(Strict):module_id:str='reserve_cooling';player_setup:PlayerSetup=Field(default_factory=PlayerSetup)
class SaveRequest(Strict):name:str=Field(default='手动存档',min_length=1,max_length=80)

def create_app(settings=None,queue=None,agent_factory=None):
 config=settings or Settings();registry={m['module_id']:m for m in available_modules()};store=Store(config.data_dir/'game.sqlite')
 q=queue or GenerationQueue(RunnerBackend(config))
 from .adventure.service import AdventureService
 services={id:(AdventureService if m.get('engine')=='scene_d20' else TurnService)(store,m,q,agent_factory,narration_mode=config.narration_mode) for id,m in registry.items()}
 service=services['reserve_cooling']
 def module_for(id):
  if id not in registry:raise RuleError('冒险不存在')
  return registry[id]
 def campaign_service(id):
  snapshot=store.snapshot(id);m=module_for(snapshot.module_id)
  if snapshot.module_hash!=m['module_hash']:raise RuleError('存档的冒险版本不匹配，未使用其他规则读取')
  return services[snapshot.module_id]
 def bundle_module(bundle):
  if not isinstance(bundle,dict) or not isinstance(bundle.get('state'),dict):raise RuleError('备份格式不正确')
  return module_for(bundle['state'].get('module_id','reserve_cooling'))
 model_state={'ready':False,'loading':False,'error':None};warm_task=None
 async def warm():
  model_state.update(loading=True,error=None)
  try:await q.warmup();model_state['ready']=True
  except Exception as exc:
   model_state.update(ready=False,error=type(exc).__name__)
   with (config.data_dir/'errors.log').open('a') as f:
    import traceback;traceback.print_exc(file=f)
  finally:model_state['loading']=False
 @asynccontextmanager
 async def lifespan(app):
  nonlocal warm_task
  store.recover();warm_task=asyncio.create_task(warm())
  yield
  # Own model process may be stopped on app shutdown; incomplete drafts recover.
  tasks=[task for svc in services.values() for task in svc.tasks.values()]
  for task in tasks:
   if not task.done():task.cancel()
  await asyncio.gather(*tasks,return_exceptions=True)
  if warm_task and not warm_task.done():warm_task.cancel();await asyncio.gather(warm_task,return_exceptions=True)
  await q.close()
 app=FastAPI(title='本地文字冒险',docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
 app.state.service=service;app.state.services=services;app.state.store=store;app.state.model_state=model_state;app.state.queue=q
 @app.middleware('http')
 async def local_only(request,call_next):
  host=request.headers.get('host','').split(':')[0]
  if host not in ('127.0.0.1','localhost','testserver'):return JSONResponse({'detail':'仅允许本机访问'},status_code=403)
  origin=request.headers.get('origin')
  if origin and origin!=f'{request.url.scheme}://{request.headers.get("host")}':return JSONResponse({'detail':'拒绝跨站请求'},status_code=403)
  if int(request.headers.get('content-length','0'))>16*1024*1024:return JSONResponse({'detail':'请求过大'},status_code=413)
  response=await call_next(request)
  response.headers.update({'X-Content-Type-Options':'nosniff','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",'Cache-Control':'no-store'})
  return response
 @app.exception_handler(Conflict)
 async def conflict(r,exc):return JSONResponse({'detail':str(exc)},status_code=409)
 @app.exception_handler(KeyError)
 async def missing(r,exc):return JSONResponse({'detail':'记录不存在'},status_code=404)
 @app.exception_handler(ValueError)
 async def invalid(r,exc):return JSONResponse({'detail':str(exc)},status_code=422)
 def ready():return model_state['ready'] and getattr(getattr(q,'backend',None),'available',getattr(getattr(q,'backend',None),'_ready',True))
 @app.get('/api/health')
 async def health():return {'status':'loading' if model_state['loading'] else 'ready' if ready() else 'unavailable','backend':config.backend,'model_available':ready(),'model_busy':bool(q.active),'mock':config.backend=='mock'}
 @app.post('/api/model/warmup')
 async def warmup():
  nonlocal warm_task
  if model_state['loading'] or q.active:return {'status':'busy'}
  warm_task=asyncio.create_task(warm());return {'status':'loading'}
 @app.get('/api/modules')
 async def modules():
  # The module manifest also contains room contents for the server. A public
  # adventure picker needs only its pitch and basic player-facing rules.
  result=[]
  for m in registry.values():
   p=m['public'];item={'module_id':p['module_id'],'title':p['title'],'opening':p['opening'],
    'description':p.get('description',p.get('premise','')),'rules':p['rules'],
    'ruleset':p.get('ruleset','reserve_2d6')}
   if m.get('engine')=='scene_d20':
    item.update(characters=p['characters'],source_url=p['source_url'],adaptations=m['world']['adaptations'])
   else:item['locations']=[{'id':id,'name':place['name']} for id,place in p['locations'].items()]
   result.append(item)
  return result
 @app.get('/api/campaigns')
 async def campaigns():return store.list_campaigns()
 @app.post('/api/campaigns')
 async def new(body:NewCampaign):
  module=module_for(body.module_id)
  if module.get('engine')=='scene_d20':
   from .adventure.engine import initial_adventure
   s=initial_adventure(module,body.player_setup.name,body.player_setup.character_id or 'noorah',mock=config.backend=='mock')
  else:
   if body.player_setup.character_id:raise RuleError('本冒险没有这张角色卡')
   s=initial_state(module,body.player_setup.name,mock=config.backend=='mock')
  store.create(s)
  return {'campaign_id':s.campaign_id,'view':services[body.module_id].view(s.campaign_id)}
 @app.get('/api/campaigns/{id}')
 async def campaign(id:str):return campaign_service(id).view(id)
 @app.post('/api/campaigns/{id}/turns',status_code=202)
 async def submit(id:str,body:PlayerCommand):
  # Existing IDs are always queryable even if the model stopped since commit.
  if not ready() and body.mode!='ooc':
   with store.tx() as db:existing=db.execute('SELECT 1 FROM turns WHERE campaign=? AND request_id=?',(id,body.client_request_id)).fetchone()
   if not existing:raise HTTPException(503,'模型正在加载或不可用；游戏存档保留。')
  return await campaign_service(id).submit_turn(id,body)
 @app.get('/api/campaigns/{id}/turns/{turn_id}')
 async def status(id:str,turn_id:str):return campaign_service(id).status(id,turn_id)
 @app.post('/api/campaigns/{id}/turns/{turn_id}/cancel')
 async def cancel(id:str,turn_id:str):return campaign_service(id).cancel(id,turn_id)
 @app.post('/api/campaigns/{id}/turns/{turn_id}/resume')
 async def resume(id:str,turn_id:str):return await campaign_service(id).resume(id,turn_id)
 @app.post('/api/campaigns/{id}/saves')
 async def save(id:str,body:SaveRequest):return store.save(id,body.name)
 @app.get('/api/campaigns/{id}/saves')
 async def saves(id:str):return store.saves(id)
 @app.post('/api/saves/{id}/fork')
 async def fork(id:str):
  with store.tx() as db:row=db.execute('SELECT bundle FROM saves WHERE id=?',(id,)).fetchone()
  if not row:raise KeyError('存档不存在')
  module=bundle_module(json.loads(row['bundle']))
  campaign=store.fork(id,module['module_hash']);return {'campaign_id':campaign,'view':campaign_service(campaign).view(campaign)}
 @app.post('/api/import')
 async def restore(request:Request):
  raw=await request.body()
  if len(raw)>16*1024*1024:raise HTTPException(413,'存档过大')
  bundle=json.loads(raw);module=bundle_module(bundle)
  campaign=store.restore(bundle,module['module_hash']);return {'campaign_id':campaign,'view':campaign_service(campaign).view(campaign)}
 @app.get('/api/campaigns/{id}/transcript')
 async def transcript(id:str):
  s=store.snapshot(id);text='# '+module_for(s.module_id)['public']['title']+' · 玩家记录\n\n'+'\n\n'.join(f"第{x['tick']}格 · {x['speaker']}\n\n{x['text']}" for x in s.transcript)
  return Response(text,media_type='text/markdown',headers={'Content-Disposition':'attachment; filename="adventure.md"'})
 @app.get('/api/campaigns/{id}/export')
 async def export(id:str):return Response(json.dumps(store.export(id),ensure_ascii=False),media_type='application/json',headers={'Content-Disposition':'attachment; filename="campaign-backup.json"'})
 @app.get('/{path:path}')
 async def static(path:str):
  root=(PROJECT_ROOT/'web/dist').resolve();file=(root/(path or 'index.html')).resolve()
  if not file.is_relative_to(root) or not file.is_file():raise HTTPException(404,'页面不存在或前端尚未构建')
  return FileResponse(file)
 return app
