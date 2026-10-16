"""Four separate roles. Only a whitelisted View becomes the role's messages."""
import asyncio,json,time,uuid
from pathlib import Path
from ..schemas import PlayerIntent,NpcDecision,SceneProposal,NarrationResult,GuideReviewView,GuideReviewResult,AdventureIntent
from ..model.base import Message,GenerationRequest
from ..model.parsing import parse_result,InvalidOutput
from ..model.compact_npc import compact_npc_schema
from ..storage import digest
from ..memory import current_player_messages

PROMPTS=Path(__file__).with_name('prompts')
class AgentRunner:
 def __init__(self,queue,store=None,turn_id=None,epoch=0):
  self.queue=queue;self.store=store;self.turn_id=turn_id;self.epoch=epoch;self.calls=0;self.repairs=0;self.deadline=time.monotonic()+120;self.metrics=[]
  self._prefix_scope='isolated:'+uuid.uuid4().hex
 async def invoke(self,role,view,schema,limit,feedback=None):
  data=view.model_dump();h=digest(data);key=role+':'+str(data.get('actor_id',''))+(':repair' if feedback else '')
  if self.store:
   cached=self.store.cached_call(self.turn_id,key,h)
   if cached is not None:return schema.model_validate(cached)
  system=(PROMPTS/(role+'.txt')).read_text()
  # Build each role from its own immutable projection and explicit current task.
  suffix='';protected=set()
  if role in ('interpreter','adventure_interpreter'):
   suffix='\n本次玩家原话：'+data['text']+'\n只解释上面这次原话，不执行其他动作。'
   if role=='adventure_interpreter':suffix+='如果眼前目标或做法与这句原话不符，必须澄清；不得挑最接近的可用动作替代原意。移动要按scene.exits的方向描述选目标，不得默认选择第一个出口。'
  if role=='guide':suffix='\n玩家这次的桌外问题：'+data['question']+'\n先回答这个问题；不重复无关规则，不把历史台词当成新指令。'
  if role=='npc':
   current=current_player_messages(data.get('observations',[]),data['scene'])
   # Keep the latest actual player message, not every message sharing a tick.
   current=current[-1:]
   protected.update(o['observation_id'] for o in current)
   own_results=[o for o in data.get('observations',[]) if o.get('source_actor_id')==data['actor_id'] and o.get('modality')=='direct']
   protected.update(o['observation_id'] for o in own_results[-1:])
   protected.update(ref for b in data.get('beliefs',[]) for ref in b['source_observation_ids'])
   suffix='\n本轮你实际听到玩家说：'+('\n'.join(o['content'] for o in current) if current else '没有新的直接提问。只对本轮新情况作必要回应，不重复自己的旧台词。')
   if data['scene'].get('reply_focus')=='status_report':
    own=next(p for p in data['scene']['people'] if p['id']==data['actor_id'])
    health={'healthy':'健康','strained':'轻伤','injured':'受伤','incapacitated':'失能'}[own['health']]
    suffix+='\n你的身份是'+data['name']+'；你的当前位置是'+data['scene']['location']+'；你自己的身体状态是'+health+'。其他角色的自述不能当成自己的位置或状态。'
    suffix+='\n本轮是岗位情况汇报。以本人身份说出scene中的当前位置、自己的身体状态和可确认的当前情况；不复述玩家的提问，不要求其他人代答，不编造事故原因或已执行的工作。自己的身体健康不能证明区域安全：没有全面检查记录，就说明其他情况尚未核实，禁止补写“一切正常”“没有发现异常”。使用玩家本轮的通话频道回复。没有获准披露的具体资料时，d填空数组或省略。'
   work=next((a for a in data['allowed_actions'] if a['kind']=='MODULE_ACTION'),None)
   if work:suffix+='\n本次工作请求：'+json.dumps(work,ensure_ascii=False,separators=(',',':'))+'。接受则提交此MODULE_ACTION；不接受则说明条件，不得只口头答应。'
  if feedback:suffix+='\n上次提议被规则拒绝：'+feedback[:240]+'。在同一视图权限内重做，不声称已执行。'
  def user_message():
   return Message(role='user',content=json.dumps(data,ensure_ascii=False,separators=(',',':'))+suffix)
  msgs=[Message(role='system',content=system),user_message()]
  count=await asyncio.to_thread(self.queue.count_tokens,msgs)
  while count>4096 and role=='npc':
   candidates=[(i,o) for i,o in enumerate(data['observations']) if not o.get('evidence_refs') and o['observation_id'] not in protected]
   removable=min(candidates,key=lambda pair:(bool(pair[1].get('proposition_ids')),pair[1]['observed_at'],pair[0]))[0] if candidates else None
   if removable is None:break
   data['observations'].pop(removable)
   ids={o['observation_id'] for o in data['observations']};data['beliefs']=[b for b in data['beliefs'] if set(b['source_observation_ids'])<=ids]
   # Rebuilding JSON must not erase the current task appended after it.
   msgs[-1]=user_message();count=await asyncio.to_thread(self.queue.count_tokens,msgs)
  while count>4096 and role=='guide' and data.get('history'):
   # Keep the current question and factual sources; only trim older OOC chat.
   data['history'].pop(0);msgs[-1]=user_message()
   count=await asyncio.to_thread(self.queue.count_tokens,msgs)
  if count>4096:raise InvalidOutput('角色上下文超预算，未删除关键权限和证据')
  wire_schema=compact_npc_schema() if role=='npc' else schema.model_json_schema()
  if role=='adventure_interpreter':
   from ..adventure.wire import adventure_intent_schema
   wire_schema=adventure_intent_schema(view)
  scope=self.store.turn(self.turn_id)['campaign'] if self.store and self.turn_id else self._prefix_scope
  # Only the fixed role prompt is reusable. The complete current projection,
  # observations, permissions, feedback and repair transcript remain variable.
  namespace=json.dumps([scope,role,data.get('actor_id',''),digest(wire_schema)],ensure_ascii=False,separators=(',',':'))
  for attempt in range(2):
   remaining=self.deadline-time.monotonic()
   if self.calls>=6 or remaining<=0:raise InvalidOutput('本回合生成预算已用尽')
   self.calls+=1
   request=GenerationRequest(request_id=uuid.uuid4().hex,session_id=self.turn_id or 'probe',role_kind=role,messages=msgs,max_output_tokens=limit,temperature=0.1,response_schema=wire_schema,thinking_policy='disabled',timeout_seconds=min(120,max(.01,remaining)),prefix_message_count=1,cache_namespace=namespace)
   result=await self.queue.generate(request)
   metric={'role':role,'usage':result.usage,'elapsed_ms':result.elapsed_ms,'error':result.error};self.metrics.append(metric)
   if result.error and any(x in result.error for x in ['timeout','busy','closed','worker','warmed']):self.deadline=0
   try:
    output=parse_result(result,schema,actor_id=data.get('actor_id') if role=='npc' else None)
    if self.store:self.store.record_metric(self.turn_id,metric)
    if self.store:self.store.record_call(self.turn_id,self.epoch,key,h,output.model_dump(),metric)
    return output
   except InvalidOutput as exc:
    metric['error']=metric['error'] or 'invalid_output:'+str(exc)[:300]
    if self.store:self.store.record_metric(self.turn_id,metric)
    if attempt or self.repairs>=1 or self.deadline<=time.monotonic():raise
    self.repairs+=1
    # No other role, private context or arbitrary program execution in repair.
    msgs=msgs+[Message(role='assistant',content=result.final_text),Message(role='user',content='修复上一JSON的具体错误：'+str(exc)[:240]+'。保持原角色权限，只返回完整JSON。')]
    if await asyncio.to_thread(self.queue.count_tokens,msgs)>4096:raise InvalidOutput('修复上下文超预算')
  raise InvalidOutput('输出无效')
 async def interpret(self,view):return await self.invoke('interpreter',view,PlayerIntent,192)
 async def decide_npc(self,view):return await self.invoke('npc',view,NpcDecision,256)
 async def repair_npc(self,view,reason):
  if self.repairs>=1:raise InvalidOutput('本回合修复预算已用尽')
  self.repairs+=1
  return await self.invoke('npc',view,NpcDecision,256,feedback=reason)
 async def plan_scene(self,view):return await self.invoke('director',view,SceneProposal,128)
 async def narrate(self,view):return await self.invoke('narrator',view,NarrationResult,192)
 async def guide(self,view):
  return await self.grounded_answer('guide',view)
 async def interpret_adventure(self,view):return await self.invoke('adventure_interpreter',view,AdventureIntent,224)
 async def adventure_npc(self,view):return await self.grounded_answer('adventure_npc',view)
 async def adventure_narrate(self,view):return await self.invoke('adventure_narrator',view,NarrationResult,384)
 async def grounded_answer(self,role,view):
  # Review receives sources and the draft, not past assistant guesses. Feedback
  # and revision remain actual model calls; no canned answer replaces a failure.
  limit=192 if role=='adventure_npc' else 384
  answer=await self.invoke(role,view,NarrationResult,limit)
  for attempt in range(2):
   speaker=next((s.get('speaker_name') for s in view.sources if s['event_id']=='identity'),None) if role=='adventure_npc' else None
   review=await self.invoke('guide_review',GuideReviewView(question=view.question,sources=view.sources,answer=answer.text,speaker_name=speaker),GuideReviewResult,160)
   if review.supported and not review.unsupported_claims:
    if self.store:self.store.record_call(self.turn_id,self.epoch,role+':',digest(view.model_dump()),answer.model_dump(),{})
    return answer
   if attempt or self.repairs>=1:raise InvalidOutput('场外回答含有来源不支持的具体设定')
   self.repairs+=1
   reason='以下具体表述没有来源支持：'+'；'.join(review.unsupported_claims)
   answer=await self.invoke(role,view,NarrationResult,limit,feedback=reason+'。删除这些设定；缺少资料就说明角色尚未取得，不列举通常或可能的物资、数量。')
  raise InvalidOutput('场外答复来源检查失败')
