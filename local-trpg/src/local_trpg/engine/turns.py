"""One ordered draft per campaign; await generation outside SQLite transactions."""
import asyncio,time,re,copy
from ..schemas import PlayerCommand,PlayerIntent,ActionIntent,InterpreterView,DirectorView,TurnReceipt,ContextBudget
from ..storage import Conflict,turn_transcript
from ..agents.runner import AgentRunner
from ..model.parsing import InvalidOutput
from .rules import Engine,RuleError
from .projection import scene,build_agent_view,build_narration_view,player_view,disclosure,disclosure_options
from .scheduler import select_npcs
from .guidance import guide_view
from ..memory import current_player_messages

class TurnService:
 def __init__(self,store,module,queue,agent_factory=None,narration_mode='model'):
  if narration_mode not in ('brief','model'):raise ValueError('未知叙述模式')
  self.narration_mode=narration_mode
  # Transport-only player speech; never write this cache to SQLite or saves.
  self.provisional={}
  self.store=store;self.module=module;self.queue=queue;self.tasks={};self.agent_factory=agent_factory or (lambda turn,epoch:AgentRunner(queue,store,turn,epoch));self.last_metrics={}
 async def submit_turn(self,campaign_id,command):
  if command.recipient_ids and not set(command.recipient_ids)<=set(self.store.snapshot(campaign_id).actors):raise RuleError('收件人不存在')
  id=self.store.submit(campaign_id,command)
  if self.store.turn(id)['status']=='queued' and (id not in self.tasks or self.tasks[id].done()):self.tasks[id]=asyncio.create_task(self.run_turn(id))
  return TurnReceipt(turn_id=id,status=self.store.turn(id)['status'])
 async def resume(self,campaign,id):
  self.store.resume(campaign,id);self.tasks[id]=asyncio.create_task(self.run_turn(id));return self.status(campaign,id)
 def status(self,campaign,id):
  status=self.store.status(campaign,id);preview=self.provisional.get(id)
  if status['status']=='completed_ooc':status['table_talk']=self.store.table_talk(campaign)
  valid=(preview and status['status']=='running' and not status['cancel_requested']
   and all(preview[key]==status[key] for key in ('campaign_id','turn_id','epoch','base_revision'))
   and status['revision']==preview['base_revision'])
  status['provisional']=copy.deepcopy(preview) if valid else None
  if preview and not valid:self.provisional.pop(id,None)
  return status
 def cancel(self,campaign,id):
  self.store.cancel(campaign,id);self.provisional.pop(id,None)
  return self.status(campaign,id)
 def publish_replies(self,id,epoch,engine):
  # Recheck the execution lease after every await, before exposing a proposal.
  self.store.progress(id,epoch,'相关船员回应')
  public=build_narration_view(engine.state,engine.events,module=self.module).events
  dialogue=[{key:event[key] for key in ('event_id','speaker','speaker_id','text')}
   for event in public if event['kind']=='heard' and event['speaker_id']!='player']
  if dialogue:self.provisional[id]={'campaign_id':engine.state.campaign_id,'turn_id':id,
   'epoch':epoch,'base_revision':engine.state.revision,'dialogue':dialogue}
 def project(self,state,active_turn_id=None):
  view=player_view(state,self.module,active_turn_id)
  view['table_talk']=self.store.table_talk(state.campaign_id)
  return view
 def view(self,campaign):return self.project(self.store.snapshot(campaign),self.store.active(campaign))
 def interpreter_view(self,state,command):
  w=self.module['world'];known=state.evidence.get('player',[])
  current=scene(state,'player',self.module)
  current['locations']=[{'id':k,'name':v['name']} for k,v in w['locations'].items()]
  current['roster']=[{'id':id,'name':p['name'],'role':p['role']} for id,p in w['actors'].items()]
  # IDs and action meanings are public rule interfaces, never private preconditions.
  actions=[{'id':id,'name':p.get('title',id)} for id,p in w['actions'].items()]+[{'id':'choose_station','name':'选择前往维修站'}]
  clarification=None
  if command.reply_to_turn_id:
   old=self.store.turn(command.reply_to_turn_id)
   if old['campaign']!=state.campaign_id or old['status']!='needs_input':raise RuleError('回复目标不是当前战役的澄清问题')
   clarification=old['reply']
  return InterpreterView(text=command.text,channel=command.speech_channel,recipient_ids=command.recipient_ids,scene=current,known_evidence=[{'id':e,'title':w['evidence'][e]['title']} for e in known],legal_actions=actions,clarification=clarification)
 def shortcut(self,command,state=None):
  # Exact public choices are also available as UI suggestions. All effects
  # still go through the same rule prerequisites, clock and transaction.
  text=command.text.strip().rstrip('。！？?')
  if state is not None and self.store and command.reply_to_turn_id and text in ('确认','确定','是','是的','好的','出发','起飞','确认出发'):
   previous=self.store.turn(command.reply_to_turn_id)
   if previous['campaign']==state.campaign_id and previous['status']=='needs_input' and previous['base_revision']==state.revision:
    if previous['reply'].startswith('前往维修站会离开当前现场'):
     return PlayerIntent(kind='USE_ITEM',module_action_id='choose_station',attempt_description='确认前往最近维修站')
    if previous['reply'].startswith('启动会离开主船并结算本章'):
     return PlayerIntent(kind='USE_ITEM',module_action_id='launch_pod',attempt_description='确认启动救生艇')
  if text in ('简图','查看简图','查看地图','船内简图'):
   return PlayerIntent(kind='INSPECT',attempt_description='查看简图')
  if state is not None and state.actors['player'].location_id=='cargo' and text in ('故障贴纸','原制冷故障贴纸','查看故障贴纸','检查原制冷故障贴纸','查看货物标签'):
   return PlayerIntent(kind='INSPECT',evidence_refs=['E05'],attempt_description=text)
  if state is not None and state.actors['player'].location_id in ('cargo','clinic') and text in ('查看货物说明','阅读货物说明','查看货物说明副本'):
   return PlayerIntent(kind='INSPECT',evidence_refs=['E05'],attempt_description=text)
  if text in ('都有谁在','这里有谁','附近有谁','现在谁在这里'):
   return PlayerIntent(kind='INSPECT',attempt_description='环顾当前位置，查看在场人物。')
  if text=='查看这里有什么可以调查的':
   return PlayerIntent(kind='INSPECT',attempt_description=text)
  if text=='使用工作频道询问当前情况':
   return PlayerIntent(kind='ASK',spoken_text='各位，请汇报你所在岗位的当前情况。',attempt_description=text)
  if text=='检查动力舱的阀组与操作卡':
   return PlayerIntent(kind='INSPECT',evidence_refs=['E01','E04'],attempt_description=text)
  if text=='询问医生如何保冷和转存':
   return PlayerIntent(kind='ASK',target_ids=['doctor'],spoken_text='方榆，细胞如何保冷和转存？',attempt_description=text)
  if text=='向方榆出示补给票E07':
   return PlayerIntent(kind='SHOW_EVIDENCE',target_ids=['doctor'],evidence_refs=['E07'],spoken_text='方榆，请核对本次储冷补给票E07。',attempt_description=text)
  if text=='向大家说明已掌握的证据' and state is not None and state.evidence.get('player'):
   return PlayerIntent(kind='SHOW_EVIDENCE',evidence_refs=state.evidence['player'],spoken_text='这些是我已经取得的记录，请大家核对。',attempt_description=text)
  if text in ('选择前往最近维修站','我选择前往最近维修站','确认前往最近维修站'):
   return PlayerIntent(kind='USE_ITEM',module_action_id='choose_station',attempt_description=text)
  if text=='确认启动救生艇':
   return PlayerIntent(kind='USE_ITEM',module_action_id='launch_pod',attempt_description=text)
  for id,p in self.module['world']['locations'].items():
   if text in ('前往'+p['name'],'去'+p['name'],'移动到'+p['name']):return PlayerIntent(kind='MOVE',target_ids=[id],attempt_description=text)
  for id,work in self.module['world']['actions'].items():
   title=work['title']
   if 'player' in work['actor_ids'] and text in ('执行'+title,'我执行'+title):
    return PlayerIntent(kind='USE_ITEM',module_action_id=id,attempt_description=text)
   for actor,profile in self.module['world']['actors'].items():
    if actor in work['actor_ids'] and text in (profile['name']+'，请执行'+title,profile['name']+',请执行'+title):
     return PlayerIntent(kind='ASK',module_action_id=id,target_ids=[actor],spoken_text=text,attempt_description=text)
  return None
 async def run_turn(self,id):
  try:t=self.store.claim(id)
  except Conflict:return
  epoch=t['epoch'];state=self.store.snapshot(t['campaign']);command=PlayerCommand.model_validate_json(t['command']);agent=self.agent_factory(id,epoch);started=time.monotonic();degraded=False
  engine=Engine(state,self.module,id,lambda key:self.store.roll(id,key));dialogue=[]
  try:
   if command.mode=='ooc':
    if not command.text.strip():self.store.finish(id,epoch,'needs_input','请写下想向主持人询问的问题。');return
    self.store.progress(id,epoch,'Qwen回答场外问题')
    try:
     view=guide_view(command.text,state,self.module,self.store.table_talk(state.campaign_id))
     output=await agent.guide(view)
     allowed={x['event_id'] for x in view.sources}|({'history'} if view.history else set())
     if not output.text.strip() or not output.source_event_ids or not set(output.source_event_ids)<=allowed:raise InvalidOutput('场外回答没有合法来源')
    except (InvalidOutput,ValueError,AttributeError):
     self.store.finish(id,epoch,'failed','本地Qwen这次未能生成可靠答复，请稍后重试。本次没有改变游戏状态。');return
    self.store.finish(id,epoch,'completed_ooc',output.text);return
   if not command.text.strip():self.store.finish(id,epoch,'needs_input','私下说明已保留在本次请求中。请另行说明角色实际要说或做什么。');return
   intent=self.shortcut(command,state)
   confirmed_action=intent.module_action_id if intent and intent.attempt_description in ('确认前往最近维修站','确认启动救生艇') else None
   if intent is None:intent=await agent.interpret(self.interpreter_view(state,command))
   self.store.progress(id,epoch,'裁决行动')
   if intent.needs_clarification or intent.kind is None:
    self.store.finish(id,epoch,'needs_input',intent.clarification_question or '请明确角色要说出口的话或实际行动。');return
   # choose_station is a registered voyage decision, not a room or NPC.
   # Only remove that exact action ID if the runner also used it as a target;
   # any other invented target still reaches the ordinary rejection path.
   if intent.module_action_id=='choose_station' and set(intent.target_ids)<={'choose_station'}:
    intent.kind='USE_ITEM';intent.target_ids=[]
   if intent.kind=='IMPROVISE':
    proposal=await agent.plan_scene(DirectorView(action=intent.model_dump(exclude={'private_intent'}),scene=scene(state,'player',self.module),available_actions=self.interpreter_view(state,command).legal_actions,private_aside=command.private_aside))
    if not proposal.module_action_id:self.store.finish(id,epoch,'needs_input',proposal.clarification or '请明确具体操作方式。');return
    intent.module_action_id=proposal.module_action_id;intent.target_ids=proposal.target_ids;intent.kind='USE_ITEM'
   # Telegraph an irreversible chapter-ending decision before resolving it.
   # Confirmation states only public consequences, not hidden cargo deadlines.
   ending_action=intent.module_action_id
   if ending_action=='choose_station' and state.flags.get('main_loop_restored') and confirmed_action!='choose_station':
    self.store.finish(id,epoch,'needs_input','前往维修站会离开当前现场并结束本章，途中时间仍会流逝。你可以先询问航程或核对准备。确定出发，回复“确认”即可；否则可以继续行动。本次尚未出发，也未消耗时间。');return
   if ending_action=='launch_pod' and state.flags.get('pod_prepared') and state.actors['player'].onboard and confirmed_action!='launch_pod':
    self.store.finish(id,epoch,'needs_input','启动会离开主船并结算本章，只带走已经登艇的人和实际装上的物品。确定起飞，回复“确认”即可；也可以先核对人员和物资。本次尚未启动，也未消耗时间。');return
   # Explicit UI fields have priority over model-invented channels/recipients.
   action=ActionIntent(actor_id='player',kind=intent.kind,target_ids=intent.target_ids,description=intent.attempt_description or command.text,
    spoken_text=intent.spoken_text,channel=command.speech_channel,recipient_ids=command.recipient_ids or intent.recipient_ids,
    evidence_refs=intent.evidence_refs,module_action_id=intent.module_action_id,item_ids=intent.item_ids)
   # Private intentions do not enter action/event/agent/narration views.
   if action.kind in ('TALK','ASK','PERSUADE','DECEIVE','SHOW_EVIDENCE') and not action.spoken_text:
    self.store.finish(id,epoch,'needs_input','请明确你要说出口的话；我不会把未标明的心理说明转告船员。');return
   engine.resolve_player(action)
   self.store.progress(id,epoch,'相关船员回应',engine.state.model_dump())
   targets=action.target_ids+action.recipient_ids
   if engine.authorization:targets=[engine.authorization['actor_id']]+targets
   reaction_limit=1 if action.channel=='private' and len(set(action.recipient_ids))==1 else 2
   # Information-only dialogue may take no game time, but listeners still
   # react to the new speech event. Do not tie hearing to clock advancement.
   spoken=any(e.kind=='SpeechOccurred' and e.actor_id=='player' for e in engine.events)
   selected=select_npcs(engine.state,engine.events,reaction_limit,self.module,targets) if (engine.state.tick>state.tick or spoken) and engine.state.phase!='epilogue' else []
   # Routine physical work already has a rule result. In brief mode, defer
   # unaddressed reactions instead of asking two NPCs to say they have nothing
   # to add. Received observations and pending replies remain in the state;
   # explicit social interactions and work requests still invoke real NPCs.
   if (self.narration_mode=='brief' and not engine.authorization
       and action.kind not in ('TALK','ASK','PERSUADE','DECEIVE','SHOW_EVIDENCE')
       and not any(target in self.module['world']['actors'] for target in targets)):
    selected=[]
   for npc in selected:
    self.store.progress(id,epoch,'相关船员回应')
    view=build_agent_view(engine.state,npc,ContextBudget(),self.module,engine.authorization,
     status_report=intent.attempt_description=='使用工作频道询问当前情况' and action.kind=='ASK' and not engine.authorization,
     reply_to_player=action.kind in ('TALK','ASK','PERSUADE','DECEIVE','SHOW_EVIDENCE'),current_event_ids={e.event_id for e in engine.events})
    work=engine.authorization if engine.authorization and engine.authorization['actor_id']==npc else None
    try:
     decision=await agent.decide_npc(view)
     try:self.apply_npc(engine,npc,decision,view)
     except (RuleError,ValueError) as exc:
      repair=getattr(agent,'repair_npc',None)
      if not repair or agent.repairs>=1 or agent.calls>=6 or time.monotonic()>=getattr(agent,'deadline',float('inf')):raise
      decision=await repair(view,str(exc))
      self.apply_npc(engine,npc,decision,view)
     self.publish_replies(id,epoch,engine)
    except (InvalidOutput,RuleError,ValueError) as exc:
     degraded=True;engine.notice(f'{engine.state.actors[npc].name}本轮回应未能可靠生成，保持原状态；你可以继续行动。')
    if work:
     # Report the adjudicated command, not an inference from a spoken promise.
     title=self.module['world']['actions'][work['module_action_id']]['title']
     status='已执行本次操作' if engine.authorization is None else '本轮未执行'
     engine.notice(f'工作结果：{engine.state.actors[npc].name} — {title}：{status}。',actor=npc)
   pending={a:n+1 for a,n in engine.state.pending_npcs.items() if a not in selected}
   for o in engine.state.observations:
    if o.event_id in {e.event_id for e in engine.events} and o.observer_id not in selected and o.observer_id!='player':pending[o.observer_id]=pending.get(o.observer_id,0)+1
   engine.emit('PlansUpdated',{'pending_npcs':pending})
   engine.finish_clock()
   self.store.progress(id,epoch,'整理可见结果',engine.state.model_dump())
   narration_view=build_narration_view(engine.state,engine.events,module=self.module)
   public= narration_view.events
   dialogue=[{'speaker':x['speaker'],'speaker_id':x['speaker_id'],'text':x['text']} for x in public if x['kind']=='heard' and x['speaker_id']!='player']
   # NPC speech is displayed verbatim. A second paraphrase can reverse who
   # requested a task or turn a promise into an accomplished action.
   narration_view=narration_view.model_copy(update={'events':[x for x in public if x['kind']!='heard']})
   summary='\n'.join(dict.fromkeys(x['text'] for x in public if x['kind']!='heard'))
   if not summary:
    if dialogue:summary='工作频道传来回应。' if action.channel=='broadcast' else '你收到以下回应。'
    elif action.spoken_text:summary='发言已送达，但本轮没有收到船员回应。'
    else:summary='本轮没有新的可见变化。'
   if self.narration_mode=='brief' or intent.kind=='MOVE' or not narration_view.events:narration=summary
   else:
    try:
     output=await agent.narrate(narration_view)
     allowed={x['event_id'] for x in narration_view.events}
     if not output.source_event_ids or not set(output.source_event_ids)<=allowed:raise InvalidOutput('叙述引用未知事件')
     if any(x in output.text for x in ['秘密通道','隐藏出口','暗藏武器','传染体','投毒','谋杀']):raise InvalidOutput('叙述加入未裁决内容')
     for person in self.module['world']['actors'].values():
      name=person['name']
      if re.search(re.escape(name)+r'(?:说|表示|声称|要求|回答)',output.text) and not any(x['kind']=='heard' and x['speaker']==name for x in public):
       raise InvalidOutput('叙述把玩家的话转给了未发言的人')
     # Always retain mechanical facts as source-backed cards even if prose omits them.
     narration=output.text+'\n\n'+summary if summary and summary not in output.text else output.text
    except (InvalidOutput,ValueError):narration=summary;degraded=True
   result={'narration':narration,'dialogue':dialogue,'checks':engine.checks,'degraded':degraded}
   prospective=engine.state.model_copy(deep=True);prospective.revision+=1
   prospective.transcript.extend(turn_transcript(id,prospective.tick,prospective.actors['player'].name,command.model_dump(),result))
   result['view']=self.project(prospective)
   self.store.commit(id,epoch,engine.state,engine.events,result)
  except RuleError as exc:self.store.finish(id,epoch,'needs_input',str(exc))
  except Conflict:pass
  except InvalidOutput as exc:self.store.finish(id,epoch,'failed','本地模型未能解释这次行动，游戏状态未改变。请重述或检查模型状态。')
  except Exception as exc:
   self.store.finish(id,epoch,'failed','回合处理失败，游戏状态未改变。错误类型：'+type(exc).__name__)
   import traceback
   log=self.store.path.parent/'errors.log'
   with log.open('a') as f:traceback.print_exc(file=f)
  finally:
   if self.provisional.get(id,{}).get('epoch')==epoch:self.provisional.pop(id,None)
   self.last_metrics[id]={'calls':agent.calls,'repairs':agent.repairs,'elapsed_ms':(time.monotonic()-started)*1000,'model_runs':agent.metrics,'degraded':degraded}
 def apply_npc(self,engine,npc,decision,view):
  self.validate_npc(engine,npc,decision,view)
  permissions=disclosure_options(engine.state,npc,self.module)
  # Stage action, speech, disclosures and beliefs as one proposal. Nothing is
  # published if any later recipient, evidence or speech check fails.
  trial=Engine(engine.state,self.module,engine.turn_id,engine.rolls)
  trial.events=list(engine.events);trial.authorization=engine.authorization
  d=decision;auth=engine.authorization
  if d.action.kind=='MODULE_ACTION':
   trial.authorized_items=auth.get('item_ids',[])
   trial.module_action(d.action.module_action_id,npc,d.action.target_ids or auth.get('target_ids',[]))
   trial.flag('cooperation_with:'+npc,True)
   trial.authorization=None
  if d.utterance:
   u=d.utterance;trial.speech(npc,u.text,u.channel,u.recipient_ids)
   speech_id=trial.events[-1].event_id if trial.events and trial.events[-1].kind=='SpeechOccurred' else None
   recipients=sorted({o.observer_id for o in trial.state.observations if o.event_id==speech_id and o.observer_id!=npc})
   for ref in dict.fromkeys(u.disclose_fact_ids):
    permission=permissions[ref]
    if permission['kind']=='evidence':
     evidence=self.module['world']['evidence'][ref]
     text=evidence['text']
     if permission['source_kind']=='testimony':text=trial.state.actors[npc].name+'的说明：'+text
     trial.emit('EvidenceObserved',{'observer_ids':recipients,'evidence_id':ref,'content':text,
      'proposition_ids':evidence['proposition_ids'],'source':npc,'source_kind':permission['source_kind'],
      'source_observation_ids':permission['source_observation_ids']},npc)
     if npc=='mechanic' and ref in ('E03','E04') and 'player' in recipients:trial.flag('mechanic_guidance',True)
    else:
     label='亲历证词' if permission['source_kind']=='firsthand' else '转述'
     trial.emit('FactDisclosed',{'observer_ids':recipients,'proposition_ids':[ref],
      'content':trial.state.actors[npc].name+label+'：'+permission['text'],
      'source_kind':permission['source_kind'],'source_observation_ids':permission['source_observation_ids']},npc)
  for update in d.belief_updates:trial.emit('BeliefRevised',{'actor_id':npc,**update.model_dump(),'updated_at':trial.state.tick},npc)
  if d.next_goal_id:
   goal=next(g['id'] for g in view.available_goals if d.next_goal_id in (g['id'],g['text']))
   trial.emit('NpcGoalChanged',{'actor_id':npc,'goal_id':goal},npc)
  engine.state=trial.state;engine.events=trial.events;engine.checks+=trial.checks;engine.authorization=trial.authorization
 def validate_npc(self,engine,npc,d,view):
  if d.actor_id!=npc:raise RuleError('不能冒充其他角色')
  known={o['observation_id'] for o in view.observations}
  for b in d.belief_updates:
   if not b.source_observation_ids or not set(b.source_observation_ids)<=known or b.proposition_id not in self.module['world']['facts']:raise RuleError('非法观察或信念引用')
  action=d.action
  if action.kind not in ('SPEAK','WAIT','WITHHOLD','REQUEST_HELP','MODULE_ACTION'):raise RuleError('本次视图没有该动作权限')
  if d.next_goal_id and not any(d.next_goal_id in (g['id'],g['text']) for g in view.available_goals):raise RuleError('未知目标')
  if action.kind=='MODULE_ACTION':
   auth=engine.authorization
   if not auth or auth['actor_id']!=npc or auth['module_action_id']!=action.module_action_id or auth['tick']!=engine.state.tick:raise RuleError('没有本回合动作授权')
   if not set(action.item_ids)<=set(auth['item_ids']):raise RuleError('超出物品授权')
   if not set(action.target_ids)<=set(auth['target_ids']):raise RuleError('超出目标授权')
  if d.utterance:
   u=d.utterance
   if view.scene.get('reply_focus')=='status_report':
    questions=[o['content'] for o in current_player_messages(view.observations,view.scene)]
    if questions and u.text.strip(' 。！？\n')==questions[-1].strip(' 。！？\n'):raise RuleError('情况汇报必须回答自己的岗位状况，不能只复述玩家的问题')
    if view.scene['location'] not in u.text:raise RuleError('岗位汇报必须准确说明你自己的当前位置：'+view.scene['location'])
    # A narrow guard for the observed status-report failure, not a general
    # natural-language fact checker. Own health does not establish room safety.
    for clause in re.split('[。！？；，]',u.text):
     if re.search('一切正常|没有发现异常|未发现异常|没有异常',clause) and not re.search('不能|无法|尚未|不敢|并非|不代表|还没|尚无|希望',clause):
      raise RuleError('没有现场全面检查依据，不能断言区域一切正常；只汇报已知位置和身体状态，其他情况说明尚未核实')
   current_question=view.scene.get('reply_to_player') and bool(current_player_messages(view.observations,view.scene))
   if current_question and 'player' in u.recipient_ids and u.channel=='local' and engine.state.actors[npc].location_id!=engine.state.actors['player'].location_id:
    raise RuleError('玩家不在你的现场，听不到现场回答；请使用工作频道或向player私人回复')
   auth=engine.authorization
   if auth and auth['actor_id']==npc and action.kind!='MODULE_ACTION':
    # Narrow consistency guard, not a general semantic classifier. Conditional
    # offers and refusals remain legal; acceptance is never forced by code.
    conditional=re.search(r'如果|只要|除非|等到|等你|等拿|等确认|需要先|必须先|先[^。！？]*再',u.text)
    promise=re.search(r'(?:^|[。！？；，\s])(?:好[的]?，?)?(?:我|我们)(?:现在|马上|立即|这就|会|来|将|负责|准备)(?:就|来|负责)?(?:执行|动手|转存|隔离|修复|恢复|治疗|准备救生艇|备艇|发出|发送|照做|做)',u.text)
    if promise and not conditional:raise RuleError('台词承诺立即工作，但没有提交授权动作；接受请用MODULE_ACTION，不接受请明确说明条件')
   if u.disclose_fact_ids and not u.text.strip():raise RuleError('事实披露必须有实际发言')
   if u.channel=='private' and not u.recipient_ids:raise RuleError('私聊必须指定收件人')
   if not set(u.recipient_ids)<=set(engine.state.actors):raise RuleError('非法收件人')
   if not set(u.disclose_fact_ids)<=set(disclosure(engine.state,npc,self.module)):raise RuleError('秘密披露条件未满足')
   if not set(u.claim_refs)<=known:raise RuleError('非法引用')
   if npc=='captain' and 'captain_enabled_hold' not in disclosure(engine.state,npc,self.module) and any(x in u.text for x in ['我亲自启用','我刷卡','我开启了保持']):raise RuleError('未获准的关键秘密')
   # Deliberately finite checks for important task claims; ordinary text is not
   # promoted to canon and this does not claim complete semantic leak detection.
   if npc!='doctor' and re.search(r'我(?:按说明|已经|现在|开始|亲自|负责|来|会|将|已|完成){0,4}转存',u.text):raise RuleError('该角色不能宣称亲自执行转存')
   if action.kind=='MODULE_ACTION' and re.search(r'(?:我|我们)?(?:已经|已)(?:经)?[^。！？]{0,10}(?:完成|转存|修好|隔离|发出|备妥|治好)|(?:转存|修复|治疗|备艇|隔离)(?:已经|已)?完成',u.text):raise RuleError('动作尚未裁决，不得宣称已经完成')
