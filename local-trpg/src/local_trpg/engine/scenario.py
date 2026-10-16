"""Registered effects, deadlines and endings; model output cannot supply patches."""
class ScenarioEffects:
 def require(self,condition,message):
  if not condition:
   from .rules import RuleError
   raise RuleError(message)
 def nearby(self,actor,location):
  self.require(self.state.actors[actor].outcome=='present','人物已经离开')
  if actor!='player':self.move(actor,location)
  self.require(self.state.actors[actor].location_id==location,'需要先前往'+self.module['world']['locations'][location]['name'])
 def module_action(self,id,actor,targets):
  self.require(self.state.actors[actor].health!='incapacitated' and self.state.actors[actor].outcome=='present','该人物不能执行当前工作')
  self._module_action(id,actor,targets)
  if actor!='player':self.flag('last_action:'+actor,self.state.tick)
 def _module_action(self,id,actor,targets):
  s=self.state
  if id=='take_cold_cases':
   self.nearby(actor,'lifeboat')
   for i in ['cold_case_1','cold_case_2','cold_core_1','cold_core_2']:
    self.require(s.items.get(i) in ('lifeboat',actor),'应急箱不在这里')
    if self.state.items[i]!=actor:self.emit('ItemTransferred',{'item_id':i,'from_owner':self.state.items[i],'to_owner':actor},actor)
   self.give_evidence('E07',[actor],actor);self.notice('取到两只保冷箱和两块已充储冷芯。');return
  if id=='transfer_cells':
   self.nearby(actor,'cargo');s=self.state
   def accessible(item):return s.items.get(item)=='cargo' or (s.items.get(item) in s.actors and s.actors[s.items[item]].location_id=='cargo')
   self.require(all(accessible(i) for i in ['cold_case_1','cold_case_2','cold_core_1','cold_core_2']),'两箱两芯必须实际在货舱')
   self.require(any(c['valid'] and (c['expiry'] is None or s.tick<c['expiry']) for c in s.cells),'细胞已失效，不能恢复')
   if actor=='player':
    self.require('E05' in s.evidence.get('player',[]),'稳妥转存需要先阅读货物说明')
    if not s.flags.get('solo_transfer_prepared'):
     self.flag('solo_transfer_prepared',True);self.notice('已按说明准备转存；再花一格完成。');return
   elif actor=='doctor':self.require('E07' in s.evidence.get('doctor',[]),'医生需要实际看到本次补给票')
   cells=[dict(c) for c in self.state.cells]
   for i,c in enumerate(cells):
    if not c['valid'] or (c['expiry'] is not None and self.state.tick>=c['expiry']):continue
    box=f'cold_case_{1+i//2}';key='cold_expiry:'+box
    if self.state.flags.get(key) is None:self.flag(key,self.state.tick+18)
    expiry=self.state.flags[key]
    if self.state.tick>=expiry:continue
    c.update(container=box,expiry=expiry)
   self.emit('CellsChanged',{'cells':cells},actor);self.flag('cells_transferred',True);self.flag('solo_transfer_prepared',False)
   protected=sum(c['valid'] and c['container'].startswith('cold_case_') for c in cells)
   self.notice(f'转存完成：{protected}份有效细胞现已置于保冷箱中；储冷期限从首次启用起计，不会因重新装箱刷新。',actor=actor);return
  if id in ('begin_safe_isolation','complete_safe_isolation','isolate_branch'):
   if actor!='player':self.move(actor,'engine')
   self.require(self.state.actors[actor].location_id in ('engine','cargo'),'需要前往已标识的手动阀')
   self.require(not self.state.flags.get('engine_destroyed'),'设备已热损坏，不能继续按原步骤隔离')
   if self.state.flags.get('cargo_branch_isolated'):self.notice('支路已经隔离；货物期限保持原值。');return
   self.require(actor=='mechanic' or self.state.items.get('multitool')==actor or self.state.actors[actor].location_id=='engine','需要普通工具')
   if id=='begin_safe_isolation':
    self.flag('safe_isolation_prepared',{'actor':actor,'location':self.state.actors[actor].location_id});self.notice('已完成安全卸压准备，下一格可以关闭手动阀。');return
   if id=='complete_safe_isolation':
    prepared=self.state.flags.get('safe_isolation_prepared')
    self.require(isinstance(prepared,dict) and prepared.get('actor')==actor and prepared.get('location')==self.state.actors[actor].location_id,'需要同一操作者在原处完成有效的卸压准备')
   if id=='isolate_branch' and actor=='player' and not self.state.flags.get('mechanic_guidance'):
    outcome=self.check(actor,'technical','急关高压阀的技术检定')
    if outcome=='failure':self.emit('HealthChanged',{'actor_id':actor,'health':'strained'});self.notice('高压阀未能关好，你受到轻伤；仍可选择先卸压再安全关闭。');return
    if outcome=='success_with_cost':self.emit('HealthChanged',{'actor_id':actor,'health':'strained'});self.notice('支路成功隔离，但热气造成轻伤。')
   self.flag('cargo_branch_isolated',True);self.flag('safe_isolation_prepared',False)
   cells=[dict(c) for c in self.state.cells]
   for c in cells:
    if c['valid'] and c['container']=='original_cargo_box' and c['expiry'] is None:c['expiry']=self.state.tick+4
   self.emit('CellsChanged',{'cells':cells});self.notice('货舱支路已手动隔离；未保护的细胞开始计算四格断冷期限。');return
  if id=='restore_main_loop':
   self.nearby(actor,'engine');self.require(self.state.flags.get('cargo_branch_isolated'),'必须先手动隔离货舱支路')
   self.require(not self.state.flags.get('engine_destroyed'),'动力舱已经损坏，无法用旧步骤恢复')
   self.require(self.state.actors[actor].health not in ('injured','incapacitated'),'重伤时需要他人代为操作')
   self.flag('main_loop_restored',True);self.emit('ClockAdvanced',{'tick':self.state.tick,'phase':'stable'});self.notice('保持已解除，排气后主循环启动，热危机停止。现在可以选择前往维修站。');return
  if id=='treat_actor':
   target=targets[0] if targets and targets[0] in self.state.actors else 'mechanic'
   if actor!='player':self.move(actor,self.state.actors[target].location_id)
   self.require(self.state.actors[actor].location_id==self.state.actors[target].location_id,'需要接触伤员')
   authorized=getattr(self,'authorized_items',[])
   supply=next((i for i,owner in self.state.items.items() if i.startswith('medical_supply_') and owner==actor and (not authorized or i in authorized)),None)
   self.require(supply is not None,'没有可用处置物资')
   levels=['healthy','strained','injured','incapacitated'];h=self.state.actors[target].health
   self.require(h!='healthy','人物无需重复治疗')
   self.emit('ItemTransferred',{'item_id':supply,'from_owner':actor,'to_owner':'consumed'},actor)
   self.emit('HealthChanged',{'actor_id':target,'health':levels[levels.index(h)-1]},actor);return
  if id=='send_distress':
   if actor=='player':self.require(self.state.actors[actor].location_id=='bridge','需要桥上通信终端')
   else:self.nearby(actor,'bridge')
   if self.state.flags.get('first_distress_tick') is None:
    self.flag('first_distress_tick',self.state.tick);self.flag('rescue_eta',self.state.tick+6)
   self.notice('标准求救已发出，救援预计在第'+str(self.state.flags['rescue_eta'])+'格到达；重发不会重置时间。',list(self.state.actors),actor);return
  if id=='prepare_pod':
   self.nearby(actor,'lifeboat');self.flag('pod_prepared',True);self.notice('六座救生艇已备妥，尚未强制任何人登艇。',list(self.state.actors),actor);return
  if id=='assist_evacuation':
   target=next((x for x in targets if x in self.state.actors and x!=actor),None);self.require(target is not None,'需要明确协助哪一名伤员')
   self.require(not self.state.flags.get('engine_destroyed') or self.state.actors[target].location_id!='engine','动力舱已封闭，无法安全进入')
   self.move(actor,self.state.actors[target].location_id)
   self.emit('ActorMoved',{'actor_id':target,'destination':'lifeboat'},actor);self.move(actor,'lifeboat');self.notice('已协助伤员到气闸。');return
  if id=='assemble_crew':
   acceptable=self.state.tick>=9 or (self.state.flags.get('first_distress_tick') is not None and self.state.flags.get('cells_transferred') and self.state.flags.get('pod_prepared'))
   self.require(acceptable,'请先说明可行的撤离方案：求救、货物处置与备艇，或等待公开撤离指令')
   self.flag('evacuation_agreed',True)
   for id,a in list(self.state.actors.items()):
    if a.outcome=='present' and a.health in ('healthy','strained') and not a.busy:
     if a.location_id=='engine' and self.state.flags.get('engine_destroyed'):continue
     self.move(id,'lifeboat')
   self.notice('能够自行行动且没有待办的人已沿走廊到气闸；重伤者需要明确安排协助。',list(self.state.actors));return
  if id=='board_pod':
   self.nearby(actor,'lifeboat');self.require(self.state.flags.get('pod_prepared'),'需要先备艇')
   for id,a in self.state.actors.items():
    if a.location_id=='lifeboat' and a.outcome=='present' and (id=='player' or self.state.flags.get('evacuation_agreed') or self.state.tick>=9):self.flag('onboard:'+id,True)
   for i,owner in list(self.state.items.items()):
    if i.startswith(('cold_case_','cold_core_')) and (owner=='lifeboat' or (owner in self.state.actors and self.state.actors[owner].onboard)):
     self.emit('ItemTransferred',{'item_id':i,'from_owner':owner,'to_owner':'pod'})
   self.notice('已按实际在场和同意登艇的人、物完成登艇登记。');return
  if id=='launch_pod':
   self.require(self.state.flags.get('pod_prepared') and self.state.actors['player'].onboard and self.state.actors['player'].location_id=='lifeboat','玩家必须实际登上已备妥救生艇')
   self.flag('pod_launched',True)
   if self.state.flags.get('first_distress_tick') is None:self.flag('first_distress_tick',self.state.tick);self.flag('rescue_eta',self.state.tick+6)
   for id,a in self.state.actors.items():
    if a.onboard:self.emit('ActorMoved',{'actor_id':id,'destination':'lifeboat'},id)
   self.notice('救生艇已经起飞，自动发出应急信标。',list(self.state.actors))
   self.ending('pod');return
  if id=='choose_station':
   self.require(self.state.flags.get('main_loop_restored'),'需要先稳定主循环')
   self.ending('station');return
  self.require(False,'未注册动作')
 def scheduled_action(self,id,actor,targets):
  from .rules import Engine,RuleError
  if self.state.actors[actor].busy or self.state.flags.get('last_action:'+actor)==self.state.tick:return
  trial=Engine(self.state,self.module,self.turn_id,self.rolls);trial.events=list(self.events)
  try:trial.module_action(id,actor,targets)
  except RuleError:
   self.notice(self.state.actors[actor].name+'当前无法完成原定工作，需要另行安排。',list(self.state.actors));return
  self.state=trial.state;self.events=trial.events
 def finish_clock(self):
  s=self.state;t=s.tick;crisis=not s.flags.get('main_loop_restored')
  if t in (3,6,8,9,10,12,15) and str(t) not in s.fired_triggers:
   self.emit('TriggerFired',{'trigger_id':str(t)})
   if t==3 and crisis:
    self.notice('工作频道：回流继续下降。',list(s.actors))
    if s.actors['doctor'].health=='healthy' and not s.actors['doctor'].busy and s.actors['mechanic'].health=='strained':self.scheduled_action('treat_actor','doctor',['mechanic'])
   if t==6 and crisis:self.notice('阀组周围灼热，急修风险增加；外侧面板仍可阅读。通讯员询问是否求救。',list(s.actors))
   if t==8 and crisis and not s.flags.get('pod_prepared') and not s.actors['guard'].busy and not s.actors['guard'].onboard:self.scheduled_action('prepare_pod','guard',[])
   if t==9 and crisis:
    if self.state.flags.get('first_distress_tick') is None and not s.actors['comms'].onboard:self.scheduled_action('send_distress','comms',[])
    self.flag('evacuation_agreed',True);self.notice('船长同意正常撤离准备：修船还剩三格，热损坏后另有三格撤离窗口。',list(s.actors),'captain')
   if t==10 and crisis:
    for id,a in list(self.state.actors.items()):
     if id!='player' and not a.busy and a.health in ('healthy','strained') and not a.onboard:self.move(id,'lifeboat')
    self.notice('没有明确待办任务的人前往气闸；仍在执行任务的人应报告撤离安排。',list(s.actors))
   if t==12 and crisis:
    self.flag('engine_destroyed',True);self.emit('ClockAdvanced',{'tick':t,'phase':'evacuation'});self.flag('evacuation_deadline',15)
    for id,a in list(self.state.actors.items()):
     if a.location_id=='engine':self.emit('HealthChanged',{'actor_id':id,'health':['healthy','strained','injured','incapacitated'][min(3,['healthy','strained','injured','incapacitated'].index(a.health)+1)]},id)
    cells=[dict(c) for c in self.state.cells]
    for c in cells:
     if c['container']=='original_cargo_box':c['valid']=False
    self.emit('CellsChanged',{'cells':cells});self.notice('主回路热损坏，动力舱封闭。撤离期限为第15格。',list(s.actors))
   if t==15 and self.state.phase=='evacuation':
    for id,a in list(self.state.actors.items()):
     if not (a.onboard and self.state.flags.get('pod_launched')):self.flag('lost:'+id,True)
    # A launched pod already has its own rescue arrival. The abandoned ship
    # still reaches its deadline, but ending() resolves that voyage only once.
    if not self.state.flags.get('pod_launched'):self.final_outcome('deadline')
  self.expire_cells()
 def ending(self,kind):
  arrival=self.state.tick+5 if kind=='station' else max(self.state.tick,self.state.flags.get('rescue_eta') or self.state.tick+6)
  # Main ship deadlines still happen after an early departure.
  finish=arrival if kind=='station' else max(arrival,15 if not self.state.flags.get('main_loop_restored') else arrival)
  while self.state.tick<arrival:self.advance(1);self.finish_clock()
  delivered=sum(c['valid'] and (c['expiry'] is None or arrival<c['expiry']) and (kind=='station' or self.state.items.get(c['container'])=='pod') for c in self.state.cells)
  while self.state.tick<finish:self.advance(1);self.finish_clock()
  self.final_outcome(kind,arrival,delivered)
 def final_outcome(self,kind,arrival=None,delivered=None):
  s=self.state;arrival=s.tick if arrival is None else arrival
  if kind=='station':crew=sum(a.outcome!='lost' for a in s.actors.values())
  else:crew=sum(a.onboard and s.flags.get('pod_launched',False) for a in s.actors.values())
  cells=sum(c['valid'] and (c['expiry'] is None or arrival<c['expiry']) and (kind=='station' or s.items.get(c['container'])=='pod') for c in s.cells)
  if delivered is not None:cells=delivered
  known=set(s.evidence.get('player',[]));out={'crew_saved':int(crew),'cells_saved':int(cells),'ship_saved':bool(s.flags.get('main_loop_restored')),'cause_verified':{'E01','E03','E04'}<=known,'captain_action_verified':'E02' in known and ('E06' in known or s.flags.get('guard_witness',False)),
   'record_shared_with':{e:[a for a,refs in s.evidence.items() if e in refs and a!='player'] for e in known},'promises_kept':['完整保留记录'] if s.flags.get('promise_preserve_records') else [],'promises_broken':[],'arrival_tick':arrival}
  self.emit('ScenarioResolved',{'outcome':out});self.notice(f"结局：救出{crew}/6人，保住{cells}/4份细胞，主船{'保住' if out['ship_saved'] else '损坏'}。抵达时刻第{arrival}格。")
