"""Existing transaction lifecycle, separate adventure rules and Qwen roles."""
import asyncio
import json
import re

from ..engine.rules import RuleError
from ..engine.turns import TurnService
from ..model.parsing import InvalidOutput
from ..memory import retrieve
from ..schemas import AdventureIntent, AdventureView, GuideView, NarrationView, PlayerCommand
from ..storage import Conflict
from .engine import AdventureEngine, item_name
from .commands import explicit_intent
from .views import available_actions, guide_view, player_view, scene_description, visible_npcs


def addressed_question(text, people):
    """An explicit addressee plus a question is a dialogue envelope, not a rule action.

    Only currently present names qualify. Imperatives still go to the action
    interpreter, including requests for another actor's mechanical ability.
    """
    parts=re.split(r'[，,:：]',text.strip(),maxsplit=1)
    if len(parts)!=2 or not re.search(r'[?？]|什么|为何|为什么|怎么|谁|哪里|哪儿|多少|是否',parts[1]):return None
    # A polite imperative may end in a question mark. It still needs the
    # action interpreter, rather than a non-executing character reply.
    if re.search(r'(?:请|麻烦(?:你)?|帮我|替我|给我)\s*(?:你)?\s*(?:用|使用|施放|施展|治疗|攻击|拿|把|去|帮)',parts[1]):return None
    def name(value):return re.sub(r'队长|先生|女士|老师|\s','',value).casefold()
    matches=[p for p in people if name(parts[0])==name(p['name'])]
    if len(matches)==1:return AdventureIntent(kind='talk',target_id=matches[0]['id'],words=text)
    return None


class AdventureService(TurnService):
    def project(self,state,active_turn_id=None):
        view=player_view(state,self.module,active_turn_id)
        view['table_talk']=self.store.table_talk(state.campaign_id)
        return view

    def interpreter_view(self,state,command):
        view=player_view(state,self.module)
        scene={**view['scene'],'combat':view['combat'],'location_id':view['player']['location_id']}
        scene['speech_channel']=command.speech_channel
        scene['message_recipient_ids']=command.recipient_ids
        if state.flags.get('backpack_open') and state.actors['player'].location_id=='mountain':
            scene['discovered_backpack']=True
        clarification=None
        if command.reply_to_turn_id:
            old=self.store.turn(command.reply_to_turn_id)
            if old['campaign']!=state.campaign_id or old['status']!='needs_input' or old['base_revision']!=state.revision:
                raise RuleError('回复目标不是当前状态的澄清问题')
            clarification=old['reply']
        return AdventureView(text=command.text,scene=scene,player=view['player'],party=view['party'],
            legal_actions=available_actions(state,self.module),npcs=view['npcs'],
            known_evidence=view['evidence'],clarification=clarification)

    def npc_view(self,state,npc,question):
        w=self.module['world'];loc=state.actors['player'].location_id
        if npc not in visible_npcs(state,self.module):raise RuleError('这名人物不在当前可交谈的范围内')
        if npc in state.flags['sheets']:
            card=w['profiles'][state.flags['sheets'][npc]]
            identity=state.actors[npc].name+'，'+card['role']+'；你是同行队员，只知道实际共同经历和共享资料。'
            identity+='自身技能：'+json.dumps(card['skills'],ensure_ascii=False)+'；自身能力：'+card['ability']['name']+'，剩余次数'+str(state.flags['ability_uses'][npc])+'。'
            identity+='实际随身物品：'+'、'.join(item_name(i,self.module) for i,o in state.items.items() if o==npc)+'。'
        else:
            p=w['npcs'][npc]
            # This authored field addresses the NPC as "you", never the player.
            # Resolve that template pronoun to its named subject before use.
            knowledge=p['knowledge'].get(loc,'').replace('你',p['name'])
            if npc=='kole' and state.flags.get('escort_accepted'):
                knowledge+=' '+p['knowledge']['forest'].replace('你',p['name'])
            identity=p['name']+'，'+p['role']+'。'+knowledge
            if npc=='kole' and loc=='forest':
                identity+=('队伍已经接受护送。' if state.flags.get('escort_accepted') else
                    '科尔请求队伍护送幼龙回巢，但队伍尚未答应；科尔本人打算回镇报告。可以提出请求，不得替玩家承诺。')
        view=player_view(state,self.module)
        sources=[{'event_id':'identity','speaker_name':state.actors[npc].name,'text':identity},
            {'event_id':'scene','text':scene_description(state,self.module)},
            {'event_id':'party','text':json.dumps([{k:r[k] for k in ('name','role','hp','max_hp')} for r in view['party']],ensure_ascii=False)}]
        for eid in state.evidence.get(npc,[]):
            evidence=w['evidence'][eid]
            sources.append({'event_id':eid,'text':evidence['title']+'：'+evidence['text']})
        # Global player transcript contains conversations this NPC may not hear.
        # Only actual actor-local observations can enter an NPC's memory.
        own=retrieve(state,npc,question,limit=8)
        for observation in own:
            sources.append({'event_id':observation['observation_id'],'kind':observation['modality'],
                'text':('听到的发言（未经独立核实）：' if observation['modality']=='heard' else '')+observation['content'][:240]})
        return GuideView(question=question,location=view['player']['location'],tick=state.tick,sources=sources)

    @staticmethod
    def sourced(output,sources):
        if not output.text.strip() or not output.source_event_ids or not set(output.source_event_ids)<=sources:
            raise InvalidOutput('回答引用不存在的来源')

    async def run_turn(self,id):
        try:t=self.store.claim(id)
        except Conflict:return
        epoch=t['epoch'];state=self.store.snapshot(t['campaign'])
        command=PlayerCommand.model_validate_json(t['command']);agent=self.agent_factory(id,epoch)
        try:
            if not command.text.strip():
                self.store.finish(id,epoch,'needs_input','请写下角色要做的事，或想询问的问题。');return
            if command.mode=='ooc':
                self.store.progress(id,epoch,'Qwen回答桌外问题')
                view=guide_view(command.text,state,self.module,self.store.table_talk(state.campaign_id))
                output=await agent.guide(view)
                self.sourced(output,{x['event_id'] for x in view.sources}|({'history'} if view.history else set()))
                self.store.finish(id,epoch,'completed_ooc',output.text);return
            self.store.progress(id,epoch,'理解冒险行动')
            current=player_view(state,self.module)
            selected=explicit_intent(command.text,current,available_actions(state,self.module))
            addressed=addressed_question(command.text,current['npcs'])
            intent=selected or addressed or await agent.interpret_adventure(self.interpreter_view(state,command))
            if intent.kind=='clarify':
                self.store.finish(id,epoch,'needs_input',intent.clarification or '请说明行动对象、方法和你希望达成的结果。');return
            engine=AdventureEngine(state,self.module,id,
                lambda sides,count,key:self.store.roll_dice(id,key,sides=sides,count=count))
            if intent.kind=='talk':
                npc=intent.target_id
                if not npc:raise RuleError('你想问在场的哪一位？可以直接写出名字。')
                if command.speech_channel=='broadcast':raise RuleError('这里没有工作无线电，请选择现场交谈或在场人物的私下交谈。')
                if command.speech_channel=='private' and command.recipient_ids!=[npc]:raise RuleError('请将私下交谈的对象选为当前提问的这名人物。')
                if state.flags['stats'].get(npc,{}).get('hp',1)<=0:raise RuleError('这位同伴目前昏迷，无法交谈。')
                self.store.progress(id,epoch,'Qwen扮演现场人物')
                view=self.npc_view(state,npc,intent.words or command.text)
                output=await agent.adventure_npc(view)
                self.sourced(output,{x['event_id'] for x in view.sources})
                engine.emit('SpeechOccurred',{'text':intent.words or command.text,'channel':command.speech_channel,'recipient_ids':[npc]})
                engine.emit('SpeechOccurred',{'text':output.text,'channel':command.speech_channel,'recipient_ids':['player']},npc)
                dialogue=[{'speaker':state.actors[npc].name,'speaker_id':npc,'text':output.text}]
                narration='你与'+state.actors[npc].name+'交谈。';degraded=False
            else:
                self.store.progress(id,epoch,'裁定冒险行动')
                engine.resolve(intent,command.text)
                # Describe only observations genuinely received by the player.
                events=[{'event_id':o.event_id,'text':o.content} for o in engine.state.observations
                    if o.observer_id=='player' and o.event_id.startswith(id+':')]
                public_notices=[e.payload['text'] for e in engine.events if e.kind=='PublicNotice']
                if not events:raise RuleError('请补充具体对象或方法，当前行动没有可结算的结果。')
                self.store.progress(id,epoch,'整理行动结果' if self.narration_mode=='brief' else 'Qwen叙述实际结果',engine.state.model_dump())
                current=player_view(engine.state,self.module)
                narration='\n'.join(public_notices) or '\n'.join(dict.fromkeys(e['text'] for e in events))
                degraded=False
                try:
                    if self.narration_mode=='brief':
                        output=None
                    else:
                        output=await agent.adventure_narrate(NarrationView(
                            scene={'description':current['scene']['description'],'combat':current['combat'],
                                   'player':current['player']},events=events))
                    if output is not None:
                        self.sourced(output,{e['event_id'] for e in events})
                        narration=output.text
                except (InvalidOutput,ValueError,AttributeError):
                    degraded=True
                dialogue=[]
            self.store.progress(id,epoch,'提交冒险进度',engine.state.model_dump())
            projected=engine.state.model_copy(deep=True);projected.revision+=1
            from ..storage import turn_transcript
            result={'narration':narration,'dialogue':dialogue,'checks':engine.checks,'degraded':degraded,
                'rule_log':[e.payload['text'] for e in engine.events if e.kind=='PublicNotice']}
            projected.transcript.extend(turn_transcript(id,projected.tick,projected.actors['player'].name,
                command.model_dump(),result))
            result['view']=self.project(projected)
            self.store.commit(id,epoch,engine.state,engine.events,result)
        except asyncio.CancelledError:raise
        except Conflict:pass  # Cancelled or superseded lease: never publish a draft.
        except RuleError as exc:self.store.finish(id,epoch,'needs_input',str(exc))
        except (InvalidOutput,ValueError,AttributeError):
            self.store.finish(id,epoch,'failed','本地Qwen这次未能生成可靠答复。本轮未提交游戏变化，可以重试。')
        except Exception:
            # Unexpected server errors must not leave a campaign stuck running.
            # Details stay in the local log, not in player-visible dialogue.
            import traceback
            with (self.store.path.parent/'errors.log').open('a',encoding='utf-8') as log:traceback.print_exc(file=log)
            self.store.finish(id,epoch,'failed','本轮处理遇到错误，未提交游戏变化。记录已保留，可以重试。')
        finally:
            self.last_metrics[id]=getattr(agent,'metrics',[])
