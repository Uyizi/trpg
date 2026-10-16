"""Knowledge-limited projections shared by the browser and model roles."""
import json

from ..schemas import GuideView
from .engine import item_name
from .commands import action_choices


# These describe the path visible from its origin, not the unreached room's
# private title. Authors may supply scene.exit_labels for additional modules.
CONNECTORS={
    'patrol:forest':'沿林道进入树林', 'forest:patrol':'返回镇外林道',
    'forest:mountain':'沿林间小径前往山脚洞口', 'mountain:forest':'沿小径返回树林',
    'mountain:cave':'从洞口进入洞内通道', 'cave:mountain':'退出洞穴，返回山脚洞口',
    'cave:wall':'沿黑暗通道向上探索', 'wall:cave':'沿来路返回洞内通道',
    'wall:mirror':'攀至顶部后沿上方通道继续前进', 'mirror:wall':'沿来路返回冰墙上方',
    'mirror:slides':'通过冰幕后继续向前', 'slides:mirror':'沿来路返回冰幕附近',
    'slides:hatchery':'到达滑道底部后沿短路前进', 'hatchery:slides':'沿来路返回滑道底部',
}


def scene_description(state, module):
    loc = state.actors['player'].location_id
    text = module['world']['scenes'][loc]['description']
    f = state.flags
    if loc=='forest' and f.get('escort_accepted'):
        text = '科尔已返回松溪镇报告。你们留在林间，幼龙在身边，通往山脚的小径清晰可见。'+('幼龙吃过东西，已经精神一些。' if f.get('dragon_fed') else '幼龙仍显得虚弱。')
    elif loc in ('mountain','hatchery') and f.get('battle:'+loc):
        text = '战斗结束，眼前没有仍在交战的敌人。'+('洞口有一个旧背包，往里是光线不足的洞穴。' if loc=='mountain' else '幼龙回到了巢旁，前方传来巨大动静。可以继续靠近龙巢。')
    elif loc=='mirror' and f.get('mirror_passed'):text = '冰幕已经打开，前方通道可以通行。'
    elif loc=='wall' and f.get('wall_passed'):text = '你已抵达冰墙顶部，同伴跟了上来。上方通道向山内继续延伸，也可以沿来路返回。'
    elif loc=='slides' and f.get('slides_passed'):text = '队伍已抵达滑道底部，前方短路通向孵化洞窟。'
    elif loc=='slides' and f.get('sliding'):text = '你已经进入危险滑道，前方是尖冰，旁边有交叉路线。需要用运动或体操尝试跳转。'
    elif loc=='mother' and f.get('mother_reassured'):text = '罗恩已了解救护经过，对队伍放下戒备。她留在龙巢旁和你们交谈。'
    return text


def visible_npcs(state,module):
    loc = state.actors['player'].location_id
    return [aid for aid,actor in state.actors.items() if aid!='player' and actor.location_id==loc
            and actor.outcome=='present' and (aid in state.flags['sheets'] or aid in module['world']['scenes'][loc]['npcs'])]


def available_actions(state,module):
    loc = state.actors['player'].location_id
    actions = module['world']['scenes'][loc]['actions']
    if state.flags.get('combat'):return []
    result=[]
    for id,action in actions.items():
        completed={'accept_escort':'escort_accepted','feed':'dragon_fed','light':'light',
            'climb':'wall_passed','answer':'mirror_passed','melt':'mirror_passed','break':'mirror_passed',
            'choose_slide':'slides_passed','recover_slide':'slides_passed'}
        if state.flags.get(completed.get(id,'')):continue
        if id=='recover_slide' and not state.flags.get('sliding'):continue
        if id=='choose_slide' and state.flags.get('sliding'):continue
        if id in ('accept_care','decline_care','return_village') and not state.flags.get('mother_reassured'):continue
        if id=='explain_rescue' and state.flags.get('mother_reassured'):continue
        result.append({'id':id,'title':action['title'],'skills':action.get('skills',[])})
    if loc=='mountain' and state.flags.get('backpack_open'):
        result.append({'id':'take_backpack','title':'取走已打开背包里的物品','skills':[]})
        result.append({'id':'take_item','title':'只取走背包中指定的一件物品','skills':[],
            'items':[{'id':i,'name':item_name(i,module)} for i,o in state.items.items()
                     if o=='mountain' and (i.startswith('torch_') or i in ('flint','meat','gold'))]})
    return result


def player_view(state,module,active_turn_id=None):
    w = module['world']; player = state.actors['player']; loc = player.location_id
    sheet = w['profiles'][state.flags['sheets']['player']]
    stats = state.flags['stats']['player']
    npcs=[]
    for aid in visible_npcs(state,module):
        actor=state.actors[aid]
        role=w['profiles'][state.flags['sheets'][aid]]['role'] if aid in state.flags['sheets'] else w['npcs'][aid]['role']
        npcs.append({'id':aid,'name':actor.name,'role':role,'location_id':loc,'present':True})
    known_exits=[{'id':id,'name':w['scenes'][loc].get('exit_labels',{}).get(id)
                 or CONNECTORS.get(loc+':'+id)
                 or (w['scenes'][id]['name'] if id in state.flags['visited']
                 else ('前方通路' if len(w['scenes'][loc]['exits'])==1 else f'前方通路{i+1}'))}
                 for i,id in enumerate(w['scenes'][loc]['exits'])
                 if not state.flags.get('combat') or id in state.flags['visited']]
    inventory=[{'id':id,'name':item_name(id,module)} for id,owner in state.items.items() if owner=='player']
    own_ability={**sheet['ability'],'remaining':state.flags['ability_uses']['player']}
    party=[]
    for aid,template in state.flags['sheets'].items():
        c=w['profiles'][template]
        party.append({'id':aid,'name':state.actors[aid].name,'role':c['role'],**state.flags['stats'][aid],
                      'ability':{**c['ability'],'remaining':state.flags['ability_uses'][aid]}})
    outcome=state.outcome
    fight=state.flags.get('combat')
    combat=({'round':fight['round'],'enemies':[{'id':e['id'],'name':e['name'],
        'condition':'受伤' if e['hp']<e['max_hp'] else '仍在战斗'} for e in fight['enemies'] if e['hp']>0]} if fight else None)
    view={'campaign_id':state.campaign_id,'revision':state.revision,'module_id':state.module_id,
        'module_title':module['public']['title'],'ruleset':'pinebrook_d20','skill_names':w['skill_names'],
        'player':{'id':'player','name':player.name,'role':sheet['role'],'skills':player.skills,
            'location_id':loc,'location':w['scenes'][loc]['name'],'health':player.health,'inventory':inventory,
            **stats,'attacks':sheet['attacks'],'ability':own_ability,'ammo':state.flags['ammo']['player']},
        'scene':{'description':scene_description(state,module),'visible_objects':w['scenes'][loc]['visible_objects'],
            'lighting':'火把已点燃' if state.flags.get('light') else '未点燃火把',
            'challenge_resolved':bool(state.flags.get({'wall':'wall_passed','mirror':'mirror_passed','slides':'slides_passed'}.get(loc,''))),
            'exits':known_exits,'investigations':[{'id':id,'title':o['title'],'skills':o.get('skills',[])} for id,o in w['scenes'][loc]['investigations'].items()
                if not o.get('requires') or state.flags.get(o['requires'])]},
        'tick':state.tick,'phase':state.phase,'clocks':[],'locations':known_exits,'npcs':npcs,
        'party':party,'combat':combat,
        'evidence':[{'id':id,**w['evidence'][id],'source':'亲自取得'} for id in state.evidence.get('player',[])],
        'suggestions':[] if outcome else ['观察眼前的情况'],
        'suggestion_commands':{'观察眼前的情况':{'text':'观察眼前的情况','mode':'in_character','speech_channel':'local','recipient_ids':[]}},
        'ending_summary':[outcome['summary']] if outcome else [],'transcript':state.transcript,
        'outcome':outcome,'active_turn_id':active_turn_id,'mock':state.mock}
    choices=action_choices(view,available_actions(state,module))
    view['suggestions']=list(choices)
    view['suggestion_commands']={label:{'text':label,'mode':'in_character','speech_channel':'local','recipient_ids':[]}
        for label in choices}
    return view


def guide_view(question,state,module,history=None):
    view=player_view(state,module)
    sources=[{'event_id':'briefing','text':module['public']['premise']+'\n'+module['public']['opening']},
        {'event_id':'rules','text':'\n'.join(module['public']['rules'])},
        {'event_id':'character','text':json.dumps(view['player'],ensure_ascii=False)},
        {'event_id':'scene','text':json.dumps({'tick':view['tick'],'scene':view['scene'],'npcs':view['npcs'],'combat':view['combat']},ensure_ascii=False)},
        {'event_id':'party','text':json.dumps(view['party'],ensure_ascii=False)}]
    for evidence in view['evidence']:sources.append({'event_id':evidence['id'],'text':evidence['title']+'：'+evidence['text']})
    for i,row in enumerate(state.transcript[-6:]):
        heard=row.get('speaker_id') not in ('player','dm',None)
        sources.append({'event_id':'log:'+str(i),'kind':'heard' if heard else 'record',
            'text':('角色发言（未经独立核实）：' if heard else '')+row['speaker']+'：'+row['text'][:450]})
    if state.outcome:sources.append({'event_id':'outcome','text':state.outcome['summary']})
    return GuideView(question=question,location=view['player']['location'],tick=state.tick,sources=sources,
        history=[{'question':r['question'],'answer':r['answer']} for r in (history or []) if r.get('reply_source')=='model'][-4:])
