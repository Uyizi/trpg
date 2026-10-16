"""Public table aids and player-knowledge projection, never an adventure solver."""
import json

from ..schemas import GuideView


def map_text(module):
    names = [place['name'] for place in module['public']['locations'].values()]
    return ('船内简图：'+'、'.join(names)+'。这些区域可经普通走廊往返，移动不耗行动格。'
            '简图不显示各舱物资和记录的内容；具体情况需要到场观察或询问船员。')


def ending_summary(state):
    if not state.outcome:
        return []
    out = state.outcome
    return [f"本次救出{out['crew_saved']}/6人，保住{out['cells_saved']}/4份细胞；主船"+('保住。' if out['ship_saved'] else '未保住。'),
            '本章已结束，目前没有后续章节。可以向主持人复盘已经发生的事情，或从手动存档创建分支。']


def guide_view(question, state, module, history=None):
    """Only public rules, the current scene and received information cross here.

    In particular: do not inspect flags/cells to calculate hints, deadlines or
    an optimal route. An OOC question cannot unlock an unobserved document.
    """
    public = module['public']
    player = state.actors['player']
    location = public['locations'][player.location_id]
    inventory = [{'id': item, 'name': module['world']['items'].get(item, {}).get('name', item)}
                 for item, owner in state.items.items() if owner == 'player']
    rules = '\n'.join(public['rules'])+'\n本地桌规：直接询问现状、普通对话和澄清不耗行动格；请求实际工作、交涉、调查新资料及操作才推进时间。桌外问题永远不执行行动，也不向NPC广播。'
    sources = [
        {'event_id': 'briefing', 'text': public['premise']+'\n'+public['opening']},
        {'event_id': 'rules', 'text': rules},
        {'event_id': 'map', 'text': map_text(module)},
        {'event_id': 'roster', 'text': json.dumps(list(public['characters'].values()), ensure_ascii=False)},
        {'event_id': 'character', 'text': json.dumps({'name': player.name, 'role': public['player']['role'],
            'health': player.health, 'skills': player.skills, 'inventory': inventory}, ensure_ascii=False)},
        {'event_id': 'scene', 'text': json.dumps({'location': location['name'], 'tick': state.tick, 'visible_objects': location['visible_objects'],
            'people': [{'name': a.name, 'health': a.health} for a in state.actors.values()
                       if a.location_id == player.location_id and a.outcome == 'present']}, ensure_ascii=False)},
    ]
    for eid in sorted(state.evidence.get('player', [])):
        record = module['world']['evidence'][eid]
        sources.append({'event_id': eid, 'text': record['title']+'：'+record['text']})
    recent = [o for o in state.observations if o.observer_id == 'player'][-8:]
    for o in recent:
        sources.append({'event_id': o.observation_id, 'text': o.content[:500], 'kind': o.modality,
                        'speaker': state.actors[o.source_actor_id].name if o.source_actor_id in state.actors else '系统'})
    if state.outcome:
        sources.append({'event_id': 'outcome', 'text': '\n'.join(ending_summary(state))})
    conversation = [{'question': row['question'], 'answer': row['answer']}
                    for row in (history or []) if row.get('reply_source') == 'model'][-4:]
    return GuideView(question=question, location=location['name'], tick=state.tick,
                     sources=sources, history=conversation)
