"""Constrain each action type to its own currently known targets.

The grammar constrains syntax/IDs; rule authority and player intent remain
separate checks. No hidden scene, solution or resource enters this schema.
"""


def adventure_intent_schema(view):
    branches=[]
    text={'type':'string','maxLength':500}
    def add(kind,properties=None,required=()):
        branches.append({'type':'object','properties':{'kind':{'const':kind},**(properties or {})},
            'required':['kind',*required],'additionalProperties':False})
    def enum(values):return {'type':'string','enum':list(dict.fromkeys(values))}
    add('look');add('wait')
    add('clarify',{'clarification':{'type':'string','minLength':1,'maxLength':300}},['clarification'])
    exits=[x['id'] for x in view.scene['exits']]
    if exits and not view.scene.get('combat'):add('move',{'target_id':enum(exits)},['target_id'])
    add('retreat',{'target_id':enum(['',*exits])})
    npcs=[n['id'] for n in view.npcs]
    if npcs:add('talk',{'target_id':enum(npcs),'words':text},['target_id','words'])
    for action in view.legal_actions:
        props={'target_id':{'const':action['id']},'method':text,'words':text}
        required=['target_id']
        if action.get('skills'):props['skill']=enum(action['skills'])
        if action['id']=='take_item':
            items=[i['id'] for i in action.get('items',[])]
            if not items:continue
            props['item_id']=enum(items);required.append('item_id')
        add('act',props,required)
    inspections={x['id']:x.get('skills',[]) for x in view.scene['investigations']}
    if any(i['id']=='draconic_page' for i in view.player['inventory']):inspections['draconic_page']=[]
    inspections.update({e['id']:[] for e in view.known_evidence})
    for id,skills in inspections.items():
        props={'target_id':{'const':id},'method':text}
        if skills:props['skill']=enum(skills)
        add('inspect',props,['target_id'])
    enemies=[e['id'] for e in (view.scene.get('combat') or {}).get('enemies',[])]
    if enemies and view.player.get('hp',0)>0:
        add('attack',{'target_id':enum(['',*enemies]),
            'attack_id':enum(['',*[a['id'] for a in view.player['attacks']]])})
    people=[p['id'] for p in view.party]
    for person in view.party:
        ability=person['ability']
        if person['hp']<=0 or not ability['remaining']:continue
        props={'ability_id':{'const':ability['id']}}
        required=['ability_id']
        if person['id']!='player':
            props['helper_id']={'const':person['id']};required.append('helper_id')
        if ability['id']=='magic_missile':
            if not enemies:continue
            props['target_id']=enum(enemies)
            props['target_ids']={'type':'array','items':enum(enemies),'minItems':3,'maxItems':3}
        else:
            targets=[person['id']] if ability['id']=='second_wind' else people
            props['target_id']=enum(targets)
        add('ability',props,required)
    return {'title':'AdventureAction','anyOf':branches}
