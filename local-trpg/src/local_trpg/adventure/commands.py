"""Explicit, visible action labels. Free prose still goes to the model.

These are commands selected by the player, never guessed from keywords. Rules
still validate every proposal. Only an answer entered by the player can solve
a puzzle; the menu contains no hidden solution or future target.
"""
from ..schemas import AdventureIntent


def action_choices(view, actions):
    if view.get('outcome'):return {}
    choices={'观察眼前的情况':{'kind':'look'}}
    alive=view['player']['hp']>0
    fight=view.get('combat')
    if alive:
        for path in view['scene']['exits']:
            label=('撤退：' if fight else '')+path['name']
            choices[label]={'kind':'retreat' if fight else 'move','target_id':path['id']}
        if not fight:
            for action in actions:
                if action['id']=='take_item':continue  # Needs an individual item choice.
                label=action['title']+('：' if action['id']=='answer' else '')
                choices[label]={'kind':'act','target_id':action['id']}
        for target in view['scene']['investigations']:
            choices[target['title']]={'kind':'inspect','target_id':target['id']}
        if any(i['id']=='draconic_page' for i in view['player']['inventory']):
            choices['阅读随身的龙语对照页']={'kind':'inspect','target_id':'draconic_page'}
        if fight:
            for enemy in fight['enemies']:
                for attack in view['player']['attacks']:
                    choices['用'+attack['name']+'攻击'+enemy['name']]={
                        'kind':'attack','attack_id':attack['id'],'target_id':enemy['id']}
    for person in view['party']:
        ability=person['ability']
        if person['hp']<=0 or not ability['remaining']:continue
        intent={'kind':'ability','ability_id':ability['id']}
        if person['id']!='player':intent['helper_id']=person['id']
        label=('自己' if person['id']=='player' else '请'+person['name'])+'使用'+ability['name']
        if ability['id']=='magic_missile':
            for enemy in (fight or {}).get('enemies',[]):
                choices[label+'攻击'+enemy['name']]={**intent,'target_id':enemy['id']}
        elif ability['id']=='second_wind':
            if person['hp']<person['max_hp']:choices[label]={**intent,'target_id':person['id']}
        else:
            for target in view['party']:
                if target['hp']<target['max_hp']:
                    choices[label+'治疗'+('我' if target['id']=='player' else target['name'])]={**intent,'target_id':target['id']}
    if fight:choices['让同伴继续战斗']={'kind':'wait'}
    return choices


def explicit_intent(text, view, actions):
    text=text.strip()
    choices=action_choices(view,actions)
    if text in choices:return AdventureIntent.model_validate(choices[text])
    for label,intent in choices.items():
        # A visible fill-in field; do not infer an answer from the adventure.
        if label.endswith('：') and text.startswith(label):
            return AdventureIntent.model_validate({**intent,'words':text[len(label):].strip()})
    return None
