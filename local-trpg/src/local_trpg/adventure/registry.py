"""Validate declarative adventure references without executing module content."""
import hashlib
import json


HANDLERS = {'accept_escort','feed','light','climb','answer','melt','break',
            'choose_slide','recover_slide','explain_rescue','accept_care',
            'decline_care','return_village'}


def validate_adventure(data):
    p, w = data['public'], data['world']
    if w.get('schema_version') != 2 or w.get('engine') != 'scene_d20':
        raise ValueError('不支持的冒险引擎')
    if not p['module_id'] == w['module_id'] or p['module_version'] != w['module_version']:
        raise ValueError('冒险版本不匹配')
    scenes, profiles = w['scenes'], w['profiles']
    if w['start'] not in scenes or not profiles:
        raise ValueError('缺少起始场景或角色卡')
    if {c['id'] for c in p['characters']} != set(profiles):
        raise ValueError('公开角色卡集合错误')
    for card in profiles.values():
        if not 1 <= card['hp'] <= 100 or not 1 <= card['ac'] <= 30 or not card['attacks']:
            raise ValueError('非法角色数值')
        if not set(card['equipment']) <= set(w['item_names']):
            raise ValueError('角色物品未注册')
    for scene in scenes.values():
        if not set(scene['exits']) <= set(scenes): raise ValueError('通路引用不存在的场景')
        if not set(scene['npcs']) <= set(w['npcs']): raise ValueError('场景人物未注册')
        for action in scene['actions'].values():
            if action['handler'] not in HANDLERS: raise ValueError('未注册规则处理器')
        for obj in scene['investigations'].values():
            if obj.get('evidence') and obj['evidence'] not in w['evidence']:
                raise ValueError('调查线索未注册')
        combat = scene.get('combat')
        if combat and not (1 <= combat['count'] <= 10 and combat['hp'] > 0):
            raise ValueError('非法遭遇数值')
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return {**data, 'module_id':w['module_id'], 'module_version':w['module_version'],
            'module_hash':hashlib.sha256(canonical).hexdigest(), 'engine':w['engine']}
