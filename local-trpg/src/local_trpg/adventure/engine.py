"""Rule authority for scene adventures; the model proposes, never mutates state."""
import re
import uuid

from ..engine.reducer import apply_event
from ..engine.rules import RuleError
from ..schemas import Actor, GameEvent, GameSnapshot


def initial_adventure(module, name, character_id='noorah', mock=False):
    w = module['world']
    if character_id not in w['profiles']: raise RuleError('请选择已提供的角色卡')
    actors, sheets, stats, uses, ammo, items = {}, {}, {}, {}, {}, {}
    for template, card in w['profiles'].items():
        aid = 'player' if template == character_id else template
        sheets[aid] = template
        actors[aid] = Actor(id=aid, name=name if aid == 'player' else card['name'],
                            location_id=w['start'], skills=card['skills'])
        stats[aid] = {'hp':card['hp'], 'max_hp':card['hp'], 'ac':card['ac']}
        uses[aid] = card['ability']['uses']
        ammo[aid] = 20
        for item in card['equipment']: items[aid+'_'+item] = aid
    for aid, profile in w['npcs'].items():
        actors[aid] = Actor(id=aid, name=profile['name'], location_id=w['start'] if aid == 'kole' else 'mother')
    items.update({'draconic_page':'kole', **{f'torch_{i}':'mountain' for i in range(1,6)},
                  'flint':'mountain', 'meat':'mountain', 'gold':'mountain',
                  **{aid+'_diamond':'mother' for aid in sheets}})
    for aid, actor in actors.items(): actor.inventory = sorted(i for i, owner in items.items() if owner == aid)
    state = GameSnapshot(campaign_id=uuid.uuid4().hex, module_id=module['module_id'],
        module_version=module['module_version'], module_hash=module['module_hash'],
        actors=actors, items=items, phase='stable', mock=mock,
        flags={'sheets':sheets, 'stats':stats, 'ability_uses':uses, 'ammo':ammo,
               'visited':[w['start']], 'combat':None})
    state.transcript = [{'turn_id':'opening', 'speaker':'DM', 'speaker_id':'dm',
                         'text':module['public']['opening'], 'tick':0}]
    return state


def item_name(item, module):
    names = module['world']['item_names']
    if item == 'draconic_page': return names[item]
    for key in sorted(names, key=len, reverse=True):
        if item == key or item.startswith(key+'_') or item.endswith('_'+key): return names[key]
    return item


class AdventureEngine:
    def __init__(self, state, module, turn_id, roll):
        self.state = state.model_copy(deep=True)
        self.module, self.world, self.turn_id, self.roll = module, module['world'], turn_id, roll
        self.events, self.checks = [], []
        self.roll_index = 0

    @property
    def location(self): return self.state.actors['player'].location_id
    @property
    def scene(self): return self.world['scenes'][self.location]
    @property
    def fighting(self): return bool(self.state.flags.get('combat'))

    def emit(self, kind, payload, actor='player'):
        event = GameEvent(event_id=f'{self.turn_id}:{len(self.events)}', turn_id=self.turn_id,
            beat_index=len(self.events), actor_id=actor, kind=kind, payload=payload,
            location_id=self.state.actors[actor].location_id, game_tick=self.state.tick)
        self.state = apply_event(self.state, event); self.events.append(event)

    def flag(self, key, value): self.emit('FlagSet', {'key':key, 'value':value})
    def notice(self, text): self.emit('PublicNotice', {'text':text, 'recipient_ids':list(self.state.flags['sheets'])})
    def advance(self): self.emit('ClockAdvanced', {'tick':self.state.tick+1})
    def dice(self, sides, count, label):
        self.roll_index += 1
        dice = self.roll(sides, count, f'{self.roll_index}:{label}')
        if len(dice) != count or any(type(x) != int or not 1 <= x <= sides for x in dice):
            raise RuleError('骰点格式不符')
        return dice

    def check(self, skill, dc, label, advantage=False, actor='player', bonus=None):
        n = 2 if advantage else 1
        dice = self.dice(20, n, label)
        bonus = self.state.actors[actor].skills.get(skill, 0) if bonus is None else bonus
        total = max(dice)+bonus
        name = self.world['skill_names'].get(skill, skill)
        outcome = 'success' if total >= dc else 'failure'
        text = f'{self.state.actors[actor].name} · {label}：d20'+('优势取高' if advantage else '')+f' {dice} + {bonus} = {total}，目标 {dc}，'+('成功。' if total >= dc else '未成功。')
        self.emit('CheckResolved', {'dice':dice, 'skill':bonus, 'modifier':0, 'total':total,
            'outcome':outcome, 'description':text, 'die_sides':20, 'difficulty':dc}, actor)
        self.checks.append({'label':label, 'skill':name, 'dice':dice, 'total':total,
                            'outcome':'成功' if total >= dc else '未成功', 'difficulty':dc})
        return total >= dc

    def hp(self, aid, value):
        stats = {k:dict(v) for k,v in self.state.flags['stats'].items()}
        stats[aid]['hp'] = max(0, min(stats[aid]['max_hp'], value))
        self.flag('stats', stats)
        hp = stats[aid]['hp']
        health = 'healthy' if hp == stats[aid]['max_hp'] else 'strained' if hp > 0 else 'incapacitated'
        self.emit('HealthChanged', {'actor_id':aid, 'health':health}, aid)

    def damage(self, aid, amount):
        self.hp(aid, self.state.flags['stats'][aid]['hp']-amount)
        self.notice(f'{self.state.actors[aid].name}受到{amount}点伤害，剩余{self.state.flags["stats"][aid]["hp"]}点生命。')

    def evidence(self, identifier):
        if identifier in self.state.evidence.get('player', []):
            self.notice(self.world['evidence'][identifier]['text']); return
        record = self.world['evidence'][identifier]
        self.emit('EvidenceObserved', {'observer_ids':list(self.state.flags['sheets']),
            'evidence_id':identifier, 'content':record['text'], 'proposition_ids':[], 'source':'player'})

    def transfer(self, item, to):
        self.emit('ItemTransferred', {'item_id':item, 'from_owner':self.state.items[item], 'to_owner':to})

    def finish(self, status, summary):
        self.emit('ScenarioResolved', {'outcome':{'kind':'adventure', 'status':status, 'summary':summary,
            'wyrmling_returned':bool(self.state.flags.get('mother_reassured')),
            'care_accepted':bool(self.state.flags.get('care_accepted'))}})
        self.notice(summary)

    def move(self, destination):
        if destination not in self.scene['exits']: raise RuleError('不能直接跳到未连通的场景。请说明准备怎样经过眼前的区域。')
        gate = self.world['gates'].get(self.location+':'+destination)
        if gate and not self.state.flags.get(gate): raise RuleError('当前通路还有未解决的阻碍。可以观察现场，或向主持人询问你已经知道的情况。')
        if self.fighting: raise RuleError('战斗尚未结束。若要离开，请明确撤退方向。')
        origin=self.location
        followers=list(self.state.flags['sheets'])
        # The patrol leader accompanies the initial patrol, then explicitly
        # returns to town when the escort is accepted.
        if (not self.state.flags.get('escort_accepted') and 'kole' in self.scene['npcs']
            and 'kole' in self.world['scenes'][destination]['npcs']
            and self.state.actors['kole'].location_id==origin):followers.append('kole')
        for aid in followers:
            self.emit('ActorMoved', {'actor_id':aid, 'destination':destination,
                'destination_name':self.world['scenes'][destination]['name']}, aid)
        self.flag('visited', list(dict.fromkeys(self.state.flags['visited']+[destination])))
        combat = self.scene.get('combat')
        if combat and not self.state.flags.get('battle:'+destination):
            previous = self.state.flags.get('encounter:'+destination)
            enemies = previous or [{'id':combat['kind']+'_'+str(i+1), 'name':combat['name']+str(i+1),
                'hp':combat['hp'], 'max_hp':combat['hp'], 'ac':combat['ac']} for i in range(combat['count'])]
            self.flag('combat', {'scene':destination, 'round':1, 'enemies':enemies})
        self.notice(self.scene['description'])

    def inspect(self, target, skill):
        # An owned paper can be read anywhere; its contents are not shown on pickup.
        if target in ('draconic_page','language') and self.state.items.get('draconic_page') == 'player':
            self.evidence('language'); return
        if target in self.state.evidence.get('player', []): self.evidence(target); return
        obj = self.scene['investigations'].get(target)
        if not obj: raise RuleError('请指出眼前要查看的物件；尚未到达的场景无法远程调查。')
        if obj.get('requires') and not self.state.flags.get(obj['requires']): raise RuleError('敌人仍在眼前，无法安稳搜查物品。')
        allowed = obj.get('skills', [])
        if skill and skill not in allowed: raise RuleError('这项调查需要适合的技能或更具体的做法。')
        if allowed:
            selected = skill or max(allowed, key=lambda s:self.state.actors['player'].skills.get(s,0))
            if not self.check(selected, obj['dc'], obj['title']):
                self.notice('暂时无法判断；眼前的实际情况仍然可见，可以换一种方法或向同伴请教。'); return
        if obj.get('evidence'): self.evidence(obj['evidence'])
        if obj.get('loot'):
            loot = [i for i, owner in self.state.items.items() if owner == self.location and (i.startswith('torch_') or i in ('flint','meat','gold'))]
            self.flag('backpack_open', True)
            self.notice('打开背包，看到：'+('、'.join(item_name(i,self.module) for i in loot) if loot else '已经空了')+'。查看不会自动取走物品。')

    def act(self, action, intent, original):
        handler = action['handler']
        if intent.skill and intent.skill not in action.get('skills', []): raise RuleError('请说明这个技能如何用于眼前的行动。')
        if handler == 'accept_escort':
            if self.state.flags.get('escort_accepted'): self.notice('你们已经接下护送，科尔已返回镇上。'); return
            if self.state.actors['kole'].location_id!=self.location:raise RuleError('科尔不在这里，无法当面交接委托和龙语页。')
            self.flag('escort_accepted', True); self.evidence('escort'); self.transfer('draconic_page','player')
            self.emit('ActorMoved', {'actor_id':'kole', 'destination':'patrol','destination_name':'镇外林道'}, 'kole')
            self.notice('科尔留下龙语页后返回松溪镇报告。幼龙还很虚弱，通往山脚的小径就在前方。')
        elif handler == 'feed':
            if self.state.flags.get('dragon_fed'): self.notice('幼龙已经吃过东西，正跟着队伍。'); return
            food = next((i for i,o in self.state.items.items() if o=='player' and (i.endswith('_rations') or i=='meat')), None)
            if not food: raise RuleError('你手中没有已取得的食物；可以请有口粮的同伴分给你。')
            self.transfer(food,'consumed'); self.flag('dragon_fed',True)
            self.notice('幼龙吃下食物，精神好了一些，发出“Nytha”的声音。它开始跟随你们。')
        elif handler == 'light':
            if not any(i.startswith('torch_') and o=='player' for i,o in self.state.items.items()) or self.state.items.get('flint')!='player':
                raise RuleError('需要先取得火把和生火工具，才能点燃照明。')
            self.flag('light',True); self.notice('火把点亮了上行通道，队伍可以继续深入。')
        elif handler == 'climb':
            if self.state.flags.get('wall_passed'): self.notice('绳索已固定，队伍可以沿原路通过。'); return
            skill = intent.skill or max(action['skills'],key=lambda s:self.state.actors['player'].skills.get(s,0))
            advantage = any(i.endswith('_climbers_kit') and o=='player' for i,o in self.state.items.items())
            self.advance()
            if not self.check(skill,action['dc'],'攀登冰墙',advantage): self.damage('player',sum(self.dice(6,1,'坠落伤害')))
            if self.state.flags['stats']['player']['hp']==0:
                self.hp('player',1); self.notice('同伴拉住你并协助到达顶部；你恢复到1点生命。这是单人适配的失败推进。')
            self.flag('wall_passed',True); self.notice('你抵达上方，同伴跟进。通道继续通向前方冰幕。')
        elif handler in ('answer','melt','break'):
            if self.state.flags.get('mirror_passed'): self.notice('冰幕已经打开。'); return
            if handler == 'answer':
                exact = self.scene['solution'] in original.casefold()
                translated = ('language' in self.state.evidence.get('player',[]) and '银龙' in original and '龙语' in original)
                if not (exact or translated):
                    self.notice('冰镜没有打开。龙首提醒你：需要说出所护送生物的种类，并使用龙语。'); return
                for aid,stat in list(self.state.flags['stats'].items()): self.hp(aid,stat['max_hp'])
                self.flag('slide_blessing',True); self.notice('冰幕化作带着暖意的水，队伍生命恢复；下一次滑道判断获得优势。')
            elif handler == 'melt':
                if not self.state.flags.get('light'): raise RuleError('你还没有点燃火把。')
                self.notice('你用火把慢慢融开通路，没有获得冰镜的祝福。')
            else:
                self.damage('player',sum(self.dice(6,1,'碎冰伤害'))); self.notice('冰幕被击碎，通路打开；没有获得祝福。')
                if self.state.flags['stats']['player']['hp']==0: self.hp('player',1)
            self.advance(); self.flag('mirror_passed',True)
        elif handler in ('choose_slide','recover_slide'):
            if self.state.flags.get('slides_passed'): self.notice('队伍已经通过滑道。'); return
            sliding = self.state.flags.get('sliding',False)
            if (handler=='recover_slide') != bool(sliding): raise RuleError('请根据眼前状态选择路线，或在已经滑行时调整路线。')
            skill = intent.skill or max(action['skills'],key=lambda s:self.state.actors['player'].skills.get(s,0))
            advantage = bool(self.state.flags.get('slide_blessing')) and not sliding
            self.advance()
            if advantage:self.flag('slide_blessing',False)
            success = self.check(skill,action['dc'],action['title'],advantage)
            if success or sliding:
                if not success:self.damage('player',sum(self.dice(6,1,'滑道伤害')))
                if self.state.flags['stats']['player']['hp']==0:self.hp('player',1)
                self.flag('slides_passed',True);self.flag('sliding',False)
                self.notice('你们抵达洞底，可以沿短路继续前往孵化洞窟。')
            else:
                self.flag('sliding',True);self.notice('你进入了一条有尖冰的滑道，前方有交叉路线。需要尝试运动或体操跳到安全路线。')
        elif handler == 'explain_rescue':
            if self.state.flags.get('mother_reassured'): self.notice('罗恩已经了解你们的来意。'); return
            self.flag('mother_reassured',True);self.evidence('rorn_story')
            for aid in self.state.flags['sheets']:self.transfer(aid+'_diamond',aid)
            self.notice('幼龙的反应和护送经过让罗恩放下戒备。她感谢队伍，给每位成员一颗价值50金币的钻石，并询问你们是否愿意回镇后照料幼龙。你可以答应或婉拒。')
        elif handler in ('accept_care','decline_care'):
            if not self.state.flags.get('mother_reassured'):raise RuleError('母龙仍在等待你解释来意。')
            self.flag('care_decided',True);self.flag('care_accepted',handler=='accept_care')
            self.notice('你们答应在镇上照顾幼龙。' if handler=='accept_care' else '你们婉拒长期照料，幼龙留在母亲身边。')
        elif handler == 'return_village':
            if not self.state.flags.get('mother_reassured'):raise RuleError('母龙仍在等待说明，尚未允许你们平静离去。')
            if not self.state.flags.get('care_decided'):raise RuleError('请先回答是否愿意照料幼龙；答应与婉拒都可以。')
            self.advance();self.finish('completed','护送完成。队伍与罗恩道别，返回松溪镇。'+('幼龙随你们回镇接受照料。' if self.state.flags.get('care_accepted') else '幼龙留在母龙身边。')+'被盗龙蛋的营救属于尚未实现的后续章节。')

    def basic_attack(self, aid, target, attack_id=''):
        card = self.world['profiles'][self.state.flags['sheets'][aid]]
        attack = next((a for a in card['attacks'] if a['id']==attack_id),None) if attack_id else card['attacks'][0]
        if not attack:raise RuleError('角色卡没有这项攻击')
        if attack.get('ammunition'):
            ammo = dict(self.state.flags['ammo'])
            if ammo[aid]<=0:raise RuleError('箭矢已用尽，请换用其他攻击')
            ammo[aid]-=1;self.flag('ammo',ammo)
        hit = self.check('', target['ac'], attack['name']+' → '+target['name'], actor=aid,bonus=attack['bonus'])
        if not hit:return
        n = 2 if self.state.flags['sheets'][aid]=='noorah' and self.state.flags['combat']['round']==1 else attack.get('dice',0)
        damage = attack.get('fixed_damage',0) if not n else sum(self.dice(6,n,aid+':伤害'))+attack['damage_bonus']
        target['hp'] = max(0,target['hp']-damage)
        self.notice(f'{self.state.actors[aid].name}命中{target["name"]}，造成{damage}点伤害；'+('目标已被击退。' if target['hp']==0 else '目标仍在战斗。'))

    def ability(self, intent, combat):
        aid = intent.helper_id or 'player'
        if aid not in self.state.flags['sheets']:raise RuleError('没有这位可协助的同伴')
        if self.state.flags['stats'][aid]['hp']<=0:raise RuleError('该角色昏迷，无法使用能力')
        card = self.world['profiles'][self.state.flags['sheets'][aid]]
        ability = card['ability']; uses = dict(self.state.flags['ability_uses'])
        if intent.ability_id and ability['id'] != intent.ability_id:raise RuleError('该角色没有此能力')
        if ability['id']=='first_strike':raise RuleError('首轮突袭是首轮命中时自动生效的能力')
        if not uses[aid]:raise RuleError('这项能力的使用次数已耗尽')
        if 'heal' in ability:
            target = intent.target_id or 'player'
            if ability['id']=='second_wind' and target!=aid:raise RuleError('恢复体力只能用于自己')
            if target not in self.state.flags['stats']:raise RuleError('请指定队伍中要治疗的人')
            self.hp(target,self.state.flags['stats'][target]['hp']+ability['heal'])
            self.notice(f'{self.state.actors[aid].name}使用{ability["name"]}治疗{self.state.actors[target].name}。')
        else:
            if not combat:raise RuleError('目前没有可见的战斗目标')
            live = [e for e in combat['enemies'] if e['hp']>0]
            allocation=intent.target_ids or [intent.target_id or live[0]['id']]*3
            if len(allocation)!=3 or any(id not in {e['id'] for e in live} for id in allocation):
                raise RuleError('请为三枚飞弹指定当前可见的目标，每枚造成3点伤害；可以多枚指向同一个目标。')
            targets={e['id']:e for e in live}
            for id in allocation:targets[id]['hp']=max(0,targets[id]['hp']-3)
            self.notice('三枚魔法飞弹按你的分配命中：'+'、'.join(targets[id]['name'] for id in allocation)+'；每枚造成3点伤害。')
        uses[aid]-=1;self.flag('ability_uses',uses)

    def combat_turn(self, intent):
        combat = self.state.flags['combat']
        if not combat:raise RuleError('当前没有战斗')
        combat = {'scene':combat['scene'], 'round':combat['round'], 'enemies':[dict(e) for e in combat['enemies']]}
        if intent.kind=='attack':
            if self.state.flags['stats']['player']['hp']<=0:raise RuleError('你目前昏迷，可以请求同伴治疗，或让清醒同伴继续战斗。')
            alive = [e for e in combat['enemies'] if e['hp']>0]
            target = next((e for e in alive if e['id']==intent.target_id),None) if intent.target_id else alive[0]
            if target is None:raise RuleError('请指定当前仍在战斗的敌人')
            self.basic_attack('player',target,intent.attack_id)
        elif intent.kind=='ability':self.ability(intent,combat)
        # The player authorizes party help by taking a combat turn; allies never
        # spend limited abilities or decide story choices on the player's behalf.
        for aid in self.state.flags['sheets']:
            if aid=='player' or aid==intent.helper_id or self.state.flags['stats'][aid]['hp']<=0:continue
            target = next((e for e in combat['enemies'] if e['hp']>0),None)
            if target:self.basic_attack(aid,target)
        self.advance()
        alive = [e for e in combat['enemies'] if e['hp']>0]
        for i, enemy in enumerate(alive):
            targets = [aid for aid,s in self.state.flags['stats'].items() if s['hp']>0]
            if not targets:break
            aid = targets[(i+combat['round']-1)%len(targets)]
            data = self.scene['combat']
            if self.check('',self.state.flags['stats'][aid]['ac'],enemy['name']+'攻击'+self.state.actors[aid].name,bonus=data['attack_bonus']):
                self.damage(aid,sum(self.dice(6,1,enemy['id']+':伤害'))+data['damage_bonus'])
        self.flag('encounter:'+self.location,combat['enemies'])
        if not alive:
            self.flag('battle:'+self.location,True);self.flag('combat',None)
            for aid,s in list(self.state.flags['stats'].items()):
                if s['hp']==0:self.hp(aid,1)
            self.notice('眼前敌人已被击退，战斗结束。')
            if self.location=='hatchery':
                # Published encounter: the mother arrives immediately after
                # victory, without requiring a hidden extra movement command.
                self.move('mother')
            else:self.notice('你们可以观察、搜查现场或继续前行。')
        elif not any(s['hp']>0 for s in self.state.flags['stats'].values()):
            self.flag('combat',None);self.finish('defeated','全队失去作战能力，本次护送中止。未判定角色死亡，也未判定幼龙已获救。')
        else:
            combat['round']+=1;self.flag('combat',combat)

    def resolve(self, intent, original=''):
        kind, target = intent.kind, intent.target_id
        if self.state.outcome:raise RuleError('本章已结束，可以在桌外对话复盘或从存档建立分支。')
        if self.state.flags['stats']['player']['hp']<=0 and kind in ('move','inspect','act','attack'):
            raise RuleError('你目前昏迷，无法自行操作。可以请求清醒同伴治疗，或让同伴继续战斗。')
        if kind=='look':
            from .views import scene_description
            self.notice(scene_description(self.state,self.module));return
        if kind=='talk':return  # The service routes an actual Qwen NPC reply.
        if kind=='inspect':self.inspect(target,intent.skill);return
        if kind=='retreat':
            options = [x for x in self.scene['exits'] if x in self.state.flags['visited']]
            if target and target not in options:raise RuleError('撤退需要选择已经走过的相邻路径')
            if not options:raise RuleError('这里没有已经走过的撤退路径')
            if self.fighting:self.flag('encounter:'+self.location,self.state.flags['combat']['enemies']);self.flag('combat',None)
            self.move(target or options[0]);self.notice('队伍撤回前一处地点；尚未击退的敌人仍留在原处。');return
        if kind=='move':self.move(target);return
        if self.fighting:
            if kind not in ('attack','ability','wait'):raise RuleError('敌人挡在眼前；可以攻击、治疗、等待同伴行动或明确撤退。')
            self.combat_turn(intent);return
        if kind=='ability':self.ability(intent,None);return
        if kind=='attack':raise RuleError('这里没有正在交战的敌人。若要做破坏或威胁，请说明对象和意图。')
        if kind=='wait':self.notice('你留在原处观察，没有执行新的操作。');return
        if kind=='act':
            if target in ('take_backpack','take_item'):
                if not self.state.flags.get('backpack_open') or self.location!='mountain':raise RuleError('需要先在现场查看背包')
                if target=='take_item':
                    item=intent.item_id
                    if self.state.items.get(item)!='mountain' or not (item.startswith('torch_') or item in ('flint','meat','gold')):
                        raise RuleError('这件物品不在已经查看的背包里。')
                    self.transfer(item,'player');self.notice('你取走'+item_name(item,self.module)+'，其余物品留在原处。');return
                for item,owner in list(self.state.items.items()):
                    if owner=='mountain' and (item.startswith('torch_') or item in ('flint','meat','gold')):self.transfer(item,'player')
                self.notice('你收起刚才看到的背包物品。');return
            action = self.scene['actions'].get(target)
            if not action:raise RuleError('这个做法尚未对应可裁定的行动。请说明对象、方法和希望达成的结果。')
            self.act(action,intent,original);return
        raise RuleError(intent.clarification or '请说明你想做什么。')
