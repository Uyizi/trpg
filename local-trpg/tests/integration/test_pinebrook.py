"""Public adventure boundaries and real deterministic rule outcomes."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from local_trpg.api import create_app
from local_trpg.config import Settings
from local_trpg.modules import load_module
from local_trpg.schemas import PlayerCommand
from local_trpg.storage import Store


def pinebrook():
    from local_trpg.modules import module_by_id
    return module_by_id('pinebrook')


def test_adventure_picker_offers_real_module_without_revealing_later_scenes(tmp_path):
    class Queue:
        active = 0
        async def warmup(self): pass
        async def close(self): pass
    app = create_app(Settings(data_dir=tmp_path, backend='mock'), queue=Queue())
    with TestClient(app) as client:
        choices = client.get('/api/modules').json()
        assert {'pinebrook', 'reserve_cooling'} <= {m['module_id'] for m in choices}
        option = next(m for m in choices if m['module_id'] == 'pinebrook')
        assert len(option['characters']) == 4
        assert 'orn darastrix' not in json.dumps(choices)
        response = client.post('/api/campaigns', json={'module_id': 'pinebrook', 'player_setup': {'name': '远行者', 'character_id': 'noorah'}})
        assert response.status_code == 200, response.text
        view = response.json()['view']
        assert view['player']['name'] == '远行者' and view['player']['ac'] == 14
        assert view['player']['hp'] == 11 and view['ruleset'] == 'pinebrook_d20'
        assert not view['clocks'] and not view['outcome']
        assert 'cells_saved' not in json.dumps(view)
        assert 'Rorn' not in json.dumps(view) and 'orn darastrix' not in json.dumps(view)
        old = client.post('/api/campaigns', json={'player_setup': {'name': '旧冒险'}})
        assert old.status_code == 200 and old.json()['view']['module_title'] == '备用冷却'


def test_private_future_facts_do_not_change_current_projection_or_model_sources():
    from local_trpg.adventure.engine import initial_adventure
    from local_trpg.adventure.views import player_view, guide_view
    module = pinebrook(); state = initial_adventure(module, '玩家', 'noorah')
    altered = copy.deepcopy(module)
    altered['world']['scenes']['mirror']['solution'] = 'UNSEEN_SOLUTION_CANARY'
    altered['world']['scenes']['forest']['name'] = 'UNVISITED_ROOM_SECRET_NAME'
    assert player_view(state, module) == player_view(state, altered)
    assert guide_view('冰镜答案是什么', state, module).model_dump() == guide_view('冰镜答案是什么', state, altered).model_dump()


def test_npc_does_not_receive_a_private_conversation_from_the_player_transcript(tmp_path):
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.adventure.service import AdventureService
    module=pinebrook();engine=AdventureEngine(initial_adventure(module,'玩家','noorah'),module,'private',lambda *_:[1])
    engine.emit('SpeechOccurred',{'text':'PRIVATE_OTHER_NPC_CANARY','channel':'private','recipient_ids':['evandon']})
    engine.state.transcript.append({'turn_id':'private','speaker':'玩家','speaker_id':'player','text':'PRIVATE_OTHER_NPC_CANARY','tick':0})
    service=AdventureService(Store(tmp_path/'private.sqlite'),module,None)
    view=service.npc_view(engine.state,'gallantine','你知道什么？')
    assert 'PRIVATE_OTHER_NPC_CANARY' not in view.model_dump_json()


def test_npc_does_not_gain_the_players_unshared_evidence(tmp_path):
    from local_trpg.adventure.engine import initial_adventure
    from local_trpg.adventure.service import AdventureService
    module=pinebrook();state=initial_adventure(module,'玩家','noorah')
    state.evidence['player']=['rorn_story']
    service=AdventureService(Store(tmp_path/'evidence.sqlite'),module,None)
    projected=service.npc_view(state,'kole','后来发生什么？')
    assert not any(source['event_id']=='rorn_story' for source in projected.sources)


def test_authored_npc_knowledge_names_the_npc_instead_of_assigning_its_job_to_player(tmp_path):
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.adventure.service import AdventureService
    from local_trpg.schemas import AdventureIntent
    module=pinebrook();engine=AdventureEngine(initial_adventure(module,'玩家','noorah'),module,'role',lambda *_:[1])
    engine.resolve(AdventureIntent(kind='move',target_id='forest'))
    service=AdventureService(Store(tmp_path/'role.sqlite'),module,None)
    projected=service.npc_view(engine.state,'kole','谁回镇报告？')
    identity=next(source['text'] for source in projected.sources if source['event_id']=='identity')
    assert '科尔队长要回镇报告' in identity
    assert '你要回镇报告' not in identity


def test_unknown_template_is_rejected_instead_of_silently_changing_character():
    from local_trpg.adventure.engine import initial_adventure
    with pytest.raises(ValueError):
        initial_adventure(pinebrook(), '玩家', 'unknown')


def test_registered_scene_gate_cannot_be_skipped_by_model():
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.schemas import AdventureIntent
    from local_trpg.engine.rules import RuleError
    module = pinebrook(); state = initial_adventure(module, '玩家', 'noorah')
    engine = AdventureEngine(state, module, 'skip', lambda *_: [20])
    with pytest.raises(RuleError):
        engine.resolve(AdventureIntent(kind='move', target_id='mirror'))
    assert engine.state.model_dump() == state.model_dump()


def test_different_dice_shapes_survive_save_restore_and_keep_old_dice_valid(tmp_path):
    store = Store(tmp_path/'dice.sqlite')
    # Shape validation and caching belong to the storage boundary.
    from local_trpg.adventure.engine import initial_adventure
    s = initial_adventure(pinebrook(), '玩家', 'noorah'); store.create(s)
    turn = store.submit(s.campaign_id, PlayerCommand(client_request_id='dice', expected_revision=0, text='观察'))
    first = store.roll_dice(turn, 'notice', sides=20, count=1, fixed=[17])
    assert first == [17]
    assert store.roll_dice(turn, 'notice', sides=20, count=1, fixed=[2]) == [17]
    assert store.roll(turn, 'old', fixed=[3, 4]) == [3, 4]
    with pytest.raises(ValueError): store.roll_dice(turn, 'bad', sides=20, count=1, fixed=[21])


def test_complete_adventure_and_d20_damage_rolls_survive_event_replay(tmp_path):
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.adventure.views import player_view
    from local_trpg.schemas import AdventureIntent
    module=pinebrook(); store=Store(tmp_path/'route.sqlite')
    state=initial_adventure(module,'路线玩家','noorah');store.create(state)
    def run(kind,target='',original='',**kw):
        nonlocal state
        cmd=PlayerCommand(client_request_id='route:'+str(state.revision),expected_revision=state.revision,text=original or target or kind)
        turn=store.submit(state.campaign_id,cmd);lease=store.claim(turn)
        engine=AdventureEngine(state,module,turn,lambda sides,count,key:store.roll_dice(turn,key,sides=sides,count=count,fixed=[sides]*count))
        engine.resolve(AdventureIntent(kind=kind,target_id=target,**kw),cmd.text)
        store.commit(turn,lease['epoch'],engine.state,engine.events,{'narration':'\n'.join(e.payload['text'] for e in engine.events if e.kind=='PublicNotice'),'dialogue':[],'checks':engine.checks,'degraded':False,'view':{}})
        state=store.snapshot(state.campaign_id)
    run('move','forest');run('act','accept_escort');run('act','feed');run('move','mountain')
    view=player_view(state,module)
    assert view['combat']['enemies'] and 'max_hp' not in view['combat']['enemies'][0]
    for _ in range(8):
        if not state.flags['combat']:break
        run('attack')
    assert state.flags['battle:mountain']
    run('inspect','backpack')
    assert state.items['torch_1']=='mountain'  # Looking is not looting.
    run('act','take_backpack');run('move','cave');run('act','light');run('move','wall')
    run('act','climb');run('move','mirror');run('inspect','draconic_page')
    run('act','answer',original='我想试一个答案',words='orn darastrix')
    assert not state.flags.get('mirror_passed')  # Model cannot supply the solution.
    run('act','answer',original='我说 orn darastrix')
    assert state.flags['slide_blessing']
    run('move','slides');run('act','choose_slide')
    assert state.flags['slides_passed'] and not state.flags['slide_blessing']
    run('move','hatchery')
    for _ in range(10):
        if not state.flags['combat']:break
        run('attack')
    assert state.flags['battle:hatchery']
    assert state.actors['player'].location_id=='mother'  # Mother returns as the encounter ends.
    run('act','explain_rescue');run('act','decline_care');run('act','return_village')
    assert state.outcome['status']=='completed' and state.outcome['wyrmling_returned']
    bundle=store.export(state.campaign_id)
    assert any(row['roll_key'].startswith('dice:20:') for row in bundle['rolls'])
    restored=store.restore(bundle,module['module_hash'])
    assert store.snapshot(restored).model_dump(exclude={'campaign_id'})==state.model_dump(exclude={'campaign_id'})
    original=load_module()
    from local_trpg.engine.rules import initial_state
    old=initial_state(original,'旧规则');store.create(old)
    assert store.restore(store.export(old.campaign_id),original['module_hash'])


def test_failed_climb_progresses_but_does_not_heal_an_unrelated_injury():
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.schemas import AdventureIntent
    state=initial_adventure(pinebrook(),'攀登者','noorah')
    state.actors['player'].location_id='wall'
    for aid in state.flags['sheets']:state.actors[aid].location_id='wall'
    engine=AdventureEngine(state,pinebrook(),'fall',lambda sides,count,_:[1 if sides==20 else 4]*count)
    engine.resolve(AdventureIntent(kind='act',target_id='climb'))
    assert engine.state.flags['wall_passed']
    assert engine.state.flags['stats']['player']['hp']==7


def test_limited_healing_cannot_be_repeated_after_uses_are_spent():
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.schemas import AdventureIntent
    from local_trpg.engine.rules import RuleError
    state=initial_adventure(pinebrook(),'治疗者','evandon')
    engine=AdventureEngine(state,pinebrook(),'healing',lambda sides,count,_:[1]*count)
    for _ in range(2):engine.resolve(AdventureIntent(kind='ability',ability_id='cure_wounds'))
    assert engine.state.flags['ability_uses']['player']==0
    before=engine.state.model_dump()
    with pytest.raises(RuleError):engine.resolve(AdventureIntent(kind='ability',ability_id='cure_wounds'))
    assert before==engine.state.model_dump()


def test_patrol_leader_follows_into_the_forest_before_handing_over_the_quest():
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.adventure.views import visible_npcs
    from local_trpg.schemas import AdventureIntent
    module=pinebrook();engine=AdventureEngine(initial_adventure(module,'玩家','noorah'),module,'follow',lambda sides,count,_:[sides]*count)
    engine.resolve(AdventureIntent(kind='move',target_id='forest'))
    assert 'kole' in visible_npcs(engine.state,module)
    engine.resolve(AdventureIntent(kind='act',target_id='accept_escort'))
    assert 'kole' not in visible_npcs(engine.state,module)


def test_three_missiles_can_split_without_becoming_three_free_casts():
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.schemas import AdventureIntent
    module=pinebrook();state=initial_adventure(module,'法师','gallantine')
    state.actors['player'].location_id='forest';state.flags['escort_accepted']=True
    engine=AdventureEngine(state,module,'split',lambda sides,count,_:[1]*count)
    engine.resolve(AdventureIntent(kind='move',target_id='mountain'))
    combat=copy.deepcopy(engine.state.flags['combat'])
    intent=AdventureIntent(kind='ability',ability_id='magic_missile',target_ids=['icicle_1','icicle_2','icicle_3'])
    engine.ability(intent,combat)
    assert [e['hp'] for e in combat['enemies']]==[4,4,4,7,7]
    assert engine.state.flags['ability_uses']['player']==1


@pytest.mark.asyncio
async def test_brief_mode_commits_real_rule_description_without_a_second_model_narration(tmp_path):
    from local_trpg.adventure.engine import initial_adventure
    from local_trpg.adventure.service import AdventureService
    from local_trpg.schemas import AdventureIntent
    class InterpreterOnly:
        metrics=[]
        async def interpret_adventure(self,view):return AdventureIntent(kind='move',target_id='forest')
    module=pinebrook();store=Store(tmp_path/'brief.sqlite')
    state=initial_adventure(module,'玩家','noorah');store.create(state)
    service=AdventureService(store,module,None,agent_factory=lambda *_:InterpreterOnly(),narration_mode='brief')
    receipt=await service.submit_turn(state.campaign_id,PlayerCommand(client_request_id='brief',expected_revision=0,text='我走入树林'))
    await service.tasks[receipt.turn_id]
    status=service.status(state.campaign_id,receipt.turn_id)
    assert status['status']=='committed' and not status['result']['degraded']
    assert '蓝灰色' in status['result']['narration']
    assert store.snapshot(state.campaign_id).actors['player'].location_id=='forest'


def test_taking_one_discovered_item_does_not_take_the_entire_container():
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.schemas import AdventureIntent
    state=initial_adventure(pinebrook(),'玩家','noorah')
    state.actors['player'].location_id='mountain';state.flags.update({'backpack_open':True,'battle:mountain':True})
    engine=AdventureEngine(state,pinebrook(),'one-item',lambda *_:[1])
    engine.resolve(AdventureIntent(kind='act',target_id='take_item',item_id='torch_1'))
    assert engine.state.items['torch_1']=='player'
    assert engine.state.items['gold']=='mountain' and engine.state.items['torch_2']=='mountain'


def test_unconscious_character_cannot_climb_or_move_under_own_power():
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.schemas import AdventureIntent
    from local_trpg.engine.rules import RuleError
    state=initial_adventure(pinebrook(),'玩家','noorah');state.flags['stats']['player']['hp']=0
    engine=AdventureEngine(state,pinebrook(),'unconscious',lambda *_:[1])
    with pytest.raises(RuleError):engine.resolve(AdventureIntent(kind='move',target_id='forest'))
    assert engine.state.actors['player'].location_id=='patrol'


def test_companion_retains_a_shared_direct_experience_after_leaving_the_scene(tmp_path):
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.adventure.service import AdventureService
    module=pinebrook();engine=AdventureEngine(initial_adventure(module,'玩家','noorah'),module,'memory',lambda *_:[1])
    engine.notice('我们共同攀过冰墙，绳索已经收好。')
    engine.move('forest')
    service=AdventureService(Store(tmp_path/'memory.sqlite'),module,None)
    assert '共同攀过冰墙' in service.npc_view(engine.state,'gallantine','刚才做过什么？').model_dump_json()


def test_combat_exposes_only_retreat_paths_and_no_automatic_mother_arrival():
    from local_trpg.adventure.engine import AdventureEngine, initial_adventure
    from local_trpg.adventure.views import player_view
    from local_trpg.adventure.wire import adventure_intent_schema
    from local_trpg.adventure.service import AdventureService
    from local_trpg.schemas import AdventureIntent
    module=pinebrook();state=initial_adventure(module,'玩家','noorah')
    for actor in state.actors.values():actor.location_id='slides'
    state.flags.update(slides_passed=True,visited=['patrol','forest','mountain','cave','wall','mirror','slides'])
    engine=AdventureEngine(state,module,'hatchery',lambda *_:[1])
    engine.resolve(AdventureIntent(kind='move',target_id='hatchery'))
    view=player_view(engine.state,module)
    assert [e['id'] for e in view['scene']['exits']]==['slides']
    assert '归来的银龙' not in json.dumps(view,ensure_ascii=False)
    service=AdventureService(None,module,None)
    wire=adventure_intent_schema(service.interpreter_view(engine.state,PlayerCommand(client_request_id='wire',expected_revision=0,text='战斗')))
    assert not any(b['properties']['kind']['const']=='move' for b in wire['anyOf'])


@pytest.mark.asyncio
async def test_addressed_npc_question_reaches_real_npc_role_without_action_classifier(tmp_path):
    from local_trpg.adventure.engine import initial_adventure
    from local_trpg.adventure.service import AdventureService
    from local_trpg.schemas import NarrationResult
    class NpcOnly:
        metrics=[]
        async def interpret_adventure(self,view):raise AssertionError('explicit dialogue is not an action classification')
        async def adventure_npc(self,view):
            assert view.question=='队长科尔，今天的巡逻要注意什么？'
            return NarrationResult(text='注意昨天发现脚印的地方。',source_event_ids=['identity'])
    module=pinebrook();store=Store(tmp_path/'dialogue.sqlite')
    state=initial_adventure(module,'玩家','noorah');store.create(state)
    service=AdventureService(store,module,None,agent_factory=lambda *_:NpcOnly(),narration_mode='brief')
    receipt=await service.submit_turn(state.campaign_id,PlayerCommand(client_request_id='question',expected_revision=0,text='队长科尔，今天的巡逻要注意什么？'))
    await service.tasks[receipt.turn_id]
    status=service.status(state.campaign_id,receipt.turn_id)
    assert status['status']=='committed'
    assert status['result']['dialogue'][0]['text']=='注意昨天发现脚印的地方。'


def test_explicit_question_routing_does_not_consume_ability_requests_or_remote_names():
    from local_trpg.adventure.service import addressed_question
    people=[{'id':'kole','name':'科尔队长'}]
    assert addressed_question('科尔，请治疗我。',people) is None
    assert addressed_question('罗恩，你是谁？',people) is None
    assert addressed_question('科尔，接下来谁负责护送？',people).target_id=='kole'


def test_explicit_options_use_only_current_targets_and_keep_mirror_answer_player_supplied():
    from local_trpg.adventure.commands import action_choices, explicit_intent
    from local_trpg.adventure.engine import initial_adventure
    from local_trpg.adventure.views import player_view,available_actions
    module=pinebrook();state=initial_adventure(module,'玩家','noorah')
    view=player_view(state,module)
    assert explicit_intent('沿林道进入树林',view,available_actions(state,module)).target_id=='forest'
    assert explicit_intent('前往母龙巢穴',view,available_actions(state,module)) is None
    for aid in state.flags['sheets']:state.actors[aid].location_id='mirror'
    view=player_view(state,module);actions=available_actions(state,module)
    choices=action_choices(view,actions)
    assert 'orn darastrix' not in json.dumps(choices,ensure_ascii=False)
    intent=explicit_intent('向冰镜说出自己的回答：orn darastrix',view,actions)
    assert intent.kind=='act' and intent.target_id=='answer' and intent.words=='orn darastrix'


@pytest.mark.asyncio
async def test_selected_explicit_action_does_not_need_model_classification(tmp_path):
    from local_trpg.adventure.engine import AdventureEngine,initial_adventure
    from local_trpg.adventure.service import AdventureService
    class NoClassifier:
        metrics=[]
        async def interpret_adventure(self,view):raise AssertionError('selected action is already explicit')
    module=pinebrook();engine=AdventureEngine(initial_adventure(module,'玩家','noorah'),module,'setup',lambda *_:[1])
    engine.move('forest')
    store=Store(tmp_path/'selected.sqlite');store.create(engine.state)
    service=AdventureService(store,module,None,agent_factory=lambda *_:NoClassifier(),narration_mode='brief')
    receipt=await service.submit_turn(engine.state.campaign_id,PlayerCommand(client_request_id='select',expected_revision=0,text='答应护送幼龙'))
    await service.tasks[receipt.turn_id]
    assert service.status(engine.state.campaign_id,receipt.turn_id)['status']=='committed'
    assert store.snapshot(engine.state.campaign_id).flags['escort_accepted']


def test_polite_ability_request_is_not_routed_to_a_non_executing_npc_reply():
    from local_trpg.adventure.service import addressed_question
    people=[{'id':'evandon','name':'Evandon Haart'}]
    assert addressed_question('Evandon Haart，请用治疗伤口治疗我，好吗？',people) is None
    assert addressed_question('Evandon Haart，治疗伤口能恢复多少生命？',people).kind=='talk'


@pytest.mark.asyncio
async def test_reading_a_new_page_displays_the_actual_read_content_in_brief_mode(tmp_path):
    from local_trpg.adventure.engine import initial_adventure
    from local_trpg.adventure.service import AdventureService
    class NoClassifier:metrics=[]
    module=pinebrook();state=initial_adventure(module,'玩家','noorah')
    state.items['draconic_page']='player';state.actors['player'].inventory.append('draconic_page')
    store=Store(tmp_path/'read.sqlite');store.create(state)
    service=AdventureService(store,module,None,agent_factory=lambda *_:NoClassifier(),narration_mode='brief')
    receipt=await service.submit_turn(state.campaign_id,PlayerCommand(client_request_id='read',expected_revision=0,text='阅读随身的龙语对照页'))
    await service.tasks[receipt.turn_id]
    status=service.status(state.campaign_id,receipt.turn_id)
    assert status['status']=='committed'
    assert 'orn' in status['result']['narration']
    assert '行动已按规则结算' not in status['result']['narration']
