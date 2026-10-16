from local_trpg.adventure.engine import AdventureEngine, initial_adventure
from local_trpg.adventure.service import AdventureService
from local_trpg.modules import module_by_id
from local_trpg.schemas import AdventureIntent, PlayerCommand
from local_trpg.storage import Store


def test_act_targets_are_scoped_to_actions_not_inspectable_objects(tmp_path):
    from local_trpg.adventure.wire import adventure_intent_schema
    module=module_by_id('pinebrook')
    engine=AdventureEngine(initial_adventure(module,'玩家','noorah'),module,'forest',lambda *_:[1])
    engine.resolve(AdventureIntent(kind='move',target_id='forest'))
    service=AdventureService(Store(tmp_path/'wire.sqlite'),module,None)
    view=service.interpreter_view(engine.state,PlayerCommand(client_request_id='wire',expected_revision=0,text='我答应护送幼龙'))
    schema=adventure_intent_schema(view)
    act=[b for b in schema['anyOf'] if b['properties']['kind']['const']=='act']
    inspect=[b for b in schema['anyOf'] if b['properties']['kind']['const']=='inspect']
    assert {b['properties']['target_id']['const'] for b in act}=={'accept_escort','feed'}
    assert any(b['properties']['target_id']['const']=='dragon' for b in inspect)
    assert all('attack_id' not in b['properties'] for b in act)
    assert all(b['additionalProperties'] is False for b in schema['anyOf'])
