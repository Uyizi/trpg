"""Wire compression must preserve the strict decision and view identity boundary."""
import json

import pytest

from local_trpg.model.base import GenerationResult
from local_trpg.model.parsing import InvalidOutput, parse_result
from local_trpg.schemas import NpcDecision


def parse(value, actor_id='doctor'):
    return parse_result(GenerationResult(final_text=json.dumps(value), finish_reason='stop'),
                        NpcDecision, actor_id=actor_id)


def test_compact_work_speech_and_beliefs_expand_without_losing_semantics():
    decision = parse({'a': 'MODULE_ACTION', 'm': 'prepare_pod', 'x': ['guard'],
        'i': ['wrench'], 'z': '检查', 't': '我来检查。', 'c': 'private',
        'r': ['player'], 'd': ['E03'], 'q': ['p1'], 'g': 'g1',
        'b': [{'proposition_id': 'p1', 'stance': 'doubted',
               'source_observation_ids': ['o1'], 'basis': 'testimony'}]})
    assert decision.actor_id == 'doctor'
    assert decision.action.model_dump() == {'kind': 'MODULE_ACTION', 'module_action_id': 'prepare_pod',
        'target_ids': ['guard'], 'item_ids': ['wrench'], 'description': '检查'}
    assert decision.utterance.model_dump() == {'text': '我来检查。', 'channel': 'private',
        'recipient_ids': ['player'], 'disclose_fact_ids': ['E03'], 'claim_refs': ['p1']}
    assert decision.belief_updates[0].source_observation_ids == ['o1']
    assert decision.belief_updates[0].basis == 'testimony'
    assert decision.next_goal_id == 'g1'


def test_compact_wait_has_no_speech_and_speech_defaults_match_legacy():
    assert parse({'a': 'WAIT'}).utterance is None
    compact = parse({'a': 'SPEAK', 't': '尚未核实。'})
    legacy = parse({'actor_id': 'doctor', 'action': {'kind': 'SPEAK'},
                    'utterance': {'text': '尚未核实。'}})
    assert compact == legacy


@pytest.mark.parametrize('value', [
    {'a': 'WAIT', 'actor_id': 'guard'}, {'a': 'WAIT', 'sql': None},
    {'a': 'SPEAK', 'c': 'private'}, {'a': 'SPEAK', 't': 'x', 'c': 'telepathy'},
    {'a': 'SPEAK', 't': 'x', 'd': [1]}, {'a': 'SPEAK', 't': 'x', 'b': [
        {'proposition_id': 'p', 'stance': 'accepted', 'source_observation_ids': [], 'actor_id': 'guard'}]},
    {'a': 'SPEAK', 't': 'x' * 801}, {'a': None},
])
def test_compact_rejects_unknown_ambiguous_or_invalid_fields(value):
    with pytest.raises(InvalidOutput):
        parse(value)


def test_compact_requires_trusted_identity_and_rejects_duplicates():
    with pytest.raises(InvalidOutput):
        parse({'a': 'WAIT'}, actor_id=None)
    with pytest.raises(InvalidOutput):
        parse_result(GenerationResult(final_text='{"a":"WAIT","a":"SPEAK"}',
            finish_reason='stop'), NpcDecision, actor_id='doctor')


def test_compact_schema_excludes_actor_and_forbids_unknown_fields():
    from local_trpg.model.compact_npc import compact_npc_schema
    schema = compact_npc_schema()
    assert schema['required'] == ['a']
    assert schema['additionalProperties'] is False
    assert 'actor_id' not in schema['properties']
    assert 'MODULE_ACTION' in schema['properties']['a']['enum']
    # Returned schemas must not let callers mutate future requests.
    schema['properties'].clear()
    assert 't' in compact_npc_schema()['properties']
