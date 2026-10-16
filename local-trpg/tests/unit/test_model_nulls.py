import pytest
from local_trpg.model.base import GenerationResult
from local_trpg.model.parsing import parse_result,InvalidOutput
from local_trpg.schemas import PlayerIntent,NpcDecision

def test_real_runner_optional_null_clarification_is_absent_not_a_failed_action():
 result=GenerationResult(final_text='{"kind":"INSPECT","attempt_description":"阅读审计页","clarification_question":null}',finish_reason='stop')
 p=parse_result(result,PlayerIntent)
 assert p.kind=='INSPECT' and p.attempt_description=='阅读审计页' and p.clarification_question==''

def test_null_normalization_never_admits_unknown_fields_or_missing_required_actor():
 for text in ('{"kind":"INSPECT","sql":null}', '{"actor_id":null,"action":{"kind":"SPEAK"}}'):
  with pytest.raises(InvalidOutput):parse_result(GenerationResult(final_text=text,finish_reason='stop'),PlayerIntent if 'sql' in text else NpcDecision)

def test_known_nested_utterance_wire_shape_keeps_same_authority():
 result=GenerationResult(final_text='{"actor_id":"mechanic","action":{"kind":"SPEAK","utterance":{"text":"改接工单禁止保持。","channel":"private","recipient_ids":["player"],"disclose_fact_ids":["E03"]}}}',finish_reason='stop')
 d=parse_result(result,NpcDecision)
 assert d.actor_id=='mechanic' and d.action.kind=='SPEAK' and d.utterance.disclose_fact_ids==['E03']

def test_ambiguous_duplicate_utterance_is_rejected():
 with pytest.raises(InvalidOutput):
  parse_result(GenerationResult(final_text='{"actor_id":"mechanic","action":{"utterance":{"text":"A"}},"utterance":{"text":"B"}}',finish_reason='stop'),NpcDecision)
