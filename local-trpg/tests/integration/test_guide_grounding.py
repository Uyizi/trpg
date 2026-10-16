import json

import pytest

from local_trpg.agents.runner import AgentRunner
from local_trpg.engine.guidance import guide_view
from local_trpg.engine.rules import initial_state
from local_trpg.model.base import GenerationResult
from local_trpg.model.parsing import InvalidOutput
from local_trpg.modules import load_module
from local_trpg.schemas import PlayerCommand
from local_trpg.storage import Store


class SourceReviewQueue:
    def __init__(self, always_bad=False):
        self.requests = []
        self.always_bad = always_bad

    def count_tokens(self, messages):
        return 200

    async def generate(self, request):
        self.requests.append(request)
        if request.role_kind == 'guide_review':
            payload = json.loads(request.messages[-1].content)
            assert 'history' not in payload
            supported = '急救包' not in payload['answer']
            output = {'supported': supported, 'unsupported_claims': [] if supported else ['急救包和氧气瓶']}
        else:
            repaired = any('没有来源支持' in message.content for message in request.messages)
            output = {'text': '你尚未查看应急柜，不能确认物资和数量。' if repaired and not self.always_bad else '应急柜里通常有急救包和氧气瓶。', 'source_event_ids': ['map']}
        return GenerationResult(final_text=json.dumps(output, ensure_ascii=False), finish_reason='stop')


@pytest.mark.asyncio
async def test_unsupported_draft_is_revised_by_model_and_only_final_answer_becomes_history(tmp_path):
    module = load_module(); state = initial_state(module, '来源')
    store = Store(tmp_path/'game.db'); store.create(state)
    cmd = PlayerCommand(client_request_id='qa', expected_revision=0, mode='ooc', text='柜内有什么？')
    turn = store.submit(state.campaign_id, cmd); lease = store.claim(turn)
    queue = SourceReviewQueue(); runner = AgentRunner(queue, store, turn, lease['epoch'])
    result = await runner.guide(guide_view(cmd.text, state, module))
    assert result.text == '你尚未查看应急柜，不能确认物资和数量。'
    assert [r.role_kind for r in queue.requests] == ['guide', 'guide_review', 'guide', 'guide_review']
    store.finish(turn, lease['epoch'], 'completed_ooc', result.text)
    history = store.table_talk(state.campaign_id)
    assert history[-1]['reply_source'] == 'model' and '急救包' not in history[-1]['answer']


@pytest.mark.asyncio
async def test_repeated_unsupported_claim_is_an_explicit_failure_not_a_published_guess():
    module = load_module(); state = initial_state(module, '来源')
    with pytest.raises(InvalidOutput, match='不支持'):
        await AgentRunner(SourceReviewQueue(True)).guide(guide_view('柜内有什么？', state, module))


@pytest.mark.asyncio
async def test_npc_review_receives_the_actual_speakers_identity():
    from local_trpg.schemas import GuideView
    class NpcReview:
        def count_tokens(self,messages):return 100
        async def generate(self,request):
            if request.role_kind=='guide_review':
                payload=json.loads(request.messages[-1].content)
                assert payload['speaker_name']=='科尔队长'
                output={'supported':True,'unsupported_claims':[]}
            else:output={'text':'我打算回镇报告，请你们护送幼龙。','source_event_ids':['identity']}
            return GenerationResult(final_text=json.dumps(output,ensure_ascii=False),finish_reason='stop')
    view=GuideView(question='谁回镇报告？',location='树林',tick=0,sources=[{'event_id':'identity','speaker_name':'科尔队长','text':'科尔队长打算返回镇子，请队伍护送幼龙。'}])
    result=await AgentRunner(NpcReview()).adventure_npc(view)
    assert '我打算回镇报告' in result.text
