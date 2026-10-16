"""Small model wire format; the engine still consumes the original NpcDecision.

Only identity comes from trusted view data. No permissions or game effects are
inferred here. Long-form output remains supported by the existing parser.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import BeliefUpdate, Channel, NpcAction

NpcActionKind = NpcAction.model_fields['kind'].annotation


class CompactNpcDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    a: NpcActionKind
    m: str | None = None
    x: list[str] | None = None
    i: list[str] | None = None
    z: str | None = None
    t: str | None = Field(default=None, max_length=800)
    c: Channel | None = None
    r: list[str] | None = None
    d: list[str] | None = None
    q: list[str] | None = None
    b: list[BeliefUpdate] | None = Field(default=None, max_length=4)
    g: str | None = None

    @model_validator(mode='after')
    def speech_has_text(self):
        if self.t is None and any(getattr(self, key) is not None for key in ('c', 'r', 'd', 'q')):
            raise ValueError('紧凑台词的频道、收件人和引用必须与t文本一起提供')
        return self


def compact_npc_schema() -> dict[str, Any]:
    """Return a fresh, actor-independent JSON schema for compact NPC output."""
    return CompactNpcDecision.model_json_schema()


def expand_npc_wire(value: Any, actor_id: str | None) -> dict[str, Any]:
    """Validate compact keys, then expand to the unchanged strict game schema.

    The caller must validate the result as NpcDecision and apply existing game
    validation. Supplying long-form identity together with compact keys fails.
    """
    if not isinstance(actor_id, str) or not actor_id:
        raise ValueError('紧凑NPC输出缺少可信视图actor_id')
    wire = CompactNpcDecision.model_validate(value).model_dump(exclude_none=True)
    action = {'kind': wire['a']}
    for short, full in (('m', 'module_action_id'), ('x', 'target_ids'),
                        ('i', 'item_ids'), ('z', 'description')):
        if short in wire:
            action[full] = wire[short]
    expanded = {'actor_id': actor_id, 'action': action}
    if 't' in wire:
        speech = {'text': wire['t']}
        for short, full in (('c', 'channel'), ('r', 'recipient_ids'),
                            ('d', 'disclose_fact_ids'), ('q', 'claim_refs')):
            if short in wire:
                speech[full] = wire[short]
        expanded['utterance'] = speech
    for short, full in (('b', 'belief_updates'), ('g', 'next_goal_id')):
        if short in wire:
            expanded[full] = wire[short]
    return expanded
