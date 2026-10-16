"""Bounded actor-local recall, preserving original observations and provenance."""


def retrieve(state, actor_id, query='', limit=16):
    if limit <= 0:
        return []
    own = [(i, o) for i, o in enumerate(state.observations) if o.observer_id == actor_id]
    recent = sorted(own, key=lambda pair: (pair[1].observed_at, pair[0]), reverse=True)
    unique, seen = [], set()
    for pair in recent:
        o = pair[1]
        # Repeated transmission of one source is one memory. Independent events
        # without shared source IDs remain distinct, even with identical wording.
        source = tuple(o.source_observation_ids) or (o.event_id,)
        key = (o.source_actor_id, o.modality, o.channel, o.source_kind, source,
               tuple(o.evidence_refs), tuple(o.proposition_ids), o.content)
        if key not in seen:
            seen.add(key)
            unique.append(pair)

    chosen = {}

    def take(rows, count):
        for i, o in rows:
            if len(chosen) >= limit or count <= 0:
                break
            if i not in chosen:
                chosen[i] = o
                count -= 1

    # The current player utterance and recent performed actions must survive
    # a large archive of evidence. Self-heard speech is not an action result.
    take((pair for pair in unique if pair[1].source_actor_id == 'player'
          and pair[1].modality == 'heard'), 1)
    take((pair for pair in unique if pair[1].source_actor_id == actor_id
          and pair[1].modality == 'direct'), max(1, limit // 4))

    terms = query.split()
    ranked = sorted(unique, key=lambda pair: (
        sum(term in pair[1].content for term in terms), pair[1].observed_at, pair[0]), reverse=True)
    evidence = [pair for pair in ranked if pair[1].evidence_refs or pair[1].proposition_ids]
    covered, diverse = set(), []
    for pair in evidence:
        o = pair[1]
        refs = {('evidence', ref) for ref in o.evidence_refs} | {('proposition', ref) for ref in o.proposition_ids}
        if refs - covered:
            diverse.append(pair)
            covered.update(refs)
    # Different evidence IDs get a chance before repeated discussion of one ID.
    diverse_ids = {i for i, _ in diverse}
    take(diverse + [pair for pair in evidence if pair[0] not in diverse_ids], max(1, limit // 2))
    # Reserve fresh context even when a query strongly matches old records.
    take(unique, max(1, limit // 4))
    take(ranked, limit)
    return [o.model_dump() for _, o in sorted(chosen.items(), key=lambda pair: (pair[1].observed_at, pair[0]))]
def current_player_messages(observations, scene):
    """A game tick can contain many free conversations; it is not a turn ID."""
    current_ids = scene.get('current_player_observation_ids')
    return [o for o in observations if o.get('source_actor_id') == 'player' and o.get('modality') == 'heard'
            and (o['observation_id'] in current_ids if current_ids is not None else o.get('observed_at') == scene['tick'])]

