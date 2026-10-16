def select_npcs(state,events,limit=2,module=None,targets=None):
 order=(module or {}).get('world',{}).get('initiative_order',['guard','doctor','mechanic','comms','captain'])
 event_ids={e.event_id for e in events}
 touched={o.observer_id for o in state.observations if o.event_id in event_ids and o.observer_id!='player'}
 direct=[a for a in (targets or []) if a in order]
 eligible=touched|set(direct)|set(state.pending_npcs)
 eligible={a for a in eligible if a in state.actors and state.actors[a].health!='incapacitated' and state.actors[a].outcome=='present'}
 return sorted(eligible,key=lambda a:(a not in direct,-state.pending_npcs.get(a,0),order.index(a)))[:min(2,limit)]
