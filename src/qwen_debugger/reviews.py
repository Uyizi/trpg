"""User review is a recorded action on exact hashes; never inferred from answers."""
from .domain import digest,utcnow,CaseImport
from .tasks import compile_case
from .transforms import source_of
from .runner import uid


def propose_review(store,parent,candidate,note=''):
    for c in (parent,candidate):
        if compile_case(CaseImport.model_validate(source_of(c)))!=c:raise ValueError('无效案例快照')
    if parent['split']!='development' or candidate['split']!='development':raise ValueError('人工变换只用于开发案例')
    if any(parent[k]!=candidate[k] for k in ('root_case_id','family_id')):raise ValueError('候选必须保留根案例与 family')
    if digest(parent)==digest(candidate):raise ValueError('候选与原题相同')
    id=uid('review');p=store.put_case(parent);c=store.put_case(candidate)
    store.put_object('review_proposal',id,1,{'id':id,'parent_hash':p,'candidate_hash':c,'note':note,
                                          'created':utcnow(),'status':'unknown'})
    return id


def proposal(store,id):
    rows=[r for r in store.rows('objects') if r['kind']=='review_proposal' and r['id']==id]
    if not rows:raise ValueError('审核候选不存在')
    return store.object(rows[0]['hash'])


def decide_review(store,id,decision,note,actor,*,parent_hash,candidate_hash):
    p=proposal(store,id)
    if decision not in ('accept','reject'):raise ValueError('审核决定必须 accept 或 reject')
    if not note.strip() or not actor.strip():raise ValueError('需填写审核人和判断理由')
    if p['parent_hash']!=parent_hash or p['candidate_hash']!=candidate_hash:raise ValueError('审核对象已变化或散列不匹配')
    if any(r['kind']=='review_decision' and r['id']==id for r in store.rows('objects')):raise ValueError('审核已完成；修改候选应新建审核')
    return store.put_object('review_decision',id,1,{'id':id,'parent_hash':parent_hash,'candidate_hash':candidate_hash,
        'decision':decision,'note':note,'actor':actor,'created':utcnow(),'source':'explicit_local_user_action'})


def review_receipt(store,id):
    p=proposal(store,id);rows=[r for r in store.rows('objects') if r['kind']=='review_decision' and r['id']==id]
    d=store.object(rows[0]['hash']) if rows else None
    return {'review_id':id,'parent_hash':p['parent_hash'],'child_hash':p['candidate_hash'],
            'status':('human_confirmed' if d['decision']=='accept' else 'invalid') if d else 'unknown',
            'review':d,'scope':'only exact reviewed pair; human judgement, not an automatic semantic proof'}
