"""Validation shared by live writes and archive restoration."""
import json
from typing import Literal
from pydantic import Field
from .domain import Strict,Name,digest,CaseImport


class SuiteSnapshot(Strict):
    id: Name
    revision: int = Field(ge=1)
    parent_hash: str | None
    purpose: Literal['development','heldout']
    created: str
    case_hashes: list[str] = Field(min_length=1,max_length=32)
    family_ids: list[str]
    root_case_ids: list[str]
    grouping_evidence: str
    source_experiment: str | None


def pool_from_rows(rows):
    return {r['hash']:{'kind':r['kind'],'id':r['id'],'revision':r['revision'],'value':json.loads(r['body'])} for r in rows}


def validate_case(c,pool):
    from .tasks import compile_case
    source={k:c[k] for k in CaseImport.model_fields if k in c}
    if compile_case(CaseImport.model_validate(source))!=c:raise ValueError('案例不是有效求解快照')
    if c['revision']==1:
        if c.get('parent_hash'):raise ValueError('首修订不能引用父修订')
    else:
        parent=pool.get(c.get('parent_hash'))
        if not parent or parent['kind']!='case':raise ValueError('案例父修订缺失')
        old=parent['value']
        if (old['case_id'],old['revision']+1,old['root_case_id'],old['family_id'])!=(c['case_id'],c['revision'],c['root_case_id'],c['family_id']):raise ValueError('父修订谱系不匹配')
    for r in pool.values():
        if r['kind']=='case':
            old=r['value']
            if old['root_case_id']==c['root_case_id'] and old['family_id']!=c['family_id']:
                raise ValueError('同一 root_case_id 必须保持同一 family_id')


def development_usage(pool):
    hashes={h for h,r in pool.items() if r['kind']=='case' and r['value']['split']=='development'}
    for r in pool.values():
        v=r['value']
        if r['kind']=='suite' and v.get('purpose')=='development':hashes.update(v.get('case_hashes',[]))
        elif r['kind']=='manifest' and 'development' in str(v.get('purpose','')):
            if v.get('case_hash'):hashes.add(v['case_hash'])
            hashes.update(s['case_hash'] for s in v.get('schedule',[]) if s.get('case_hash'))
    cases=[pool[h]['value'] for h in hashes if h in pool and pool[h]['kind']=='case']
    return {c['root_case_id'] for c in cases},{c['family_id'] for c in cases}


def validate_suite(suite,pool,*,check_exposure=True):
    SuiteSnapshot.model_validate(suite)
    if len(set(suite['case_hashes']))!=len(suite['case_hashes']):raise ValueError('套件不能包含重复案例快照')
    cases=[]
    for h in suite['case_hashes']:
        if h not in pool or pool[h]['kind']!='case':raise ValueError('套件引用的案例不存在')
        c=pool[h]['value'];validate_case(c,pool);cases.append(c)
    if suite['family_ids']!=sorted({c['family_id'] for c in cases}) or suite['root_case_ids']!=sorted({c['root_case_id'] for c in cases}):
        raise ValueError('套件根案例/家族清单与实际案例不符')
    splits={}
    for c in cases:splits.setdefault(c['family_id'],set()).add(c['split'])
    if any(len(s)>1 for s in splits.values()):raise ValueError('同家族不可跨开发与留出组')
    if suite['revision']==1:
        if suite['parent_hash'] is not None:raise ValueError('套件首修订不能有父修订')
    else:
        old=pool.get(suite['parent_hash'])
        if not old or old['kind']!='suite' or (old['value']['id'],old['value']['revision']+1)!=(suite['id'],suite['revision']):raise ValueError('套件父修订谱系缺失或不匹配')
    if check_exposure and suite['purpose']=='heldout':
        if any(c['split']!='heldout' or c['selected_on_outcome'] for c in cases):raise ValueError('独立评估不得含开发或按结果挑选案例')
        roots,families=development_usage(pool)
        if roots.intersection(suite['root_case_ids']) or families.intersection(suite['family_ids']):
            raise ValueError('根案例或家族已经用于开发，不能作为独立评估')
    return cases
