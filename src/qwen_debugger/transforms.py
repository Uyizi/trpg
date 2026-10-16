"""Certified deletion on a narrow closed-world task, never arbitrary prose."""
import copy
from .domain import CaseImport,digest
from .tasks import compile_case


def source_of(case):
    return {key:copy.deepcopy(case[key]) for key in CaseImport.model_fields if key in case}


def require_supported(case):
    if case['input']['kind']!='structured' or case['input'].get('task_type')!='closed_world_owner_v1':
        raise ValueError('自动缩减当前仅支持结构化实体转移题；普通文本需要人工语义审核')
    if case['split']!='development':
        raise ValueError('留出案例不能直接用于搜索；请明确建立开发案例')
    if compile_case(CaseImport.model_validate(source_of(case)))!=case:
        raise ValueError('案例不是有效的已编译快照')


def deletion_candidates(parent):
    require_supported(parent)
    noise=parent['input']['payload'].get('independent_noise',[])
    # Paragraph-level deletion first, then individual approved sentences.
    groups=[tuple(range(len(noise)))]+[(i,) for i in range(len(noise))]
    seen=set()
    for removed in groups:
        if not removed or removed in seen: continue
        seen.add(removed)
        source=source_of(parent)
        source['input']['payload']['independent_noise']=[n for i,n in enumerate(noise) if i not in removed]
        source.update(case_id='shrink_'+digest({'parent':digest(parent),'removed':removed})[:24],
                      revision=1,parent_hash=None,selected_on_outcome=True,
                      provenance={'kind':'derived','parent_case_hash':digest(parent),'transform':'delete_safe_background_v1',
                                  'removed_indices':list(removed)})
        yield compile_case(CaseImport.model_validate(source))


def validate_deletion(parent,child):
    checks={}
    try:
        require_supported(parent); require_supported(child)
        checks['fresh_compile']=True
        for key in ('root_case_id','family_id','split','gold','renderer'):
            checks[key]=parent[key]==child[key]
        before=copy.deepcopy(parent['input']); after=copy.deepcopy(child['input'])
        old=before['payload'].pop('independent_noise',[]); new=after['payload'].pop('independent_noise',[])
        checks['all_other_ir_unchanged']=before==after
        cursor=iter(old)
        checks['ordered_deletion']=len(new)<len(old) and all(any(x==entry for x in cursor) for entry in new)
        checks['solver_answer_unchanged']=parent['solver']['answer']==child['solver']['answer']
    except (ValueError,KeyError,TypeError): checks['fresh_compile']=False
    return {'transform':'delete_safe_background_v1','validator':'owner_noise_preservation_v1',
            'parent_hash':digest(parent),'child_hash':digest(child),
            'status':'rule_verified' if checks and all(checks.values()) else 'invalid',
            'checks':checks,'label_mapping':'identity','inverse':'not_invertible_without_parent',
            'scope':'only fixed approved background sentences; no event, query, or instruction deletion'}
