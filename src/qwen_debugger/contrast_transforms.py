"""One-factor, closed-world transformations with auditable label mappings."""
import copy
from .domain import CaseImport, digest
from .tasks import compile_case, solve_owner
from .transforms import require_supported, source_of

RENDERERS=('owner_transfer_zh_v1','owner_transfer_zh_noise_first_v1')
NAMES=('周宁','陈舟','沈岚','苏禾','方乔','江月','唐星','陆远','白榆','叶青','夏言','宋安')


def contrast_variant(parent,factor):
    if factor not in ('rename','noise-position') or parent['input'].get('renderer_id')=='owner_choice_zh_v1' or parent['input'].get('task_type')!='closed_world_owner_v1':
        from .semantic_transforms import make_variant
        return make_variant(parent,factor)
    require_supported(parent)
    source=source_of(parent);payload=source['input']['payload']
    if factor=='rename':
        # IDs and all edges stay fixed; displayed labels are a bijection.
        forbidden={e['name'] for e in payload['entities']}|{payload['object']['name']}
        names=[name for name in NAMES if name not in forbidden]
        if len(names)<len(payload['entities']): raise ValueError('可用替换名字不足；当前只使用固定名字表')
        for entity,name in zip(payload['entities'],names): entity['name']=name
        source['gold']['expected']=solve_owner(payload)[1]
    elif factor=='noise-position':
        if not payload.get('independent_noise'): raise ValueError('没有独立背景句，无法进行位置对照')
        source['input']['renderer_id']=RENDERERS[1-RENDERERS.index(source['input']['renderer_id'])]
    else: raise ValueError('不支持的对照因素：'+str(factor))
    source.update(case_id='contrast_'+digest({'parent':digest(parent),'factor':factor,'input':source['input']})[:24],
                  revision=1,parent_hash=None,
                  provenance={'kind':'derived','parent_case_hash':digest(parent),'transform':factor+'_v1'})
    child=compile_case(CaseImport.model_validate(source))
    if validate_contrast(parent,child,factor)['status']!='rule_verified': raise ValueError('变换未通过题意保持规则')
    return child


def validate_contrast(parent,child,factor):
    if factor not in ('rename','noise-position') or parent['input'].get('renderer_id')=='owner_choice_zh_v1' or parent['input'].get('task_type')!='closed_world_owner_v1':
        from .semantic_transforms import validate_variant
        return validate_variant(parent,child,factor)
    checks={};mapping={}
    try:
        require_supported(parent);require_supported(child)
        checks['fresh_compile']=True
        for key in ('root_case_id','family_id','split','selected_on_outcome'):
            checks[key]=parent[key]==child[key]
        before=copy.deepcopy(parent['input']);after=copy.deepcopy(child['input'])
        old=before['payload']['entities'];new=after['payload']['entities']
        checks['entity_ids_unchanged']=[e['id'] for e in old]==[e['id'] for e in new]
        mapping={a['name']:b['name'] for a,b in zip(old,new)}
        checks['bijective_labels']=len(mapping)==len(old)==len(new)==len(set(mapping.values()))
        if factor=='rename':
            checks['all_names_changed']=all(a['name']!=b['name'] for a,b in zip(old,new))
            before['payload']['entities']=[e['id'] for e in old]
            after['payload']['entities']=[e['id'] for e in new]
            checks['only_names_changed']=before==after
        elif factor=='noise-position':
            checks['different_known_renderers']=(before['renderer_id'] in RENDERERS and after['renderer_id'] in RENDERERS
                                                and before['renderer_id']!=after['renderer_id'])
            before.pop('renderer_id');after.pop('renderer_id')
            checks['identical_task_ir']=before==after
            checks['has_noise']=bool(before['payload'].get('independent_noise'))
            x=parent['messages'][0]['content'].splitlines();y=child['messages'][0]['content'].splitlines()
            checks['same_lines_new_position']=sorted(x)==sorted(y) and x!=y and x[0]==y[0] and x[-1]==y[-1]
        else: checks['supported_factor']=False
        checks['same_entity_trace']=parent['solver']['trace']==child['solver']['trace']
        expected=mapping.get(parent['gold']['expected'])
        checks['mapped_answer']=expected==child['gold']['expected']==child['solver']['answer']
        checks['same_scoring_rule']={**parent['gold'],'expected':expected}==child['gold']
    except (ValueError,KeyError,TypeError,IndexError): checks['fresh_compile']=False
    return {'transform':factor+'_v1','validator':'owner_single_factor_v1','parent_hash':digest(parent),
            'child_hash':digest(child),'status':'rule_verified' if checks and all(checks.values()) else 'invalid',
            'checks':checks,'label_mapping':mapping,'inverse_label_mapping':{v:k for k,v in mapping.items()},
            'scope':'closed-world owner task only; does not certify general prose or equal token lengths'}
