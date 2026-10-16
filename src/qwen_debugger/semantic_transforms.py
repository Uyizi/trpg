"""Registry for strictly scoped option, noise, unordered-fact and counterfactual changes."""
from .domain import CaseImport,digest
from .tasks import compile_case,solve_owner,solve_owner_choice,solve_rules,SAFE_NOISE
from .transforms import source_of

FACTORS=('rename','noise-position','permute-options','insert-noise','reorder-facts','flip-query')


def supported(case):
    if case['input']['kind']!='structured' or case['split']!='development':
        raise ValueError('变换只支持开发集的结构化任务；原始文本使用人工候选审核')
    if compile_case(CaseImport.model_validate(source_of(case)))!=case:raise ValueError('快照与重新求解不符')


def transformed_source(parent,factor):
    supported(parent);s=source_of(parent);p=s['input']['payload'];task=s['input']['task_type']
    if factor=='rename' and task=='closed_world_owner_v1':
        from .contrast_transforms import NAMES
        blocked={e['name'] for e in p['entities']}|{p['object']['name']}
        names=[n for n in NAMES if n not in blocked]
        if len(names)<len(p['entities']):raise ValueError('替换名字不足')
        for e,n in zip(p['entities'],names):e['name']=n
    elif factor=='permute-options' and s['input']['renderer_id']=='owner_choice_zh_v1':
        p['options']=p['options'][1:]+p['options'][:1]
    elif factor=='insert-noise' and task=='closed_world_owner_v1':
        old=p.get('independent_noise',[])
        remaining=[t for t in sorted(SAFE_NOISE) if t not in [n['text'] for n in old]]
        if not remaining:raise ValueError('已包含全部可认证背景句')
        p['independent_noise']=[*old,{'kind':'background','text':remaining[0]}]
    elif factor=='reorder-facts' and task=='scoped_rules_v1':
        if len(p['facts'])<2 or p['facts']==list(reversed(p['facts'])):raise ValueError('独立事实不足或重排无变化')
        p['facts']=list(reversed(p['facts']))
    elif factor=='flip-query' and task=='scoped_rules_v1':
        # Counterfactual query, not a semantics-preserving rewrite. Solve the new label.
        alternatives=[e['id'] for e in p['predicates'] if e['id']!=p['query']['predicate']]
        answer=solve_rules(p)[1];chosen=None
        for id in alternatives:
            q={**p,'query':{**p['query'],'predicate':id}}
            if solve_rules(q)[1]!=answer:chosen=id;break
        if chosen is None:raise ValueError('没有真值相反的已声明查询命题')
        p['query']['predicate']=chosen
    else:raise ValueError('此任务不支持该变换：'+str(factor))
    solver=solve_rules if task=='scoped_rules_v1' else (solve_owner_choice if s['input']['renderer_id']=='owner_choice_zh_v1' else solve_owner)
    s['gold']['expected']=solver(p)[1]
    return s


def make_variant(parent,factor):
    s=transformed_source(parent,factor)
    s.update(case_id='variant_'+digest({'parent':digest(parent),'factor':factor,'input':s['input']})[:24],revision=1,parent_hash=None,
             provenance={'kind':'derived','parent_case_hash':digest(parent),'transform':factor+'_v1'})
    return compile_case(CaseImport.model_validate(s))


def validate_variant(parent,child,factor):
    checks={};mapping={}
    try:
        supported(parent);supported(child);expected=transformed_source(parent,factor)
        checks['exact_declared_transform']=expected['input']==child['input'] and expected['gold']==child['gold']
        checks['lineage']=all(parent[k]==child[k] for k in ('root_case_id','family_id','split','selected_on_outcome'))
        if parent['input']['renderer_id']=='owner_choice_zh_v1':
            a=parent['input']['payload']['options'];b=child['input']['payload']['options']
            mapping={chr(65+i):chr(65+b.index(id)) for i,id in enumerate(a)}
        elif factor=='flip-query':mapping={'是':'否','否':'是'}
        else:mapping={v:v for v in parent['choices']}
        checks['answer_mapping']=mapping.get(parent['gold']['expected'])==child['gold']['expected']
    except (ValueError,KeyError,TypeError):checks['supported']=False
    return {'transform':factor+'_v1','validator':'structured_registry_v1','parent_hash':digest(parent),'child_hash':digest(child),
            'status':'rule_verified' if checks and all(checks.values()) else 'invalid','checks':checks,
            'label_mapping':mapping,'inverse_label_mapping':{v:k for k,v in mapping.items()},
            'scope':'counterfactual query with recomputed answer' if factor=='flip-query' else 'fixed structured transformation only'}
