"""Owner investigation: replay, safe shrink, confirmation reserve, fixed triggers."""
from .domain import Budget,digest
from .contrast_transforms import contrast_variant,validate_contrast
from .shrink import shrink_plan,create_investigation


def full_plan(case,profile,budget,*,factors=None,pairs=1,seed=100):
    factors=factors or ['rename','noise-position','insert-noise']
    if not 1<=len(factors)<=3 or len(set(factors))!=len(factors) or not 1<=pairs<=5:raise ValueError('调查支持1—3个因素，每因素1—5对')
    if not 0<=seed<2**63-pairs*len(factors):raise ValueError('种子范围无效')
    plan=shrink_plan(case,profile,budget);triggers=[]
    for i,f in enumerate(factors):
        child=contrast_variant(case,f);receipt=validate_contrast(case,child,f)
        if receipt['status']!='rule_verified':raise ValueError('无效触发变换')
        triggers.append({'factor':f,'candidate':child,'receipt':receipt,'seeds':list(range(seed+i*pairs,seed+(i+1)*pairs))})
    calls=2*pairs*len(factors);needed=13+calls
    if budget.requests<needed or budget.output_tokens<needed*profile.max_new_tokens:raise ValueError('预算不能容纳原题3次、预留确认10次与全部触发配对')
    plan.update(trigger_plans=triggers,trigger_requests=calls,minimum_calls_reserved=needed,
                confirmation_reserve={'requests':10+calls,'output_tokens':(10+calls)*profile.max_new_tokens},
                purpose='full_development_investigation',trigger_policy='original case; fixed factors and seeds; no outcome-based selection')
    return plan


def create_workflow(store,case,profile,budget,**options):
    plan=full_plan(case,profile,budget,**options)
    return create_investigation(store,case,profile,budget,workflow_plan=plan)
