"""Fixed-order trigger investigation; no model-generated oracle or hidden search."""
from .domain import digest
from .contrast_transforms import contrast_variant,validate_contrast
from .fixed import create_fixed
from .reviews import review_receipt
from .validity import check_receipt


def investigation_plan(case,profile,budget,*,factors=None,pairs=2,seed=100):
    factors=factors or ['rename','noise-position','insert-noise']
    if not 1<=len(factors)<=3 or len(set(factors))!=len(factors):raise ValueError('选择1—3个不同因素')
    if not 1<=pairs<=10 or not 0<=seed<2**63-len(factors)*pairs:raise ValueError('配对数或种子范围无效')
    planned=[]
    for f in factors:
        b=contrast_variant(case,f);r=validate_contrast(case,b,f)
        if r['status']!='rule_verified':raise ValueError('变换未通过规则：'+f)
        planned.append({'factor':f,'candidate':b,'receipt':r})
    calls=2*pairs*len(factors)
    if calls>budget.requests or calls*profile.max_new_tokens>budget.output_tokens:raise ValueError('预算不能容纳全部调查因素')
    return {'factors':factors,'pairs_per_factor':pairs,'seed':seed,'planned_observations':calls,
            'budget':budget.model_dump(),'transforms':planned,'purpose':'development_investigation',
            'policy':'fixed order, one factor per paired request; descriptive only; no adaptive hypothesis selection'}


def create_matrix(store,case,profile,budget,**kwargs):
    plan=investigation_plan(case,profile,budget,**kwargs);slots=[];cases={'original':case};metadata=[]
    for i,t in enumerate(plan['transforms']):
        f=t['factor'];cases[f]=t['candidate'];r=t['receipt'];rh=store.put_object('validity_receipt',digest(r),1,r)
        metadata.append({'factor':f,'receipt_hash':rh})
        for j in range(plan['pairs_per_factor']):
            s=plan['seed']+i*plan['pairs_per_factor']+j
            for arm in (('original','candidate') if j%2==0 else ('candidate','original')):
                slots.append({'case':'original' if arm=='original' else f,'profile':'shared','arm':arm,'seed':s,'pair_id':f+':'+str(s),'receipt_hash':rh})
    meta={k:v for k,v in plan.items() if k!='transforms'};meta['pairs']=metadata
    return create_fixed(store,kind='matrix',profiles={'shared':profile},cases=cases,slots=slots,budget=budget,metadata=meta)


def create_review_comparison(store,review_id,profile,budget,*,pairs=2):
    r=review_receipt(store,review_id);a,b=check_receipt(store,r)
    rh=store.put_object('validity_receipt',digest(r),1,r);slots=[]
    if not 1<=pairs<=24:raise ValueError('pairs 必须为1—24')
    for s in range(42,42+pairs):
        for arm in ('original','candidate'):slots.append({'case':arm,'profile':'shared','arm':arm,'seed':s,'pair_id':'human:'+str(s),'receipt_hash':rh})
    return create_fixed(store,kind='matrix',profiles={'shared':profile},cases={'original':a,'candidate':b},slots=slots,budget=budget,
                        metadata={'review_id':review_id,'pairs':[{'factor':'human','receipt_hash':rh}],'purpose':'development_reviewed_comparison'})
