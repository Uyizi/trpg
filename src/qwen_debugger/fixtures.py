"""24 synthetic cases from two declared template families, never real failures."""
from .domain import CaseImport
from .tasks import compile_case,solve_owner,solve_rules,solve_owner_choice


def fixture_cases(prefix='fixture'):
    cases=[]
    for i in range(12):
        entities=[{'id':'p'+str(j),'name':n} for j,n in enumerate(('林夏','顾川','许遥','周宁'))]
        length=1+i%5;start=i%4;holder=start;events=[]
        for j in range(length):
            next_=(holder+1+j%2)%4
            events.append({'kind':'transfer','from_id':'p'+str(holder),'to_id':'p'+str(next_)});holder=next_
        payload={'entities':entities,'object':{'id':'book','name':'笔记本'},'initial_owner_id':'p'+str(start),'events':events,
                 'independent_noise':[{'kind':'background','text':'窗外正在下雨。'}],
                 'query':{'kind':'final_owner','object_id':'book','answer_format':'entity_name_only'}}
        renderer='owner_transfer_zh_v1';solver=solve_owner
        if i>=8:
            payload['options']=['p0','p1','p2','p3'];payload['query']['answer_format']='option_letter';renderer='owner_choice_zh_v1';solver=solve_owner_choice
        cases.append(build(prefix+'_owner_'+str(i+1),'owner_fixture_template_v1','closed_world_owner_v1',renderer,payload,solver(payload)[1]))
    for i in range(12):
        predicates=[{'id':'p'+str(j),'name':n} for j,n in enumerate(('甲成立','乙成立','丙成立','丁成立','戊成立'))]
        p={'predicates':predicates,'facts':['p0','p3'] if i%2 else ['p0'],
           'rules':[{'if_all':['p0'],'then':'p1'},{'if_all':['p1','p3'],'then':'p2'}],
           'query':{'kind':'is_entailed','predicate':'p'+str(i%5),'answer_format':'yes_no_only'}}
        cases.append(build(prefix+'_rules_'+str(i+1),'rules_fixture_template_v1','scoped_rules_v1','scoped_rules_zh_v1',p,solve_rules(p)[1]))
    return cases


def build(id,family,task,renderer,payload,answer):
    return compile_case(CaseImport.model_validate({'schema_version':'1.0','case_id':id,'revision':1,'root_case_id':id,'family_id':family,
        'provenance':{'kind':'synthetic','generator':'debugger_fixture_v1'},'selected_on_outcome':False,'split':'development',
        'input':{'kind':'structured','task_type':task,'renderer_id':renderer,'payload':payload},
        'gold':{'kind':'exact_text','expected':answer,'normalizers':['strip_outer_whitespace'],'source':'deterministic_solver'}}))


def install_fixtures(store,prefix='fixture'):
    cases=fixture_cases(prefix)
    for c in cases:store.put_case(c)
    return {'cases':[c['case_id'] for c in cases],'count':len(cases),'families':2,'source':'synthetic, not observed model failures'}
