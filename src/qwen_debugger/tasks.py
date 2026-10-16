"""Small closed-world solvers; the generating model never supplies the oracle."""
from typing import Literal
from pydantic import Field
from .domain import Strict, CaseImport, digest

SAFE_NOISE = frozenset(['房间里有一张圆桌。', '窗外正在下雨。'])


class Entity(Strict):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=20, pattern=r'^[\u4e00-\u9fffA-Za-z0-9]+$')


class Event(Strict):
    kind: Literal['transfer']
    from_id: str
    to_id: str


class Noise(Strict):
    kind: Literal['background']
    text: str


class OwnerQuery(Strict):
    kind: Literal['final_owner']
    object_id: str
    answer_format: Literal['entity_name_only','option_letter']


class OwnerTask(Strict):
    entities: list[Entity] = Field(min_length=2)
    object: Entity
    initial_owner_id: str
    events: list[Event]
    independent_noise: list[Noise] = Field(default_factory=list)
    query: OwnerQuery
    options: list[str] | None = None


class Rule(Strict):
    if_all: list[str] = Field(min_length=1)
    then: str


class RuleQuery(Strict):
    kind: Literal['is_entailed']
    predicate: str
    answer_format: Literal['yes_no_only']


class RulesTask(Strict):
    predicates: list[Entity] = Field(min_length=1)
    facts: list[str]
    rules: list[Rule]
    query: RuleQuery


def unique_entities(entities):
    names = {e.id:e.name for e in entities}
    if len(names)!=len(entities) or len(set(names.values()))!=len(entities):
        raise ValueError('实体 ID 和名称必须分别唯一')
    return names


def solve_owner(payload, *, noise_first=False):
    ir=OwnerTask.model_validate(payload)
    if ir.query.answer_format!='entity_name_only' or ir.options is not None:
        raise ValueError('姓名 renderer 不接受选项；请使用选择题 renderer')
    names=unique_entities(ir.entities)
    if ir.query.object_id!=ir.object.id or ir.initial_owner_id not in names:
        raise ValueError('目标对象或初始持有者无效')
    if ir.object.name in names.values():
        raise ValueError('对象名不能与人物名相同')
    for noise in ir.independent_noise:
        if noise.text not in SAFE_NOISE:
            raise ValueError('背景句不在固定审核词表中，不能认证为独立噪声')
    holder=ir.initial_owner_id
    lines=['仅以以下事件为准，按列出顺序发生，未描述其他转移。',
           f'最初，{names[holder]}持有{ir.object.name}。']
    trace=[holder]
    for event in ir.events:
        if event.from_id!=holder or event.to_id not in names or event.to_id==holder:
            raise ValueError('非法转移链')
        lines.append(f'{names[event.from_id]}把{ir.object.name}交给{names[event.to_id]}。')
        holder=event.to_id; trace.append(holder)
    if noise_first:
        # Move only the approved background block; preserve all event ordering.
        lines[1:1]=[n.text for n in ir.independent_noise]
    else:
        lines.extend(n.text for n in ir.independent_noise)
    lines.append(f'最后谁持有{ir.object.name}？只输出姓名。')
    return '\n'.join(lines), names[holder], list(names.values()), {'version':'owner_v1','trace':trace}


def solve_owner_choice(payload):
    import copy
    ir=OwnerTask.model_validate(payload)
    names=unique_entities(ir.entities)
    if ir.query.answer_format!='option_letter' or not ir.options or len(ir.options)>26 or set(ir.options)!=set(names) or len(ir.options)!=len(names):
        raise ValueError('选择题必须以每个实体恰好一次组成全部选项')
    base=copy.deepcopy(payload);base.pop('options',None);base['query']['answer_format']='entity_name_only'
    prompt,_,_,evidence=solve_owner(base)
    labels=[chr(65+i) for i in range(len(ir.options))]
    lines=prompt.splitlines()[:-1]+[f'{label}. {names[id]}' for label,id in zip(labels,ir.options)]
    lines.append(f'最后谁持有{ir.object.name}？只输出选项字母。')
    answer=labels[ir.options.index(evidence['trace'][-1])]
    return '\n'.join(lines),answer,labels,{**evidence,'version':'owner_choice_v1'}


def solve_rules(payload):
    ir=RulesTask.model_validate(payload); names=unique_entities(ir.predicates)
    mentioned=set(ir.facts)|{ir.query.predicate}
    for rule in ir.rules: mentioned.update(rule.if_all); mentioned.add(rule.then)
    if not mentioned<=names.keys(): raise ValueError('规则引用未知命题')
    known=set(ir.facts)
    # 有限正向规则，到不动点；无外部常识、否定或开放世界推断。
    while True:
        nxt=known|{r.then for r in ir.rules if set(r.if_all)<=known}
        if nxt==known: break
        known=nxt
    lines=['只使用以下事实和规则。能推导出查询命题则回答“是”，否则回答“否”。',
           '已知事实：'+('、'.join(names[f] for f in ir.facts) or '无')+'。']
    for rule in ir.rules:
        lines.append('如果'+ '且'.join(names[p] for p in rule.if_all)+'，那么'+names[rule.then]+'。')
    lines.append(f'能推导出“{names[ir.query.predicate]}”吗？只输出是或否。')
    return '\n'.join(lines), '是' if ir.query.predicate in known else '否', ['是','否'], {'version':'rules_v1','closure':sorted(known)}


def compile_case(imported: CaseImport):
    c=imported.model_dump(mode='json')
    if imported.input.kind=='raw':
        if imported.gold.source=='deterministic_solver':
            raise ValueError('原始文本不能自称经过确定性求解器验证')
        c.update(messages=c['input']['messages'], renderer=None, solver=None, choices=[], allowed_transforms=[])
    else:
        task=imported.input
        if imported.gold.kind!='exact_text':raise ValueError('当前结构化求解器只支持 exact_text 标签')
        registry={'closed_world_owner_v1':('owner_transfer_zh_v1',solve_owner),
                  'scoped_rules_v1':('scoped_rules_zh_v1',solve_rules)}
        renderer,solve=registry[task.task_type]
        noise_first=task.task_type=='closed_world_owner_v1' and task.renderer_id=='owner_transfer_zh_noise_first_v1'
        if task.task_type=='closed_world_owner_v1' and task.renderer_id=='owner_choice_zh_v1':
            renderer=task.renderer_id
            prompt,answer,choices,evidence=solve_owner_choice(task.payload)
        elif noise_first:
            renderer=task.renderer_id
            prompt,answer,choices,evidence=solve_owner(task.payload,noise_first=True)
        else:
            if task.renderer_id!=renderer: raise ValueError('renderer 与任务类型不匹配')
            prompt,answer,choices,evidence=solve(task.payload)
        if imported.gold.expected!=answer: raise ValueError('导入 gold 与实际求解结果不一致')
        c.update(messages=[{'role':'user','content':prompt}], renderer={'id':renderer,'version':'1'},
                 solver={**evidence,'answer':answer}, choices=choices, allowed_transforms=[])
    c['messages_hash']=digest(c['messages'])
    c['content_hash']=digest(c)
    return c


def grade(case, result):
    kind=case['gold']['kind']
    g={'oracle':oracle_id(case),'extractor':'exact_v1' if kind=='exact_text' else kind+'_v1','outcome':'INCONCLUSIVE',
       'failure_kind':None,'evaluable':False,'correct':None,'gold_source':case['gold']['source']}
    finish=result.get('finish_reason')
    if finish=='length': return {**g,'outcome':'TRUNCATED'}
    if finish!='stop': return g
    if case['gold']['source']=='unknown': return g
    text=result['text']
    if kind!='exact_text':
        from .scoring import score
        correct,failure=score(case['gold'],text)
        return {**g,'outcome':'PASS' if correct else 'FAIL','evaluable':True,'correct':correct,'failure_kind':failure}
    if 'strip_outer_whitespace' in case['gold']['normalizers']: text=text.strip()
    expected=case['gold']['expected']
    if text==expected: return {**g,'outcome':'PASS','evaluable':True,'correct':True}
    choices=case['choices']
    if text in choices: kind='semantic_mismatch'
    elif text in ('答案是'+expected, '答案是'+expected+'。', expected+'。'):
        kind='required_format_missing'
    else: kind='unclassified_response' if choices else 'exact_text_mismatch'
    return {**g,'outcome':'FAIL','failure_kind':kind,'evaluable':True,'correct':False}


def oracle_id(case):
    return {'exact_text':'closed_exact_v1','json_fields':'closed_json_fields_v1','numeric':'closed_numeric_v1'}[case['gold']['kind']]
