"""Explicit, bounded scalar/field scoring; never infer a tolerance from prose."""
import json
import re
from decimal import Decimal, localcontext


def number(text):
    if not isinstance(text,str) or len(text)>256:
        raise ValueError('数值必须是长度不超过256的十进制文本')
    text=text.strip()
    if not re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?',text):
        raise ValueError('只接受有限十进制数值，不接受解释文字或单位')
    n=Decimal(text)
    if not n.is_finite() or abs(n.as_tuple().exponent)>256:
        raise ValueError('数值精度/指数超出支持范围')
    return n


def fields(text):
    if len(text)>65536:raise ValueError('JSON 对象超过64 KiB字符上限')
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('JSON 字段重复')
            out[k]=v
        return out
    def constant(value):raise ValueError('JSON 不接受非有限数值')
    try:
        obj=json.loads(text,object_pairs_hook=pairs,parse_float=number,parse_int=number,parse_constant=constant)
    except (json.JSONDecodeError,RecursionError) as exc:
        raise ValueError('需要单个 JSON 对象') from exc
    if not isinstance(obj,dict) or not 1<=len(obj)<=64:
        raise ValueError('需要1—64个字段的 JSON 对象')
    if any(type(v) not in (str,bool,Decimal,type(None)) for v in obj.values()):
        raise ValueError('字段仅支持字符串、布尔、有限数值或 null')
    return obj


def score(gold,text):
    try:
        if gold['kind']=='numeric':
            actual,expected,tolerance=number(text),number(gold['expected']),number(gold['absolute_tolerance'])
            # Enough precision for the explicitly bounded decimal inputs.
            with localcontext() as ctx:
                ctx.prec=1024
                correct=abs(actual-expected)<=tolerance
        else:
            actual,expected=fields(text),fields(gold['expected'])
            if actual.keys()!=expected.keys():return False,'required_format_missing'
            correct=all(type(actual[k]) is type(v) and actual[k]==v for k,v in expected.items())
    except ValueError:
        return False,'required_format_missing'
    return correct,None if correct else 'semantic_mismatch'
