import json
from typing import get_args
from pydantic import BaseModel
from .base import extract_final
class InvalidOutput(ValueError):pass

def parse_result(result,schema,*,actor_id=None):
 if result.error:raise InvalidOutput(result.error)
 if result.finish_reason!='stop':raise InvalidOutput('输出未正常结束：'+result.finish_reason)
 text=extract_final(result.final_text)
 if text.startswith('```') and text.endswith('```'):
  lines=text.splitlines();text='\n'.join(lines[1:-1])
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('重复JSON字段')
   d[k]=v
  return d
 def optional_nulls(value,model):
  if not isinstance(value,dict):return value
  value=dict(value)
  for name,field in model.model_fields.items():
   if name not in value:continue
   # In model proposals only, an explicit null for a defaulted field means
   # omitted. Required identity, unknown keys and enums still validate strictly.
   if value[name] is None and not field.is_required():
    value.pop(name);continue
   candidates=(field.annotation,*get_args(field.annotation))
   nested=next((t for t in candidates if isinstance(t,type) and issubclass(t,BaseModel)),None)
   if nested:value[name]=optional_nulls(value[name],nested)
  return value
 try:
  value=json.loads(text,object_pairs_hook=pairs)
  if schema.__name__=='NpcDecision' and isinstance(value,dict) and 'a' in value:
   from .compact_npc import expand_npc_wire
   value=expand_npc_wire(value,actor_id)
  # This runner sometimes nests speech under action. Normalize that single
  # unambiguous wire variant, then apply the unchanged strict game schema.
  if schema.__name__=='NpcDecision' and isinstance(value,dict) and isinstance(value.get('action'),dict) and 'utterance' in value['action']:
   if 'utterance' in value:raise ValueError('重复或歧义的utterance')
   value['utterance']=value['action'].pop('utterance')
  return schema.model_validate(optional_nulls(value,schema))
 except (ValueError,TypeError) as exc:raise InvalidOutput('输出不符合结构：'+str(exc)[:180]) from exc
