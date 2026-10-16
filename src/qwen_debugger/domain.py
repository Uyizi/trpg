"""Strict input contracts. Persist canonical snapshots, never mutable model objects."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, allow_inf_nan=False)


Name = Annotated[str, Field(min_length=1, max_length=120, pattern=r'^[\w.-]+$')]


class Message(Strict):
    role: Literal['system', 'user', 'assistant']
    content: str


class RawInput(Strict):
    kind: Literal['raw']
    messages: list[Message] = Field(min_length=1)

    @model_validator(mode='after')
    def final_user(self):
        if self.messages[-1].role != 'user':
            raise ValueError('最后一条消息必须是 user；历史回答可放在此前')
        return self


class StructuredInput(Strict):
    kind: Literal['structured']
    task_type: Literal['closed_world_owner_v1', 'scoped_rules_v1']
    renderer_id: Literal['owner_transfer_zh_v1', 'owner_transfer_zh_noise_first_v1', 'owner_choice_zh_v1', 'scoped_rules_zh_v1']
    payload: dict


class Gold(Strict):
    kind: Literal['exact_text','json_fields','numeric'] = 'exact_text'
    expected: str = Field(min_length=1)
    normalizers: list[Literal['strip_outer_whitespace']] = Field(default_factory=list)
    source: Literal['deterministic_solver', 'user_provided', 'unknown']
    absolute_tolerance: str | None = None

    @model_validator(mode='after')
    def explicit_rule(self):
        from .scoring import fields,number
        if self.kind=='numeric':
            number(self.expected)
            if self.absolute_tolerance is None or number(self.absolute_tolerance)<0:
                raise ValueError('numeric 必须显式声明非负 absolute_tolerance')
        elif self.absolute_tolerance is not None:
            raise ValueError('只有 numeric 可以声明容差')
        if self.kind=='json_fields':fields(self.expected)
        return self

    @model_serializer(mode='wrap')
    def snapshot(self,handler):
        data=handler(self)
        if self.kind!='numeric':data.pop('absolute_tolerance',None)
        return data


class CaseImport(Strict):
    schema_version: Literal['1.0'] = '1.0'
    case_id: Name
    revision: int = Field(ge=1)
    root_case_id: Name
    family_id: Name
    provenance: dict
    selected_on_outcome: bool
    split: Literal['development', 'heldout'] = 'development'
    parent_hash: str | None = None
    input: Annotated[Union[RawInput, StructuredInput], Field(discriminator='kind')]
    gold: Gold


class RuntimeProfile(Strict):
    id: Name
    backend: Literal['local', 'mock', 'api']
    revision: int = Field(default=1, ge=1)
    model_path: str | None = None
    context_length: int = Field(default=65536, ge=128)
    max_input_tokens: int = Field(default=4096, ge=1)
    max_new_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0, le=5)
    thinking: bool = False
    kv_cache: Literal['q4', 'q8', 'bf16'] = 'q8'
    fast_kernels: bool = True
    prefill_chunk_size: int = Field(default=1024, ge=1)
    timeout_seconds: int = Field(default=120, ge=1, le=1800)
    mock_text: str = '许遥'
    mock_finish: Literal['stop', 'length', 'unknown', 'error', 'timeout'] = 'stop'
    mock_usage_unknown: bool = False
    api_base: str | None = None
    api_model: str | None = None
    api_key_env: str | None = None

    @model_serializer(mode='wrap')
    def snapshot(self,handler):
        data=handler(self)
        # Preserve existing non-API snapshots and their hashes.
        if self.backend!='api':
            for key in ('api_base','api_model','api_key_env'):data.pop(key,None)
        return data

    @model_validator(mode='after')
    def budget_and_backend(self):
        if self.max_new_tokens >= self.context_length:
            raise ValueError('输出预算必须小于上下文窗口')
        if self.backend == 'local' and (self.mock_text != '许遥' or self.mock_finish != 'stop' or self.mock_usage_unknown):
            raise ValueError('local profile 不能配置 mock 行为')
        if self.backend=='api':
            from urllib.parse import urlsplit
            import re
            u=urlsplit(self.api_base or '')
            if u.scheme not in ('http','https') or u.hostname not in ('localhost','127.0.0.1','::1') or u.username or u.password or u.query or u.fragment:
                raise ValueError('API 地址必须是本机 loopback HTTP(S)，不得含认证值或查询参数')
            if not self.api_model or not self.api_model.strip():raise ValueError('需明确 API 模型名称')
            if self.api_key_env and not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',self.api_key_env):raise ValueError('api_key_env 必须为环境变量名称')
        elif any(v is not None for v in (self.api_base,self.api_model,self.api_key_env)):
            raise ValueError('非 API profile 不能设置 API 参数')
        return self


class Budget(Strict):
    requests: int = Field(default=3, ge=1, le=64)
    output_tokens: int = Field(default=3072, ge=1)
    seconds: int = Field(default=600, ge=1, le=1800)
