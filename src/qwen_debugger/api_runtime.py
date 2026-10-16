"""Narrow loopback OpenAI-compatible adapter; no proxy, redirects or hidden retries."""
import json
import os
import socket
import time
import subprocess
import sys
import urllib.error
import urllib.request
from .domain import digest
from .runtime import NotSent,UnknownCompletion,ResponseError


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


class ApiRuntime:
    def __init__(self,profile):
        self.profile=profile
        self.proc=None
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    def prepare(self,case,seed,heartbeat=None):
        payload={'model':self.profile.api_model,'messages':case['messages'],'temperature':self.profile.temperature,
                 'max_tokens':self.profile.max_new_tokens,'seed':seed,'stream':False,
                 'chat_template_kwargs':{'enable_thinking':self.profile.thinking}}
        return {'payload':payload,'fingerprint':digest(payload),'rendered_prompt':None,'input_ids':None,'input_tokens':None,
                'endpoint':self.profile.api_base.rstrip('/')+'/chat/completions','identity':'service-declared, not verified weights',
                'count_source':'unobservable before dispatch','limits':'input/context limits cannot be verified client-side without the server tokenizer',
                'capabilities':{'seed':'requested, use unverified','thinking':'requested, template unverified','cache':'unknown'}}
    def generate(self,request,heartbeat=None):
        deadline=min(getattr(self,'deadline',float('inf')),time.monotonic()+self.profile.timeout_seconds)
        if getattr(self,'cancelled',lambda:False)() or time.monotonic()>=deadline:raise NotSent('请求前取消或到达预算')
        self.proc=subprocess.Popen([sys.executable,'-m','qwen_debugger.api_worker'],stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                   env={**os.environ,'QDEBUG_PARENT_PID':str(os.getpid())})
        wire=json.dumps({'profile':self.profile.model_dump(mode='json'),'request':request},ensure_ascii=False).encode()
        try:
            while True:
                if getattr(self,'cancelled',lambda:False)() or time.monotonic()>=deadline:
                    raise UnknownCompletion('API 达到绝对期限或被取消；完成状态未知，不自动重发')
                if heartbeat:heartbeat()
                try:
                    out,_=self.proc.communicate(input=wire,timeout=min(.1,max(.001,deadline-time.monotonic())))
                    break
                except subprocess.TimeoutExpired:wire=None
            if time.monotonic()>=deadline:raise UnknownCompletion('API 响应在绝对期限后收到')
            try:envelope=json.loads(out)
            except (ValueError,UnicodeError):raise UnknownCompletion('API 请求进程未返回完整证据') from None
            if envelope['kind']=='result':return envelope['result']
            if envelope['kind']=='not_sent':raise NotSent(envelope['message'])
            if envelope['kind']=='response_error':raise ResponseError(envelope['message'],envelope['evidence'])
            raise UnknownCompletion(envelope['message'])
        finally:self.close()

    def _generate_blocking(self,request,heartbeat=None):
        if getattr(self,'cancelled',lambda:False)() or time.monotonic()>=getattr(self,'deadline',float('inf')):
            raise NotSent('请求前取消或到达预算')
        headers={'Content-Type':'application/json'}
        if self.profile.api_key_env:
            key=os.environ.get(self.profile.api_key_env)
            if not key:raise NotSent('未设置 profile 指定的 API 密钥环境变量')
            headers['Authorization']='Bearer '+key
        req=urllib.request.Request(request['endpoint'],data=json.dumps(request['payload'],ensure_ascii=False).encode(),headers=headers,method='POST')
        timeout=min(self.profile.timeout_seconds,max(.01,getattr(self,'deadline',float('inf'))-time.monotonic()))
        if heartbeat:heartbeat()
        try:
            with self.opener.open(req,timeout=timeout) as response:
                raw=response.read(16*1024*1024+1)
                if len(raw)>16*1024*1024:raise UnknownCompletion('API 响应超过16MiB，无法取得完整证据')
        except (TimeoutError,socket.timeout) as exc:raise UnknownCompletion('API 请求超时；不自动重发') from exc
        except urllib.error.HTTPError as exc:
            raw=exc.read(1024*1024)
            raise ResponseError('API HTTP '+str(exc.code),{'text':'','backend':'api','http_status':exc.code,
                'raw_response_text':raw.decode('utf-8',errors='replace'),'error_body_limit':1048576}) from None
        except urllib.error.URLError as exc:raise UnknownCompletion('API 连接异常，完成状态未知') from None
        try:
            body=json.loads(raw);choice=body['choices'][0];message=choice['message'];text=message.get('content')
            if text is None:text=''
            if not isinstance(text,str):raise ValueError('non-text content')
        except (ValueError,KeyError,IndexError,TypeError) as exc:
            raise ResponseError('API 响应格式无法解析',{'text':'','backend':'api','raw_response_text':raw.decode('utf-8',errors='replace')}) from exc
        finish=choice.get('finish_reason');usage=body.get('usage') or {}
        return {'text':text,'raw_text':text,'reasoning':message.get('reasoning_content',message.get('reasoning')),
                'finish_reason':finish if finish in ('stop','length') else 'unknown','backend':'api',
                'usage':{'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens')},
                'usage_source':'service-reported','raw_response':body,'raw_response_text':raw.decode('utf-8'),
                'output_ids':None,'seed_capability':'unverified','cache_state':'unknown','first_token_latency':None}
    def close(self):
        if self.proc:
            if self.proc.poll() is None:self.proc.kill()
            self.proc.communicate()
            self.proc=None
