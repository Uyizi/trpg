"""Adapters generate evidence, never correctness labels."""
import json
import os
import select
import subprocess
import time
from pathlib import Path
from .domain import digest
from .inventory import PROJECT


class NotSent(RuntimeError):
    """No model generation was dispatched; reservation can be released."""


class UnknownCompletion(TimeoutError):
    """Generation was dispatched, but no trustworthy response was received."""


class ResponseError(RuntimeError):
    def __init__(self,message,evidence):
        super().__init__(message)
        self.evidence=evidence


class MockRuntime:
    def __init__(self,profile): self.profile=profile
    def prepare(self,case,seed,heartbeat=None):
        payload={'messages':case['messages'],'seed':seed,'profile':self.profile.model_dump(mode='json')}
        return {'payload':payload,'fingerprint':digest(payload),'rendered_prompt':None,'input_ids':None,
                'input_tokens':None,'count_source':'unknown (mock)','identity':'mock'}
    def generate(self,request,heartbeat=None):
        if self.profile.mock_finish=='timeout': raise UnknownCompletion('synthetic timeout')
        if self.profile.mock_finish=='error': raise RuntimeError('synthetic runtime failure')
        return {'text':self.profile.mock_text,'raw_text':self.profile.mock_text,'reasoning':None,
                'finish_reason':self.profile.mock_finish,
                'usage':{'input_tokens':None,'output_tokens':None if self.profile.mock_usage_unknown else 1},
                'backend':'mock','usage_source':'synthetic_fixture','output_ids':None}
    def close(self): pass


class LocalRuntime:
    def __init__(self,profile,log_path):
        self.profile=profile; self.proc=None
        self.log_path=Path(log_path); self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.log=None

    def _send(self,payload,heartbeat=None):
        def check_dispatch():
            if time.monotonic()>=getattr(self,'deadline',float('inf')):
                raise NotSent('预算期限已到，未发送生成')
            if getattr(self,'cancelled',lambda:False)():
                raise NotSent('已取消，未发送生成')
        check_dispatch()
        if self.proc is None:
            env=os.environ.copy()
            env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                       HF_HUB_DISABLE_PROGRESS_BARS='1',PYTHONDONTWRITEBYTECODE='1',QDEBUG_PARENT_PID=str(os.getpid()),
                       QDEBUG_PROJECT_ROOT=str(PROJECT))
            self.log=self.log_path.open('a',encoding='utf-8')
            self.proc=subprocess.Popen([str(PROJECT/'.venv-qwen35/bin/python'),'-u',
                str(Path(__file__).with_name('local_worker.py'))],cwd=PROJECT,env=env,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
                pass_fds=(self.lease_fd,) if hasattr(self,'lease_fd') else ())
        wire=json.dumps(payload,ensure_ascii=False)+'\n'
        check_dispatch()
        try:
            self.proc.stdin.write(wire); self.proc.stdin.flush()
        except (BrokenPipeError,OSError) as exc:
            raise UnknownCompletion('worker 管道断开，无法确认发送状态') from exc
        start=time.monotonic()
        while time.monotonic()-start<self.profile.timeout_seconds and time.monotonic()<getattr(self,'deadline',float('inf')):
            if heartbeat: heartbeat()
            ready,_,_=select.select([self.proc.stdout],[],[],0.25)
            if ready:
                line=self.proc.stdout.readline()
                if not line: raise UnknownCompletion('本地 worker 退出；详情见 '+str(self.log_path))
                try: result=json.loads(line)
                except ValueError as exc: raise UnknownCompletion('本地 worker 返回无效协议') from exc
                if result.get('error'): raise RuntimeError(result['error'])
                return result
            if self.proc.poll() is not None: raise UnknownCompletion('本地 worker 意外退出')
        self.close()
        raise UnknownCompletion('本地请求超时；已停止调试器自己的 worker，未重试')

    def prepare(self,case,seed,heartbeat=None):
        return self._send({'op':'prepare','messages':case['messages'],'seed':seed,
                           'profile':self.profile.model_dump(mode='json')},heartbeat)

    def generate(self,request,heartbeat=None):
        return self._send({'op':'generate','request':request},heartbeat)

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try: self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
            self.proc.stdin.close(); self.proc.stdout.close(); self.proc=None
        if self.log is not None: self.log.close(); self.log=None
