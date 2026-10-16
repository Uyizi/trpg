"""Cloud-local Unix socket keeps one model in GPU memory across CLI sessions."""
import fcntl,json,os,socket,socketserver,subprocess,sys,time,gc
from pathlib import Path
ROOT=Path('/mnt/workspace/qwen35_update')
SOCKET=ROOT/'logs/chat.sock'
LIMIT=4*1024*1024


def ensure_service():
    # Startup guard avoids duplicate services; the daemon also holds a lifetime lock.
    with (ROOT/'logs/chat_start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            with socket.socket(socket.AF_UNIX) as s:s.connect(str(SOCKET))
            return
        except OSError:pass
        with (ROOT/'logs/chat_service.log').open('ab') as log:
            subprocess.Popen([sys.executable,'-m','models.resident_chat'],cwd=ROOT/'app',stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(100):
            try:
                with socket.socket(socket.AF_UNIX) as s:s.connect(str(SOCKET))
                return
            except OSError:time.sleep(.1)
        raise RuntimeError('Cloud model service did not start; inspect logs/chat_service.log')


def request(payload,stream=False,on_text=None):
    ensure_service()
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(1800);s.connect(str(SOCKET))
        s.sendall((json.dumps(payload,ensure_ascii=False)+'\n').encode())
        with s.makefile('rb') as f:
            while line:=f.readline(LIMIT):
                event=json.loads(line)
                if event.get('type')=='text':
                    if on_text is not None:on_text(event['text'])
                    elif stream:print(event['text'],end='',flush=True)
                if event.get('type')=='error':raise ValueError(event['message'])
                if event.get('type')=='done':return event
    raise RuntimeError('Cloud model service disconnected')


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        line=self.rfile.readline(LIMIT)
        if not line:return
        def send(value):
            self.wfile.write((json.dumps(value,ensure_ascii=False)+'\n').encode());self.wfile.flush()
        try:
            if len(line)>=LIMIT:raise ValueError('Request too large')
            value=json.loads(line)
            for name in ('qwen35_v2_queue.json', 'qwen35_advanced_queue.json'):
                path=ROOT/'logs'/name
                if path.exists():
                    state=json.loads(path.read_text())
                    if state.get('state') in ('running','waiting') and Path('/proc',str(state.get('pid',-1))).exists():
                        raise ValueError('云端训练队列仍在运行，完成后再启动常驻聊天。')
            from .pretrained import PretrainedChatModel
            model=str(Path(value['model']).resolve())
            identity=(model,bool(value.get('thinking')))
            if self.server.identity!=identity:
                self.server.backend=None;self.server.identity=None;gc.collect()
                import torch
                torch.cuda.empty_cache()
                self.server.backend=PretrainedChatModel(model,device='cuda',context_length=value['context_length'],local_files_only=True,thinking=identity[1])
                self.server.identity=identity
            backend=self.server.backend
            backend.context_length=value['context_length']
            backend.speculative=value.get('speculative',0)
            backend.prefill_chunk_size=value.get('prefill_chunk_size',2048)
            answer=backend.answer(value['messages'],value['max_new_tokens'],value['temperature'],
                on_text=lambda text:send({'type':'text','text':text}))
            send({'type':'done','answer':answer,'truncated':backend.last_response_truncated})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as exc:
            try:send({'type':'error','message':str(exc)[:500]})
            except OSError:pass


def main():
    with (ROOT/'logs/chat_service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if SOCKET.exists():SOCKET.unlink()
        with socketserver.UnixStreamServer(str(SOCKET),Handler) as server:
            os.chmod(SOCKET,0o600);server.backend=None;server.identity=None
            (ROOT/'logs/chat_service.pid').write_text(str(os.getpid()))
            server.serve_forever()
if __name__=='__main__':main()
