import argparse,socket,os
import uvicorn
from .config import Settings
from .api import create_app

def main():
 p=argparse.ArgumentParser(description='本地多智能体文字冒险');p.add_argument('--port',type=int,default=int(os.environ.get('LOCAL_TRPG_PORT','8768')));args=p.parse_args()
 config=Settings(port=args.port)
 with socket.socket() as s:
  # Match uvicorn: recent closed connections in TIME_WAIT do not mean a
  # second server is listening. A live listener still fails this bind.
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:s.bind((config.host,config.port))
  except OSError:raise SystemExit(f'端口{config.port}已被占用；请继续使用已有窗口，或指定 --port 新端口。')
 config.data_dir.mkdir(parents=True,exist_ok=True)
 (config.data_dir/'server.pid').write_text(str(os.getpid()))
 print(f'本地文字冒险：http://{config.host}:{config.port}；初次加载模型约需数分钟。',flush=True)
 uvicorn.run(create_app(config),host=config.host,port=config.port,workers=1,log_level='warning')
if __name__=='__main__':main()
