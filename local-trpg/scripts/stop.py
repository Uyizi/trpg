"""Stop only the server recorded by this project; never stop another model app."""
import os,signal
from pathlib import Path
root=Path(__file__).resolve().parents[1]
pid_file=Path(os.environ.get('LOCAL_TRPG_DATA_DIR',str(root/'data')))/'server.pid'
try:
 pid=int(pid_file.read_text());cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 if b'local_trpg' not in cmd or Path(f'/proc/{pid}/cwd').resolve()!=root:raise RuntimeError('PID不属于本项目，未停止')
 os.kill(pid,signal.SIGTERM);print('已请求退出备用冷却，已提交存档保留。')
except (FileNotFoundError,ProcessLookupError):print('没有发现正在运行的本项目服务。')
