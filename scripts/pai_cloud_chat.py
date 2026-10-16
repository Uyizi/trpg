"""Interactive cloud Qwen3.5 chat through the existing PAI SSH credentials."""
import os
import shlex
import subprocess
import sys
import threading

from pai_direct_proxy import Relay, Server


def main():
    env = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')}
    command = ('cd /mnt/workspace/qwen35_update/app && '
               'model=/mnt/workspace/qwen35_update/base/Qwen3.5-4B; '
               'if [ -f /mnt/workspace/qwen35_update/active-model/moe_manifest.json ]; then '
               'model=/mnt/workspace/qwen35_update/active-model; fi; '
               'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false '
               '../.venv35/bin/python -m models.chat_pretrained '
               '--model "$model" '
               '--device cuda --local-files-only --resident ' + shlex.join(sys.argv[1:]))
    with Server(('127.0.0.1', 0), Relay) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        proxy = f'http://127.0.0.1:{server.server_address[1]}'
        env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy, NO_PROXY='', no_proxy='')
        # Never print or persist ProxyClient diagnostics: they may contain credentials.
        # With a terminal, remote errors are carried by its PTY on SSH stdout.
        try:
            result = subprocess.run(['ssh', '-tt' if sys.stdin.isatty() else '-T',
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                'YOUR_DSW_SSH_ALIAS', command], env=env, stderr=subprocess.DEVNULL)
            if result.returncode:
                print('云端聊天未启动或已中断；确认 DSW 正在运行。', file=sys.stderr)
            return result.returncode
        finally:
            server.shutdown()
            thread.join(timeout=1)


if __name__ == '__main__':
    raise SystemExit(main())
