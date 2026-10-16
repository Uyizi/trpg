#!/usr/bin/env bash
# 从任意目录启动；直接调用专用环境，无需手动 activate。
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ "${1:-}" == "--launcher-help" ]]; then
    echo 'chat                     启动本地 Qwen3.5 NF4 MoE，新建会话'
    echo 'chat --cloud             在阿里云运行 Qwen3.5-4B（无需下载到本机）'
    echo 'chat --web off           关闭联网，使用本机模型和资料'
    echo 'chat --resume homework   主动恢复指定会话'
    echo '聊天中输入 /exit 退出。'
    exit 0
fi

if [[ "${1:-}" == "--cloud" ]]; then
    shift
    exec python3 scripts/pai_cloud_chat.py "$@"
fi

if [[ ! -x .venv-qwen35/bin/python ]]; then
    echo '缺少本机 Qwen3.5 环境 .venv-qwen35，请按 docs/LOCAL_QWEN35.md 配置。' >&2
    exit 1
fi

# 用户要求所有下载绕过 VPN；启动器不再隐式下载依赖。
if ! .venv-qwen35/bin/python -c 'import torch, peft, bitsandbytes; from transformers import Qwen3_5ForConditionalGeneration' >/dev/null 2>&1; then
    echo 'Qwen3.5 推理依赖不完整。本次未启动下载；请检查 .venv-qwen35。' >&2
    exit 1
fi

# 即使本地权重缺失，也直接报错，不向模型仓库发起请求。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec .venv-qwen35/bin/python -m models.chat_pretrained --local-files-only --device cuda "$@"
