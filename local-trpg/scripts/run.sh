#!/usr/bin/env bash
set -euo pipefail
trpg_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$trpg_root"
if [[ ! -x .venv/bin/python ]]; then echo '应用环境缺失。请按 README 安装独立环境。' >&2; exit 1; fi
if [[ ! -f web/dist/index.html ]]; then echo '前端尚未构建，请按 README 执行构建。' >&2; exit 1; fi
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -m local_trpg "$@"
