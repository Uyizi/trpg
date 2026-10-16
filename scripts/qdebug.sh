#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${QDEBUG_PYTHON:-}" ]]; then
    interpreter="$QDEBUG_PYTHON"
elif [[ -x "$project_dir/.venv-debugger/bin/python" ]]; then
    interpreter="$project_dir/.venv-debugger/bin/python"
elif python3 -c 'import pydantic; assert pydantic.__version__.startswith("2.")' >/dev/null 2>&1; then
    interpreter=python3
elif [[ -x "$HOME/miniconda3/bin/python3" ]]; then
    interpreter="$HOME/miniconda3/bin/python3"
else
    echo '需要 Python 3.11+ 和 Pydantic 2；可用 QDEBUG_PYTHON 指定解释器。未执行下载。' >&2
    exit 1
fi
exec "$interpreter" -m qwen_debugger.cli "$@"
