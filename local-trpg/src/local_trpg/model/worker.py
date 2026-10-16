"""Stdlib wire protocol run exclusively inside the existing model environment.

No application dependency is installed in that environment. stdout is protocol
only; model diagnostics go to stderr. The tokenizer process never loads weights.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

protocol = sys.stdout
sys.stdout = sys.stderr
project, model_path = Path(sys.argv[1]), Path(sys.argv[2])
tokenizer_only = "--tokenizer-only" in sys.argv[3:]
# A directly executed script otherwise exposes model/queue.py as stdlib queue.
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(project))
from models.inference_environment import configure_cuda_allocator
configure_cuda_allocator()

parent_pid = int(os.environ.get("LOCAL_TRPG_PARENT_PID", os.getppid()))
if sys.platform == "linux":
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot set parent-death signal")
    if os.getppid() != parent_pid:
        sys.exit(1)

model = None
tokenizer = None
template = None
gpu_lease = None


def prepare_tokenizer():
    global tokenizer, template
    if tokenizer is not None:
        return
    from transformers import AutoConfig, AutoTokenizer
    from models.runtime_paths import read_bundle
    base = read_bundle(model_path)["base"] if (model_path / "moe_manifest.json").exists() else str(model_path)
    config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    template = tokenizer.chat_template
    marker = "if loop.index0 > ns.last_query_index"
    if config.model_type == "qwen3_5" and template.count(marker) == 1:
        template = template.replace(marker, marker + " or (enable_thinking is defined and enable_thinking is false)")


def encode(messages):
    prepare_tokenizer()
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False, chat_template=template)


def warmup():
    global model, gpu_lease
    if tokenizer_only:
        raise ValueError("tokenizer-only worker cannot load weights")
    if model is None:
        import fcntl
        lock_path = Path.home() / ".cache/qwen-debugger/gpu.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        gpu_lease = lock_path.open("a+")
        try:
            fcntl.flock(gpu_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            gpu_lease.close()
            gpu_lease = None
            raise RuntimeError("gpu_busy: another local model owner holds the GPU lease")
        from models.pretrained import PretrainedChatModel
        model = PretrainedChatModel(model_id=str(model_path), device="cuda", local_files_only=True,
            context_length=6144, thinking=False, kv_cache="q8", prefill_chunk_size=1024, fast_kernels=True)
    import torch
    from models.structured_decoding import _load_xgrammar
    _load_xgrammar()  # Required local capability; missing runtime fails explicitly.
    return {"identity": model.identity, "runtime": model.description,"structured_output_mode":"json_schema",
        "loaded_torch_memory_mib": round(torch.cuda.memory_allocated() / 2**20),
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes", "tokenizers")},
        "cuda": torch.version.cuda,
        "template_sha256": hashlib.sha256(model.inference_template.encode()).hexdigest()}


def generate(command):
    if model is None:
        raise RuntimeError("backend_not_warmed")
    request = command["request"]
    if request["thinking_policy"] == "enabled":
        raise ValueError("thinking_enabled_not_supported_by_fixed_profile")
    messages = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
    input_ids = model.encode(messages)
    if len(input_ids) + request["max_output_tokens"] > 6144:
        raise ValueError("context_budget_exceeded")
    model.clear_cache()
    try:
        text = model.answer(messages, max_new_tokens=request["max_output_tokens"], temperature=request["temperature"],
            response_schema=request.get('response_schema'),prefix_cache_key=request.get('cache_namespace'),
            prefix_messages=request.get('prefix_message_count',0))
        stats = dict(model.last_response_stats)
        return {"text": text, "finish_reason": "length" if model.last_response_truncated else "stop",
            "usage": {"input_tokens": stats["input_tokens"], "output_tokens": stats["output_tokens"],
                      "cached_input_tokens":stats.get('reused_tokens',0)}, "stats": stats}
    finally:
        model.clear_cache()


for line in sys.stdin:
    started = time.monotonic()
    try:
        command = json.loads(line)
        if command["op"] == "count":
            result = {"count": len(encode(command["messages"]))}
        elif command["op"] == "warmup":
            result = warmup()
        elif command["op"] == "generate":
            result = generate(command)
        else:
            raise ValueError("unknown_operation")
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result = {"error": f"{type(exc).__name__}: {exc}"}
    result["elapsed_ms"] = (time.monotonic() - started) * 1000
    protocol.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    protocol.flush()
