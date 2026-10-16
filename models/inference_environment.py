"""Process-local inference defaults; call before importing CUDA libraries."""
import os


def configure_cuda_allocator():
    # Growing KV and dequantization buffers otherwise leave increasingly
    # large inactive allocations. Respect either spelling of a user override.
    if not any(key in os.environ for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF')):
        os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
