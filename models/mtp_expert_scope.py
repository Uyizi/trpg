"""Temporary expert dispatch opt-in for a two-token MTP target verification."""
from contextlib import contextmanager

from .qwen35_moe import expert_modules


@contextmanager
def expert_verification_scope(model, enabled=True):
    """Enable two-row shared-base inference, restoring every expert on exit.

    Top1Experts still enforces eval/no-grad and its existing shared NF4/adapter
    guards. Single-token decode and larger prefill retain their usual dispatch.
    The scope changes flags only; it never copies or caches adapter parameters.
    """
    previous = [(module, module.tinybatch_verification) for module in expert_modules(model)]
    try:
        for module, _ in previous:
            module.tinybatch_verification = bool(enabled)
        yield model
    finally:
        for module, value in previous:
            module.tinybatch_verification = value
