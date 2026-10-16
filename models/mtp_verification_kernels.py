"""Request-local opt-in for cached two-token MTP verification kernels.

Entering this scope changes no module methods or global kernel configuration.
The installed Qwen3.5 adapter reads the flag on each call; nested scopes,
exceptions, and independent threads/tasks retain their own setting.
"""
from contextlib import contextmanager
from contextvars import ContextVar


_MTP_VERIFICATION_ENABLED = ContextVar('mtp_verification_kernels_enabled', default=False)


def mtp_verification_enabled():
    return _MTP_VERIFICATION_ENABLED.get()


@contextmanager
def mtp_verification_scope(enabled=True):
    """Allow cached two-token delta calls to use guarded recurrent dispatch."""
    token = _MTP_VERIFICATION_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _MTP_VERIFICATION_ENABLED.reset(token)
