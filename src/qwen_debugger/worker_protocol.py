"""Dependency-free output capture and worker lifetime safeguards."""
import os
import signal
import sys


def decode_output(tokenizer, ids):
    # Whitespace normalization belongs exclusively to the case's frozen oracle.
    return {'text':tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
            'raw_text':tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)}


def exit_with_parent():
    if sys.platform.startswith('linux'):
        import ctypes
        expected=int(os.environ['QDEBUG_PARENT_PID'])
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:
            raise OSError(ctypes.get_errno(),'cannot set parent-death signal')
        if os.getppid()!=expected: os._exit(1)
