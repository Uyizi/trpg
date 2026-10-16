"""Measured prefill default; zero requests auto, positive values stay fixed."""


def choose_prefill_chunk(tokens, requested, available_bytes):
    if requested<0 or tokens<=0:
        raise ValueError('Invalid prefill size or input length')
    if requested:return requested
    # On this trained NF4 model, 32K input took 26.954/29.448/36.931 s
    # with 1024/2048/4096 chunks. More free memory does not imply faster
    # prefill. Keep the measured best default; explicit overrides remain.
    return 1024
