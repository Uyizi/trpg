"""Keep known optional-kernel notices out of streamed assistant text."""
import logging
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / 'outputs/chat-runtime.log'


class KernelNoticeFilter(logging.Filter):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def filter(self, record):
        message = record.getMessage()
        known = any(message.startswith(f'`{name}` is falling back to its reference PyTorch implementation')
                    for name in ('causal_conv1d_fn', 'causal_conv1d_update'))
        if known:
            self.sink.handle(record)
            return False
        # All other warnings and errors remain visible.
        return True


def configure_chat_diagnostics():
    logger = logging.getLogger('transformers.integrations.hub_kernels')
    if any(isinstance(f, KernelNoticeFilter) for f in logger.filters):
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addFilter(KernelNoticeFilter(handler))
