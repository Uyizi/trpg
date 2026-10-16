"""Native XGrammar token masks for the custom generation loop.

Reference: https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html
Only compiled grammars/tokenizer metadata are shared. Every request has its own
matcher and bitmask. This enforces wire structure, not truth or game authority.
"""
from __future__ import annotations

from collections import OrderedDict
import importlib
import json
import threading
from typing import Any


class StructuredDecodingError(RuntimeError):
    """A requested grammar cannot be enforced; never continue unmasked."""


# Strong tokenizer references prevent object-id reuse while cached. Native
# compiler caching is disabled, so these explicit LRU bounds own all retention.
_TOKENIZER_LIMIT = 2
_SCHEMA_LIMIT = 8
_CACHE: OrderedDict = OrderedDict()
_CACHE_LOCK = threading.RLock()


def _load_xgrammar():
    try:
        return importlib.import_module('xgrammar')
    except Exception as exc:
        raise StructuredDecodingError(
            'XGrammar is required for response_schema, but its native runtime is unavailable: '
            + str(exc)) from exc


def create_constraint(tokenizer, schema: dict[str, Any], *, vocab_size: int | None = None):
    """Compile/cache a schema and return an independent request constraint.

    Pass the model's output vocab_size, which may include padding beyond the
    tokenizer vocabulary. No downloads, model loading or optional fallback.
    """
    xgr = _load_xgrammar()
    try:
        width = len(tokenizer) if vocab_size is None else vocab_size
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError('vocab_size must be a positive integer')
        if not isinstance(schema, dict):
            raise ValueError('response schema must be an object')
        # Preserve property order: native ordered-object generation checks both
        # required keys and duplicate keys. Canonicalization must not reorder it.
        schema_json = json.dumps(schema, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        key = (id(tokenizer), width)
        with _CACHE_LOCK:
            entry = _CACHE.get(key)
            if entry is None:
                info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=width)
                compiler = xgr.GrammarCompiler(info, max_threads=2, cache_enabled=False)
                entry = (tokenizer, info, compiler, OrderedDict())
                _CACHE[key] = entry
                while len(_CACHE) > _TOKENIZER_LIMIT:
                    _CACHE.popitem(last=False)
            _CACHE.move_to_end(key)
            _, info, compiler, compiled_cache = entry
            compiled = compiled_cache.get(schema_json)
            if compiled is None:
                compiled = compiler.compile_json_schema(schema_json, any_whitespace=False,
                    separators=(',', ':'), strict_mode=True)
                compiled_cache[schema_json] = compiled
                while len(compiled_cache) > _SCHEMA_LIMIT:
                    compiled_cache.popitem(last=False)
            compiled_cache.move_to_end(schema_json)
        return _NativeConstraint(xgr, compiled, info.vocab_size)
    except StructuredDecodingError:
        raise
    except Exception as exc:
        raise StructuredDecodingError('XGrammar schema compilation failed: ' + str(exc)) from exc


class _NativeConstraint:
    def __init__(self, xgr, compiled, width):
        self._xgr = xgr
        self._width = width
        self._matcher = xgr.GrammarMatcher(compiled)
        self._bitmask = xgr.allocate_token_bitmask(1, width)
        self._accepted: list[int] = []
        self._failed = False

    def apply(self, logits, generated_ids: list[int]):
        """Standalone checked mask; callers need no separate validated sampler."""
        try:
            logits=self.mask_for_sampling(logits,generated_ids)
            if not logits.isfinite().any().item():
                raise ValueError('grammar permits no finite next-token score')
            return logits
        except StructuredDecodingError:
            raise
        except Exception as exc:
            self.reject_sampling(exc)

    def reject_sampling(self,error):
        """Invalidate this request after the validated sampler rejects scores."""
        self._failed=True
        raise StructuredDecodingError('XGrammar token constraint failed: '+str(error)) from error

    def mask_for_sampling(self, logits, generated_ids: list[int]):
        """Mask [vocabulary] scores after accepting only newly generated tokens.

        generated_ids contains completion tokens only, never prompt tokens.
        Repeated calls at the same position are allowed; changing past tokens
        requires a new request constraint. Scores are masked in place. This
        deferred form MUST be followed by validated sampling; sampling failure
        must call reject_sampling before another matcher step is permitted.
        """
        if self._failed:
            raise StructuredDecodingError('XGrammar constraint already failed')
        try:
            if logits.ndim != 1 or logits.shape[0] != self._width:
                raise ValueError('constraint expects one logits vector with the configured vocab_size')
            position = len(self._accepted)
            if generated_ids[:position] != self._accepted:
                raise ValueError('completion history changed; use a fresh constraint')
            for token_id in generated_ids[position:]:
                if not self._matcher.accept_token(int(token_id)):
                    raise ValueError('completion token rejected by grammar')
                self._accepted.append(int(token_id))
            if self._matcher.is_terminated():
                raise ValueError('grammar has already consumed its stop token')
            if self._matcher.fill_next_token_bitmask(self._bitmask):
                bitmask = self._bitmask.to(logits.device, non_blocking=True)
                self._xgr.apply_token_bitmask_inplace(logits.unsqueeze(0), bitmask)
            return logits
        except Exception as exc:
            self._failed = True
            raise StructuredDecodingError('XGrammar token constraint failed: ' + str(exc)) from exc
