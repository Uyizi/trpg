"""Strict, inference-only native dense Qwen3.5 MTP head.

Architecture/shift: official vLLM qwen3_5_mtp.py and SGLang
qwen3_5_mtp.py / speculative/eagle_worker_v2.py; references and hidden-tap
evidence are in docs/MTP_HEAD_2026-10-15.md. Reuses the installed upstream
Transformers decoder for gated full attention, zero-centered norms and RoPE.
This module does not enable speculative decoding in the production generator.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
)

from .mtp_probe import expected_mtp_shapes, read_safetensors_header


@dataclass
class MTPKVCache:
    """Only this head's full-attention KV; never a target hybrid cache."""
    kv: DynamicCache
    owner: object
    next_positions: torch.Tensor | None = None


@dataclass
class MTPOutput:
    hidden_states: torch.Tensor
    logits: torch.Tensor
    past_key_values: MTPKVCache | None


class FinalHiddenCapture:
    """Scoped output hook on the target TEXT final norm, preserving its result.

    Unlike output_hidden_states=True this retains only one layer's normalized
    hidden tensor. No parameters, configuration or target cache are changed.
    """

    def __init__(self, text_final_norm):
        self.norm = text_final_norm
        self.hidden_states = None
        self._hook = None

    def __enter__(self):
        if self._hook is not None:
            raise RuntimeError('Hidden capture context is already active')
        def capture(module, args, output):
            self.hidden_states = self._tensor(output).detach().clone()
        self._hook = self.norm.register_forward_hook(capture)
        return self

    def _tensor(self,output):
        return output

    def __exit__(self, *exc):
        self._hook.remove()
        self._hook = None


class TextHiddenCapture(FinalHiddenCapture):
    """Capture text-backbone output, leaving its final norm hooks untouched.

    This is the same final-normalized tensor. Hooking the parent preserves the
    existing fused RMSNorm dispatch, which deliberately rejects norm hooks.
    """
    def _tensor(self,output):
        hidden=getattr(output,'last_hidden_state',None)
        if hidden is None and isinstance(output,(tuple,list)) and output:
            hidden=output[0]
        if not isinstance(hidden,torch.Tensor) or hidden.ndim!=3:
            raise ValueError('Text backbone did not return [batch,tokens,hidden] final states')
        return hidden


class Qwen35MTPHead(nn.Module):
    """One full-attention dense MTP layer with 15 mandatory BF16 tensors.

    ``target_hidden_states`` must be the target's FINAL-NORMALIZED output.
    At slot t pair it with embedding(x[t+1]); prefill() does this rotation.
    Explicit positions refer to the original hidden/cache slots, with NO +1
    shift during prefill. Subsequent draft/extend slots increment normally.

    Shared embedding/head are intentionally outside this module's registered
    children: .to(), .eval(), requires_grad_(), state_dict() and strict weight
    loading affect only the MTP branch and cannot alter target modules.
    """

    def __init__(self, config, embed_tokens, lm_head, weights, *, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        config = getattr(config, 'text_config', config)
        expected = expected_mtp_shapes(config.to_dict())
        if set(weights) != set(expected):
            missing, extra = set(expected) - set(weights), set(weights) - set(expected)
            raise ValueError(f'Incomplete MTP weights: missing={sorted(missing)}, extra={sorted(extra)}')
        for name, shape in expected.items():
            tensor = weights[name]
            if not isinstance(tensor, torch.Tensor) or list(tensor.shape) != shape:
                raise ValueError(f'MTP shape mismatch: {name}')
            if tensor.device.type != 'cpu' or tensor.dtype != torch.bfloat16:
                raise ValueError(f'MTP source must be real CPU BF16 tensors: {name}')
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Nonfinite MTP tensor: {name}')
        expected_embedding_shape = (config.vocab_size, config.hidden_size)
        if tuple(embed_tokens.weight.shape) != expected_embedding_shape:
            raise ValueError('Target embedding shape is incompatible')
        if lm_head.weight is not embed_tokens.weight:
            raise ValueError('Target embedding/lm_head must be tied to the same parameter')
        if dtype not in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
            raise ValueError('MTP compute dtype must be floating point')
        # Deep copy prevents overwriting the target's 32-layer hybrid layout.
        self.config = copy.deepcopy(config)
        self.config.num_hidden_layers = 1
        self.config.layer_types = ['full_attention']
        self.config.full_attention_interval = 1
        self.config._attn_implementation = 'eager' if torch.device(device).type == 'cpu' else 'sdpa'
        self._cache_owner = object()
        object.__setattr__(self, 'embed_tokens', embed_tokens)
        object.__setattr__(self, 'lm_head', lm_head)
        # No random initialization fallback, even transiently at real size.
        with torch.device('meta'):
            self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
            self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.layers = nn.ModuleList([Qwen3_5DecoderLayer(self.config, 0)])
            self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        state = {name.removeprefix('mtp.'): tensor for name, tensor in weights.items()}
        self.load_state_dict(state, strict=True, assign=True)
        self.to(device=device, dtype=dtype)
        # Construct real nonpersistent buffers AFTER casting parameters: rounded
        # BF16 inverse frequencies accumulate a wrong phase at long positions.
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config).to(device=device)
        self.requires_grad_(False)
        self.eval()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, 'rotary_emb'):
            # Recompute from config, rather than widening already rounded
            # frequencies, when callers subsequently move/cast this head.
            self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config).to(device=self.fc.weight.device)
        return result

    @classmethod
    def from_safetensors(cls, path, config, embed_tokens, lm_head, **kwargs):
        """Read all payloads only after complete file/header/layout validation."""
        config = getattr(config, 'text_config', config)
        expected = expected_mtp_shapes(config.to_dict())
        header = read_safetensors_header(Path(path))
        if set(header) != set(expected):
            raise ValueError('Checkpoint must contain exactly the 15 native MTP tensors')
        for name, shape in expected.items():
            if header[name]['shape'] != shape or header[name]['dtype'] != 'BF16':
                raise ValueError(f'Checkpoint MTP shape/dtype mismatch: {name}')
        with safe_open(path, framework='pt', device='cpu') as file:
            weights = {name: file.get_tensor(name) for name in expected}
        return cls(config, embed_tokens, lm_head, weights, **kwargs)

    def new_cache(self):
        return MTPKVCache(DynamicCache(config=self.config), self._cache_owner)

    @torch.inference_mode()
    def prefill(self, input_ids, target_hidden_states, next_token_ids, position_ids, **kwargs):
        """Rotate each complete, unpadded request left and append target bonus."""
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('Prefill requires nonempty [batch, tokens] input_ids')
        if next_token_ids.numel() != input_ids.shape[0]:
            raise ValueError('Exactly one next token is required per request')
        shifted = torch.cat([input_ids[:, 1:], next_token_ids.reshape(-1, 1)], dim=1)
        return self(shifted, target_hidden_states, position_ids, **kwargs)

    @torch.inference_mode()
    def forward(self, input_ids, target_hidden_states, position_ids, *,
                past_key_values=None, use_cache=False, logits_to_keep=0):
        """Unpadded text only. Inputs are already shifted; use prefill otherwise.

        Caller owns request boundaries. For chunked prefill the final token of
        each chunk must be the NEXT chunk's first token, except the final chunk
        where it is the target-generated bonus. No padding/vision/grammar here.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('MTP requires nonempty [batch, tokens] input_ids')
        batch, length = input_ids.shape
        if target_hidden_states.shape != (batch, length, self.config.hidden_size):
            raise ValueError('Target hidden shape disagrees with shifted input_ids')
        if position_ids.shape != input_ids.shape or position_ids.dtype != torch.long:
            raise ValueError('Explicit text position_ids must be int64 [batch, tokens]')
        if (position_ids < 0).any() or (position_ids[:, 1:] != position_ids[:, :-1] + 1).any():
            raise ValueError('MTP positions must be nonnegative and contiguous')
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError('logits_to_keep must be a nonnegative integer')
        cache = past_key_values
        if cache is None and use_cache:
            cache = self.new_cache()
        if cache is not None:
            if not isinstance(cache, MTPKVCache) or cache.owner is not self._cache_owner:
                raise ValueError('MTP cache belongs to another head or target')
            if cache.next_positions is not None and not torch.equal(position_ids[:, 0], cache.next_positions):
                raise ValueError('MTP cache position discontinuity')
        past_length = cache.kv.get_seq_length() if cache is not None else 0
        hidden = target_hidden_states.to(device=self.fc.weight.device, dtype=self.fc.weight.dtype)
        embedding = self.embed_tokens(input_ids).to(dtype=hidden.dtype, device=hidden.device)
        hidden = self.fc(torch.cat([self.pre_fc_norm_embedding(embedding),
                                    self.pre_fc_norm_hidden(hidden)], dim=-1))
        # Causality is defined by cache slot order, independent of rotary offset.
        slots = torch.arange(length, device=hidden.device)[:, None] + past_length
        keys = torch.arange(past_length + length, device=hidden.device)[None, :]
        mask = torch.zeros(length, past_length + length, dtype=hidden.dtype, device=hidden.device)
        mask.masked_fill_(keys > slots, float('-inf'))
        mask = mask[None, None]
        rotary_positions = position_ids.to(hidden.device)[None].expand(3, -1, -1)
        position_embeddings = self.rotary_emb(hidden, rotary_positions)
        hidden = self.layers[0](hidden, position_embeddings=position_embeddings,
                                attention_mask=mask, past_key_values=cache.kv if cache else None)
        hidden = self.norm(hidden)
        projected = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
        logits = self.lm_head(projected.to(dtype=self.lm_head.weight.dtype,
                                           device=self.lm_head.weight.device))
        if cache is not None:
            cache.next_positions = position_ids[:, -1].clone() + 1
        return MTPOutput(hidden, logits, cache)
