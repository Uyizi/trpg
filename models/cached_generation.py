"""Single-session, prefix-validated generation with hybrid quantized KV."""
import time
import torch
from .quantized_kv import make_cache, cache_bytes


def enable_structured_gqa():
    """Keep compressed-cache KV heads for supported structured SDPA masks."""
    from transformers.integrations import sdpa_attention
    from torch.nn.attention.bias import CausalBias
    if not getattr(sdpa_attention, '_structured_gqa', False):
        original = sdpa_attention.use_gqa_in_sdpa
        def supported(mask, key, value):
            return (key.is_cuda and isinstance(mask, CausalBias) and key.shape[-1] == value.shape[-1] <= 256) or original(mask, key, value)
        sdpa_attention.use_gqa_in_sdpa = supported
        sdpa_attention._structured_gqa = True


class CachedGenerator:
    def __init__(self, model, *, kv_mode='q8', chunk_size=1024,reuse_decode_buffers=True,optimized_sampling=None):
        self.model = model
        self.kv_mode, self.chunk_size = kv_mode, chunk_size
        self.reuse_decode_buffers=reuse_decode_buffers
        self.optimized_sampling=optimized_sampling
        from .prefix_cache import PrefixSnapshotPool
        self.prefix_pool=PrefixSnapshotPool()
        self._prefix_weights=None
        # HF conservatively repeats KV heads for any non-None mask. PyTorch's
        # structured CausalBias supports GQA directly; keep the four KV heads.
        enable_structured_gqa()
        self.clear()

    def clear(self):
        self.cache = None
        self.prefix = []

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, *, temperature, streamer=None,
                 prefix_cache_key=None, prefix_length=0, constraint=None):
        ids = input_ids[0].tolist()
        scoped=prefix_cache_key is not None
        snapshot_hit=False
        versioned=True
        if scoped:
            if not isinstance(prefix_cache_key,str) or not prefix_cache_key or not 0<prefix_length<len(ids):
                raise ValueError('Scoped prefix must have a nonempty key and leave variable input tokens')
            from .prefix_cache import model_cache_signature
            signature=model_cache_signature(self.model)
            versioned=signature is not None
            if signature!=self._prefix_weights or not versioned:
                self.prefix_pool.clear();self._prefix_weights=signature
            self.clear()
            try:self.cache=self.prefix_pool.get(prefix_cache_key,ids[:prefix_length])
            except (MemoryError,torch.OutOfMemoryError):
                self.prefix_pool.clear();self.cache=None
            if self.cache is not None:
                self.prefix=ids[:prefix_length];snapshot_hit=True
        reused = len(self.prefix)
        # Hybrid recurrent state cannot be cropped arbitrarily. Only reuse a
        # complete, exact cached prefix; edits/session switches rebuild it.
        if not reused or reused >= len(ids) or ids[:reused] != self.prefix:
            self.clear()
            reused = 0
            self.cache = make_cache(self.model.config.get_text_config(), self.kv_mode)
        # Ordinary chat originally had one host token read. Only constrained
        # generation benefits from combining its separate validity check. None
        # keeps that distinction; bool overrides support controlled profiles.
        optimized_sampling=(constraint is not None) if self.optimized_sampling is None else self.optimized_sampling
        stats = {'reused_tokens': reused, 'prefill_tokens': len(ids) - reused, 'kv_mode': self.kv_mode,
                 'fixed_prefix_hit':snapshot_hit,'decode_input_buffers':self.reuse_decode_buffers,
                 'sampling_backend':'validated-device-token' if optimized_sampling else (
                     'reference-device-token' if self.reuse_decode_buffers else 'reference')}
        stats['fixed_prefix_versioned']=versioned
        started = time.monotonic()
        device = input_ids.device
        from .prefill_policy import choose_prefill_chunk
        available=torch.cuda.mem_get_info(device)[0] if device.type=='cuda' and self.chunk_size==0 else None
        chunk_size=choose_prefill_chunk(len(ids)-reused,self.chunk_size,available)
        stats['prefill_chunk_size']=chunk_size

        def forward(tokens, start, tensor=None, positions=None):
            if tensor is None:
                tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            if positions is None:
                positions = torch.arange(start, start + len(tokens), device=device).view(1, 1, -1).expand(4, 1, -1)
            # PyTorch's lower-right causal bias dispatches to memory-efficient
            # SDPA without materializing a [query_length, history_length] mask.
            from torch.nn.attention.bias import causal_lower_right
            masks = {'full_attention': causal_lower_right(len(tokens), start + len(tokens)),
                     'linear_attention': None}
            output = self.model(input_ids=tensor, position_ids=positions,
                                attention_mask=masks, past_key_values=self.cache,
                                use_cache=True, logits_to_keep=1)
            self.cache = output.past_key_values
            self.prefix.extend(tokens)
            return output.logits[0, -1]

        generated = []
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        try:
            if scoped and not snapshot_hit:
                for start in range(0,prefix_length,chunk_size):
                    end=min(start+chunk_size,prefix_length)
                    tensor=input_ids[:,start:end] if self.reuse_decode_buffers else None
                    logits=forward(ids[start:end],start,tensor=tensor)
                if versioned:
                    try:self.prefix_pool.put(prefix_cache_key,ids[:prefix_length],self.cache)
                    except (MemoryError,torch.OutOfMemoryError):pass
                start_at=prefix_length
            else:start_at=reused
            for start in range(start_at, len(ids), chunk_size):
                tensor=input_ids[:,start:start+chunk_size] if self.reuse_decode_buffers else None
                logits = forward(ids[start:start + chunk_size], start,tensor=tensor)
            # A small request-local position vector replaces per-token device
            # arange allocations. Each decode forward takes one immutable view.
            decode_positions=None
            if self.reuse_decode_buffers and max_new_tokens>1:
                decode_positions=torch.arange(len(ids),len(ids)+max_new_tokens-1,device=device).view(1,1,-1).expand(4,1,-1)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            stats['prefill_seconds'] = round(time.monotonic() - started, 3)
            decode_start = time.monotonic()
            if streamer is not None:
                streamer.put(input_ids.cpu())
            from .token_sampling import sample_token,TokenSamplingError
            for step in range(max_new_tokens):
                if constraint is not None:
                    mask=constraint.mask_for_sampling if optimized_sampling and hasattr(constraint,'mask_for_sampling') else constraint.apply
                    logits=mask(logits,generated)
                device_token=None
                if optimized_sampling:
                    try:sampled=sample_token(logits,temperature=temperature)
                    except TokenSamplingError as exc:
                        if constraint is not None and hasattr(constraint,'reject_sampling'):
                            constraint.reject_sampling(exc)
                        raise
                    token=sampled.token_id
                    device_token=sampled.device_token.view(1,1)
                elif temperature > 0:
                    values, indices = torch.topk(logits.float() / temperature, min(20, logits.numel()))
                    probs = values.softmax(-1)
                    probs.masked_fill_(probs.cumsum(-1) - probs > .8, 0)
                    device_token=indices[torch.multinomial(probs, 1)]
                    token = int(device_token)
                else:
                    device_token=logits.argmax()
                    token = int(device_token)
                generated.append(token)
                if streamer is not None:
                    streamer.put(torch.tensor([token]))
                if token in eos or step + 1 == max_new_tokens:
                    break
                positions=None if decode_positions is None else decode_positions[:,:,step:step+1]
                tensor=device_token.view(1,1) if self.reuse_decode_buffers else None
                logits = forward([token], len(self.prefix),tensor=tensor,positions=positions)
            if streamer is not None:
                streamer.end()
        except BaseException:
            # Cancelled/failed output is not committed to history or KV reuse.
            self.clear()
            raise
        stats['decode_seconds'] = round(time.monotonic() - decode_start, 3)
        stats['decode_tokens_per_second'] = round(len(generated) / max(time.monotonic() - decode_start, 1e-6), 2)
        stats['cache_mib'] = round(cache_bytes(self.cache) / 2**20, 2)
        stats['prefix_snapshots']=self.prefix_pool.info()
        return torch.tensor([ids + generated], device=device), stats
