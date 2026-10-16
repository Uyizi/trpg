"""Bounded immutable snapshots of exact, scoped hybrid prompt prefixes."""
from collections import OrderedDict
import copy
import logging
import torch
from .quantized_kv import cache_bytes, QuantizedAttentionLayer


def _packed_prefix_tensors(cache):
    for layer in cache.layers:
        if isinstance(layer,QuantizedAttentionLayer):
            for bank in (layer.key_bank,layer.value_bank):
                for value in (bank.data,bank.meta):
                    if value is not None:
                        yield value,value[:,:,:bank.units]


def _snapshot_size(cache):
    # Active banks reserve growing slabs; immutable prefixes retain only used
    # rows. All ordinary KV and dictionary-held hybrid states still count.
    return cache_bytes(cache)-sum(value.untyped_storage().nbytes()-used.numel()*used.element_size()
                                 for value,used in _packed_prefix_tensors(cache))


def _snapshot_clone(cache):
    memo={id(value):used.clone(memory_format=torch.contiguous_format)
          for value,used in _packed_prefix_tensors(cache)}
    return copy.deepcopy(cache,memo)


def exact_message_prefix(encode,messages,count,full_ids):
    """Find a fixed boundary without rendering an invalid system-only chat.

    Some official templates require a user query. Two distinct dummy queries
    expose only their common fixed prefix (possibly including the user header).
    Intersect with actual tokens to account for BPE boundary merges.
    """
    if not 0<count<len(messages):raise ValueError('Fixed messages must leave a variable user message')
    left=encode(messages[:count]+[{'role':'user','content':'A'}],generation=False)
    right=encode(messages[:count]+[{'role':'user','content':'B'}],generation=False)
    length=0
    for a,b,c in zip(left,right,full_ids):
        if a!=b or a!=c:break
        length+=1
    if not 0<length<len(full_ids):raise ValueError('Template has no reusable fixed token boundary')
    return full_ids[:length]


def model_cache_signature(model):
    """Frozen computation state; return None when mutation cannot be tracked."""
    def freeze(value):
        if isinstance(value,torch.Tensor):
            return (value.data_ptr(),value._version,tuple(value.shape),tuple(value.stride()),
                    value.dtype,value.device,value.requires_grad)
        if isinstance(value,dict):return tuple((k,freeze(v)) for k,v in sorted(value.items()))
        if isinstance(value,(tuple,list)):return tuple(freeze(v) for v in value)
        if value is None or isinstance(value,(str,int,float,bool,torch.dtype,torch.device)):return value
        return (type(value),id(value))
    def quant(state):
        names=('absmax','code','blocksize','quant_type','dtype','shape','offset','nested')
        return tuple((name,freeze(getattr(state,name,None))) for name in names)+(('state2',
                    quant(state.state2) if getattr(state,'state2',None) is not None else None),)
    attrs=('active_adapter','active_adapters','disable_adapters','merged','merged_adapters',
           '_disable_adapters','scaling','use_dora','routing_strength','shared_frozen','fused_lora',
           'eps','variance_epsilon','attention_dropout','compute_dtype','rope_type','attention_scaling')
    try:
        signature=[]
        for module_name,module in model.named_modules():
            # Unknown adapter variants may carry arbitrary mutable computation.
            if getattr(module,'lora_variant',None):return None
            signature.append((id(module),type(module),module.training,
                              freeze(module.__dict__.get('forward')),
                              freeze(module.__dict__.get('_norm'))))
            for name in attrs:
                value=getattr(module,name,None)
                if not callable(value):signature.append((name,freeze(value)))
            config=getattr(module,'config',None)
            if config is not None:
                signature.append(('attention_config',freeze(getattr(config,'_attn_implementation',None))))
            for name,tensor in (*module._parameters.items(),*module._buffers.items()):
                if tensor is not None:
                    signature.append((name,freeze(tensor)))
                    state=getattr(tensor,'quant_state',None)
                    if state is not None:signature.append(('quant_state',quant(state)))
        return tuple(signature)
    except RuntimeError as exc:
        # Inference-created weights have no version counter. Recompute instead
        # of assuming they cannot be mutated or serving a stale role prefix.
        logging.getLogger(__name__).debug('Unversioned prefix state in %s: %s',module_name,exc)
        return None


class PrefixSnapshotPool:
    def __init__(self, *, max_entries=8, max_bytes=256*2**20):
        if max_entries<0 or max_bytes<0:
            raise ValueError('Prefix cache limits must be nonnegative')
        self.max_entries,self.max_bytes=max_entries,max_bytes
        self.entries=OrderedDict();self.bytes=0
        self.hits=self.misses=self.evictions=0

    def get(self, namespace, tokens):
        key=(namespace,tuple(tokens))
        entry=self.entries.get(key)
        if entry is None:
            self.misses+=1
            return None
        # All state, including convolution/recurrent state and packed storage,
        # is private to this execution. The retained snapshot never advances.
        result=copy.deepcopy(entry[0])
        self.entries.move_to_end(key);self.hits+=1
        return result

    def put(self, namespace, tokens, cache):
        size=_snapshot_size(cache)
        if not self.max_entries or size>self.max_bytes:
            return False
        key=(namespace,tuple(tokens))
        old=self.entries.pop(key,None)
        if old:self.bytes-=old[1]
        while self.entries and (len(self.entries)>=self.max_entries or self.bytes+size>self.max_bytes):
            _,(_,removed)=self.entries.popitem(last=False)
            self.bytes-=removed;self.evictions+=1
        snapshot=_snapshot_clone(cache)
        actual=cache_bytes(snapshot)
        if actual!=size:
            raise RuntimeError('Prefix snapshot storage accounting mismatch')
        self.entries[key]=(snapshot,size);self.bytes+=size
        return True

    def clear(self):
        self.entries.clear();self.bytes=0

    def info(self):
        return dict(entries=len(self.entries),bytes=self.bytes,max_bytes=self.max_bytes,
                    max_entries=self.max_entries,hits=self.hits,misses=self.misses,evictions=self.evictions)
