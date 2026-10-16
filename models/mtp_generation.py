"""Opt-in greedy MTP verification: one target token plus one draft per call.

Only target-verified tokens reach the stream. A rejected block restores the
complete hybrid state and replays its known target token. No sampling/grammar,
cross-request cache reuse, unverified hidden chaining, training or downloads.
"""
import math
import time

import torch
from torch.nn.attention.bias import causal_lower_right

from .hybrid_cache_transaction import HybridCacheTransaction
from .mtp_expert_scope import expert_verification_scope
from .mtp_verification_kernels import mtp_verification_scope
from .qwen35_mtp import TextHiddenCapture
from .quantized_kv import QuantizedAttentionLayer, cache_bytes, make_cache


def crosses_quantized_flush(cache):
    """A singleton would compact after token 1, changing token 2's history."""
    return any(isinstance(layer,QuantizedAttentionLayer) and layer.keys is not None
               and layer.keys.shape[-2]>=255 for layer in cache.layers)


class MTPGenerator:
    def __init__(self,model,head,*,kv_mode='q8',chunk_size=1024,fast_verification=True):
        if kv_mode not in ('bf16','q8','q4') or chunk_size<0:
            raise ValueError('Unsupported MTP KV mode or prefill chunk size')
        if model.training or head.training:
            raise ValueError('MTP verification requires eval mode')
        self.model,self.head=model,head
        self.kv_mode,self.chunk_size=kv_mode,chunk_size
        self.fast_verification=fast_verification
        from .cached_generation import enable_structured_gqa
        enable_structured_gqa()
        norms=[m for name,m in model.named_modules() if name.endswith('language_model.norm')]
        if len(norms)!=1:raise ValueError('Need exactly one target text final norm')
        self.final_norm=norms[0]
        backbones=[m for name,m in model.named_modules() if name.endswith('language_model')]
        if len(backbones)!=1:raise ValueError('Need exactly one target text backbone')
        self.hidden_module=backbones[0]
        self.clear()

    def clear(self):
        self.cache=None
        self.prefix=[]

    @torch.inference_mode()
    def generate(self,input_ids,max_new_tokens,*,temperature,streamer=None,
                 prefix_cache_key=None,prefix_length=0,constraint=None):
        if not math.isfinite(temperature) or temperature!=0:
            raise ValueError('原生 MTP 当前仅支持 --temperature 0')
        if constraint is not None or prefix_cache_key is not None or prefix_length:
            raise ValueError('原生 MTP 尚不支持结构化约束或固定提示缓存，请关闭 --mtp')
        if input_ids.ndim!=2 or input_ids.shape[0]!=1 or not input_ids.shape[1] or input_ids.dtype!=torch.long:
            raise ValueError('MTP requires nonempty, unpadded int64 [1,tokens] input')
        if type(max_new_tokens)is not int or max_new_tokens<1:
            raise ValueError('max_new_tokens must be a positive integer')
        self.clear()
        ids=input_ids[0].tolist();device=input_ids.device
        stats={'backend':'native-mtp-greedy-two-token','kv_mode':self.kv_mode,
               'reused_tokens':0,'prefill_tokens':len(ids),'target_calls':0,
               'verification_calls':0,'replay_calls':0,'head_calls':0,
               'proposed_tokens':0,'accepted_tokens':0,'quantized_boundary_steps':0,
               'snapshot_copied_bytes_peak':0,'sampling_backend':'greedy-target-verified'}
        stats['fast_verification']=self.fast_verification
        from .prefill_policy import choose_prefill_chunk
        available=torch.cuda.mem_get_info(device)[0] if device.type=='cuda' and self.chunk_size==0 else None
        chunk=choose_prefill_chunk(len(ids),self.chunk_size,available)
        stats['prefill_chunk_size']=chunk
        eos=self.model.generation_config.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos or [])
        generated=[];draft_cache=None;transaction=None
        started=time.monotonic()
        positions=torch.arange(len(ids)+max_new_tokens+1,device=device)[None]
        self.cache=make_cache(self.model.config.get_text_config(),self.kv_mode)

        def forward(tokens,*,keep=1,tensor=None):
            start=len(self.prefix)
            if tensor is None:tensor=torch.tensor([tokens],device=device,dtype=torch.long)
            masks={'full_attention':causal_lower_right(len(tokens),start+len(tokens)),
                   'linear_attention':None}
            verifying=len(tokens)==2 and start>=len(ids)
            if verifying:
                with expert_verification_scope(self.model,self.fast_verification),mtp_verification_scope(self.fast_verification):
                    out=self.model(input_ids=tensor,position_ids=positions[:,start:start+len(tokens)][None].expand(4,-1,-1),
                                   attention_mask=masks,past_key_values=self.cache,use_cache=True,logits_to_keep=keep)
            else:
                out=self.model(input_ids=tensor,position_ids=positions[:,start:start+len(tokens)][None].expand(4,-1,-1),
                               attention_mask=masks,past_key_values=self.cache,use_cache=True,logits_to_keep=keep)
            self.cache=out.past_key_values
            self.prefix.extend(tokens)
            stats['target_calls']+=1
            return out.logits[0]

        def head_forward(tokens,hidden,start):
            nonlocal draft_cache
            out=self.head(torch.tensor([tokens],device=device,dtype=torch.long),hidden,
                          positions[:,start:start+len(tokens)],past_key_values=draft_cache,
                          use_cache=True,logits_to_keep=1)
            draft_cache=out.past_key_values;stats['head_calls']+=1
            return out.logits[0,-1]

        def emit(token):
            generated.append(token)
            if streamer is not None:streamer.put(torch.tensor([token]))
            return token in eos or len(generated)==max_new_tokens

        try:
            with TextHiddenCapture(self.hidden_module) as capture:
                # Head prefill is chunked along with target. Earlier chunks use
                # the next prompt token as their final shifted embedding.
                for start in range(0,len(ids),chunk):
                    end=min(start+chunk,len(ids))
                    logits=forward(ids[start:end],tensor=input_ids[:,start:end])
                    pending=int(logits[-1].argmax())
                    if max_new_tokens>1:
                        tail=ids[end] if end<len(ids) else pending
                        shifted=ids[start+1:end]+[tail]
                        draft_logits=head_forward(shifted,capture.hidden_states,start)
                if device.type=='cuda':torch.cuda.synchronize(device)
                stats['prefill_seconds']=round(time.monotonic()-started,4)
                decode_start=time.monotonic()
                if streamer is not None:streamer.put(input_ids.cpu())
                while True:
                    # Pending is already determined by the target. A draft is
                    # streamed only after the batched target call approves it.
                    if emit(pending):break
                    start=len(self.prefix)
                    if crosses_quantized_flush(self.cache):
                        stats['quantized_boundary_steps']+=1
                        logits=forward([pending])
                        accepted_inputs=[pending]
                        pending=int(logits[-1].argmax())
                        hidden=capture.hidden_states
                    else:
                        proposal=int(draft_logits.argmax())
                        transaction=HybridCacheTransaction(self.cache)
                        stats['snapshot_copied_bytes_peak']=max(stats['snapshot_copied_bytes_peak'],
                                                              getattr(transaction,'copied_bytes',0))
                        logits=forward([pending,proposal],keep=0)
                        stats['verification_calls']+=1;stats['proposed_tokens']+=1
                        if int(logits[0].argmax())==proposal:
                            transaction.commit();transaction=None
                            stats['accepted_tokens']+=1
                            accepted_inputs=[pending,proposal]
                            pending=int(logits[1].argmax())
                            hidden=capture.hidden_states
                            if emit(proposal):break
                        else:
                            transaction.rollback();transaction=None
                            del self.prefix[start:]
                            logits=forward([pending])
                            stats['replay_calls']+=1
                            accepted_inputs=[pending]
                            # Singleton replay defines the correction. Its
                            # kernels can round differently from the batch.
                            pending=int(logits[-1].argmax())
                            hidden=capture.hidden_states
                    if pending not in eos and len(generated)+1<max_new_tokens:
                        draft_logits=head_forward(accepted_inputs[1:]+[pending],hidden,start)
                elapsed=time.monotonic()-decode_start
                stats['decode_seconds']=round(elapsed,4)
                stats['decode_tokens_per_second']=round(len(generated)/max(elapsed,1e-9),3)
                stats['cache_mib']=round(cache_bytes(self.cache)/2**20,3)
                stats['output_tokens']=len(generated)
                stats['acceptance_rate']=stats['accepted_tokens']/max(stats['proposed_tokens'],1)
                return torch.tensor([ids+generated],device=device),stats
        finally:
            if transaction is not None:transaction.rollback()
            self.clear()
            if streamer is not None:streamer.end()
