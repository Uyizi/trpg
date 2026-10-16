"""Text-only chunked prefill and exact-greedy prompt-lookup speculation.

Hybrid recurrent caches cannot simply be cropped after rejected drafts. Snapshot
and restore the complete cache, then replay the accepted prefix plus correction.
Prompt lookup is optional; sampling uses ordinary target-model decoding.
"""
import copy,struct
import torch
from transformers import DynamicCache


def draft_from_prompt(tokens,count):
    if count<=0 or len(tokens)<8:return []
    data=struct.pack(f'<{len(tokens)}I',*tokens)
    for n in (4,3,2):
        pattern=data[-4*n:]
        offset=data.rfind(pattern,0,len(data)-4*n)
        if offset>=0 and offset%4==0:
            start=offset//4+n
            return tokens[start:min(start+count,len(tokens))]
    return []


@torch.inference_mode()
def generate(model,input_ids,max_new_tokens,*,temperature=.7,chunk_size=2048,speculative=0,streamer=None):
    if speculative and temperature!=0:raise ValueError('推测解码当前支持贪心模式，请同时设置 --temperature 0')
    text_config=model.config.get_text_config()
    cache=DynamicCache(config=text_config)
    device=input_ids.device
    stats={'target_calls':0,'proposed_tokens':0,'accepted_tokens':0,'prefill_chunk_size':chunk_size}
    def forward(tokens,cache,start,keep=1):
        ids=torch.tensor([tokens],dtype=torch.long,device=device)
        # Text-only mRoPE uses identical temporal/height/width positions.
        positions=torch.arange(start,start+len(tokens),device=device).view(1,1,-1).expand(3,1,-1)
        out=model(input_ids=ids,attention_mask=torch.ones((1,start+len(tokens)),dtype=torch.long,device=device),
                  position_ids=positions,past_key_values=cache,use_cache=True,logits_to_keep=keep)
        stats['target_calls']+=1
        return out.logits[0],out.past_key_values
    original=input_ids[0].tolist();tokens=list(original)
    for start in range(0,len(tokens),chunk_size):
        logits,cache=forward(tokens[start:start+chunk_size],cache,start)
    next_logits=logits[-1]
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos or [])
    generated=[]
    if streamer is not None:streamer.put(input_ids.cpu())
    def emit(values):
        for value in values:
            generated.append(value);tokens.append(value)
            if streamer is not None:streamer.put(torch.tensor([value]))
            if value in eos or len(generated)>=max_new_tokens:return True
        return False
    while len(generated)<max_new_tokens:
        proposals=draft_from_prompt(tokens,min(speculative,max_new_tokens-len(generated)))
        expected=int(next_logits.argmax())
        if proposals and proposals[0]==expected:
            before=copy.deepcopy(cache)
            logits,cache=forward(proposals,cache,len(tokens),keep=0)
            predictions=[expected]+logits[:-1].argmax(-1).tolist()
            accepted=0
            for proposed,predicted in zip(proposals,predictions):
                if proposed!=predicted:break
                accepted+=1
            stats['proposed_tokens']+=len(proposals);stats['accepted_tokens']+=accepted
            if accepted==len(proposals):
                del before
                if emit(proposals):break
                next_logits=logits[-1]
            else:
                values=proposals[:accepted]+[predictions[accepted]]
                cache=before
                logits,cache=forward(values,cache,len(tokens))
                if emit(values):break
                next_logits=logits[-1]
        else:
            if temperature>0:
                scores=next_logits.float()/temperature
                top_values,top_indices=torch.topk(scores,min(20,len(scores)))
                probs=torch.softmax(top_values,dim=-1)
                probs.masked_fill_(probs.cumsum(-1)-probs>0.8,0)
                token=int(top_indices[torch.multinomial(probs,1)])
            else:token=expected
            if emit([token]):break
            logits,cache=forward([token],cache,len(tokens)-1)
            next_logits=logits[-1]
    if streamer is not None:streamer.end()
    return torch.tensor([original+generated],device=device),stats
