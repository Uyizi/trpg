"""Exact teacher-forced loss for batch-one SFT without prompt vocabulary logits.

Only positions preceding supervised tokens need the large vocabulary projection.
The transformer still reads the entire prompt and gradients pass through it.
"""
import torch
import torch.nn.functional as F


def token_log_probs(model,input_ids,labels,attention_mask=None):
    if input_ids.shape[0]!=1:raise ValueError('Selective assistant loss requires microbatch size 1')
    targets=labels[:,1:]
    positions=torch.where(targets[0]!=-100)[0]
    if not positions.numel():raise ValueError('No supervised assistant tokens')
    output=model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,logits_to_keep=positions)
    return -F.cross_entropy(output.logits[0].float(),targets[0,positions],reduction='none')
