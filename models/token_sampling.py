"""Validated single-vector sampling with one explicit device-to-host transfer.

The device-side token is retained for the next model forward, avoiding a second
CPU-to-device copy. Torch's multinomial implementation may perform its own
internal synchronization; this module does not replace its RNG algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch


class TokenSamplingError(ValueError):
    """No safe next token can be selected from the supplied scores."""


@dataclass(frozen=True, slots=True)
class SampledToken:
    token_id: int
    # Shape [1], dtype long, same device as logits; view(1, 1) for model input.
    device_token: torch.Tensor


@torch.inference_mode()
def sample_token(logits: torch.Tensor, *, temperature: float,
                 generator: torch.Generator | None = None) -> SampledToken:
    """Select one token, rejecting invalid scores before returning any result.

    Positive temperature preserves the generator's top-20, top-p .8 rule and
    torch.multinomial RNG. Nonpositive temperature uses first-maximum greedy
    selection. Negative infinity is allowed as a mask; all-masked scores, NaN,
    positive infinity, and overflow during float32 temperature scaling fail
    closed. Inputs are not mutated. Failed stochastic selection may consume an
    RNG draw, but its placeholder token is never returned to the caller.

    Masking callers can defer their finite-score host check to this function,
    and must invalidate their request constraint if TokenSamplingError escapes.
    """
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() == 0:
        raise TokenSamplingError('sampling expects one nonempty logits vector')
    if not logits.is_floating_point():
        raise TokenSamplingError('sampling expects floating-point logits')
    if not isinstance(temperature, Real) or not math.isfinite(temperature):
        raise TokenSamplingError('temperature must be a finite real number')

    if temperature > 0:
        # Retain division before topk: moving it afterwards changes float32
        # overflow/tie behavior and can silently change seeded token selection.
        scaled = logits.float() / temperature
        valid = scaled.amax().isfinite()
        values, indices = torch.topk(scaled, min(20, logits.numel()))
        # Give multinomial legal input even for a failed row; return its failure
        # flag together with the token instead of synchronizing to branch here.
        values = torch.where(valid, values, 0.)
        probs = values.softmax(-1)
        probs.masked_fill_(probs.cumsum(-1) - probs > .8, 0)
        token = indices.gather(0, torch.multinomial(probs, 1, generator=generator))
    else:
        maximum, index = torch.max(logits, dim=0)
        valid = maximum.isfinite()
        token = index.reshape(1)

    # A finite maximum proves at least one finite score and excludes NaN/+inf.
    # This is the only explicit host transfer/inspection on the CUDA path.
    # -1 carries the validity flag in a value disjoint from every vocabulary ID.
    # Encoding it avoids launching separate bool->long and concatenation kernels.
    token_id = torch.where(valid, token, -1).cpu().tolist()[0]
    if token_id < 0:
        raise TokenSamplingError('no finite next-token score or invalid NaN/positive infinity scores')
    return SampledToken(token_id, token)
