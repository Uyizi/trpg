"""Guarded adapters for installed FLA kernels; never resolve kernels from the Hub.

Transformers 5.17 already selects installed FLA for gated delta attention. These
adapters retain that backend on CUDA and restore the actual Torch references on
CPU or when gradients are required. They also accelerate causal convolution.
"""
from functools import wraps
import inspect

import torch

from .mtp_verification_kernels import mtp_verification_enabled


# These are model-call metadata, not changes to the recurrence/convolution math.
_MODEL_METADATA = frozenset(('use_cache', 'output_attentions', 'output_hidden_states',
                             'return_dict', 'cache_position'))


def _needs_grad(*tensors):
    return torch.is_grad_enabled() and any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in tensors)


def _previous_backend(function):
    try:
        implementation = inspect.getclosurevars(function).nonlocals.get('implementation', function)
    except (TypeError, ValueError):
        implementation = function
    return f'{implementation.__module__}.{implementation.__name__}'


def _packed_lengths(cu_seqlens, query, initial_state, cu_seqlens_cpu=None):
    """Validate before Triton sees offsets that could address outside a tensor."""
    if cu_seqlens is None:
        if cu_seqlens_cpu is not None:
            raise ValueError('cu_seqlens_cpu requires cu_seqlens')
        return None
    if (not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64) or cu_seqlens.numel() < 2):
        raise ValueError('cu_seqlens must be a one-dimensional integer tensor with at least two offsets')
    lengths = cu_seqlens.detach().cpu().tolist()
    if (query.shape[0] != 1 or lengths[0] != 0 or lengths[-1] != query.shape[1]
            or any(end <= start for start, end in zip(lengths, lengths[1:]))):
        raise ValueError('cu_seqlens requires batch=1, increasing offsets, and endpoints 0 and sequence length')
    if initial_state is not None and initial_state.shape[0] != len(lengths) - 1:
        raise ValueError('initial_state must contain one state per packed sequence')
    if cu_seqlens_cpu is not None:
        if (not isinstance(cu_seqlens_cpu, torch.Tensor) or cu_seqlens_cpu.device.type != 'cpu'
                or cu_seqlens_cpu.ndim != 1 or cu_seqlens_cpu.dtype not in (torch.int32, torch.int64)
                or cu_seqlens_cpu.tolist() != lengths):
            raise ValueError('cu_seqlens_cpu must match cu_seqlens')
    return lengths


def _reference_delta(reference, tensors, initial_state, output_final_state, normalized, lengths, options):
    options = dict(options, output_final_state=output_final_state, use_qk_l2norm_in_kernel=normalized)
    if lengths is None:
        return reference(*tensors, initial_state=initial_state, **options)
    parts = [reference(*(tensor[:, start:end] for tensor in tensors),
                       initial_state=None if initial_state is None else initial_state[index:index + 1],
                       **options)
             for index, (start, end) in enumerate(zip(lengths, lengths[1:]))]
    return (torch.cat([part[0] for part in parts], dim=1),
            torch.cat([part[1] for part in parts], dim=0) if output_final_state else None)


def _record_dispatch(stats, path, reason=None):
    stats[path] += 1
    if reason is not None:
        reasons = stats['fallback_reasons']
        reasons[reason] = reasons.get(reason, 0) + 1


def _delta_adapter(original, kernel, chunked, stats):
    reference = inspect.unwrap(original)

    def run(query, key, value, g, beta, initial_state, output_final_state, normalized, chunk_size, kwargs):
        options = dict(kwargs)
        cu_seqlens = options.pop('cu_seqlens', None)
        cu_cpu = options.pop('cu_seqlens_cpu', None)
        if options.pop('cp_context', None) is not None:
            raise NotImplementedError('Local Qwen3.5 adapters do not support context parallelism')
        # These FLA options change the public Torch reference contract. Never
        # silently interpret a differently laid out state or pre-activation gate.
        for name in ('gk', 'gv', 'scale', 'state_v_first', 'transpose_state_layout',
                     'use_gate_in_kernel', 'A_log', 'dt_bias',
                     'use_beta_sigmoid_in_kernel', 'allow_neg_eigval', 'head_first'):
            option = options.pop(name, None)
            if option is not None and option is not False:
                raise ValueError(f'{name} is not supported by the Qwen3.5 Torch reference contract')
        lengths = _packed_lengths(cu_seqlens, query, initial_state, cu_cpu)
        tensors = (query, key, value, g, beta)
        shape = query.shape
        fast = (query.is_cuda and query.dtype in (torch.float16, torch.bfloat16)
                and query.ndim == 4 and query.shape == key.shape
                and value.ndim == 4 and value.shape[:3] == shape[:3]
                and shape[1] > 0 and g.shape == beta.shape == shape[:3]
                and key.dtype == value.dtype == query.dtype
                and g.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and beta.dtype in (torch.float16, torch.bfloat16, torch.float32)
                and all(tensor.device == query.device for tensor in tensors)
                and not _needs_grad(*tensors, initial_state)
                and all(name in _MODEL_METADATA or option is None for name, option in options.items())
                and (not chunked or chunk_size in (16, 32, 64)))
        if initial_state is not None:
            num_sequences = len(lengths) - 1 if lengths else shape[0]
            fast = (fast and initial_state.device == query.device
                    and initial_state.dtype == torch.float32
                    and initial_state.shape == (num_sequences, shape[2], shape[3], value.shape[-1]))
        if not fast:
            if chunked:
                options['chunk_size'] = chunk_size
            unknown = sorted(name for name, option in options.items()
                             if name not in _MODEL_METADATA and name != 'chunk_size' and option is not None)
            if unknown:
                reason = 'unsupported_kwargs:' + ','.join(unknown)
            elif not query.is_cuda:
                reason = 'cpu'
            elif _needs_grad(*tensors, initial_state):
                reason = 'gradients'
            elif query.dtype not in (torch.float16, torch.bfloat16):
                reason = f'query_dtype:{query.dtype}'
            elif initial_state is not None and initial_state.dtype != torch.float32:
                reason = f'initial_state_dtype:{initial_state.dtype}'
            elif chunked and chunk_size not in (16, 32, 64):
                reason = f'chunk_size:{chunk_size}'
            else:
                reason = 'shape_device_or_dtype'
            result = _reference_delta(reference, tensors, initial_state, output_final_state,
                                      normalized, lengths, options)
            _record_dispatch(stats, 'fallback', reason)
            return result
        kernel_options = dict(initial_state=initial_state, output_final_state=output_final_state,
                              use_qk_l2norm_in_kernel=normalized)
        if cu_seqlens is not None:
            kernel_options['cu_seqlens'] = cu_seqlens.to(device=query.device).contiguous()
            if chunked:
                kernel_options['cu_seqlens_cpu'] = cu_cpu
        if chunked:
            kernel_options['chunk_size'] = chunk_size
        # Always use named g/beta: recurrent FLA's fifth positional argument is gk.
        result = kernel(query, key, value, g=g, beta=beta, **kernel_options)
        _record_dispatch(stats, 'fast')
        return result

    if chunked:
        @wraps(original)
        def chunk(query, key, value, g, beta, chunk_size=64, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
            if (mtp_verification_enabled() and query.ndim == 4 and query.shape[1] == 2
                    and initial_state is not None and chunk_size in (16, 32, 64)):
                # FLA's recurrent kernel scans arbitrary T, including T=2.
                # Keep all mathematical validation and CPU/gradient fallbacks
                # in the existing recurrent adapter. Uncached prefill and
                # every call outside the request-local scope stay unchanged.
                from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
                result = modeling.torch_recurrent_gated_delta_rule(
                    query, key, value, g=g, beta=beta, initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
                stats['mtp_recurrent'] = stats.get('mtp_recurrent', 0) + 1
                return result
            return run(query, key, value, g, beta, initial_state, output_final_state,
                       use_qk_l2norm_in_kernel, chunk_size, kwargs)
        return chunk

    @wraps(original)
    def recurrent(query, key, value, g, beta, initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=False, **kwargs):
        return run(query, key, value, g, beta, initial_state, output_final_state,
                   use_qk_l2norm_in_kernel, None, kwargs)
    return recurrent


def install_convolution_kernels():
    """Install local CUDA convolution and delta-rule adapters, once per module.

    The historical entrypoint is retained for existing callers. Imports are local
    and optional: callers can keep their existing ImportError fallback when FLA is
    absent. Actual installed functions and their prior backends are inspectable.
    """
    from fla.modules.conv import causal_conv1d
    from fla.modules.conv.triton.ops import causal_conv1d_update
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    if getattr(modeling, '_local_fla_kernels', None):
        return
    names = ('causal_conv1d_fn', 'causal_conv1d_update',
             'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')
    originals = {name: getattr(modeling, name) for name in names}
    dispatch = {name: {'fast': 0, 'fallback': 0, 'fallback_reasons': {}} for name in names}
    reference_forward = inspect.unwrap(originals['causal_conv1d_fn'])
    reference_update = inspect.unwrap(originals['causal_conv1d_update'])

    @wraps(originals['causal_conv1d_fn'])
    def forward(x, weight, bias=None, activation=None, **kwargs):
        # HF passes cu_seq_lens_q before renaming it for gated delta attention.
        # Keep packed convolution correct with a segmented reference fallback.
        kwargs = dict(kwargs)
        cu_seqlens = kwargs.pop('cu_seqlens', None)
        alias = kwargs.pop('cu_seq_lens_q', None)
        if alias is not None:
            if cu_seqlens is not None and not torch.equal(cu_seqlens.cpu(), alias.cpu()):
                raise ValueError('Conflicting convolution sequence offsets')
            cu_seqlens = alias
        lengths = _packed_lengths(cu_seqlens, x.transpose(1, 2), None,
                                  kwargs.pop('cu_seqlens_cpu', None))
        if kwargs.get('cp_context') is not None:
            raise NotImplementedError('Local Qwen3.5 convolution does not support context parallelism')
        if lengths is not None:
            result = torch.cat([reference_forward(x[..., start:end], weight, bias, activation, **kwargs)
                                for start, end in zip(lengths, lengths[1:])], dim=-1)
            _record_dispatch(dispatch['causal_conv1d_fn'], 'fallback', 'packed_sequences')
            return result
        if (not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or weight.device != x.device or weight.dtype != x.dtype
                or activation not in (None, 'silu', 'swish')
                or _needs_grad(x, weight, bias)
                or any(name not in _MODEL_METADATA and option is not None for name, option in kwargs.items())):
            result = reference_forward(x, weight, bias, activation, **kwargs)
            unknown = sorted(name for name, option in kwargs.items()
                             if name not in _MODEL_METADATA and option is not None)
            reason = ('unsupported_kwargs:' + ','.join(unknown) if unknown else
                      'cpu' if not x.is_cuda else 'gradients' if _needs_grad(x, weight, bias) else
                      'activation_device_or_dtype')
            _record_dispatch(dispatch['causal_conv1d_fn'], 'fallback', reason)
            return result
        output, _ = causal_conv1d(x.transpose(1, 2), weight=weight, bias=bias,
                                 activation=activation, backend='triton')
        _record_dispatch(dispatch['causal_conv1d_fn'], 'fast')
        return output.transpose(1, 2)

    @wraps(originals['causal_conv1d_update'])
    def update(x, state, weight, bias=None, activation=None):
        if (not x.is_cuda or x.shape[-1] != 1 or weight.device != x.device
                or weight.dtype != x.dtype or state.device != x.device or state.dtype != x.dtype
                or activation not in (None, 'silu', 'swish') or _needs_grad(x, state, weight, bias)):
            result = reference_update(x, state, weight, bias, activation)
            reason = ('cpu' if not x.is_cuda else 'gradients' if _needs_grad(x, state, weight, bias) else
                      'sequence_length_activation_device_or_dtype')
            _record_dispatch(dispatch['causal_conv1d_update'], 'fallback', reason)
            return result
        output, _ = causal_conv1d_update(x.transpose(1, 2), state, weight=weight,
                                       bias=bias, activation=activation)
        _record_dispatch(dispatch['causal_conv1d_update'], 'fast')
        return output.transpose(1, 2)

    modeling.causal_conv1d_fn = forward
    modeling.causal_conv1d_update = update
    modeling.torch_chunk_gated_delta_rule = _delta_adapter(
        originals['torch_chunk_gated_delta_rule'], chunk_gated_delta_rule, True, dispatch['torch_chunk_gated_delta_rule'])
    modeling.torch_recurrent_gated_delta_rule = _delta_adapter(
        originals['torch_recurrent_gated_delta_rule'], fused_recurrent_gated_delta_rule, False,
        dispatch['torch_recurrent_gated_delta_rule'])
    modeling._local_fla_dispatch = dispatch
    modeling._local_fla_originals = originals
    modeling._local_fla_previous_backends = {name: _previous_backend(function)
                                            for name, function in originals.items()}
    modeling._local_fla_kernels = names
    modeling._local_fla_conv = True


def kernel_metadata():
    """Describe actual installation without claiming a new delta-rule speedup."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    installed = list(getattr(modeling, '_local_fla_kernels', ()))
    return {'installed': installed, 'backend': 'local-fla-triton' if installed else 'transformers-default',
            'previous_backends': dict(getattr(modeling, '_local_fla_previous_backends', {})),
            'dispatch': {name: dict(fast=counts['fast'], fallback=counts['fallback'],
                                    mtp_recurrent=counts.get('mtp_recurrent', 0),
                                    fallback_reasons=dict(counts['fallback_reasons']))
                         for name, counts in getattr(modeling, '_local_fla_dispatch', {}).items()},
            'fallback': 'torch-reference on CPU, gradients, or unsupported arguments'}
