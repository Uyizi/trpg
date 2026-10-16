"""Bounded, opt-in CUDA Graphs around stateless frozen Top1Experts decode.

This does not capture attention, KV caches, or recurrent state. Configuration and
CPU fallbacks do not query CUDA. Mutating weights through ``.data``, raw storage,
or another thread while inference runs is outside the frozen-weight contract.
"""
from collections import Counter
import math
import threading

import torch
from torch import nn

from .qwen35_moe import Top1Experts


_DEFAULT_BYTES = 256 * 1024 * 1024


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # An inference tensor has no mutation counter. It cannot be guarded.
        raise ValueError('unversioned-weight') from None


def _tensor_state(tensor):
    return (tensor.data_ptr(), _version(tensor), tensor.device, tensor.dtype,
            tuple(tensor.shape), tuple(tensor.stride()), tensor.requires_grad)


def _has_hooks(module):
    return (module._forward_hooks or module._forward_pre_hooks
            or module._backward_hooks or module._backward_pre_hooks)


def _global_hooks():
    from torch.nn.modules import module
    return (module._global_forward_hooks or module._global_forward_pre_hooks
            or module._global_backward_hooks or module._global_backward_pre_hooks)


class _Snapshot:
    """Cache only this MLP's bindings; never walk the full model on replay."""
    def __init__(self, module):
        from bitsandbytes.nn import Linear4bit
        from peft.tuners.lora import Linear4bit as LoraLinear4bit
        from peft.tuners.lora.layer import Linear as LoraLinear
        if (type(module) is not Top1Experts or not module.shared_frozen
                or not module.fused_lora or len(module.experts) != 2):
            raise ValueError('unsupported-experts')
        self.module = module
        self.nodes = tuple(module.modules())
        self.bindings, self.tensors, self.quant = [], [], []
        self.layers, self.dropouts, self.activations = [], [], []
        from peft.utils.other import ModulesToSaveWrapper
        router = module.router
        if type(router) is nn.Linear:
            self.router_state = None
        elif type(router) is ModulesToSaveWrapper:
            if any(type(value) is not nn.Linear for value in router.modules_to_save.values()):
                raise ValueError('unsupported-router')
            self.router_state = (router.disable_adapters, tuple(router.active_adapters), frozenset(router._adapters))
        else:
            raise ValueError('unsupported-router')
        seen = set()
        for node in self.nodes:
            if node.training or _has_hooks(node):
                raise ValueError('training-or-hooks')
            self.bindings.append((node, tuple(node._modules.items()),
                                  tuple(node._parameters.items()), tuple(node._buffers.items()),
                                  node.__dict__.get('forward')))
            for tensor in (*node._parameters.values(), *node._buffers.values()):
                if tensor is not None and id(tensor) not in seen:
                    seen.add(id(tensor))
                    self.tensors.append((tensor, _tensor_state(tensor)))
        if _global_hooks():
            raise ValueError('global-hooks')
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            left, right = (getattr(expert, name) for expert in module.experts)
            for layer in (left, right):
                if (type(layer) not in (LoraLinear4bit, LoraLinear) or type(layer.base_layer) is not Linear4bit
                        or layer.disable_adapters or layer.merged or layer.lora_variant
                        or len(layer.active_adapters) != 1):
                    raise ValueError('unsupported-adapter')
                adapter = layer.active_adapters[0]
                if adapter not in layer.lora_A or adapter not in layer.lora_B:
                    raise ValueError('missing-adapter')
                a, b = layer.lora_A[adapter], layer.lora_B[adapter]
                dropout = layer.lora_dropout[adapter]
                scale = layer.scaling[adapter]
                if (type(a) is not nn.Linear or type(b) is not nn.Linear
                        or a.bias is not None or b.bias is not None
                        or a.weight.dtype != torch.float32 or b.weight.dtype != torch.float32
                        or not 1 <= a.weight.shape[0] <= 64
                        or type(dropout) not in (nn.Dropout, nn.Identity)
                        or dropout.training or not math.isfinite(scale)):
                    raise ValueError('unsupported-lora')
                self.layers.append((layer, adapter, scale, layer.base_layer.compute_dtype,
                                    layer.base_layer.compute_type_is_set))
                if isinstance(dropout, nn.Dropout):
                    self.dropouts.append((dropout, dropout.p))
                base = layer.base_layer
                state = base.weight.quant_state
                if state is None or state.quant_type != 'nf4':
                    raise ValueError('not-nf4')
                self.quant.append((base, state, base.quant_state))
                self._quant_state(state, seen)
            if left.base_layer.weight is not right.base_layer.weight:
                raise ValueError('base-not-shared')
        for expert in module.experts:
            activation = expert.act_fn
            # Known stateless activation only; custom callbacks may have effects.
            from transformers.activations import SiLUActivation
            if activation is not torch.nn.functional.silu and type(activation) not in (nn.SiLU, SiLUActivation):
                raise ValueError('unsupported-activation')
            self.activations.append((expert, activation, getattr(activation, 'inplace', None)))
        if any(t.requires_grad or not t.is_cuda or not t.is_contiguous()
               for t, _ in self.tensors):
            raise ValueError('non-frozen-or-layout')
        self.strength = module.routing_strength

    def _quant_state(self, state, seen):
        # Quantization metadata includes nested double-quantization tensors.
        fields = ('absmax', 'code', 'blocksize', 'quant_type', 'dtype', 'shape', 'offset', 'state2', 'nested')
        values = tuple((name, getattr(state, name, None)) for name in fields)
        self.quant.append((state, values))
        for _, value in values:
            if isinstance(value, torch.Tensor) and id(value) not in seen:
                seen.add(id(value))
                self.tensors.append((value, _tensor_state(value)))
        if state.state2 is not None:
            self._quant_state(state.state2, seen)

    def valid(self):
        module = self.module
        if (not module.shared_frozen or not module.fused_lora
                or module.routing_strength != self.strength or _global_hooks()):
            return False
        for node, children, parameters, buffers, forward in self.bindings:
            if (node.training or _has_hooks(node)
                    or node.__dict__.get('forward') is not forward
                    or len(node._modules) != len(children)
                    or len(node._parameters) != len(parameters) or len(node._buffers) != len(buffers)):
                return False
            for mapping, items in ((node._modules, children), (node._parameters, parameters), (node._buffers, buffers)):
                if any(mapping.get(key) is not value for key, value in items):
                    return False
        for layer, adapter, scale, dtype, compute_set in self.layers:
            if (layer.disable_adapters or layer.merged or layer.lora_variant
                    or layer.active_adapters != [adapter] or layer.scaling.get(adapter) != scale
                    or layer.base_layer.compute_dtype != dtype
                    or layer.base_layer.compute_type_is_set != compute_set):
                return False
        for entry in self.quant:
            if len(entry) == 3:
                base, state, base_state = entry
                if base.weight.quant_state is not state or base.quant_state is not base_state:
                    return False
            else:
                state, values = entry
                for name, value in values:
                    current = getattr(state, name, None)
                    if isinstance(value, torch.Tensor) or name == 'state2':
                        if current is not value:
                            return False
                    elif current != value:
                        return False
        if any(node.p != p for node, p in self.dropouts):
            return False
        if self.router_state is not None:
            router = module.router
            if (router.disable_adapters, tuple(router.active_adapters), frozenset(router._adapters)) != self.router_state:
                return False
        if any(expert.act_fn is not activation or getattr(activation, 'inplace', None) != inplace
               for expert, activation, inplace in self.activations):
            return False
        return all(_tensor_state(tensor) == state for tensor, state in self.tensors)


class _Layer:
    def __init__(self, manager, module):
        self.manager, self.module, self.original = manager, module, module.forward
        self.graph = self.input = self.output = self.snapshot = self.stream = None
        self.memory_bytes = 0
        self.failed = False
        self.replays = self.captures = 0
        self.fallbacks = Counter()
        self.error = None
        self.wrapper = self.forward

    def release(self):
        if self.graph is not None:
            # Configuration is infrequent; drain outstanding use before freeing.
            self.stream.synchronize()
            self.graph.reset()
        self.manager.memory_bytes -= self.memory_bytes
        self.memory_bytes = 0
        self.graph = self.input = self.output = self.snapshot = self.stream = None

    def fallback(self, reason, *args, **kwargs):
        self.fallbacks[reason] += 1
        return self.original(*args, **kwargs)

    def forward(self, *args, **kwargs):
        hidden = args[0] if len(args) == 1 and not kwargs else kwargs.get('hidden') if not args and len(kwargs) == 1 else None
        if (not isinstance(hidden, torch.Tensor) or not hidden.is_cuda
                or hidden.ndim != 3 or hidden.shape[:2] != (1, 1)
                or hidden.dtype not in (torch.bfloat16, torch.float16)
                or not hidden.is_contiguous() or hidden.requires_grad):
            return self.fallback('input', *args, **kwargs)
        if (self.module.training or torch.is_grad_enabled()
                or torch.is_autocast_enabled('cuda') or self.failed):
            return self.fallback('execution-mode', *args, **kwargs)
        if not self.manager.lock.acquire(blocking=False):
            return self.fallback('concurrent', *args, **kwargs)
        try:
            if torch.cuda.is_current_stream_capturing():
                return self.fallback('outer-capture', *args, **kwargs)
            if self.graph is not None:
                if (hidden.device != self.input.device or hidden.dtype != self.input.dtype
                        or hidden.shape != self.input.shape or not self.snapshot.valid()):
                    self.release()
                    self.failed = True
                    return self.fallback('changed-state', *args, **kwargs)
                if torch.cuda.current_stream(hidden.device) != self.stream:
                    return self.fallback('stream', *args, **kwargs)
            else:
                if (sum(layer.graph is not None for layer in self.manager.layers) >= self.manager.max_graphs
                        or self.manager.memory_bytes >= self.manager.max_memory_bytes):
                    return self.fallback('budget', *args, **kwargs)
                try:
                    self.capture(hidden)
                except Exception as error:
                    self.error = f'{type(error).__name__}: {error}'
                    self.failed = True
                    self.release()
                    return self.fallback('capture-failed', *args, **kwargs)
            self.input.copy_(hidden)
            self.graph.replay()
            self.replays += 1
            self.module.fused_lora_projections += 3
            # A caller may retain the result across the next token/layer replay.
            return self.output.clone()
        finally:
            self.manager.lock.release()

    def capture(self, hidden):
        # Reject before running any warmup, so hooks never fire extra times.
        _Snapshot(self.module)
        device = hidden.device
        before = torch.cuda.memory_reserved(device)
        counter = self.module.fused_lora_projections
        try:
            self.stream = torch.cuda.current_stream(device)
            self.input = torch.empty_like(hidden)
            self.input.copy_(hidden)
            warmup = torch.cuda.Stream(device=device)
            warmup.wait_stream(self.stream)
            try:
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        self.original(self.input)
            finally:
                # A Python-side warmup failure may occur after kernels queued.
                # Drain that stream before fallback can free/reuse static input.
                self.stream.wait_stream(warmup)
                warmup.synchronize()
            self.snapshot = _Snapshot(self.module)
            self.graph = torch.cuda.CUDAGraph()
            start_counter = self.module.fused_lora_projections
            with torch.cuda.graph(self.graph):
                self.output = self.original(self.input)
            if self.module.fused_lora_projections - start_counter != 3:
                raise ValueError('fused-projections-not-used')
            self.memory_bytes = max(0, torch.cuda.memory_reserved(device) - before)
            self.manager.memory_bytes += self.memory_bytes
            if self.manager.memory_bytes > self.manager.max_memory_bytes:
                raise ValueError('graph-memory-budget')
            self.captures += 1
        finally:
            self.module.fused_lora_projections = counter


class _Manager:
    def __init__(self, model, max_graphs, max_memory_bytes):
        self.max_graphs, self.max_memory_bytes = max_graphs, max_memory_bytes
        self.memory_bytes = 0
        self.lock = threading.Lock()
        self.layers = [_Layer(self, module) for module in model.modules() if type(module) is Top1Experts]


def configure_cuda_graph_decode(model, enabled=True, *, max_graphs=64, max_memory_bytes=_DEFAULT_BYTES):
    """Install/restore lazy graphs. Reconfigure off/on after intentional mutation.

    The owner must serialize model configuration and frozen inference. Only
    exact B1Q1 BF16/FP16 stateless MLP calls are candidates. Each layer owns one
    pool and one input shape/dtype. No CUDA operation occurs during enable.
    """
    if max_graphs <= 0 or max_memory_bytes <= 0:
        raise ValueError('CUDA graph limits must be positive')
    manager = getattr(model, '_cuda_graph_decode', None)
    if manager is not None:
        if enabled and (manager.max_graphs, manager.max_memory_bytes) == (max_graphs, max_memory_bytes):
            return cuda_graph_decode_info(model)
        for layer in manager.layers:
            if layer.module.forward == layer.wrapper:
                layer.module.forward = layer.original
            layer.release()
        delattr(model, '_cuda_graph_decode')
    if enabled:
        manager = _Manager(model, max_graphs, max_memory_bytes)
        model._cuda_graph_decode = manager
        for layer in manager.layers:
            layer.module.forward = layer.wrapper
    return cuda_graph_decode_info(model)


def cuda_graph_decode_info(model):
    """Return counters without querying/synchronizing CUDA or copying tensors."""
    manager = getattr(model, '_cuda_graph_decode', None)
    layers = manager.layers if manager is not None else ()
    fallbacks = Counter()
    for layer in layers:
        fallbacks.update(layer.fallbacks)
    return dict(backend='frozen-top1-cuda-graph', configured_layers=len(layers),
                captured_layers=sum(layer.graph is not None for layer in layers),
                captures=sum(layer.captures for layer in layers), replays=sum(layer.replays for layer in layers),
                memory_bytes=manager.memory_bytes if manager else 0,
                max_graphs=manager.max_graphs if manager else 0,
                max_memory_bytes=manager.max_memory_bytes if manager else 0,
                fallbacks=dict(fallbacks), errors=[layer.error for layer in layers if layer.error],
                scope='stateless frozen MLP only; no attention/KV/recurrent state')
