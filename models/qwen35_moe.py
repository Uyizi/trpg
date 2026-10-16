"""Two-expert Top-1 upcycling of Qwen3.5 language FFNs.

Derived from Sparse Upcycling (2212.05055) and Switch Transformers (2101.03961).
Language attention/DeltaNet are unchanged. No token dropping or capacity truncation.
The runtime loads the text decoder only, keeping trained adapter key paths intact.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Top1Experts(nn.Module):
    def __init__(self, dense, *, share_frozen=False):
        super().__init__()
        # Distinct modules let PEFT install distinct LoRAs; only their frozen
        # parameters/buffers share storage. Training keeps its original copies.
        memo = {id(t): t for t in list(dense.parameters()) + list(dense.buffers())} if share_frozen else {}
        self.experts = nn.ModuleList([dense, copy.deepcopy(dense, memo)])
        self.shared_frozen = share_frozen
        hidden = dense.gate_proj.in_features
        self.router = nn.Linear(hidden, 2, bias=False, device=dense.gate_proj.weight.device,
                                dtype=torch.float32)
        nn.init.normal_(self.router.weight, std=1e-4)
        self.balance_loss = None
        self.token_mask = None
        self.last_counts = None
        # Kept outside PEFT adapter tensors; saved with every routing checkpoint.
        self.register_buffer('routing_bias', torch.zeros(2, device=dense.gate_proj.weight.device), persistent=False)
        self.register_buffer('routing_counts', torch.zeros(2, device=dense.gate_proj.weight.device), persistent=False)
        self.routing_strength = 0.0  # v1 stays bit-for-bit on its original selection rule.
        self.collect_routing = False
        self.fused_lora = False
        self.fused_lora_projections = 0
        self.tinybatch_verification = False

    def forward(self, hidden):
        shape = hidden.shape
        x = hidden.reshape(-1, shape[-1])
        # Keep routing probabilities in FP32 even under BF16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            probabilities = F.softmax(self.router(x.float()), dim=-1)
        selected = (probabilities + self.routing_strength * self.routing_bias.detach()).argmax(dim=-1)
        tiny_verification = (self.tinybatch_verification and x.shape[0] == 2
                             and not torch.is_grad_enabled())
        fast = self.shared_frozen and not self.training and (x.shape[0] == 1 or tiny_verification)
        if fast:
            projections = [getattr(e, n) for e in self.experts for n in ('gate_proj', 'up_proj', 'down_proj')]
            fast = all(hasattr(p, 'base_layer') and hasattr(p.base_layer.weight, 'quant_state')
                       and not p.disable_adapters and not p.merged and not p.lora_variant for p in projections)
        if fast:
            # No torch.where/nonzero CPU synchronization on each decode layer.
            # Evaluate tiny LoRA branches on GPU around one common base GEMM.
            def projection(name, inputs):
                a, b = (getattr(e, name) for e in self.experts)
                if not hasattr(a, 'base_layer'):
                    return a(inputs)
                result = a.base_layer(inputs)
                if self.fused_lora:
                    from .routed_lora import try_routed_lora
                    fused = try_routed_lora(inputs, result, selected, a, b)
                    if fused is not None:
                        self.fused_lora_projections += 1
                        return fused
                for expert_id, layer in enumerate((a, b)):
                    for adapter in layer.active_adapters:
                        if adapter in layer.lora_A:
                            value = layer.lora_B[adapter](layer.lora_A[adapter](
                                layer.lora_dropout[adapter](inputs.to(layer.lora_A[adapter].weight.dtype))))
                            delta = (value * layer.scaling[adapter]).to(result.dtype)
                            result = result + delta * (selected == expert_id)[:, None]
                return result
            hidden = self.experts[0].act_fn(projection('gate_proj', x)) * projection('up_proj', x)
            values = projection('down_proj', hidden)
            gate = 2 * probabilities.gather(1, selected[:, None])
            return (values * gate.to(values.dtype)).view(shape)
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            indices = torch.where(selected == expert_id)[0]
            if indices.numel():
                values = expert(x.index_select(0, indices))
                # Switch uses the selected softmax probability (not Top-1 renormalization).
                # Factor 2 preserves the copied Dense output at uniform initial routing.
                # This is our explicit initialization adaptation, not a paper benchmark claim.
                gate = (2 * probabilities[indices, expert_id]).to(values.dtype)
                out.index_add_(0, indices, values * gate.unsqueeze(-1))
        self.balance_loss = None
        if self.training and torch.is_grad_enabled():
            valid = self.token_mask.reshape(-1).bool() if self.token_mask is not None else torch.ones_like(selected, dtype=torch.bool)
            counts = F.one_hot(selected[valid], 2).float().mean(dim=0)
            mean_prob = probabilities[valid].mean(dim=0)
            self.balance_loss = 2 * (counts.detach() * mean_prob).sum()
            self.last_counts = counts.detach()
            if self.collect_routing:
                with torch.no_grad():
                    self.routing_counts.add_(torch.bincount(selected[valid], minlength=2).float())
        return out.view(shape)


def upcycle(model, *, share_frozen=False):
    for layer in model.model.language_model.layers:
        if isinstance(layer.mlp, Top1Experts):
            raise ValueError('Model already upcycled')
        layer.mlp = Top1Experts(layer.mlp, share_frozen=share_frozen)
    return model


def expert_modules(model):
    return [module for module in model.modules() if isinstance(module, Top1Experts)]


def configure_expert_decode(model, enabled):
    for module in expert_modules(model):
        module.fused_lora = bool(enabled and module.shared_frozen)


def expert_decode_metadata(model):
    modules = expert_modules(model)
    return dict(backend='selected-lora-triton',
                configured_layers=sum(m.fused_lora for m in modules),
                fused_projection_calls=sum(m.fused_lora_projections for m in modules),
                fallback='original PyTorch expert path on unsupported inputs/adapters')


def load_dense(base, adapter=None, *, trainable=False, quantized=False):
    from transformers import BitsAndBytesConfig
    from peft import PeftModel
    from .text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
    options = {}
    if quantized:
        options.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=['lm_head']))
    model = TextOnlyQwen3_5ForCausalLM.from_pretrained(
        base, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', **options)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable, local_files_only=True)
    return model


def build_moe(base, dense_adapter, *, merged_base=None, quantized=False, share_frozen=False):
    cached = None
    if merged_base is not None:
        if quantized and share_frozen:
            from .runtime_weight_cache import validated_cache
            cached = validated_cache(merged_base)
        if cached is not None:
            try:
                model = load_dense(cached, quantized=True)
            except torch.OutOfMemoryError:
                raise
            except (OSError, ValueError, RuntimeError, KeyError):
                cached = None
        if cached is None:
            model = load_dense(merged_base, quantized=quantized)
    else:
        if quantized:
            raise ValueError('Quantize the merged Dense export, never merge adapters into 4-bit weights')
        model = load_dense(base, dense_adapter).merge_and_unload(safe_merge=True)
    model.runtime_weight_cache = str(cached) if cached is not None else None
    return upcycle(model, share_frozen=share_frozen)


def routing_state(model):
    return [{'bias': m.routing_bias.detach().cpu().tolist(), 'strength': m.routing_strength}
            for m in expert_modules(model)]


def restore_routing(model, states):
    modules = expert_modules(model)
    if len(states) != len(modules):
        raise ValueError('Routing state layer count mismatch')
    for module, state in zip(modules, states):
        bias = torch.tensor(state['bias'], device=module.routing_bias.device)
        if bias.shape != (2,) or not torch.isfinite(bias).all():
            raise ValueError('Invalid routing bias')
        module.routing_bias.copy_(bias)
        module.routing_strength = float(state['strength'])


def load_bundle(folder, *, optimize_inference=True):
    """Reconstruct frozen expert copies, then load their trained LoRAs and routers."""
    from peft import PeftModel
    from .runtime_paths import read_bundle
    manifest = read_bundle(folder)
    if manifest.get('format') not in ('qwen35-top1-v1', 'qwen35-top1-nf4-v2') or manifest.get('state') != 'completed':
        raise ValueError('Incomplete or unsupported MoE bundle')
    for name in ('dense_adapter', 'moe_adapter'):
        with (Path(manifest[name]) / 'adapter_model.safetensors').open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != manifest[name + '_sha256']:
                raise ValueError(f'{name} changed after the bundle was published')
    model = build_moe(manifest['base'], manifest['dense_adapter'],
                      merged_base=manifest.get('merged_base'), quantized=manifest.get('quantization') == 'nf4',
                      share_frozen=optimize_inference)
    model = PeftModel.from_pretrained(model, manifest['moe_adapter'], local_files_only=True)
    if 'routing_state' in manifest:
        path = Path(manifest['routing_state'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['routing_sha256']:
            raise ValueError('Routing state changed after publication')
        restore_routing(model, json.loads(path.read_text()))
    model.config.use_cache = True
    model.config.text_config.use_cache = True
    return model.eval(), manifest
