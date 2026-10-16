"""Qwen3.5 language inference with the original trained checkpoint key layout.

The official text CausalLM moves ``model.language_model`` to ``model``. Keeping
that extra level lets existing Dense and expert LoRA checkpoints load unchanged.
Only the upstream text decoder is constructed; visual tensors in the original
safetensors checkpoint have no destination and are skipped by Transformers.
"""
from torch import nn
from transformers import Qwen3_5Config
from transformers.generation import GenerationMixin
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
)
from transformers.utils import can_return_tuple


_MULTIMODAL_INPUTS = (
    'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw',
    'mm_token_type_ids',
)


def _reject_multimodal(inputs):
    for name in _MULTIMODAL_INPUTS:
        if inputs.get(name) is not None:
            raise ValueError(f'This model is text-only; {name} is unsupported')


class TextOnlyQwen3_5Model(Qwen3_5PreTrainedModel):
    config_class = Qwen3_5Config
    input_modalities = ('text',)
    accepts_loss_kwargs = False
    _no_split_modules = ['Qwen3_5DecoderLayer']

    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen3_5TextModel(config.text_config)
        self.post_init()

    @can_return_tuple
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, **kwargs):
        _reject_multimodal(kwargs)
        # The parent LM passes these optional arguments even for plain text.
        for name in _MULTIMODAL_INPUTS:
            kwargs.pop(name, None)
        return self.language_model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, **kwargs,
        )


class TextOnlyQwen3_5ForCausalLM(Qwen3_5ForConditionalGeneration):
    """Original text forward/loss and adapter paths, without a vision encoder."""

    config_class = Qwen3_5Config
    input_modalities = ('text',)
    _no_split_modules = ['Qwen3_5DecoderLayer']
    _keys_to_ignore_on_load_unexpected = [r'^mtp.*', r'^model\.visual\.']

    def __init__(self, config):
        # Deliberately skip the multimodal constructor: even a temporary visual
        # module would allocate parameters before NF4 placement/upcycling.
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = TextOnlyQwen3_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size,
                                 config.text_config.vocab_size, bias=False)
        self.post_init()

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        _reject_multimodal(model_kwargs)
        # The text decoder expands ordinary positions for its rotary embedding.
        return GenerationMixin._prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs)

    @staticmethod
    def _expand_inputs_for_generation(expand_size=1, is_encoder_decoder=False,
                                      input_ids=None, **model_kwargs):
        _reject_multimodal(model_kwargs)
        positions = model_kwargs.get('position_ids')
        if positions is not None and positions.ndim == 3:
            model_kwargs.pop('position_ids')
        input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids, **model_kwargs,
        )
        if positions is not None and positions.ndim == 3:
            # Explicit Qwen positions are [rotary axes, batch, tokens].
            model_kwargs['position_ids'] = positions.repeat_interleave(expand_size, dim=1)
        return input_ids, model_kwargs

    def get_image_features(self, *args, **kwargs):
        raise ValueError('This model is text-only; image input is unsupported')

    def get_video_features(self, *args, **kwargs):
        raise ValueError('This model is text-only; video input is unsupported')
