"""Small CPU checks for language-only loading of the original checkpoint layout."""
import copy
import importlib.util
import inspect
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


def tiny_config():
    return Qwen3_5Config(
        text_config=dict(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, layer_types=['linear_attention', 'full_attention'],
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=2,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                             'partial_rotary_factor': 1.0, 'mrope_section': [4, 2, 2]},
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=48, num_heads=4,
                           patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                           out_hidden_size=32, num_position_embeddings=16),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
        pad_token_id=0, eos_token_id=2,
    )


class TextOnlyQwen35Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # Installed FLA is selected by upstream decorators even for CPU inputs.
        # Compare the actual upstream PyTorch math without invoking Triton.
        from transformers.models.qwen3_5 import modeling_qwen3_5
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            patcher = patch.object(modeling_qwen3_5, name,
                                   inspect.unwrap(getattr(modeling_qwen3_5, name)))
            patcher.start()
            cls.addClassCleanup(patcher.stop)
        torch.manual_seed(17)
        cls.original = Qwen3_5ForConditionalGeneration(tiny_config()).eval()
        cls.ids = torch.tensor([[5, 9, 11, 8]])

    def setUp(self):
        self.original.model.rope_deltas = None

    def model_class(self):
        self.assertIsNotNone(importlib.util.find_spec('models.text_only_qwen35'),
                             'The language-only Qwen3.5 loader must exist')
        from models.text_only_qwen35 import TextOnlyQwen3_5ForCausalLM
        return TextOnlyQwen3_5ForCausalLM

    def text_model(self):
        model = self.model_class()(copy.deepcopy(self.original.config)).eval()
        info = model.load_state_dict(self.original.state_dict(), strict=False)
        self.assertEqual(info.missing_keys, [])
        self.assertTrue(all(key.startswith('model.visual.') for key in info.unexpected_keys))
        return model

    def test_never_constructs_vision_or_allocates_visual_parameters(self):
        model_class = self.model_class()
        with patch.object(Qwen3_5VisionModel, '__init__', side_effect=AssertionError('vision allocated')):
            model = model_class(tiny_config())
        self.assertFalse(hasattr(model.model, 'visual'))
        self.assertFalse(any('visual' in name or 'vision' in name for name, _ in model.named_parameters()))
        self.assertTrue(any(name.startswith('model.language_model.layers.') for name, _ in model.named_parameters()))

    @torch.inference_mode()
    def test_identical_text_logits_loss_and_cached_decode(self):
        model = self.text_model()
        expected = self.original(self.ids, labels=self.ids, use_cache=True)
        actual = model(self.ids, labels=self.ids, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
        for token in (7, 13):
            next_id = torch.tensor([[token]])
            expected = self.original(next_id, past_key_values=expected.past_key_values, use_cache=True)
            actual = model(next_id, past_key_values=actual.past_key_values, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)

    @torch.inference_mode()
    def test_generate_matches_original_including_padding_and_beams(self):
        model = self.text_model()
        ids = torch.tensor([[0, 0, 5, 9], [5, 9, 11, 8]])
        mask = (ids != 0).long()
        for beams in (1, 2):
            options = dict(attention_mask=mask, max_new_tokens=3, do_sample=False, num_beams=beams)
            torch.testing.assert_close(model.generate(ids, **options), self.original.generate(ids, **options))

    @torch.inference_mode()
    def test_tied_embeddings_checkpoint_retains_original_storage_and_logits(self):
        config = tiny_config()
        config.tie_word_embeddings = True
        config.text_config.tie_word_embeddings = True
        original = Qwen3_5ForConditionalGeneration(config).eval()
        with tempfile.TemporaryDirectory() as folder:
            original.save_pretrained(folder)
            text = self.model_class().from_pretrained(folder, local_files_only=True)
            self.assertIs(text.lm_head.weight, text.model.language_model.embed_tokens.weight)
            torch.testing.assert_close(text(self.ids).logits, original(self.ids).logits, rtol=0, atol=0)

    @torch.inference_mode()
    def test_beam_generation_preserves_explicit_qwen_position_axes(self):
        model = self.text_model()
        positions = torch.arange(4).view(1, 1, 4).expand(4, 1, 4)
        options = dict(position_ids=positions, max_new_tokens=2, do_sample=False, num_beams=2)
        torch.testing.assert_close(model.generate(self.ids, **options), self.original.generate(self.ids, **options))

    @torch.inference_mode()
    def test_loads_official_checkpoint_with_original_keys_and_no_vision(self):
        model_class = self.model_class()
        with tempfile.TemporaryDirectory() as folder:
            self.original.save_pretrained(folder)
            with patch.object(Qwen3_5VisionModel, '__init__', side_effect=AssertionError('vision allocated')):
                model, info = model_class.from_pretrained(folder, local_files_only=True,
                                                          output_loading_info=True)
            self.assertEqual(info['missing_keys'], set())
            self.assertEqual(info['unexpected_keys'], set())
            self.assertEqual(info['mismatched_keys'], set())
            torch.testing.assert_close(model(self.ids).logits, self.original(self.ids).logits, rtol=0, atol=0)

    def test_rejects_multimodal_forward_generation_and_feature_requests(self):
        model = self.text_model()
        for name in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'mm_token_type_ids'):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'text-only'):
                    model(self.ids, **{name: torch.ones(1)})
                with self.assertRaisesRegex(ValueError, 'text-only'):
                    model.generate(self.ids, max_new_tokens=1, **{name: torch.ones(1)})
        with self.assertRaisesRegex(ValueError, 'text-only'):
            model.get_image_features(torch.ones(1))
        with self.assertRaisesRegex(ValueError, 'text-only'):
            model.get_video_features(torch.ones(1))

    @torch.inference_mode()
    def test_dense_and_expert_lora_checkpoint_hierarchies_are_preserved(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from models.qwen35_moe import upcycle
        for experts in (False, True):
            with self.subTest(experts=experts), tempfile.TemporaryDirectory() as folder:
                original = copy.deepcopy(self.original)
                text = self.text_model()
                if experts:
                    upcycle(original, share_frozen=True)
                    upcycle(text, share_frozen=True)
                    text.load_state_dict({key: value for key, value in original.state_dict().items()
                                          if not key.startswith('model.visual.')}, strict=True)
                    targets = r'model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)'
                else:
                    targets = r'model\.language_model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)'
                original = get_peft_model(original, LoraConfig(
                    task_type='CAUSAL_LM', r=2, lora_alpha=4, target_modules=targets,
                    modules_to_save=['router'] if experts else None))
                for name, parameter in original.named_parameters():
                    if 'lora_B' in name:
                        parameter.fill_(0.03)
                original.eval().save_pretrained(folder)
                text = PeftModel.from_pretrained(text, folder, local_files_only=True).eval()
                torch.testing.assert_close(text(self.ids).logits, original(self.ids).logits, rtol=0, atol=0)
                if experts:
                    layer = text.base_model.model.model.language_model.layers[0].mlp
                    self.assertIs(layer.experts[0].gate_proj.base_layer.weight,
                                  layer.experts[1].gate_proj.base_layer.weight)

    @torch.inference_mode()
    def test_cached_generator_reuses_exact_text_prefix(self):
        from models.cached_generation import CachedGenerator
        model = self.text_model()
        generator = CachedGenerator(model, kv_mode='bf16', chunk_size=2)
        first, _ = generator.generate(self.ids, 2, temperature=0)
        continuation = torch.cat([first, torch.tensor([[17, 18]])], dim=1)
        actual, stats = generator.generate(continuation, 2, temperature=0)
        expected = self.original.generate(continuation, max_new_tokens=2, do_sample=False)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(stats['reused_tokens'], 0)

    def test_chat_loader_uses_text_model_and_reports_configured_architecture(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from models.pretrained import PretrainedChatModel
        with tempfile.TemporaryDirectory() as folder:
            self.original.save_pretrained(folder)
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({f'tok{i}': i for i in range(64)}, unk_token='tok1')),
                unk_token='tok1', pad_token='tok0', eos_token='tok2',
                chat_template='{% for message in messages %}{{ message.content }}{% endfor %}')
            tokenizer.save_pretrained(folder)
            with patch.object(Qwen3_5VisionModel, '__init__', side_effect=AssertionError('vision allocated')):
                chat = PretrainedChatModel(folder, device='cpu', context_length=64, local_files_only=True,
                                           kv_cache='bf16', fast_kernels=False)
            self.assertIsInstance(chat.model, self.model_class())
            self.assertTrue(chat.description['text_only'])
            self.assertEqual(chat.description['native_context_length'], 128)
            self.assertEqual(chat.description['position_encoding'], 'default')
            self.assertEqual(chat.description['full_attention_layers'], 1)
            self.assertEqual(chat.description['linear_attention_layers'], 1)
            self.assertEqual(chat.description['convolution_kernel'], 'default')

    @torch.inference_mode()
    def test_bundle_loader_preserves_experts_routes_and_source_checksums(self):
        from peft import LoraConfig, get_peft_model
        from models.qwen35_moe import load_bundle, routing_state, upcycle
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            base, dense_path, moe_path = (folder / name for name in ('base', 'dense', 'moe'))
            self.original.save_pretrained(base)
            original = Qwen3_5ForConditionalGeneration.from_pretrained(
                base, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
            dense = get_peft_model(original, LoraConfig(
                task_type='CAUSAL_LM', r=2, target_modules=['gate_proj', 'up_proj', 'down_proj']))
            dense.save_pretrained(dense_path)
            original = dense.merge_and_unload()
            upcycle(original, share_frozen=True)
            original = get_peft_model(original, LoraConfig(
                task_type='CAUSAL_LM', r=2, modules_to_save=['router'],
                target_modules=r'model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)'))
            for name, parameter in original.named_parameters():
                if 'lora_B' in name:
                    parameter.fill_(0.05)
            original.eval().save_pretrained(moe_path)
            routes = routing_state(original)
            routes[0] = {'bias': [0.125, -0.125], 'strength': 0.3}
            route_path = folder / 'routing.json'
            route_path.write_text(json.dumps(routes))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = dict(format='qwen35-top1-v1', state='completed',
                            base=str(base), merged_base=str(base), dense_adapter=str(dense_path),
                            moe_adapter=str(moe_path), routing_state=str(route_path),
                            dense_adapter_sha256=digest(dense_path / 'adapter_model.safetensors'),
                            moe_adapter_sha256=digest(moe_path / 'adapter_model.safetensors'),
                            routing_sha256=digest(route_path))
            (folder / 'moe_manifest.json').write_text(json.dumps(manifest))
            checkpoints = [base / 'model.safetensors', dense_path / 'adapter_model.safetensors',
                           moe_path / 'adapter_model.safetensors', route_path]
            before = [digest(path) for path in checkpoints]
            with patch.object(Qwen3_5VisionModel, '__init__', side_effect=AssertionError('vision allocated')):
                actual, _ = load_bundle(folder)
            self.assertFalse(hasattr(actual.base_model.model.model, 'visual'))
            self.assertEqual(routing_state(actual), routes)
            from models.qwen35_moe import restore_routing
            restore_routing(original, routes)
            for name, value in actual.state_dict().items():
                if '.router.original_module.' in name:
                    continue  # PEFT uses the restored modules_to_save router.
                torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0, msg=name)
            torch.testing.assert_close(actual(self.ids).logits, original(self.ids).logits, rtol=0, atol=0)
            self.assertEqual([digest(path) for path in checkpoints], before)
            with (moe_path / 'adapter_model.safetensors').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed after the bundle was published'):
                load_bundle(folder)


if __name__ == '__main__':
    unittest.main()
