"""CPU coverage of cache provenance, damage detection, and inference-only use."""
import importlib
import json
import os
import fcntl
import io
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file


class RuntimeWeightCacheTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('models.runtime_weight_cache'),
                             'Optional validated NF4 cache support must exist')
        self.cache = importlib.import_module('models.runtime_weight_cache')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        root_patch = patch.object(self.cache, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.source = self.root / 'exports' / 'dense'
        self.source.mkdir(parents=True)
        (self.source / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
        save_file({'model.language_model.layers.0.mlp.gate_proj.weight': torch.ones(4, 4)},
                  self.source / 'model-00001-of-00001.safetensors')
        index = {'weight_map': {'model.language_model.layers.0.mlp.gate_proj.weight':
                                'model-00001-of-00001.safetensors'}}
        (self.source / 'model.safetensors.index.json').write_text(json.dumps(index))
        self.folder = self.cache.cache_path(self.source)
        self.folder.mkdir(parents=True)
        config = {'model_type': 'qwen3_5', 'quantization_config': self.cache.QUANTIZATION}
        (self.folder / 'config.json').write_text(json.dumps(config))
        save_file({
            'model.language_model.layers.0.mlp.gate_proj.weight': torch.ones(4, 1, dtype=torch.uint8),
            'model.language_model.layers.0.mlp.gate_proj.weight.quant_state.bitsandbytes__nf4': torch.ones(1, dtype=torch.uint8),
        }, self.folder / 'model.safetensors')
        self.write_manifest()

    def write_manifest(self):
        manifest = self.cache.cache_manifest(self.source, self.folder)
        (self.folder / 'runtime_cache.json').write_text(json.dumps(manifest))

    def test_valid_cache_is_selected_without_changing_source(self):
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        self.assertEqual(self.cache.validated_cache(self.source), self.folder)
        self.assertEqual({path.name: path.read_bytes() for path in self.source.iterdir()}, before)

    def test_source_config_index_and_shard_stat_changes_invalidate_cache(self):
        for filename in ('config.json', 'model.safetensors.index.json', 'model-00001-of-00001.safetensors'):
            with self.subTest(filename=filename):
                self.write_manifest()
                path = self.source / filename
                stamp = path.stat()
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000))
                self.assertIsNone(self.cache.validated_cache(self.source))
        self.write_manifest()
        (self.source / 'config.json').write_text('{"model_type":"changed"}')
        self.assertIsNone(self.cache.validated_cache(self.source))

    def test_wrong_source_or_quantization_versions_invalidate_cache(self):
        manifest_file = self.folder / 'runtime_cache.json'
        for field in ('source', 'semantics'):
            with self.subTest(field=field):
                self.write_manifest()
                manifest = json.loads(manifest_file.read_text())
                manifest[field] = {'changed': True}
                manifest_file.write_text(json.dumps(manifest))
                self.assertIsNone(self.cache.validated_cache(self.source))

    def test_missing_and_corrupt_cache_files_fall_back(self):
        for filename in ('config.json', 'model.safetensors', 'runtime_cache.json'):
            with self.subTest(filename=filename):
                self.write_manifest()
                path = self.folder / filename
                original = path.read_bytes()
                path.unlink()
                self.assertIsNone(self.cache.validated_cache(self.source))
                path.write_bytes(b'corrupt')
                self.assertIsNone(self.cache.validated_cache(self.source))
                path.write_bytes(original)

    def test_non_text_or_adapter_tensors_cannot_be_published(self):
        for name in ('model.visual.weight', 'model.language_model.layers.0.mlp.router.weight',
                     'model.language_model.layers.0.mlp.experts.0.gate_proj.weight',
                     'model.language_model.layers.0.mlp.gate_proj.lora_A.default.weight'):
            with self.subTest(name=name):
                save_file({name: torch.ones(1)}, self.folder / 'model.safetensors')
                with self.assertRaises(ValueError):
                    self.cache.cache_manifest(self.source, self.folder)

    def test_inference_uses_cache_and_training_always_uses_source(self):
        from models.qwen35_moe import build_moe
        for quantized, shared, expected in ((True, True, self.folder), (True, False, self.source),
                                             (False, True, self.source)):
            with self.subTest(quantized=quantized, shared=shared):
                model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[])))
                with patch('models.qwen35_moe.load_dense', return_value=model) as loader:
                    actual = build_moe('unused', 'unused', merged_base=self.source,
                                       quantized=quantized, share_frozen=shared)
                self.assertEqual(Path(loader.call_args.args[0]), expected)
                self.assertEqual(getattr(actual, 'runtime_weight_cache', None),
                                 str(self.folder) if expected == self.folder else None)

    def test_cache_loading_error_uses_original_source(self):
        from models.qwen35_moe import build_moe
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[])))
        with patch('models.qwen35_moe.load_dense', side_effect=[ValueError('bad cache'), model]) as loader:
            actual = build_moe('unused', 'unused', merged_base=self.source, quantized=True, share_frozen=True)
        self.assertEqual(Path(loader.call_args_list[0].args[0]), self.folder)
        self.assertEqual(Path(loader.call_args_list[1].args[0]), self.source)
        self.assertIsNone(actual.runtime_weight_cache)

    def test_cache_shard_index_must_match_headers_and_stay_inside_directory(self):
        index = self.folder / 'model.safetensors.index.json'
        for mapping in ({'invented.tensor': 'model.safetensors'},
                        {'some.tensor': '../outside.safetensors'}):
            with self.subTest(mapping=mapping):
                index.write_text(json.dumps({'weight_map': mapping}))
                with self.assertRaises((ValueError, OSError)):
                    self.cache.cache_manifest(self.source, self.folder)
                self.assertIsNone(self.cache.validated_cache(self.source))

    def prep_bundle(self):
        bundle = self.root / 'bundle'
        bundle.mkdir()
        (bundle / 'moe_manifest.json').write_text(json.dumps(dict(
            state='completed', format='qwen35-top1-nf4-v2', quantization='nf4',
            merged_base=str(self.source))))
        return bundle

    def test_prepare_publishes_atomically_without_touching_source(self):
        from scripts.prepare_text_runtime_cache import main
        bundle = self.prep_bundle()
        payload = self.root / 'serialized'
        shutil.copytree(self.folder, payload)
        shutil.rmtree(self.folder)
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        class FrozenBase:
            is_loaded_in_4bit = True
            def eval(self): return self
            def requires_grad_(self, value): return self
            def save_pretrained(self, staging, **options):
                for path in payload.iterdir():
                    if path.name != 'runtime_cache.json':
                        shutil.copy2(path, Path(staging) / path.name)
        with patch.object(sys, 'argv', ['prepare', '--bundle', str(bundle)]), \
             patch('pathlib.Path.home', return_value=self.root), \
             patch('models.qwen35_moe.load_dense', return_value=FrozenBase()), \
             patch('sys.stdout', new=io.StringIO()):
            main()
        self.assertEqual(self.cache.validated_cache(self.source), self.folder)
        self.assertEqual({path.name: path.read_bytes() for path in self.source.iterdir()}, before)
        self.assertEqual(list(self.folder.parent.glob('*.staging-*')), [])

    def test_prepare_refuses_existing_destination_and_busy_gpu_lease(self):
        from scripts.prepare_text_runtime_cache import main
        bundle = self.prep_bundle()
        with patch.object(sys, 'argv', ['prepare', '--bundle', str(bundle)]), \
             patch('pathlib.Path.home', return_value=self.root):
            before = (self.folder / 'runtime_cache.json').read_bytes()
            with self.assertRaises(FileExistsError):
                main()
            self.assertEqual((self.folder / 'runtime_cache.json').read_bytes(), before)
            shutil.rmtree(self.folder)
            lease = self.root / '.cache/qwen-debugger/gpu.lock'
            lease.parent.mkdir(parents=True)
            with lease.open('a+') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    main()
            self.assertFalse(self.folder.exists())


if __name__ == '__main__':
    unittest.main()
