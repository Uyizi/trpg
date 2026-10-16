import os
import unittest
from unittest.mock import patch


class InferenceEnvironmentTests(unittest.TestCase):
    def test_default_prevents_growing_cache_fragmentation(self):
        from models.inference_environment import configure_cuda_allocator
        with patch.dict(os.environ,{},clear=True):
            configure_cuda_allocator()
            self.assertEqual(os.environ['PYTORCH_ALLOC_CONF'],'expandable_segments:True')

    def test_explicit_current_and_legacy_options_remain_untouched(self):
        from models.inference_environment import configure_cuda_allocator
        for key in ('PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF'):
            for value in ('backend:cudaMallocAsync','expandable_segments:False',''):
                with self.subTest(key=key,value=value),patch.dict(os.environ,{key:value},clear=True):
                    configure_cuda_allocator()
                    self.assertEqual(dict(os.environ),{key:value})


if __name__=='__main__':unittest.main()
