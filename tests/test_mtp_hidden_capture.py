"""Capture final normalized text output without disabling norm kernels."""
import unittest

import torch

from tests.test_mtp_generation import tiny_target


class MTPHiddenCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_text_output_capture_leaves_final_norm_hooks_untouched(self):
        from models.qwen35_mtp import TextHiddenCapture
        model=tiny_target()
        text=model.model.language_model
        with torch.inference_mode(),TextHiddenCapture(text) as capture:
            self.assertFalse(text.norm._forward_hooks)
            out=text(input_ids=torch.tensor([[2,3,8]]),use_cache=False)
            torch.testing.assert_close(capture.hidden_states,out.last_hidden_state,rtol=0,atol=0)
            self.assertNotEqual(capture.hidden_states.data_ptr(),out.last_hidden_state.data_ptr())
        self.assertFalse(text._forward_hooks)

    def test_generation_attaches_capture_to_text_model_instead_of_norm(self):
        from models.mtp_generation import MTPGenerator
        from tests.test_mtp_generation import ScriptedHead
        model=tiny_target()
        generator=MTPGenerator(model,ScriptedHead(3,[2,3,8]),kv_mode='bf16')
        self.assertIs(generator.hidden_module,model.model.language_model)
        generator.generate(torch.tensor([[2,3,8]]),2,temperature=0)
        self.assertFalse(model.model.language_model._forward_hooks)
        self.assertFalse(model.model.language_model.norm._forward_hooks)


if __name__=='__main__':unittest.main()
