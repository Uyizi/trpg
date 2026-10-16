import unittest


class PrefillPolicyTests(unittest.TestCase):
    def test_auto_keeps_measured_best_chunk_even_with_memory_margin(self):
        from models.prefill_policy import choose_prefill_chunk
        self.assertEqual(choose_prefill_chunk(6144,0,8*2**30),1024)
        self.assertEqual(choose_prefill_chunk(8192,0,8*2**30),1024)
        self.assertEqual(choose_prefill_chunk(32768,0,8*2**30),1024)
        self.assertEqual(choose_prefill_chunk(262144,0,3*2**30),1024)

    def test_explicit_sizes_are_retained_and_invalid_values_rejected(self):
        from models.prefill_policy import choose_prefill_chunk
        self.assertEqual(choose_prefill_chunk(262144,1024,8*2**30),1024)
        self.assertEqual(choose_prefill_chunk(64,2,None),2)
        with self.assertRaises(ValueError):choose_prefill_chunk(64,-1,None)


if __name__=='__main__':unittest.main()
