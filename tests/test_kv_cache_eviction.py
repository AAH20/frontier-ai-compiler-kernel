"""Unit tests for KV-Cache Submodular Eviction Solver."""

import unittest
from frontier_ai_compiler_kernel.core.kv_cache_submodular_eviction import KvCacheSubmodularEvictor
from frontier_ai_compiler_kernel.core.models import KvEvictionConfig


class TestKvCacheEviction(unittest.TestCase):
    def setUp(self):
        self.evictor = KvCacheSubmodularEvictor()
        # 100 tokens with sinks at start, local window at end
        self.weights = [10.0 if i < 4 or i >= 80 else 0.1 for i in range(100)]
        self.config = KvEvictionConfig(target_cache_size=30, sink_tokens_preserved=4, local_window_preserved=20)

    def test_eviction_sinks_and_window_preserved(self):
        res = self.evictor.evict_kv_cache(self.weights, self.config)

        # Sinks (0..3) must be retained
        for s in range(4):
            self.assertIn(s, res.retained_token_indices)

        # Local window (80..99) must be retained
        for w in range(80, 100):
            self.assertIn(w, res.retained_token_indices)

        self.assertGreaterEqual(res.attention_coverage_pct, 90.0)
        self.assertEqual(len(res.retained_token_indices), 30)


if __name__ == "__main__":
    unittest.main()
