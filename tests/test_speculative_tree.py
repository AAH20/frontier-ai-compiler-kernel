"""Unit tests for Speculative Decoding Tree Scheduler."""

import unittest
from frontier_ai_compiler_kernel.core.speculative_decoding_tree import SpeculativeDecodingTreeScheduler


class TestSpeculativeTreeScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = SpeculativeDecodingTreeScheduler(max_tree_nodes=32, max_depth=4)
        self.candidate_probs = {
            0: [("high", 0.90), ("low", 0.10)],
            1: [("prob", 0.85), ("token", 0.15)],
            2: [("sequence", 0.95)],
        }

    def test_tree_construction_and_speedup(self):
        res = self.scheduler.build_optimal_speculative_tree(self.candidate_probs, kv_cache_budget_bytes=8192)

        self.assertGreater(len(res.nodes), 1)
        self.assertLessEqual(len(res.nodes), 32)
        # Expected accepted tokens must exceed single autoregressive token (1.0)
        self.assertGreater(res.expected_accepted_tokens, 1.0)
        self.assertGreater(res.speedup_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
