"""Unit tests for FlashAttention SRAM Tiling Optimizer."""

import unittest
from frontier_ai_compiler_kernel.core.flash_attention_sram_tiling import FlashAttentionSramTilingOptimizer


class TestFlashAttentionSramTiling(unittest.TestCase):
    def setUp(self):
        self.optimizer = FlashAttentionSramTilingOptimizer()

    def test_bank_conflict_elimination(self):
        plan = self.optimizer.optimize_tile_and_swizzle(head_dim=128, target_sram_limit_kb=192.0)

        self.assertTrue(plan.is_conflict_free)
        self.assertEqual(plan.bank_conflicts_count, 0)
        self.assertEqual(plan.memory_coalescing_efficiency_pct, 100.0)
        self.assertLessEqual(plan.shared_memory_kb, 192.0)


if __name__ == "__main__":
    unittest.main()
