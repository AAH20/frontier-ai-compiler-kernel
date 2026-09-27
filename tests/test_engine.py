"""Integration tests for FrontierAiCompilerEngine."""

import unittest
from frontier_ai_compiler_kernel.engine import FrontierAiCompilerEngine


class TestFrontierAiCompilerEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FrontierAiCompilerEngine()

    def test_full_benchmark_run(self):
        report = self.engine.run_full_benchmark()

        # 1. Megatron 4D
        self.assertGreater(report.megatron_mfu_estimate_pct, 40.0)

        # 2. Speculative Tree
        self.assertGreater(report.speculative_expected_tokens, 1.0)
        self.assertGreater(report.speculative_speedup_ratio, 1.0)

        # 3. FlashAttention Tiling
        self.assertEqual(report.flash_bank_conflicts, 0)
        self.assertEqual(report.flash_coalescing_efficiency_pct, 100.0)

        # 4. MoE Dispatch
        self.assertEqual(report.moe_dropped_tokens, 0)

        # 5. KV Eviction
        self.assertGreaterEqual(report.kv_attention_coverage_pct, 90.0)
        self.assertGreater(report.kv_memory_saved_pct, 50.0)

        # 6. CoWoS Floorplan
        self.assertTrue(report.cowos_thermally_safe)

        # Pipeline sub-second check
        self.assertLess(report.total_pipeline_time_ms, 1000.0)


if __name__ == "__main__":
    unittest.main()
