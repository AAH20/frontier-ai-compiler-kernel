"""Unit tests for Megatron 4D Parallelism Optimizer."""

import unittest
from frontier_ai_compiler_kernel.core.megatron_4d_parallelism import Megatron4DParallelismOptimizer
from frontier_ai_compiler_kernel.core.models import GpuClusterTopology, ModelWorkload


class TestMegatron4DParallelism(unittest.TestCase):
    def setUp(self):
        self.optimizer = Megatron4DParallelismOptimizer()
        self.topology = GpuClusterTopology(total_gpus=512, gpus_per_node=8)
        self.workload = ModelWorkload(
            model_name="test_llm_70b",
            num_parameters_billions=70.0,
            num_layers=80,
            hidden_dim=8192,
            num_attention_heads=64,
            sequence_length=4096,
            micro_batch_size=1,
            global_batch_size=512,
        )

    def test_optimization_and_sharding_validity(self):
        res = self.optimizer.optimize_4d_sharding(self.topology, self.workload)

        self.assertTrue(res.is_valid)
        # Total GPUs must multiply to 512
        self.assertEqual(res.tp * res.pp * res.dp * res.cp, 512)
        # TP must be within intra-node NVLink domain (<= 8)
        self.assertLessEqual(res.tp, 8)
        # MFU must exceed baseline
        self.assertGreaterEqual(res.estimated_mfu_pct, 40.0)


if __name__ == "__main__":
    unittest.main()
