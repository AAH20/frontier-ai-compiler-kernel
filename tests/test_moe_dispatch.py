"""Unit tests for MoE Token Dispatch Balancer."""

import unittest
from frontier_ai_compiler_kernel.core.moe_token_dispatch_balancer import MoeTokenDispatchBalancer
from frontier_ai_compiler_kernel.core.models import MoeTokenRequest


class TestMoeDispatchBalancer(unittest.TestCase):
    def setUp(self):
        self.balancer = MoeTokenDispatchBalancer(capacity_factor=1.25)
        # Create 100 requests where many prefer expert 0
        self.requests = [
            MoeTokenRequest(token_id=i, top_experts=[0, (i + 1) % 8], routing_weights=[0.8, 0.2])
            for i in range(100)
        ]

    def test_zero_token_drops(self):
        report = self.balancer.balance_and_dispatch(
            self.requests, num_experts=8, tokens_per_expert_capacity=20
        )

        self.assertTrue(report.dispatch_success)
        self.assertEqual(report.dropped_tokens_count, 0)
        self.assertGreater(report.max_expert_load, 0)


if __name__ == "__main__":
    unittest.main()
