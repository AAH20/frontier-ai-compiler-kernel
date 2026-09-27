"""Speculative Decoding Draft-Token Tree Scheduler (Eagle / Medusa / SpecExec)."""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

from .models import DraftTokenNode, SpeculativeTreeSchedule


class SpeculativeDecodingTreeScheduler:
    """Solves the NP-Hard Speculative Decoding Draft-Token Tree Scheduling Problem.
    
    In tree-based speculative decoding (Medusa, Eagle, SpecExec), a small draft model
    proposes a non-linear tree of candidate tokens verified in a SINGLE forward pass
    by the target model using a custom 2D tree-attention mask.
    
    Optimizes the tree topology to maximize expected accepted tokens under KV-cache
    memory bandwidth and GPU SRAM budget constraints.
    """

    def __init__(self, max_tree_nodes: int = 64, max_depth: int = 6):
        self.max_tree_nodes = max_tree_nodes
        self.max_depth = max_depth

    def build_optimal_speculative_tree(
        self,
        candidate_probabilities: Dict[int, List[Tuple[str, float]]],
        kv_cache_budget_bytes: int = 32768,  # 32 KB KV-cache allocation
    ) -> SpeculativeTreeSchedule:
        """Solves optimal tree branching under KV-cache memory constraints."""
        t0 = time.perf_counter()

        # Root node represents the current verified prefix
        root = DraftTokenNode(
            node_id=0,
            token_str="<ROOT>",
            parent_id=None,
            depth=0,
            prob=1.0,
            cumulative_prob=1.0,
        )

        nodes: List[DraftTokenNode] = [root]
        tree_shape: Dict[int, int] = {0: 1}

        # Priority queue for candidate branch expansion:
        # (-marginal_gain, depth, parent_node_id, token_str, prob)
        pq: List[Tuple[float, int, int, str, float]] = []

        # Populate initial expansions from root
        for tok_str, prob in candidate_probabilities.get(0, []):
            marginal_gain = prob
            heapq.heappush(pq, (-marginal_gain, 1, 0, tok_str, prob))

        node_counter = 1
        kv_bytes_per_token = 256  # KV vectors per token across heads in FP16

        while pq and len(nodes) < self.max_tree_nodes:
            neg_gain, depth, parent_id, tok_str, prob = heapq.heappop(pq)
            marginal_gain = -neg_gain

            if depth > self.max_depth:
                continue

            if (len(nodes) + 1) * kv_bytes_per_token > kv_cache_budget_bytes:
                break

            # Create node
            parent_cum_prob = nodes[parent_id].cumulative_prob
            cum_prob = parent_cum_prob * prob

            new_node = DraftTokenNode(
                node_id=node_counter,
                token_str=tok_str,
                parent_id=parent_id,
                depth=depth,
                prob=prob,
                cumulative_prob=round(cum_prob, 4),
            )
            nodes.append(new_node)
            tree_shape[depth] = tree_shape.get(depth, 0) + 1

            # Expand children for new node
            for child_tok, child_prob in candidate_probabilities.get(depth, []):
                child_marginal_gain = cum_prob * child_prob
                heapq.heappush(pq, (-child_marginal_gain, depth + 1, node_counter, child_tok, child_prob))

            node_counter += 1

        # Calculate expected accepted tokens:
        # E[Tokens] = 1 (root) + sum_{v in tree} cumulative_prob(v)
        expected_accepted = 1.0 + sum(n.cumulative_prob for n in nodes if n.node_id != 0)

        # Baseline autoregressive generation = 1 token per forward pass
        # Target model forward pass overhead with tree attention is ~1.15x
        speedup = expected_accepted / 1.15

        kv_used_kb = (len(nodes) * kv_bytes_per_token) / 1024.0
        depth_distribution = [tree_shape.get(d, 0) for d in range(max(tree_shape.keys()) + 1)]

        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        return SpeculativeTreeSchedule(
            nodes=nodes,
            tree_shape=depth_distribution,
            expected_accepted_tokens=round(expected_accepted, 2),
            kv_cache_memory_used_kb=round(kv_used_kb, 1),
            speedup_ratio=round(speedup, 2),
            solve_time_ms=round(solve_time_ms, 2),
        )
