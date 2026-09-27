"""KV-Cache Dynamic Compaction & Attention Submodular Eviction Solver."""

from __future__ import annotations

import time
from typing import List, Set

from .models import KvEvictionConfig, KvEvictionResult


class KvCacheSubmodularEvictor:
    """Solves the NP-Hard KV-Cache Dynamic Attention Submodular Eviction Problem.
    
    In long-context LLM inference (1M-2M tokens), Key-Value (KV) cache memory
    rapidly exhausts GPU HBM.
    
    This solver formulates KV-cache compaction as Submodular Attention Coverage:
      1. Preserves Initial Attention Sinks (tokens 0..sink_size) to prevent perplexity spikes.
      2. Preserves the Recent Local Window (tokens T-W..T) for immediate syntactic fluency.
      3. Maximizes Cumulative Attention Mass (Heavy Hitters) across the middle history
         using a greedy (1 - 1/e) submodular coverage algorithm.
    """

    def evict_kv_cache(
        self,
        accumulated_attention_weights: List[float],  # Score for each token in history
        config: KvEvictionConfig,
    ) -> KvEvictionResult:
        """Selects optimal subset of tokens to retain, evicting low-impact middle tokens."""
        t0 = time.perf_counter()

        T = len(accumulated_attention_weights)
        if T <= config.target_cache_size:
            # Entire context fits in cache
            return KvEvictionResult(
                original_seq_len=T,
                retained_token_indices=list(range(T)),
                evicted_token_indices=[],
                attention_coverage_pct=100.0,
                memory_compression_ratio_pct=0.0,
                solve_time_ms=0.0,
            )

        retained: Set[int] = set()

        # Step 1: Always retain Attention Sinks (tokens 0 to sink_tokens_preserved - 1)
        sink_count = min(config.sink_tokens_preserved, T)
        for i in range(sink_count):
            retained.add(i)

        # Step 2: Always retain Recent Local Window (tokens T - local_window to T - 1)
        window_start = max(sink_count, T - config.local_window_preserved)
        for i in range(window_start, T):
            retained.add(i)

        # Step 3: Fill remaining quota with Heavy Hitters from candidate middle pool
        remaining_quota = config.target_cache_size - len(retained)

        if remaining_quota > 0:
            candidate_pool = [
                i for i in range(sink_count, window_start)
                if i not in retained
            ]

            # Greedy submodular choice: pick candidate tokens with largest cumulative attention
            sorted_candidates = sorted(
                candidate_pool,
                key=lambda idx: accumulated_attention_weights[idx],
                reverse=True,
            )

            for idx in sorted_candidates[:remaining_quota]:
                retained.add(idx)

        all_tokens = set(range(T))
        evicted = sorted(list(all_tokens - retained))
        retained_sorted = sorted(list(retained))

        # Calculate attention coverage percentage
        total_attention_mass = sum(accumulated_attention_weights)
        retained_attention_mass = sum(accumulated_attention_weights[i] for i in retained)
        coverage_pct = (retained_attention_mass / max(1e-6, total_attention_mass)) * 100.0

        compression_ratio = ((T - len(retained)) / max(1, T)) * 100.0
        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        return KvEvictionResult(
            original_seq_len=T,
            retained_token_indices=retained_sorted,
            evicted_token_indices=evicted,
            attention_coverage_pct=round(coverage_pct, 1),
            memory_compression_ratio_pct=round(compression_ratio, 1),
            solve_time_ms=round(solve_time_ms, 2),
        )
