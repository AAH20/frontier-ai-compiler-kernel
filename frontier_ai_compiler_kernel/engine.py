"""Frontier AI Compiler & Supercomputing Optimization Engine."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Tuple

from .core.models import (
    ChipletDie,
    GpuClusterTopology,
    KvEvictionConfig,
    ModelWorkload,
    MoeTokenRequest,
)
from .core.megatron_4d_parallelism import Megatron4DParallelismOptimizer
from .core.speculative_decoding_tree import SpeculativeDecodingTreeScheduler
from .core.flash_attention_sram_tiling import FlashAttentionSramTilingOptimizer
from .core.moe_token_dispatch_balancer import MoeTokenDispatchBalancer
from .core.kv_cache_submodular_eviction import KvCacheSubmodularEvictor
from .core.cowos_chiplet_floorplanner import CowosChipletFloorplanner


@dataclass
class FrontierAiCompilerBenchmarkReport:
    """Consolidated KPI report across all 6 Frontier AI Compiler & Silicon Solvers."""
    # 1. Megatron 4D Parallelism
    megatron_sharding_tuple: str
    megatron_mfu_estimate_pct: float
    megatron_pipeline_bubble_pct: float
    megatron_solve_time_ms: float
    # 2. Speculative Decoding Tree
    speculative_expected_tokens: float
    speculative_speedup_ratio: float
    speculative_solve_time_ms: float
    # 3. FlashAttention-3 SRAM Tiling
    flash_tile_dimensions: str
    flash_bank_conflicts: int
    flash_coalescing_efficiency_pct: float
    flash_solve_time_ms: float
    # 4. MoE Token Dispatch
    moe_dropped_tokens: int
    moe_load_imbalance_ratio: float
    moe_solve_time_ms: float
    # 5. KV-Cache Submodular Eviction
    kv_attention_coverage_pct: float
    kv_memory_saved_pct: float
    kv_solve_time_ms: float
    # 6. CoWoS Chiplet Floorplanner
    cowos_peak_temp_c: float
    cowos_thermally_safe: bool
    cowos_solve_time_ms: float
    # Total
    total_pipeline_time_ms: float


class FrontierAiCompilerEngine:
    """Unified Orchestration Engine for Frontier AI Silicon & Compiler Solvers."""

    def __init__(self):
        self.parallelism_optimizer = Megatron4DParallelismOptimizer()
        self.speculative_scheduler = SpeculativeDecodingTreeScheduler(max_tree_nodes=64, max_depth=5)
        self.sram_optimizer = FlashAttentionSramTilingOptimizer()
        self.moe_balancer = MoeTokenDispatchBalancer(capacity_factor=1.25)
        self.kv_evictor = KvCacheSubmodularEvictor()
        self.cowos_floorplanner = CowosChipletFloorplanner()

    def run_full_benchmark(self) -> FrontierAiCompilerBenchmarkReport:
        """Executes full benchmark suite across all 6 frontier compiler & hardware optimization solvers."""
        t0 = time.perf_counter()

        # 1. Megatron 4D Parallelism Benchmark (1-Trillion Parameter Model on 1,024 Blackwell GPUs)
        topology = GpuClusterTopology(total_gpus=1024, gpus_per_node=8, nvlink_bw_gbps=1800.0, infiniband_bw_gbps=400.0)
        workload = ModelWorkload(
            model_name="frontier_trillion_dense",
            num_parameters_billions=1000.0,
            num_layers=128,
            hidden_dim=16384,
            num_attention_heads=128,
            sequence_length=8192,
            micro_batch_size=1,
            global_batch_size=2048,
        )
        p4d_res = self.parallelism_optimizer.optimize_4d_sharding(topology, workload)

        # 2. Speculative Decoding Draft-Token Tree Benchmark
        candidate_probs = {
            0: [("the", 0.85), ("a", 0.10), ("in", 0.05)],
            1: [("model", 0.80), ("system", 0.15), ("network", 0.05)],
            2: [("generates", 0.90), ("produces", 0.08)],
            3: [("tokens", 0.95), ("output", 0.04)],
            4: [("fast", 0.75), ("efficiently", 0.20)],
        }
        spec_res = self.speculative_scheduler.build_optimal_speculative_tree(candidate_probs, kv_cache_budget_bytes=16384)

        # 3. FlashAttention-3 SRAM Tiling Benchmark (Hopper / Blackwell SM)
        sram_res = self.sram_optimizer.optimize_tile_and_swizzle(head_dim=128, target_sram_limit_kb=192.0)

        # 4. MoE All-to-All Token Dispatch Benchmark (DeepSeek-style 64 experts, 1024 tokens)
        moe_requests = [
            MoeTokenRequest(
                token_id=i,
                top_experts=[(i * 3) % 64, (i * 7 + 1) % 64],
                routing_weights=[0.85, 0.15],
            )
            for i in range(1024)
        ]
        moe_res = self.moe_balancer.balance_and_dispatch(moe_requests, num_experts=64, tokens_per_expert_capacity=32)

        # 5. KV-Cache Submodular Attention Eviction Benchmark (4,096 tokens -> 512 cached)
        # Synthetic attention curve: high on sinks (0..4) and recent window (4000..4095), sparse in between
        attn_scores = []
        for i in range(4096):
            if i < 4:
                attn_scores.append(10.0 - i)  # Sink
            elif i >= 4032:
                attn_scores.append(5.0 + (i - 4032) * 0.1)  # Recent
            elif i % 128 == 0:
                attn_scores.append(4.0)  # Periodic retrieval needle
            else:
                attn_scores.append(0.01)
        kv_config = KvEvictionConfig(target_cache_size=512, sink_tokens_preserved=4, local_window_preserved=64)
        kv_res = self.kv_evictor.evict_kv_cache(attn_scores, kv_config)

        # 6. CoWoS Multi-Chiplet Floorplanning Benchmark (2x GB200 GPU Dies + 8x HBM3e + 1x NVLink Switch)
        dies = [
            ChipletDie("gpu_die_0", "GPU_CORE", width_mm=25.0, height_mm=32.0, power_watts=550.0),
            ChipletDie("gpu_die_1", "GPU_CORE", width_mm=25.0, height_mm=32.0, power_watts=550.0),
            ChipletDie("nvlink_switch", "NVLINK_SWITCH", width_mm=15.0, height_mm=15.0, power_watts=100.0),
        ] + [
            ChipletDie(f"hbm3e_{i}", "HBM3E", width_mm=11.0, height_mm=11.0, power_watts=30.0)
            for i in range(8)
        ]
        connectivity = {
            ("gpu_die_0", "gpu_die_1"): 10.0,
            ("gpu_die_0", "nvlink_switch"): 8.0,
            ("gpu_die_1", "nvlink_switch"): 8.0,
        }
        for i in range(4):
            connectivity[(f"hbm3e_{i}", "gpu_die_0")] = 5.0
            connectivity[(f"hbm3e_{i+4}", "gpu_die_1")] = 5.0

        cowos_res = self.cowos_floorplanner.floorplan_chiplets(dies, connectivity)

        total_pipeline_time_ms = (time.perf_counter() - t0) * 1000.0

        sharding_str = f"TP={p4d_res.tp} | PP={p4d_res.pp} | DP={p4d_res.dp} | CP={p4d_res.cp}"
        tile_str = f"{sram_res.tile_m}x{sram_res.tile_n}x{sram_res.tile_k} (+pad={sram_res.padding_elements})"

        return FrontierAiCompilerBenchmarkReport(
            megatron_sharding_tuple=sharding_str,
            megatron_mfu_estimate_pct=p4d_res.estimated_mfu_pct,
            megatron_pipeline_bubble_pct=p4d_res.pipeline_bubble_pct,
            megatron_solve_time_ms=p4d_res.solve_time_ms,
            speculative_expected_tokens=spec_res.expected_accepted_tokens,
            speculative_speedup_ratio=spec_res.speedup_ratio,
            speculative_solve_time_ms=spec_res.solve_time_ms,
            flash_tile_dimensions=tile_str,
            flash_bank_conflicts=sram_res.bank_conflicts_count,
            flash_coalescing_efficiency_pct=sram_res.memory_coalescing_efficiency_pct,
            flash_solve_time_ms=sram_res.solve_time_ms,
            moe_dropped_tokens=moe_res.dropped_tokens_count,
            moe_load_imbalance_ratio=moe_res.load_imbalance_ratio,
            moe_solve_time_ms=moe_res.solve_time_ms,
            kv_attention_coverage_pct=kv_res.attention_coverage_pct,
            kv_memory_saved_pct=kv_res.memory_compression_ratio_pct,
            kv_solve_time_ms=kv_res.solve_time_ms,
            cowos_peak_temp_c=cowos_res.peak_die_temperature_c,
            cowos_thermally_safe=cowos_res.is_thermally_safe,
            cowos_solve_time_ms=cowos_res.solve_time_ms,
            total_pipeline_time_ms=round(total_pipeline_time_ms, 2),
        )
