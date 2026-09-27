"""Strongly-typed data models for Frontier AI Silicon & Compiler NP-Hard Kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. Megatron 4D Parallelism Models
# ============================================================================

class InterconnectTier(Enum):
    NVLINK = "NVLink-5 (1800 GB/s)"
    PCIE = "PCIe Gen5 (64 GB/s)"
    INFINIBAND = "InfiniBand NDR (400 Gbps)"
    ETHERNET = "Ethernet RoCE (100 Gbps)"


@dataclass
class GpuClusterTopology:
    total_gpus: int
    gpus_per_node: int = 8
    nvlink_bw_gbps: float = 1800.0   # Blackwell GB200 NVLink-5 bidirectional
    infiniband_bw_gbps: float = 400.0 # Quantum-2 NDR InfiniBand per rail
    hbm_capacity_gb: float = 192.0    # HBM3e capacity per GPU


@dataclass
class ModelWorkload:
    model_name: str
    num_parameters_billions: float
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    sequence_length: int
    micro_batch_size: int
    global_batch_size: int


@dataclass
class Parallelism4DConfig:
    tp: int  # Tensor Parallelism
    pp: int  # Pipeline Parallelism
    dp: int  # Data Parallelism (FSDP / ZeRO-3)
    cp: int  # Context Parallelism
    pipeline_bubble_pct: float
    inter_node_traffic_gb: float
    estimated_mfu_pct: float
    is_valid: bool
    solve_time_ms: float = 0.0


# ============================================================================
# 2. Speculative Decoding Draft-Token Tree Models
# ============================================================================

@dataclass
class DraftTokenNode:
    node_id: int
    token_str: str
    parent_id: Optional[int]
    depth: int
    prob: float
    cumulative_prob: float


@dataclass
class SpeculativeTreeSchedule:
    nodes: List[DraftTokenNode]
    tree_shape: List[int]  # Token counts per depth tier
    expected_accepted_tokens: float
    kv_cache_memory_used_kb: float
    speedup_ratio: float
    solve_time_ms: float


# ============================================================================
# 3. FlashAttention-3 SRAM Bank Conflict Models
# ============================================================================

@dataclass
class SramTilingPlan:
    tile_m: int
    tile_n: int
    tile_k: int
    padding_elements: int
    bank_conflicts_count: int
    shared_memory_kb: float
    memory_coalescing_efficiency_pct: float
    is_conflict_free: bool
    solve_time_ms: float


# ============================================================================
# 4. MoE All-to-All Token Dispatch Balancing Models
# ============================================================================

@dataclass
class MoeTokenRequest:
    token_id: int
    top_experts: List[int]
    routing_weights: List[float]


@dataclass
class MoeDispatchReport:
    total_tokens: int
    dropped_tokens_count: int
    max_expert_load: int
    average_expert_load: float
    load_imbalance_ratio: float
    all_to_all_comm_volume_mb: float
    dispatch_success: bool
    solve_time_ms: float


# ============================================================================
# 5. KV-Cache Submodular Eviction Models
# ============================================================================

@dataclass
class KvEvictionConfig:
    target_cache_size: int
    sink_tokens_preserved: int = 4
    local_window_preserved: int = 64


@dataclass
class KvEvictionResult:
    original_seq_len: int
    retained_token_indices: List[int]
    evicted_token_indices: List[int]
    attention_coverage_pct: float
    memory_compression_ratio_pct: float
    solve_time_ms: float


# ============================================================================
# 6. CoWoS 2.5D/3D Chiplet Floorplanning Models
# ============================================================================

@dataclass
class ChipletDie:
    die_id: str
    die_type: str  # "GPU_CORE", "HBM3E", "NVLINK_SWITCH"
    width_mm: float
    height_mm: float
    power_watts: float


@dataclass
class ChipletPlacement:
    die_id: str
    x_mm: float
    y_mm: float
    peak_temp_celsius: float


@dataclass
class CowosFloorplanReport:
    placements: List[ChipletPlacement]
    peak_die_temperature_c: float
    total_wirelength_mm: float
    is_thermally_safe: bool
    solve_time_ms: float
