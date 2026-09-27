"""Frontier AI Silicon, Polyhedral Compiler & 4D Parallelism Optimization Kernel.

Production-grade algorithmic solvers for the 6 Apex NP-Hard and APX-Hard
problems across modern LLM training clusters and ultra-low-latency inference engines.
"""

from .core import (
    ChipletDie,
    ChipletPlacement,
    CowosChipletFloorplanner,
    CowosFloorplanReport,
    DraftTokenNode,
    FlashAttentionSramTilingOptimizer,
    GpuClusterTopology,
    InterconnectTier,
    KvCacheSubmodularEvictor,
    KvEvictionConfig,
    KvEvictionResult,
    Megatron4DParallelismOptimizer,
    ModelWorkload,
    MoeDispatchReport,
    MoeTokenDispatchBalancer,
    MoeTokenRequest,
    Parallelism4DConfig,
    SpeculativeDecodingTreeScheduler,
    SpeculativeTreeSchedule,
    SramTilingPlan,
)

__version__ = "0.1.0"

__all__ = [
    "Megatron4DParallelismOptimizer",
    "SpeculativeDecodingTreeScheduler",
    "FlashAttentionSramTilingOptimizer",
    "MoeTokenDispatchBalancer",
    "KvCacheSubmodularEvictor",
    "CowosChipletFloorplanner",
    # Models
    "InterconnectTier",
    "GpuClusterTopology",
    "ModelWorkload",
    "Parallelism4DConfig",
    "DraftTokenNode",
    "SpeculativeTreeSchedule",
    "SramTilingPlan",
    "MoeTokenRequest",
    "MoeDispatchReport",
    "KvEvictionConfig",
    "KvEvictionResult",
    "ChipletDie",
    "ChipletPlacement",
    "CowosFloorplanReport",
]
