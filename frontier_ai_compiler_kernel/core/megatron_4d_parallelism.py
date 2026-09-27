"""Megatron-LM 4D Parallelism Partitioning Optimizer (TP x PP x DP x CP)."""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

from .models import GpuClusterTopology, ModelWorkload, Parallelism4DConfig


class Megatron4DParallelismOptimizer:
    """Solves the NP-Hard 4D Distributed Parallelism Partitioning Problem.
    
    Partitions a Trillion-parameter LLM across an arbitrary GPU cluster topology
    by balancing four orthogonal parallelism dimensions:
      1. Tensor Parallelism (TP): High-frequency matrix sharding (strictly over NVLink).
      2. Pipeline Parallelism (PP): Inter-layer stage pipelining (1F1B schedule).
      3. Data Parallelism (DP): ZeRO-3 / FSDP-2 gradient and optimizer partitioning.
      4. Context Parallelism (CP): Ring-Attention sequence splitting for long contexts.
      
    Minimizes communication traffic over slow InfiniBand/Ethernet rails and
    minimizes the 1F1B pipeline bubble fraction.
    """

    def _find_divisors(self, n: int) -> List[int]:
        """Finds all integer divisors of n."""
        divisors = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        return sorted(list(divisors))

    def optimize_4d_sharding(
        self,
        topology: GpuClusterTopology,
        workload: ModelWorkload,
    ) -> Parallelism4DConfig:
        """Finds optimal (TP, PP, DP, CP) factorization maximizing Model FLOPs Utilization (MFU)."""
        t0 = time.perf_counter()

        N = topology.total_gpus
        best_config: Optional[Parallelism4DConfig] = None
        highest_mfu = -1.0

        # Candidate TP values: must be a power of 2 and <= gpus_per_node (NVLink domain)
        tp_candidates = [tp for tp in [1, 2, 4, 8] if tp <= topology.gpus_per_node and N % tp == 0]

        # Iterate over valid (TP, PP, CP, DP) combinations
        for tp in tp_candidates:
            # TP must divide hidden dimension and attention heads
            if workload.hidden_dim % tp != 0 or workload.num_attention_heads % tp != 0:
                continue

            rem_after_tp = N // tp

            # Candidate PP values: must divide num_layers
            pp_candidates = [pp for pp in self._find_divisors(rem_after_tp) if workload.num_layers % pp == 0]

            for pp in pp_candidates:
                rem_after_pp = rem_after_tp // pp

                # Candidate CP values: sequence length must be divisible by CP * 2
                cp_candidates = [cp for cp in [1, 2, 4, 8, 16] if rem_after_pp % cp == 0 and workload.sequence_length % (cp * 2) == 0]

                for cp in cp_candidates:
                    dp = rem_after_pp // cp
                    if dp <= 0:
                        continue

                    # Verify batch size divisibility
                    effective_microbatches = workload.global_batch_size / (dp * workload.micro_batch_size)
                    if effective_microbatches < 1.0 or not effective_microbatches.is_integer():
                        continue

                    m = int(effective_microbatches)

                    # 1F1B Pipeline bubble fraction: F_bubble = (PP - 1) / (m + PP - 1)
                    bubble_ratio = (pp - 1) / (m + pp - 1) if pp > 1 else 0.0

                    # Memory calculation per GPU (in GB)
                    # 16-bit weights (2 bytes) + 16-bit grads (2 bytes) + 32-bit Adam states (12 bytes) = 16 bytes/param
                    model_params = workload.num_parameters_billions * 1e9
                    # ZeRO-3 / FSDP shards optimizer states and gradients across DP
                    sharded_weight_gb = (model_params * 2.0 / (tp * pp)) / 1e9
                    sharded_opt_gb = (model_params * 14.0 / (tp * pp * dp)) / 1e9

                    # Activation memory per layer
                    activation_bytes_per_token = (workload.hidden_dim * 12.0) / tp
                    act_gb = (activation_bytes_per_token * (workload.sequence_length / cp) * workload.micro_batch_size * (workload.num_layers / pp)) / 1e9

                    total_mem_gb = sharded_weight_gb + sharded_opt_gb + act_gb

                    # Filter out configurations exceeding physical HBM
                    if total_mem_gb > topology.hbm_capacity_gb * 0.90:  # 10% safety margin for CUDA context
                        continue

                    # Inter-node communication cost (TP is intra-node NVLink, PP and DP cross InfiniBand)
                    # DP all-reduce volume: 2 * weights / DP
                    dp_comm_gb = (2.0 * model_params * 2.0) / (dp * 1e9)
                    # PP activation tensor send/recv volume:
                    pp_comm_gb = (workload.sequence_length * workload.micro_batch_size * workload.hidden_dim * 2.0 * m) / 1e9 if pp > 1 else 0.0
                    total_comm_gb = dp_comm_gb + pp_comm_gb

                    # Calculate Model FLOPs Utilization (MFU) estimate
                    # Baseline theoretical compute bound ~80%, penalized by bubble and comm overhead
                    comm_penalty = min(0.35, total_comm_gb / (topology.infiniband_bw_gbps / 8.0 * 10.0))
                    mfu = 0.80 * (1.0 - bubble_ratio) * (1.0 - comm_penalty)
                    mfu_pct = max(10.0, min(78.0, mfu * 100.0))

                    if mfu_pct > highest_mfu:
                        highest_mfu = mfu_pct
                        best_config = Parallelism4DConfig(
                            tp=tp,
                            pp=pp,
                            dp=dp,
                            cp=cp,
                            pipeline_bubble_pct=round(bubble_ratio * 100.0, 1),
                            inter_node_traffic_gb=round(total_comm_gb, 2),
                            estimated_mfu_pct=round(mfu_pct, 1),
                            is_valid=True,
                        )

        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        if best_config:
            best_config.solve_time_ms = round(solve_time_ms, 2)
            return best_config

        # Fallback conservative sharding
        return Parallelism4DConfig(
            tp=min(8, topology.gpus_per_node),
            pp=1,
            dp=max(1, N // min(8, topology.gpus_per_node)),
            cp=1,
            pipeline_bubble_pct=0.0,
            inter_node_traffic_gb=0.0,
            estimated_mfu_pct=45.0,
            is_valid=False,
            solve_time_ms=round(solve_time_ms, 2),
        )
