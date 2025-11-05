# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual harness to validate ring FlashAttention causal prefill on 2 GPUs.

Run with:
    torchrun --nproc_per_node=2 tests/manual/test_ring_flash_attn_prefill_cp.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from vllm.v1.attention.backends.ring_flash_attn import (
    RingFlashAttentionImpl,
    RingFlashAttentionMetadata,
    RingFlashAttentionMetadataBuilder,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass
class _DummyCudaGraphMode:
    def has_full_cudagraphs(self) -> bool:
        return False


class _DummyCompilationConfig:
    def __init__(self) -> None:
        self.cudagraph_mode = _DummyCudaGraphMode()
        self.max_cudagraph_capture_size = 0


@dataclass
class _DummySchedulerConfig:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 8192
    chunked_prefill_enabled: bool = False
    enable_chunked_prefill: bool = False


@dataclass
class _DummyCacheConfig:
    block_size: int
    cache_dtype: str = "auto"
    enable_prefix_caching: bool = False
    num_gpu_blocks: int = 64
    num_cpu_blocks: int = 0
    mamba_block_size: int | None = None


@dataclass
class _DummyParallelConfig:
    tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    data_parallel_size_local: int = 1
    data_parallel_rank: int = 0
    data_parallel_rank_local: int | None = None
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_master_port: int = 29500
    data_parallel_rpc_port: int = 29550
    data_parallel_backend: str = "mp"
    data_parallel_external_lb: bool = False
    data_parallel_hybrid_lb: bool = False
    all2all_backend: str | None = None
    enable_expert_parallel: bool = False
    enable_eplb: bool = False
    worker_cls: str = "auto"
    sd_worker_cls: str = "auto"
    worker_extension_cls: str = ""
    distributed_executor_backend: str = "mp"
    rank: int = 0

    def __post_init__(self) -> None:
        self.world_size = (
            self.tensor_parallel_size
            * self.context_parallel_size
            * self.pipeline_parallel_size
        )


class _DummyModelConfig:
    def __init__(self, num_heads: int, head_size: int) -> None:
        self._num_heads = num_heads
        self._head_size = head_size
        self.dtype = torch.float16
        self.max_model_len = 4096

    def get_num_attention_heads(self, parallel_config: _DummyParallelConfig) -> int:
        return self._num_heads // parallel_config.tensor_parallel_size

    def get_num_kv_heads(self, parallel_config: _DummyParallelConfig) -> int:
        return self._num_heads // parallel_config.tensor_parallel_size

    def get_head_size(self) -> int:
        return self._head_size

    def get_sliding_window(self) -> None:
        return None

    def get_num_layers(self) -> int:
        return 1

    def get_sliding_window_for_layer(self, _: int) -> None:
        return None

    def get_logits_soft_cap_for_layer(self, _: int) -> float:
        return 0.0

    def get_sm_scale_for_layer(self, _: int) -> float:
        return 1.0 / (self._head_size**0.5)


@dataclass
class _DummyVllmConfig:
    model_config: _DummyModelConfig
    cache_config: _DummyCacheConfig
    parallel_config: _DummyParallelConfig
    scheduler_config: _DummySchedulerConfig = field(default_factory=_DummySchedulerConfig)
    compilation_config: _DummyCompilationConfig = field(default_factory=_DummyCompilationConfig)


class _DummyAttentionLayer(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        scale = torch.tensor(1.0, device=device)
        self._q_scale = scale
        self._k_scale = scale
        self._v_scale = scale


def _gather_ring_output(output: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(output)
    return output


def _sdpa_prefill_reference(
    common: CommonAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    head_dim = query.shape[-1]
    scale = 1.0 / head_dim**0.5

    offsets = common.query_start_loc_cpu
    seq_lens_cpu = common.seq_lens_cpu
    outputs: list[torch.Tensor] = []
    for i, seq_len_tensor in enumerate(seq_lens_cpu):
        seq_len = int(seq_len_tensor.item())
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        q_slice = query[start:end]
        k_slice = key[start:end]
        v_slice = value[start:end]
        q_sdpa = q_slice.unsqueeze(0).transpose(1, 2)
        k_sdpa = k_slice.unsqueeze(0).transpose(1, 2)
        v_sdpa = v_slice.unsqueeze(0).transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            is_causal=True,
            dropout_p=0.0,
            scale=scale,
        )
        outputs.append(attn_out.transpose(1, 2).squeeze(0))
    return torch.cat(outputs, dim=0)


def _build_attn_metadata(
    builder: RingFlashAttentionMetadataBuilder,
    batch: BatchSpec,
    block_size: int,
    device: torch.device,
) -> tuple[RingFlashAttentionMetadata, CommonAttentionMetadata]:
    common = create_common_attn_metadata(batch, block_size, device)
    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
    return metadata, common


def _validate_case(
    impl: RingFlashAttentionImpl,
    builder: RingFlashAttentionMetadataBuilder,
    layer: _DummyAttentionLayer,
    batch: BatchSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    block_size = builder.kv_cache_spec.block_size
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    metadata, common = _build_attn_metadata(builder, batch, block_size, device)
    num_tokens = metadata.num_actual_tokens

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    query = torch.randn(num_tokens, impl.num_heads, impl.head_size, dtype=dtype, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    output = torch.zeros_like(query)

    num_blocks = builder.vllm_config.cache_config.num_gpu_blocks
    kv_cache = torch.zeros(
        2,
        num_blocks,
        block_size,
        impl.num_kv_heads,
        impl.head_size,
        dtype=dtype,
        device=device,
    )

    impl.forward(layer, query, key, value, kv_cache, metadata, output)
    ring_result = _gather_ring_output(output.clone()).float()

    if dist.get_rank() == 0:
        reference = _sdpa_prefill_reference(common, query, key, value).float()
    else:
        reference = torch.zeros_like(ring_result)
    dist.broadcast(reference, src=0)

    torch.testing.assert_close(
        ring_result,
        reference,
        atol=2e-3,
        rtol=2e-3,
        msg="Ring FlashAttention output does not match reference prefill attention.",
    )


def _run_test() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dtype = torch.float16
    num_heads = 8
    head_size = 64
    block_size = 8

    model_config = _DummyModelConfig(num_heads=num_heads, head_size=head_size)
    cache_config = _DummyCacheConfig(block_size=block_size, num_gpu_blocks=128)
    parallel_config = _DummyParallelConfig(context_parallel_size=world_size)
    vllm_config = _DummyVllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
    )

    with set_current_vllm_config(vllm_config):
        init_distributed_environment()
        initialize_model_parallel(
            tensor_model_parallel_size=1, context_parallel_size=world_size
        )

        kv_cache_spec = FullAttentionSpec(
            block_size=cache_config.block_size,
            num_kv_heads=model_config.get_num_kv_heads(parallel_config),
            head_size=model_config.get_head_size(),
            dtype=dtype,
        )

        builder = RingFlashAttentionMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=["layers.0"],
            vllm_config=vllm_config,
            device=device,
        )

        impl = RingFlashAttentionImpl(
            num_heads=model_config.get_num_attention_heads(parallel_config),
            head_size=model_config.get_head_size(),
            scale=1.0 / (model_config.get_head_size() ** 0.5),
            num_kv_heads=model_config.get_num_kv_heads(parallel_config),
            alibi_slopes=None,
            sliding_window=model_config.get_sliding_window(),
            kv_cache_dtype=cache_config.cache_dtype,
        )

        layer = _DummyAttentionLayer(device)

        cases = [
            BatchSpec(seq_lens=[32], query_lens=[32], name="single_seq"),
            BatchSpec(seq_lens=[24, 40], query_lens=[24, 40], name="mixed_seq"),
            BatchSpec(seq_lens=[17, 9, 33], query_lens=[17, 9, 33], name="varied_seq"),
        ]

        for case in cases:
            _validate_case(impl, builder, layer, case, device, dtype)

        if dist.get_rank() == 0:
            print("Ring FlashAttention causal prefill matches reference across test cases.")


if __name__ == "__main__":
    try:
        _run_test()
    finally:
        cleanup_dist_env_and_memory()
