# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.ops.common import (
    CPTritonContext,
    correct_attn_out,
)
from vllm.attention.utils.fa_utils import flash_attn_varlen_func
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_cp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, AttentionCGSupport

from .flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)


@dataclass(kw_only=True)
class RingFlashAttentionMetadata(FlashAttentionMetadata):
    """Metadata that augments FlashAttention metadata with CP partitions."""

    cp_query_start_loc: torch.Tensor
    cp_query_start_loc_cpu: torch.Tensor
    cp_seq_lens: torch.Tensor
    cp_seq_lens_cpu: torch.Tensor
    cp_seq_starts_cpu: torch.Tensor
    cp_block_table: torch.Tensor
    cp_slot_mapping: torch.Tensor
    cp_token_indices: torch.Tensor
    cp_num_actual_tokens: int
    cp_seq_lens_all_ranks: torch.Tensor
    cp_max_tokens: int


def _divide_blocks(blocks: int, world_size: int, rank: int) -> tuple[int, int]:
    """Return (start_block, num_blocks) assigned to ``rank``."""
    if blocks == 0:
        return 0, 0
    base = blocks // world_size
    remainder = blocks % world_size
    num = base + (1 if rank < remainder else 0)
    start = base * rank + min(rank, remainder)
    return start, num


class RingFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Builds metadata and precomputes the CP partition for each rank."""

    # cuda graph不支持
    cudagraph_support: AttentionCGSupport = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> RingFlashAttentionMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        # Ring backend only supports prefill. 
        # During decode， fall back to base FA metadata.
        if metadata.max_query_len == 1:
            return metadata

        cp_group = get_cp_group()
        cp_world_size = cp_group.world_size
        cp_rank = cp_group.rank_in_group

        logger.info_once(
            "RingFlashAttention enabled | cp_world_size=%d cp_rank=%d block_size=%d",
            cp_world_size,
            cp_rank,
            self.block_size,
        )

        if common_attn_metadata.num_computed_tokens_cpu.sum().item() != 0:
            logger.info_once(
                "RingFlashAttention falling back to base FlashAttention."
            )
            return metadata

        block_size = self.block_size
        num_reqs = common_attn_metadata.num_reqs

        seq_lens_cpu = common_attn_metadata.seq_lens_cpu.to(torch.int32)
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu.to(torch.int32)
        block_table = metadata.block_table

        cp_seq_lens_all = torch.zeros(
            cp_world_size, num_reqs, dtype=torch.int32, device=torch.device("cpu")
        )

        cp_seq_lens: list[int] = []
        cp_seq_starts: list[int] = []
        cp_block_rows: list[torch.Tensor] = []
        token_indices: list[torch.Tensor] = []

        for req_idx in range(num_reqs):
            seq_len = int(seq_lens_cpu[req_idx].item())
            req_start = int(query_start_loc_cpu[req_idx].item())

            total_blocks = cdiv(seq_len, block_size)
            start_block, num_blocks = _divide_blocks(total_blocks, cp_world_size, cp_rank)

            start_token = min(start_block * block_size, seq_len)
            end_token = min(start_token + num_blocks * block_size, seq_len)
            local_len = max(0, end_token - start_token)

            # rank partitions
            for other_rank in range(cp_world_size):
                o_start_block, o_num_blocks = _divide_blocks(
                    total_blocks, cp_world_size, other_rank
                )
                o_start_token = min(o_start_block * block_size, seq_len)
                o_end_token = min(o_start_token + o_num_blocks * block_size, seq_len)
                cp_seq_lens_all[other_rank, req_idx] = max(
                    0, o_end_token - o_start_token
                )

            cp_seq_lens.append(local_len)
            cp_seq_starts.append(start_token)

            if local_len == 0:
                cp_block_rows.append(torch.full_like(block_table[req_idx], PAD_SLOT_ID))
                continue

            token_range = torch.arange(
                start_token, end_token, dtype=torch.int64
            ) + req_start
            token_indices.append(token_range)

            block_row = block_table[req_idx]
            local_block_end = start_block + cdiv(local_len, block_size)
            local_blocks = (
                block_row[start_block:local_block_end].clone()
                if local_block_end > start_block
                else torch.empty(0, dtype=block_row.dtype, device=block_row.device)
            )
            padded_row = torch.full_like(block_row, PAD_SLOT_ID)
            if local_blocks.numel() > 0:
                padded_row[: local_blocks.shape[0]] = local_blocks
            cp_block_rows.append(padded_row)

        if token_indices:
            cp_token_indices = torch.cat(token_indices, dim=0)
        else:
            cp_token_indices = torch.empty(
                0, dtype=torch.int64, device=metadata.slot_mapping.device
            )

        cp_slot_mapping = (
            metadata.slot_mapping.index_select(
                0, cp_token_indices.to(metadata.slot_mapping.device)
            )
            if cp_token_indices.numel() > 0
            else metadata.slot_mapping.new_empty((0,))
        )

        cp_seq_lens_cpu = torch.tensor(cp_seq_lens, dtype=torch.int32)
        cp_seq_lens_tensor = cp_seq_lens_cpu.to(metadata.seq_lens.device)
        cp_seq_starts_cpu = torch.tensor(cp_seq_starts, dtype=torch.int32)

        cp_query_start_loc_cpu = torch.zeros(
            len(cp_seq_lens) + 1, dtype=torch.int32
        )
        if len(cp_seq_lens):
            cp_query_start_loc_cpu[1:] = torch.cumsum(cp_seq_lens_cpu, dim=0)
        cp_query_start_loc = cp_query_start_loc_cpu.to(metadata.query_start_loc.device)

        cp_block_table = torch.stack(cp_block_rows, dim=0).to(block_table.device)
        cp_total_tokens_all_ranks = cp_seq_lens_all.sum(dim=1)
        cp_max_tokens = int(cp_total_tokens_all_ranks.max().item())

        ring_meta = RingFlashAttentionMetadata(
            num_actual_tokens=metadata.num_actual_tokens,
            max_query_len=metadata.max_query_len,
            query_start_loc=metadata.query_start_loc,
            max_seq_len=metadata.max_seq_len,
            seq_lens=metadata.seq_lens,
            block_table=metadata.block_table,
            slot_mapping=metadata.slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            scheduler_metadata=None,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            prefix_scheduler_metadata=None,
            max_num_splits=0,
            causal=True,
            max_dcp_context_kv_len=0,
            dcp_context_kv_lens=None,
            cp_query_start_loc=cp_query_start_loc,
            cp_query_start_loc_cpu=cp_query_start_loc_cpu,
            cp_seq_lens=cp_seq_lens_tensor,
            cp_seq_lens_cpu=cp_seq_lens_cpu,
            cp_seq_starts_cpu=cp_seq_starts_cpu,
            cp_block_table=cp_block_table,
            cp_slot_mapping=cp_slot_mapping,
            cp_token_indices=cp_token_indices,
            cp_num_actual_tokens=int(cp_query_start_loc_cpu[-1].item()),
            cp_seq_lens_all_ranks=cp_seq_lens_all,
            cp_max_tokens=cp_max_tokens,
        )

        if logger.isEnabledFor(logging.DEBUG):
            try:
                assert int(ring_meta.cp_query_start_loc_cpu[-1].item()) == int(
                    ring_meta.cp_token_indices.numel()
                ), "cp_num_actual_tokens mismatch with token_indices length"
            except Exception:
                logger.exception(
                    "RingFA metadata invariant failed: cp_num_actual_tokens=%d token_indices=%d",
                    int(ring_meta.cp_query_start_loc_cpu[-1].item()),
                    int(ring_meta.cp_token_indices.numel()),
                )
            logger.debug(
                "RingFA partition summary | rank=%d ws=%d | local_tokens=%d | per-req lens=%s | starts=%s | cp_max_tokens=%d",
                cp_rank,
                cp_world_size,
                int(ring_meta.cp_num_actual_tokens),
                ring_meta.cp_seq_lens_cpu.tolist(),
                ring_meta.cp_seq_starts_cpu.tolist(),
                ring_meta.cp_max_tokens,
            )

        return ring_meta


class _RingContext:
    """Helper that keeps track of the running attention state."""

    def __init__(self) -> None:
        self.out: Optional[torch.Tensor] = None
        self.lse: Optional[torch.Tensor] = None

    def update(self, block_out: torch.Tensor, block_lse: torch.Tensor) -> None:
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(0, 1).unsqueeze(-1).to(torch.float32)
        if self.out is None:
            self.out = block_out
            self.lse = block_lse
            return
        assert self.lse is not None
        self.out = self.out - torch.sigmoid(block_lse - self.lse) * (
            self.out - block_out
        )
        
        # online update
        self.lse = self.lse - F.logsigmoid(self.lse - block_lse)

    def result(self, dtype: torch.dtype) -> torch.Tensor:
        assert self.out is not None
        return self.out.to(dtype)

    def result_lse(self) -> torch.Tensor:
        assert self.lse is not None
        return self.lse.squeeze(-1).to(torch.float32)


class RingComm:
    """wrapper over GroupCoordinator."""

    def __init__(self, group) -> None:
        self.group = group
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        self.next_rank = (self.rank + 1) % self.world_size
        self.prev_rank = (self.rank - 1 + self.world_size) % self.world_size

    def send(self, tensor_dict: dict[str, torch.Tensor]) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            try:
                total_bytes = 0
                details: list[str] = []
                for k, v in tensor_dict.items():
                    if isinstance(v, torch.Tensor):
                        numel = v.numel()
                        b = v.element_size() * numel
                        total_bytes += b
                        details.append(
                            f"{k}:{tuple(v.shape)}/{str(v.dtype).removeprefix('torch.')}@{v.device}"
                        )
                    else:
                        details.append(f"{k}:obj")
                logger.debug(
                    "RingFA comm send | rank=%d -> %d | items=%s | total=%.2fMB",
                    self.rank,
                    self.next_rank,
                    ", ".join(details),
                    total_bytes / 1e6,
                )
            except Exception:
                logger.debug(
                    "RingFA comm send | rank=%d -> %d", self.rank, self.next_rank)
        self.group.send_tensor_dict(tensor_dict, dst=self.next_rank)

    def recv(self) -> dict[str, torch.Tensor]:
        recv = self.group.recv_tensor_dict(src=self.prev_rank)
        if logger.isEnabledFor(logging.DEBUG) and recv is not None:
            try:
                total_bytes = 0
                details: list[str] = []
                for k, v in recv.items():
                    if isinstance(v, torch.Tensor):
                        numel = v.numel()
                        b = v.element_size() * numel
                        total_bytes += b
                        details.append(
                            f"{k}:{tuple(v.shape)}/{str(v.dtype).removeprefix('torch.')}@{v.device}"
                        )
                    else:
                        details.append(f"{k}:obj")
                logger.debug(
                    "RingFA comm recv | rank=%d <- %d | items=%s | total=%.2fMB",
                    self.rank,
                    self.prev_rank,
                    ", ".join(details),
                    total_bytes / 1e6,
                )
            except Exception:
                logger.debug(
                    "RingFA comm recv | rank=%d <- %d", self.rank, self.prev_rank)
        if recv is None:
            return {}
        return recv


class RingFlashAttentionImpl(FlashAttentionImpl):
    """Runs FlashAttention by exchanging KV slices in a ring across CP ranks."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        self._ring_triton_ctx = CPTritonContext()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RingFlashAttentionMetadata | FlashAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            raise ValueError("RingFlashAttentionImpl requires an output buffer.")

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Quantized output is not supported.")

        if self.kv_cache_dtype != "auto":
            raise NotImplementedError("Ring backend does not support KV quantisation.")

        if attn_metadata is None:
            return output.zero_()
        if (
            not isinstance(attn_metadata, RingFlashAttentionMetadata)
            or getattr(attn_metadata, "max_query_len", 0) == 1
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        cp_group = get_cp_group()
        if cp_group.world_size == 1:
            raise RuntimeError("RingFlashAttentionImpl expects cp_world_size > 1.")

        local_tokens = attn_metadata.cp_num_actual_tokens
        if local_tokens == 0:
            return output.zero_()

        token_indices = attn_metadata.cp_token_indices.to(query.device, non_blocking=True)
        local_query = query.index_select(0, token_indices).contiguous()
        local_key = key.index_select(0, token_indices).contiguous()
        local_value = value.index_select(0, token_indices).contiguous()
        # Log local/global Q/K/V shapes per CP rank for troubleshooting
        logger.info(
            "RingFA shapes | cp_rank=%d/%d | q_full=%s q_local=%s | k_local=%s v_local=%s",
            get_cp_group().rank_in_group,
            get_cp_group().world_size,
            tuple(query.shape),
            tuple(local_query.shape),
            tuple(local_key.shape),
            tuple(local_value.shape),
        )



        key_cache, value_cache = kv_cache.unbind(0)
        # Log KV cache buffer shapes per CP rank
        logger.info(
            "KV cache buffers | cp_rank=%d/%d | key_cache=%s | value_cache=%s | slot_map=%s",
            get_cp_group().rank_in_group,
            get_cp_group().world_size,
            tuple(key_cache.shape),
            tuple(value_cache.shape),
            tuple(attn_metadata.cp_slot_mapping.shape),
        )
        reshape_and_cache = torch.ops._C_cache_ops.reshape_and_cache_flash
        reshape_and_cache(
            local_key,
            local_value,
            key_cache,
            value_cache,
            attn_metadata.cp_slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        ring_ctx = _RingContext()
        world_size = cp_group.world_size
        curr_k = local_key
        curr_v = local_value
        curr_seq_lens = attn_metadata.cp_seq_lens_cpu.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        curr_seq_starts = attn_metadata.cp_seq_starts_cpu.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        q_seq_starts = curr_seq_starts.clone()
        q_seq_lens = curr_seq_lens.clone()
        q_seq_ends = q_seq_starts + q_seq_lens

        cu_seqlens_q = attn_metadata.cp_query_start_loc.to(query.device)
        max_seqlen_q = int(attn_metadata.cp_seq_lens_cpu.max().item())
        ring_comm = RingComm(cp_group)

        cp_rank = cp_group.rank_in_group
        owner_rank = cp_rank

        logger.debug_once(
            "RingFA forward begin | rank=%d/%d | local_tokens=%d | num_reqs=%d | max_seqlen_q=%d | q.shape=%s k.shape=%s v.shape=%s",  # noqa: E501
            cp_rank,
            world_size,
            int(attn_metadata.cp_num_actual_tokens),
            int(attn_metadata.cp_seq_lens_cpu.numel()),
            int(max_seqlen_q),
            tuple(query.shape),
            tuple(key.shape),
            tuple(value.shape),
        )

        ring_steps = 0
        total_sent_bytes = 0
        total_recv_bytes = 0
        for _ in range(world_size):
            token_count = int(curr_seq_lens.sum().item())
            if token_count > 0:
                chunk_seq_starts = curr_seq_starts
                chunk_seq_lens = curr_seq_lens
                chunk_seq_ends = chunk_seq_starts + chunk_seq_lens

                valid_mask = chunk_seq_lens > 0
                before_mask = chunk_seq_ends <= q_seq_starts
                after_mask = chunk_seq_starts >= q_seq_ends

                process_mask = valid_mask & ~after_mask
                if bool(process_mask.any()):
                    if bool((after_mask & valid_mask).any()):
                        chunk_seq_lens = chunk_seq_lens.clone()
                        chunk_seq_lens[after_mask] = 0

                    chunk_max_k = int(chunk_seq_lens.max().item())
                    if chunk_max_k > 0:
                        chunk_is_causal = bool(
                            (valid_mask & ~(before_mask | after_mask)).any()
                        )
                        #  cu_seqlens_k：batch 推理确定每个请求的起始位置
                        cu_seqlens_k = torch.zeros(
                            chunk_seq_lens.shape[0] + 1,
                            dtype=torch.int32,
                            device=query.device,
                        )
                        if bool((chunk_seq_lens > 0).any()):
                            cu_seqlens_k[1:] = torch.cumsum(chunk_seq_lens, dim=0)

                        chunk_out, chunk_lse = flash_attn_varlen_func(
                            q=local_query,
                            k=curr_k,
                            v=curr_v,
                            cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_q=max_seqlen_q,
                            cu_seqlens_k=cu_seqlens_k,
                            max_seqlen_k=chunk_max_k,
                            softmax_scale=self.scale,
                            causal=chunk_is_causal,
                            return_softmax_lse=True,
                            alibi_slopes=self.alibi_slopes,
                            window_size=self.sliding_window,
                            fa_version=self.vllm_flash_attn_version,
                        )
                        # online update
                        ring_ctx.update(chunk_out, chunk_lse)

            if world_size == 1:
                break

            # log current local KV slice shapes for this ring step
            logger.debug(
                "RingFA step | cp_rank=%d/%d | step=%d/%d | curr_k=%s curr_v=%s",
                cp_rank,
                world_size,
                ring_steps,
                world_size,
                tuple(curr_k.shape),
                tuple(curr_v.shape),
            )

            payload = {
                "k": curr_k,
                "v": curr_v,
                "seq_lens": curr_seq_lens,
                "seq_starts": curr_seq_starts,
            }
            # 统计发送的字节数
            if logger.isEnabledFor(logging.DEBUG):
                for _name, _t in payload.items():
                    if isinstance(_t, torch.Tensor):
                        total_sent_bytes += _t.element_size() * _t.numel()

            # 避免死锁，交替send和recv
            if (cp_rank % 2) == 0:
                recv_dict = ring_comm.recv()
                ring_comm.send(payload)
            else:
                ring_comm.send(payload)
                recv_dict = ring_comm.recv()
            if not recv_dict:
                break
            curr_k = recv_dict["k"].contiguous()
            curr_v = recv_dict["v"].contiguous()
            curr_seq_lens = recv_dict["seq_lens"].to(
                device=query.device, dtype=torch.int32
            ).contiguous()
            curr_seq_starts = recv_dict["seq_starts"].to(
                device=query.device, dtype=torch.int32
            ).contiguous()
            # 统计接收的字节数
            if logger.isEnabledFor(logging.DEBUG):
                for _name, _t in recv_dict.items():
                    if isinstance(_t, torch.Tensor):
                        total_recv_bytes += _t.element_size() * _t.numel()

            owner_rank = (owner_rank - 1 + world_size) % world_size
            ring_steps += 1

        final_out = ring_ctx.result(query.dtype)
        final_lse = ring_ctx.result_lse()

        logger.debug(
            "RingFA forward end | rank=%d/%d | ring_steps=%d | final_out.shape=%s | final_lse.shape=%s",
            cp_rank,
            world_size,
            ring_steps,
            tuple(final_out.shape),
            tuple(final_lse.shape),
        )

        
        logger.debug(
            "RingFA comm summary | rank=%d/%d | ring_steps=%d | sent=%.2fMB | recv=%.2fMB",
            cp_rank,
            world_size,
            ring_steps,
            total_sent_bytes / 1e6,
            total_recv_bytes / 1e6,
        )

        if world_size > 1 and attn_metadata.cp_max_tokens > 0:
            max_tokens = attn_metadata.cp_max_tokens
            local_tokens = final_out.shape[0]
            if local_tokens < max_tokens:
                padded_out = torch.zeros(
                    (max_tokens, self.num_heads, self.head_size),
                    dtype=final_out.dtype,
                    device=final_out.device,
                )
                padded_out[:local_tokens] = final_out
            else:
                padded_out = final_out

            final_lse_fp32 = final_lse.to(torch.float32)
            if local_tokens < max_tokens:
                padded_lse = torch.full(
                    (max_tokens, self.num_heads),
                    float("-inf"),
                    dtype=torch.float32,
                    device=final_out.device,
                )
                padded_lse[:local_tokens] = final_lse_fp32
            else:
                padded_lse = final_lse_fp32

            gathered_lse = cp_group.all_gather(
                padded_lse.contiguous(), dim=0
            ).view(world_size, max_tokens, self.num_heads)

            corrected_out, _ = correct_attn_out(
                padded_out, gathered_lse, cp_group.rank_in_group, self._ring_triton_ctx
            )
            final_out = corrected_out[:local_tokens].contiguous()
        else:
            final_out = final_out.contiguous()

        output.index_copy_(0, token_indices, final_out)
        logger.debug(
            "RingFA writeback | rank=%d/%d | token_indices=%d | output.shape=%s",
            cp_rank,
            world_size,
            int(token_indices.numel()),
            tuple(output.shape),
        )
        return output


class RingFlashAttentionBackend(FlashAttentionBackend):
    """registers the ring FlashAttention backend."""

    @staticmethod
    def get_name() -> str:
        return "RING_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type[RingFlashAttentionImpl]:
        return RingFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[RingFlashAttentionMetadata]:
        return RingFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[RingFlashAttentionMetadataBuilder]:
        return RingFlashAttentionMetadataBuilder
