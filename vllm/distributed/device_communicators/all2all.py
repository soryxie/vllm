# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils import has_deep_ep, has_pplx

from .base_device_communicator import All2AllManagerBase, Cache

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
else:
    FusedMoE = None


TOKEN_DISTRIBUTION = [
    [9810, 5338, 1231, 3042, 4452, 2004, 841, 6050],
    [9278, 6282, 839, 3722, 3821, 2903, 909, 5014],
    [10684, 5165, 1044, 2634, 4344, 3333, 675, 4889],
    [10897, 5629, 1086, 2114, 2762, 3463, 1059, 5758],
    [8684, 5666, 1046, 3955, 4334, 2952, 944, 5187],
    [8361, 5908, 777, 3043, 3984, 3422, 897, 6376],
    [10208, 5507, 1117, 4148, 3967, 3119, 904, 3798],
    [9781, 5211, 902, 3343, 4289, 2839, 1008, 5395],
]


def _scale_distribution(base_distribution: list[int], target_total: int) -> list[int]:
    """Scale base_distribution to sum to target_total while keeping proportions."""
    length = len(base_distribution)
    if length == 0:
        return []
    if target_total == 0:
        return [0] * length
    base_total = sum(base_distribution)
    if base_total == 0:
        base_distribution = [1] * length
        base_total = length

    scaled = [value * target_total / base_total for value in base_distribution]
    floored = [int(value) for value in scaled]
    remainder = target_total - sum(floored)
    if remainder <= 0:
        return floored

    fractional = [(idx, scaled[idx] - floored[idx]) for idx in range(length)]
    fractional.sort(key=lambda item: (-item[1], item[0]))
    for idx in range(remainder):
        destination = fractional[idx % length][0]
        floored[destination] += 1
    return floored

def map_send_token_sizes_to_ranks(rank: int, world_size: int, total_tokens: int) -> list[int]:
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} must be within [0, {world_size}).")
    if world_size > len(TOKEN_DISTRIBUTION):
        raise ValueError("world_size exceeds TOKEN_DISTRIBUTION definitions.")

    base_row = TOKEN_DISTRIBUTION[rank][:world_size]
    return _scale_distribution(base_row, total_tokens)


def map_recv_token_sizes_to_ranks(rank: int, world_size: int, total_tokens: int) -> list[int]:
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} must be within [0, {world_size}).")

    recv_sizes = []
    for sender_rank in range(world_size):
        send_distribution = map_send_token_sizes_to_ranks(
            rank=sender_rank,
            world_size=world_size,
            total_tokens=total_tokens,
        )
        recv_sizes.append(send_distribution[rank])
    return recv_sizes


class NaiveAll2AllManager(All2AllManagerBase):
    """
    A naive implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)
        self.router_logits: torch.Tensor = None
        self.all2all_buffer: torch.Tensor = None

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
        assert (len(x.shape) == 2)
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)

        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]
        buffer[start:end, :].copy_(x)
        for idx in range(self.dp_world_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            self.dp_group.broadcast(buffer[start:end, :], idx)

        return buffer
    
    def all_to_all(self, input_: torch.Tensor, reversed: bool) -> torch.Tensor:
        world_size = self.dp_group.world_size
        if world_size == 1:
            return input_
        total_send_length = input_.size(0)
        hidden_size = input_.size(1)

        # 规定每个rank上发送的token数量相同
        if reversed:
            send_lengths = map_recv_token_sizes_to_ranks(
                rank=self.rank,
                world_size=world_size,
                total_tokens=total_send_length,
            )
            recv_lengths = map_send_token_sizes_to_ranks(
                rank=self.rank,
                world_size=world_size,
                total_tokens=total_send_length,
            )
        else:
            send_lengths = map_send_token_sizes_to_ranks(
                rank=self.rank,
                world_size=world_size,
                total_tokens=total_send_length,
            )
            recv_lengths = map_recv_token_sizes_to_ranks(
                rank=self.rank,
                world_size=world_size,
                total_tokens=total_send_length,
            )
        total_recv_length = sum(recv_lengths)
        print(f"Rank {self.rank} send lengths: {send_lengths}, recv lengths: {recv_lengths}")

        # fetch recv buffer
        if self.all2all_buffer is None or \
              self.all2all_buffer.size(0) < total_recv_length * hidden_size:
            self.all2all_buffer = torch.empty(
                total_recv_length*hidden_size*4,
                dtype=input_.dtype,
                device=self.dp_group.device,
            )
        recv_buffer = self.all2all_buffer[:total_recv_length*hidden_size].view(-1, hidden_size)

        torch.distributed.all_to_all_single(
            recv_buffer,
            input_,
            output_split_sizes=recv_lengths,
            input_split_sizes=send_lengths,
            group=self.dp_group.device_group,
        )

        return recv_buffer

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        if self.top_k is None or self.global_num_experts is None:
            raise ValueError(
                "top_k and global_num_experts must be set before dispatch."
            )

        hidden_states = hidden_states.repeat(self.top_k, 1)
        hidden_states = self.all_to_all(hidden_states, reversed=False)

        num_tokens = hidden_states.shape[0]

        if self.router_logits is None or self.router_logits.shape[0] < num_tokens:
            ep_local_size = self.global_num_experts // self.dp_group.world_size
            rank = self.dp_group.rank
            start = rank * ep_local_size
            device = router_logits.device
            num_experts = router_logits.shape[1]

            ep_ids = torch.arange(start, start + ep_local_size, device=device)
            offsets = torch.arange(self.top_k, device=device)
            ep_id_matrix = (ep_ids.unsqueeze(1) + offsets) % num_experts

            prob_rows = torch.zeros(ep_local_size, num_experts,
                                    dtype=router_logits.dtype, device=device)
            row_idx = torch.arange(ep_local_size, device=device).unsqueeze(1)
            prob_rows[row_idx, ep_id_matrix] = 1.0
            repeat_times = num_tokens // ep_local_size + 1
            self.router_logits = prob_rows.repeat(repeat_times, 1)

        return hidden_states, self.router_logits[:num_tokens]

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.all_to_all(hidden_states, reversed=True)
        return hidden_states

    def destroy(self):
        pass


class PPLXAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on PPLX kernels.
    """

    def __init__(self, cpu_group):
        assert has_pplx(
        ), "pplx_kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install pplx_kernels."  # noqa
        super().__init__(cpu_group)

        if self.internode:
            # inter-node communication needs nvshmem,
            # intra-node communication uses p2p mapping directly
            from pplx_kernels.nvshmem import (nvshmem_alloc_empty_unique_id,
                                              nvshmem_get_unique_id,
                                              nvshmem_init)
            logger.debug(
                "Initialize NVSHMEM for pplx_kernels: "
                "rank=%d, world size=%d", self.rank, self.world_size)
            uid = nvshmem_get_unique_id(
            ) if self.rank == 0 else nvshmem_alloc_empty_unique_id()
            dist.broadcast(uid,
                           src=dist.get_process_group_ranks(self.cpu_group)[0],
                           group=self.cpu_group)
            logger.debug("PPLX NVSHMEM UID = %s", uid)
            nvshmem_init(uid, self.rank, self.world_size)

        self.handle_cache = Cache()

    def get_handle(self, kwargs):
        import pplx_kernels as pplx
        return self.handle_cache.get_or_create(
            kwargs, pplx.AllToAll.internode
            if self.internode else pplx.AllToAll.intranode)

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()

        if self.internode:
            from pplx_kernels.nvshmem import nvshmem_finalize
            logger.debug("PPLX NVSHMEM finalize")
            nvshmem_finalize()


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group):
        assert has_deep_ep(
        ), "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install DeepEP kernels."  # noqa
        super().__init__(cpu_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        pass


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = 1024 * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode:
            num_rdma_bytes = 1024 * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        return dict(group=self.cpu_group,
                    num_nvl_bytes=num_nvl_bytes,
                    num_rdma_bytes=num_rdma_bytes,
                    low_latency_mode=False,
                    num_qps_per_rank=num_qps_per_rank)

    def get_handle(self, kwargs):

        assert len(kwargs) == 0, (
            "DeepEPHTAll2AllManager expects no arguments. All the required "
            "args are computed in the Manager itself.")

        import deep_ep
        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer)
        # It is dangerous to set num sms outside this function. num_sms is not
        # a part of the hash-key that identifies this object. If we are in a
        # situation where we make objects with different num_sms, the hash key
        # in get_or_create must be updated.
        handle.set_num_sms(self.num_sms)
        return handle


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP Low-Latency kernels.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """
        max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
          can dispatch all the ranks must hold the same value.
        token_hidden_size: the hidden dimension of each token.
        num_ep_ranks: the number of EP group ranks.
        num_global_experts: Number of experts in the model.
        num_local_experts: Number of experts in an EP rank.
        """
        import deep_ep

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = 1024 * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts)

        assert num_rdma_bytes is not None
        return dict(group=self.cpu_group,
                    num_nvl_bytes=num_nvl_bytes,
                    num_rdma_bytes=num_rdma_bytes,
                    low_latency_mode=True,
                    num_qps_per_rank=num_qps_per_rank)

    def get_handle(self, kwargs):
        """
        The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep
        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer)
        # It is dangerous to set num sms outside this function. num_sms is not
        # a part of the hash-key that identifies this object. If we are in a
        # situation where we make objects with different num_sms, the hash key
        # in get_or_create must be updated.
        handle.set_num_sms(self.num_sms)
        return handle
