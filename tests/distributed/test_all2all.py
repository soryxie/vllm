import pytest
import torch
import tempfile
import torch.distributed as dist

from tests.kernels.moe.parallel_utils import ProcessGroupInfo, parallel_launch
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.all2all import NaiveAll2AllManager
from vllm.platforms import current_platform
from vllm.distributed import (init_distributed_environment,
                              initialize_model_parallel)


def _test_native_all2all_worker(pgi: ProcessGroupInfo, dp_size: int):
    world_size = pgi.world_size
    rank = pgi.rank
    device = pgi.device

    cfg = VllmConfig(parallel_config=ParallelConfig(
        data_parallel_size=world_size,
        enable_expert_parallel=True))
    with set_current_vllm_config(cfg):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"file://{temp_file}",
            local_rank=rank,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)

    num_tokens_per_rank = 256
    hidden_size = 2048
    num_experts = 4 * world_size
    top_k = 2

    # init comm manager
    cpu_group = dist.new_group(backend="gloo")
    manager = NaiveAll2AllManager(cpu_group)
    manager.top_k = top_k
    manager.global_num_experts = num_experts

    hidden_states = torch.full(
        (num_tokens_per_rank, hidden_size),
        float(rank),
        dtype=torch.float32,
        device=device
    )
    hidden_states_prev = hidden_states.clone()
    print(f"Rank {rank} input hidden_states {hidden_states.shape}")
    router_logits = torch.rand(
        (num_tokens_per_rank, num_experts),
        dtype=torch.float32,
        device=device
    )

    recv_hidden_states, recv_router_logits = manager.dispatch(hidden_states, router_logits)

    # check that total sent tokens equals total received tokens
    total_sent = torch.tensor(hidden_states.shape[0] * top_k,
                            dtype=torch.long,
                            device=device)
    total_recv = torch.tensor(recv_hidden_states.shape[0],
                            dtype=torch.long,
                            device=device)
    dist.all_reduce(total_sent)
    dist.all_reduce(total_recv)
    assert total_sent == total_recv, f"Rank {rank}: total sent {total_sent} != total recv {total_recv}"

    # check that the router logits are correct
    ep_local_size = num_experts // world_size
    start = rank * ep_local_size
    end = (rank + 1) * ep_local_size
    topk_ids = torch.topk(recv_router_logits, top_k, dim=1).indices
    assert torch.all(topk_ids >= start) and torch.all(topk_ids < end), f"Rank {rank}: recv router logits have invalid expert ids: {topk_ids}"

    combined_hidden_states = manager.combine(recv_hidden_states)
    assert combined_hidden_states.shape == hidden_states_prev.shape

    if rank == 0:
        print("✅ native all2all dispatch and combine passed.")


@pytest.mark.parametrize("world_dp_size", [(4, 4)])
@pytest.mark.parametrize("use_internode", [False])
def test_native_all2all(
    world_dp_size: tuple[int, int],
    use_internode: bool,
):
    current_platform.seed_everything(7)
    world_size, dp_size = world_dp_size
    if use_internode:
        if world_size > 1:
            pytest.skip("Internode tests require multiple nodes.")
        return

    parallel_launch(
        world_size,
        _test_native_all2all_worker,
        dp_size,
    )
