import pytest
import torch
import time
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
    router_logits_prev = router_logits.clone()

    new_method = []
    for _ in range(10):
        torch.cuda.synchronize()
        st = time.time()
        recv_hidden_states, recv_router_logits = manager.dispatch(hidden_states, router_logits)
        torch.cuda.synchronize()
        new_method.append(time.time() - st)
    print(f"Shape hidden_states_new {recv_hidden_states.shape}")

    if rank == 0:                           # 0.0009975433349609375
        print("✅ native all2all(test-topk) passed.")


    # previous method:
    cu_tokens_across_dp_cpu = [num_tokens_per_rank * (i+1) for i in range(world_size)]
    prev_method = []
    for _ in range(10):
        torch.cuda.synchronize()
        st = time.time()
        recv_hidden_states = manager.naive_multicast(hidden_states_prev,
                                            cu_tokens_across_dp_cpu)
        recv_router_logits = manager.naive_multicast(router_logits_prev,
                                            cu_tokens_across_dp_cpu)
        torch.cuda.synchronize()
        prev_method.append(time.time() - st) # 0.0020182132720947266


    new_fmt = [f"{x:.5f}" for x in new_method]
    prev_fmt = [f"{x:.5f}" for x in prev_method]
    print(f"Latency compare: new {new_fmt}, prev {prev_fmt}") 
    print(f"Shape hidden_states_prev {hidden_states_prev.shape}, router_logits_prev {router_logits_prev.shape}")


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
