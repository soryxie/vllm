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
    manager.set_global_expert_map(
        ep_size=world_size, 
        global_num_experts=num_experts,
        dtype=torch.int32,
        device=device)

    # 输入：每个 rank 都用自己的 rank 值填充，便于校验来源
    hidden_states = torch.full(
        (num_tokens_per_rank, hidden_size),
        float(rank),
        dtype=torch.float32,
        device=device
    )
    hidden_states_prev = hidden_states.clone()

    router_logits = torch.rand(
        (num_tokens_per_rank, num_experts),
        dtype=torch.float32,
        device=device
    )
    router_logits_prev = router_logits.clone()

    # === 测试端也复算 topk -> 映射到 EP rank -> 统计发送计数，并交换计数得到期望接收计数 ===
    _, topk_ids = torch.topk(router_logits, top_k, dim=-1)
    ep2ranks = manager.global2ep[topk_ids]

    send_counts_local = torch.bincount(
        ep2ranks.reshape(-1),
        minlength=world_size
    ).to(torch.long) 

    expected_recv_counts = torch.empty_like(send_counts_local)
    dist.all_to_all_single(
        expected_recv_counts,
        send_counts_local,
        group=cpu_group
    )

    torch.cuda.synchronize()
    st = time.time()
    recv_hidden_states, _ = manager.all2all(hidden_states, router_logits, top_k)

    exp_total = int(expected_recv_counts.sum().item())
    assert recv_hidden_states.shape == (exp_total, hidden_size), (
        f"rank {rank}: recv shape {tuple(recv_hidden_states.shape)} != {(exp_total, hidden_size)}"
    )
    torch.cuda.synchronize()
    new_method = time.time() - st

    offset = 0
    for s in range(world_size):
        cnt = int(expected_recv_counts[s].item())
        if cnt == 0:
            continue
        block = recv_hidden_states[offset:offset + cnt]
        assert torch.all(block == float(s)), (
            f"rank {rank}: block from sender {s} has wrong values"
        )
        offset += cnt
    assert offset == exp_total, f"rank {rank}: offset={offset}, exp_total={exp_total}"

    if rank == 0:
        print("✅ native all2all(test-topk) passed.")


    # previous method:
    cu_tokens_across_dp_cpu = [num_tokens_per_rank * (i+1) for i in range(world_size)]
    torch.cuda.synchronize()
    st = time.time()
    hidden_states_prev = manager.naive_multicast(hidden_states_prev,
                                         cu_tokens_across_dp_cpu)
    router_logits_prev = manager.naive_multicast(router_logits_prev,
                                         cu_tokens_across_dp_cpu)
    torch.cuda.synchronize()
    prev_method = time.time() - st

    print(f"Latency compare: new {new_method}, prev {prev_method}")



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
