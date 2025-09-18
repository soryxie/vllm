import os
import sys
import json
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import torch.cuda.nvtx as nvtx

def load_expert_assignments(filepath):
    data = json.load(open(filepath, 'r'))
    for rank in range(len(data)):
        for idx in range(len(data[rank])):
            data[rank][idx] = data[rank][idx]["topk_ids"]
    return data


class Node:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.send_state = {
            i: {
                "count": 0,
                "token_indices": []
            } for i in range(world_size)
        }
        self.recv_state = {
            i: {
                "count": 0,
                "token_indices": []
            } for i in range(world_size)
        }
        self.total_send = 0
        self.total_recv = 0


class Profiler:
    def __init__(self, rank, world_size, only_all2all, reused_ratio, topk, device):
        self.rank = rank
        self.world_size = world_size
        self.only_all2all = only_all2all
        self.reused_ratio = reused_ratio
        self.topk = topk
        self.device = device
        self.nodes = None

        if only_all2all:
            self.all2all_group = dist.new_group(ranks=list(range(world_size)))
            self.all2all_world_size = world_size
        else:
            self.all2all_group = dist.new_group(ranks=list(range(0, world_size // 2)))
            self.all2all_world_size = world_size // 2
        self.g_p2d = dist.new_group(ranks=list(range(world_size)))
        self.g_reuse = dist.new_group(ranks=list(range(world_size)))
        self.g_other0 = dist.new_group(ranks=list(range(world_size)))
        self.g_other1 = dist.new_group(ranks=list(range(world_size)))
        self.g_other2 = dist.new_group(ranks=list(range(world_size)))
        self.g_other3 = dist.new_group(ranks=list(range(world_size)))

        # tokens
        self.num_tokens_all2all = 1024
        self.num_tokens_total = int(self.num_tokens_all2all / (1 - self.reused_ratio))
        self.num_tokens_reuse = self.num_tokens_total - self.num_tokens_all2all

        # model
        self.kv_heads = 16
        self.kv_dim = 128
        self.hidden_size = 6144
        self.dtype = torch.bfloat16
        self.num_experts = 128
        self.kv_heads = 8
        self.head_dim = 128
        self.kv_block_size = 128
        self.experts_per_rank = self.num_experts // self.all2all_world_size
        self.local_expert_start = rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank 
        self.local_expert_ids = set(range(self.local_expert_start, self.local_expert_end)) 

        # data
        self.assignments = load_expert_assignments("/profile_json/ep_states_prompt_8_token_1024.jsonl")
        self.layer_num = len(self.assignments[0])

        self.prof_logdir = os.environ.get("PROF_LOGDIR", "/profile_json/logs")
        os.makedirs(self.prof_logdir, exist_ok=True)

    def warmup(self):
        send_tensor = torch.zeros((1024, 2048), dtype=self.dtype, device=self.device)
        recv_tensor = torch.zeros((1024, 2048), dtype=self.dtype, device=self.device)
        dist.all_to_all_single(recv_tensor, send_tensor)
    
    def init_send_recv(self, layer_idx, topk):
        all2all_world_size = self.all2all_world_size
        self.nodes = [Node(i, all2all_world_size) for i in range(all2all_world_size)]
        for rank in range(self.all2all_world_size):
            for token_idx, topk_ids in enumerate(self.assignments[rank][layer_idx]): 
                for expert_id in topk_ids[:topk]:  # top-2 experts, communication optimization
                    target_rank = expert_id // self.experts_per_rank
                    if target_rank == rank:
                        continue
                    self.nodes[rank].send_state[target_rank]["count"] += 1
                    self.nodes[rank].send_state[target_rank]["token_indices"].append(token_idx)
                    self.nodes[target_rank].recv_state[rank]["count"] += 1
                    self.nodes[target_rank].recv_state[rank]["token_indices"].append(token_idx)

    def init_tensors(self):
        all2all_world_size = self.all2all_world_size

        # 1. KV reuse           irecv  |  isend
        kv_reuse_tokens = self.num_tokens_reuse
        kv_reuse_block_nums = (kv_reuse_tokens + self.kv_block_size - 1) // self.kv_block_size
        self.kv_reuse_tensors = [torch.zeros(
            (2, self.kv_block_size, self.kv_heads, self.head_dim), 
            dtype=self.dtype, device=self.device) for _ in range(kv_reuse_block_nums)
        ]

        # 2. All to all         async  |  none
        if self.rank < all2all_world_size:
            self.send_split_sizes = [256 * self.topk for _ in range(all2all_world_size)]
            self.recv_split_sizes = [256 * self.topk for _ in range(all2all_world_size)]
            self.send_split_sizes[self.rank] = 0
            self.recv_split_sizes[self.rank] = 0
            self.all2all_send_tensor = torch.zeros(
                (sum(self.send_split_sizes), self.hidden_size), 
                dtype=self.dtype, device=self.device
            )
            self.all2all_recv_tensor = torch.zeros(
                (sum(self.recv_split_sizes), self.hidden_size), 
                dtype=self.dtype, device=self.device
            )

        # 3. P->D KV Transfer   isend  |  irecv
        p2d_kv_tokens = self.num_tokens_total
        p2d_kv_block_nums = (p2d_kv_tokens + self.kv_block_size - 1) // self.kv_block_size
        self.p2d_kv_tensors = [torch.zeros(
            (2, self.kv_block_size, self.kv_heads, self.head_dim), 
            dtype=self.dtype, device=self.device) for _ in range(p2d_kv_block_nums)
        ]

        # 4. Other kv read      isend  |  irecv
        other_kv_tokens = self.num_tokens_reuse * 4
        other_kv_block_nums = (other_kv_tokens + self.kv_block_size - 1) // self.kv_block_size
        self.other_kv_tensors = [torch.zeros(
            (2, self.kv_block_size, self.kv_heads, self.head_dim), 
            dtype=self.dtype, device=self.device) 
            for _ in range(other_kv_block_nums)
        ]


    def profile(self):
        rank = self.rank
        all2all_world_size = self.all2all_world_size

        if rank == 0:
            reuse_ratio = int(self.reused_ratio * 100)
            output_filename = f"/profile_json/Qwen_P2D_{'ALL2ALL' if self.only_all2all else 'ALLFLOW'}.jsonl"
            outfile = open(output_filename, "w")

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=10, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self.prof_logdir, worker_name=f"rank{rank}"),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        prof.__enter__()

        for layer_idx in [0 for _ in range(11)]:# range(self.layer_num):
            if rank == 0:
                print(f"Processing layer {layer_idx}")

            self.init_send_recv(layer_idx, self.topk)
            self.init_tensors()

            dist.barrier()
            time.sleep(1.5)

            nvtx.range_push(f"ROUND_L{layer_idx}")
            with torch.autograd.profiler.record_function(f"ROUND_L{layer_idx}"):
                if self.only_all2all:
                    """
                    Only All to all       (sync)
                    """
                    send_split_sizes = [256 * self.topk for _ in range(all2all_world_size)]
                    recv_split_sizes = [256 * self.topk for _ in range(all2all_world_size)]
                    send_split_sizes[self.rank] = 0
                    recv_split_sizes[self.rank] = 0

                    print(f"Rank {self.rank} send_split_sizes: {send_split_sizes}, recv_split_sizes: {recv_split_sizes}")
                    send_tensor = torch.zeros((sum(send_split_sizes), self.hidden_size), dtype=self.dtype, device=self.device)
                    recv_tensor = torch.zeros((sum(recv_split_sizes), self.hidden_size), dtype=self.dtype, device=self.device)

                    with torch.autograd.profiler.record_function("all_to_all_sync"):
                        dist.all_to_all_single(recv_tensor, send_tensor, recv_split_sizes, send_split_sizes, group=self.all2all_group)

                elif rank < all2all_world_size:
                    """
                    1. KV reuse         |  total_num_tokens * reuse_ratio       |  (irecv)
                    2. All to all       |  total_num_tokens                     |  (async)
                    3. P->D KV Transfer |  total_num_tokens * (1 - reuse_ratio) |  (isend)
                    4. Other kv read    |  total num tokens * reuse ratio * 4   |  (isend)
                    """
                    print(f"{rank}: isend {rank} -> {rank + all2all_world_size}, irecv {rank} <- {rank + all2all_world_size}, all2all")
                    p2p_ops_group_0 = []   # p2d isend
                    p2p_ops_group_1 = []   # reuse irecv 
                    p2p_ops_group_2 = []   # other_0 isend
                    p2p_ops_group_3 = []   # other_1 isend
                    p2p_ops_group_4 = []   # other_2 isend
                    p2p_ops_group_5 = []   # other_3 isend

                    handles = []
                    peer = rank + all2all_world_size
                    print(f"irecv {rank} <- {peer}, alltoall {rank}, isend {rank} -> {peer}")

                    nvtx.range_push("p2d_isend(batch)")
                    with torch.autograd.profiler.record_function("p2d_isend(batch)"):
                        for t in self.p2d_kv_tensors:
                            p2p_ops_group_0.append(dist.P2POp(dist.isend, t, peer, self.g_p2d))
                    nvtx.range_pop()

                    nvtx.range_push("kv_reuse_irecv(batch)")
                    with torch.autograd.profiler.record_function("kv_reuse_irecv(batch)"):
                        for t in self.kv_reuse_tensors:
                            p2p_ops_group_1.append(dist.P2POp(dist.irecv, t, peer, self.g_reuse))
                    nvtx.range_pop()

                    nvtx.range_push("other_isend(batch)")
                    with torch.autograd.profiler.record_function("other_isend(batch)"):
                        for i, t in enumerate(self.other_kv_tensors):
                            m = i % 4
                            if m == 0:
                                p2p_ops_group_2.append(dist.P2POp(dist.isend, t, peer, self.g_other0))
                            elif m == 1:
                                p2p_ops_group_3.append(dist.P2POp(dist.isend, t, peer, self.g_other1))
                            elif m == 2:
                                p2p_ops_group_4.append(dist.P2POp(dist.isend, t, peer, self.g_other2))
                            else:
                                p2p_ops_group_5.append(dist.P2POp(dist.isend, t, peer, self.g_other3))
                    nvtx.range_pop()

                    nvtx.range_push("all_to_all_async")
                    with torch.autograd.profiler.record_function("all_to_all_async"):
                        a2a_work = dist.all_to_all_single(
                            self.all2all_recv_tensor, self.all2all_send_tensor,
                            self.recv_split_sizes, self.send_split_sizes,
                            group=self.all2all_group, async_op=True
                        )
                        handles.append(a2a_work)
                    nvtx.range_pop()

                    if p2p_ops_group_0:
                        nvtx.range_push("p2d_batch_launch")
                        with torch.autograd.profiler.record_function("p2d_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_0)
                            handles.extend(works)
                        nvtx.range_pop()

                    if p2p_ops_group_1:
                        nvtx.range_push("reuse_batch_launch")
                        with torch.autograd.profiler.record_function("reuse_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_1)
                            handles.extend(works)
                        nvtx.range_pop()

                    if p2p_ops_group_2:
                        nvtx.range_push("other_0_batch_launch")
                        with torch.autograd.profiler.record_function("other_0_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_2)
                            handles.extend(works)
                        nvtx.range_pop()

                    if p2p_ops_group_3:
                        nvtx.range_push("other_1_batch_launch")
                        with torch.autograd.profiler.record_function("other_1_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_3)
                            handles.extend(works)
                        nvtx.range_pop()

                    if p2p_ops_group_4:
                        nvtx.range_push("other_2_batch_launch")
                        with torch.autograd.profiler.record_function("other_2_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_4)
                            handles.extend(works)
                        nvtx.range_pop()

                    if p2p_ops_group_5:
                        nvtx.range_push("other_3_batch_launch")
                        with torch.autograd.profiler.record_function("other_3_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_5)
                            handles.extend(works)
                        nvtx.range_pop()

                else:
                    """
                    1. KV reuse         (isend)
                    2. P->D KV Transfer (irecv)
                    3. Other kv read    (irecv)
                    """
                    p2p_ops_group_0 = []   # p2d isend
                    p2p_ops_group_1 = []   # reuse irecv 
                    p2p_ops_group_2 = []   # other_0 isend
                    p2p_ops_group_3 = []   # other_1 isend
                    p2p_ops_group_4 = []   # other_2 isend
                    p2p_ops_group_5 = []   # other_3 isend

                    handles = []
                    peer = rank - all2all_world_size
                    print(f"{rank}: isend {rank} -> {peer}, irecv {rank} <- {peer}")

                    nvtx.range_push("p2d_irecv(batch)")
                    with torch.autograd.profiler.record_function("p2d_irecv(batch)"):
                        for t in self.p2d_kv_tensors:
                            p2p_ops_group_0.append(dist.P2POp(dist.irecv, t, peer, self.g_p2d))
                    nvtx.range_pop()

                    nvtx.range_push("kv_reuse_isend(batch)")
                    with torch.autograd.profiler.record_function("kv_reuse_isend(batch)"):
                        for t in self.kv_reuse_tensors:
                            p2p_ops_group_1.append(dist.P2POp(dist.isend, t, peer, self.g_reuse))
                    nvtx.range_pop()

                    nvtx.range_push("other_irecv(batch)")
                    with torch.autograd.profiler.record_function("other_irecv(batch)"):
                        for i, t in enumerate(self.other_kv_tensors):
                            m = i % 4
                            if m == 0:
                                p2p_ops_group_2.append(dist.P2POp(dist.irecv, t, peer, self.g_other0))
                            elif m == 1:
                                p2p_ops_group_3.append(dist.P2POp(dist.irecv, t, peer, self.g_other1))
                            elif m == 2:
                                p2p_ops_group_4.append(dist.P2POp(dist.irecv, t, peer, self.g_other2))
                            else:
                                p2p_ops_group_5.append(dist.P2POp(dist.irecv, t, peer, self.g_other3))
                    nvtx.range_pop()

                    if p2p_ops_group_0:
                        nvtx.range_push("p2d_batch_launch")
                        with torch.autograd.profiler.record_function("p2d_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_0)
                            handles.extend(works)
                        nvtx.range_pop()

                    if p2p_ops_group_1:
                        nvtx.range_push("reuse_batch_launch")
                        with torch.autograd.profiler.record_function("reuse_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_1)
                            handles.extend(works)
                        nvtx.range_pop()

                    if p2p_ops_group_2:
                        nvtx.range_push("other_0_batch_launch")
                        with torch.autograd.profiler.record_function("other_0_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_2)
                            handles.extend(works)
                        nvtx.range_pop()
                    if p2p_ops_group_3:
                        nvtx.range_push("other_1_batch_launch")
                        with torch.autograd.profiler.record_function("other_1_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_3)
                            handles.extend(works)
                        nvtx.range_pop()
                    if p2p_ops_group_4:
                        nvtx.range_push("other_2_batch_launch")
                        with torch.autograd.profiler.record_function("other_2_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_4)
                            handles.extend(works)
                        nvtx.range_pop()
                    if p2p_ops_group_5:
                        nvtx.range_push("other_3_batch_launch")
                        with torch.autograd.profiler.record_function("other_3_batch_launch"):
                            works = dist.batch_isend_irecv(p2p_ops_group_5)
                            handles.extend(works)
                        nvtx.range_pop()

                torch.cuda.synchronize()

            nvtx.range_pop()
            prof.step() 
        prof.__exit__(None, None, None)

        if rank == 0:
            outfile.close()


if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f"Rank {rank}/{world_size} starting on local rank {local_rank}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, device_id=device)
    dist.barrier()
    
    only_all2all = False
    topk = 4
    reuse_ratio = 0.6
    profiler = Profiler(rank, world_size, only_all2all=only_all2all, reused_ratio=reuse_ratio, topk=topk, device=device)
    profiler.warmup()

    dist.barrier()
    print(f"Distributed environment initialized. World size: {world_size}, warmup done.")
    
    profiler.profile()

    print("Profiling complete.")

    dist.destroy_process_group()
