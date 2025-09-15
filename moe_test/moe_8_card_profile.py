import os
import sys
import json
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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
    def __init__(self, rank, world_size, only_all2all, reused_ratio, device):
        self.rank = rank
        self.world_size = world_size
        self.only_all2all = only_all2all
        self.reused_ratio = reused_ratio
        self.device = device
        self.nodes = None

        if only_all2all:
            self.all2all_group = dist.new_group(ranks=list(range(world_size)))
            self.all2all_world_size = world_size
        else:
            self.all2all_group = dist.new_group(ranks=list(range(0, world_size // 2)))
            self.all2all_world_size = world_size // 2

        # tokens
        self.num_tokens_all2all = 1024
        self.num_tokens_total = int(self.num_tokens_all2all / (1 - self.reused_ratio))
        self.num_tokens_reuse = self.num_tokens_total - self.num_tokens_all2all

        # model
        self.kv_heads = 16
        self.kv_dim = 128
        self.hidden_size = 2048
        self.dtype = torch.bfloat16
        self.num_experts = 128
        self.experts_per_rank = self.num_experts // self.all2all_world_size
        self.local_expert_start = rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank 
        self.local_expert_ids = set(range(self.local_expert_start, self.local_expert_end)) 

        # data
        self.assignments = load_expert_assignments("/profile_json/ep_states_prompt_8_token_1024.jsonl")
        self.layer_num = len(self.assignments[0])

    def warmup(self):
        send_tensor = torch.zeros((1024, 2048), dtype=self.dtype, device=self.device)
        recv_tensor = torch.zeros((1024, 2048), dtype=self.dtype, device=self.device)
        dist.all_to_all_single(recv_tensor, send_tensor)
    
    def init_send_recv(self, layer_idx):
        all2all_world_size = self.all2all_world_size
        self.nodes = [Node(i, all2all_world_size) for i in range(all2all_world_size)]
        for rank in range(self.all2all_world_size):
            for token_idx, topk_ids in enumerate(self.assignments[rank][layer_idx]): 
                for expert_id in topk_ids[:4]:  # top-4 experts, communication optimization
                    target_rank = expert_id // self.experts_per_rank
                    if target_rank == rank:
                        continue
                    self.nodes[rank].send_state[target_rank]["count"] += 1
                    self.nodes[rank].send_state[target_rank]["token_indices"].append(token_idx)
                    self.nodes[target_rank].recv_state[rank]["count"] += 1
                    self.nodes[target_rank].recv_state[rank]["token_indices"].append(token_idx)

    def profile(self):
        rank = self.rank
        all2all_world_size = self.all2all_world_size

        if rank == 0:
            output_filename = f"/profile_json/moe_profile_8card.jsonl"
            outfile = open(output_filename, "w")

        for layer_idx in [0 for _ in range(11)]:# range(self.layer_num):
            if rank == 0:
                print(f"Processing layer {layer_idx}")

            self.init_send_recv(layer_idx)

            if self.only_all2all:
                """
                Only All to all       (sync)
                """
                send_split_sizes = [self.nodes[self.rank].send_state[dst]["count"] for dst in range(all2all_world_size)]
                recv_split_sizes = [self.nodes[self.rank].recv_state[src]["count"] for src in range(all2all_world_size)]
                
                print(f"Rank {self.rank} send_split_sizes: {send_split_sizes}, recv_split_sizes: {recv_split_sizes}")
                send_tensor = torch.zeros((sum(send_split_sizes), self.hidden_size), dtype=self.dtype, device=self.device)
                recv_tensor = torch.zeros((sum(recv_split_sizes), self.hidden_size), dtype=self.dtype, device=self.device)

                torch.cuda.synchronize()
                start_time = time.time()
                dist.all_to_all_single(recv_tensor, send_tensor, recv_split_sizes, send_split_sizes, group=self.all2all_group)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                result = {
                    "layer": layer_idx,
                    "rank": self.rank,
                    "send_counts": send_split_sizes,
                    "senf_shapes": list(send_tensor.shape),
                    "recv_counts": recv_split_sizes,
                    "recv_shapes": list(recv_tensor.shape),
                    "time_ms": elapsed_time,
                }
                if rank == 0:
                    outfile.write(json.dumps(result) + "\n")
                    outfile.flush()

            elif rank < all2all_world_size:
                """
                1. KV reuse         |  total_num_tokens * reuse_ratio       |  (irecv)
                2. All to all       |  total_num_tokens                     |  (async)
                3. P->D KV Transfer |  total_num_tokens * (1 - reuse_ratio) |  (send)
                """
                # 1. KV reuse         (irecv)
                kv_reuse_recv_tokens = self.num_tokens_reuse
                kv_reuse_recv_block_nums = (kv_reuse_recv_tokens + 127) // 128
                kv_reuse_recv_tensors = [torch.zeros(
                    (2, 128, 4, 128), dtype=self.dtype, device=self.device)
                    for _ in range(kv_reuse_recv_block_nums)]

                # 2. All to all       (async)
                send_split_sizes = [self.nodes[self.rank].send_state[dst]["count"] for dst in range(all2all_world_size)]
                recv_split_sizes = [self.nodes[self.rank].recv_state[src]["count"] for src in range(all2all_world_size)]
                
                print(f"Rank {self.rank} send_split_sizes: {send_split_sizes}, recv_split_sizes: {recv_split_sizes}")
                all2all_send_tensor = torch.zeros((sum(send_split_sizes), self.hidden_size), dtype=self.dtype, device=self.device)
                all2all_recv_tensor = torch.zeros((sum(recv_split_sizes), self.hidden_size), dtype=self.dtype, device=self.device)

                # 3. P->D KV Transfer (send)
                p2d_kv_send_tokens = self.num_tokens_all2all
                p2d_kv_send_block_nums = (p2d_kv_send_tokens + 127) // 128
                p2d_kv_send_tensors = [torch.zeros(
                    (2, 128, 4, 128), dtype=self.dtype, device=self.device)
                    for _ in range(p2d_kv_send_block_nums)]

                # Execute
                handles = []
                print(f"irecv {rank} <- {rank + all2all_world_size}, alltoall {rank}, isend {rank} -> {rank + all2all_world_size}")
                torch.cuda.synchronize()
                start_time = time.time()
                for kv_reuse_recv_tensor in kv_reuse_recv_tensors:
                    handles.append(dist.irecv(kv_reuse_recv_tensor, src=rank + all2all_world_size))
                all2all_handle = dist.all_to_all_single(all2all_recv_tensor, all2all_send_tensor, recv_split_sizes, send_split_sizes, group=self.all2all_group, async_op=True)
                for p2d_kv_send_tensor in p2d_kv_send_tensors:
                    handles.append(dist.isend(p2d_kv_send_tensor, dst=rank + all2all_world_size))
                all2all_handle.wait()
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                for h in handles:
                    h.wait()
                result = {
                    "layer": layer_idx,
                    "rank": self.rank,
                    "send_counts": send_split_sizes,
                    "send_shapes": list(all2all_send_tensor.shape),
                    "recv_counts": recv_split_sizes,
                    "recv_shapes": list(all2all_recv_tensor.shape),
                    "kv_reuse": [list(t.shape) for t in kv_reuse_recv_tensors],
                    "p2d": [list(t.shape) for t in p2d_kv_send_tensors],
                    "time": elapsed_time,
                }
                if rank == 0:
                    outfile.write(json.dumps(result) + "\n")
                    outfile.flush()

            else:
                """
                1. KV reuse         (isend)
                2. P->D KV Transfer (irecv)
                """
                # 1. KV reuse         (isend)
                kv_reuse_send_tokens = self.num_tokens_reuse
                kv_reuse_send_block_nums = (kv_reuse_send_tokens + 127) // 128
                kv_reuse_send_tensors = [torch.zeros(
                    (2, 128, 4, 128), dtype=self.dtype, device=self.device)
                    for _ in range(kv_reuse_send_block_nums)]

                # 2. P->D KV Transfer (irecv)
                p2d_kv_recv_tokens = self.num_tokens_all2all
                p2d_kv_recv_block_nums = (p2d_kv_recv_tokens + 127) // 128
                p2d_kv_recv_tensors = [torch.zeros(
                    (2, 128, 4, 128), dtype=self.dtype, device=self.device)
                    for _ in range(p2d_kv_recv_block_nums)]

                # Execute
                handles = []
                print(f"isend {rank} -> {rank - all2all_world_size}, irecv {rank} <- {rank - all2all_world_size}")
                torch.cuda.synchronize()
                start_time = time.time()
                for kv_reuse_send_tensor in kv_reuse_send_tensors:
                    handles.append(dist.isend(kv_reuse_send_tensor, dst=rank - all2all_world_size))
                for p2d_kv_recv_tensor in p2d_kv_recv_tensors:
                    handles.append(dist.irecv(p2d_kv_recv_tensor, src=rank - all2all_world_size))
                for h in handles:
                    h.wait()
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time

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
    profiler = Profiler(rank, world_size, only_all2all=only_all2all, reused_ratio=0.6, device=device)
    profiler.warmup()

    dist.barrier()
    print(f"Distributed environment initialized. World size: {world_size}, warmup done.")
    
    profiler.profile()

    print("Profiling complete.")

    dist.destroy_process_group()
