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
    def __init__(self, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.nodes = None
        self.hidden_size = 2048
        self.dtype = torch.bfloat16
        self.num_experts = 128
        self.assignments = load_expert_assignments("vllm/profile_json/ep_states_prompt_8_token_1024.jsonl")
        self.layer_num = len(self.assignments[0])

        self.experts_per_rank = self.num_experts // self.world_size
        self.local_expert_start = rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank 
        self.local_expert_ids = set(range(self.local_expert_start, self.local_expert_end)) 

    def warmup(self):
        send_tensor = torch.zeros((1024, 2048), dtype=torch.bfloat16).cuda()
        recv_tensor = torch.zeros((1024, 2048), dtype=torch.bfloat16).cuda()
        dist.all_to_all_single(recv_tensor, send_tensor)
    
    def init_send_recv(self, layer_idx):
        self.nodes = [Node(i, self.world_size) for i in range(self.world_size)]
        for rank in range(self.world_size):
            for token_idx, topk_ids in enumerate(self.assignments[rank][layer_idx]): 
                for expert_id in topk_ids: 
                    target_rank = expert_id // self.experts_per_rank
                    if target_rank == rank:
                        continue
                    self.nodes[rank].send_state[target_rank]["count"] += 1
                    self.nodes[rank].send_state[target_rank]["token_indices"].append(token_idx)
                    self.nodes[target_rank].recv_state[rank]["count"] += 1
                    self.nodes[target_rank].recv_state[rank]["token_indices"].append(token_idx)

    def profile(self):
        if rank == 0:
            output_filename = f"vllm/profile_json/moe_profile_8card.jsonl"
            outfile = open(output_filename, "w")

        for layer_idx in range(self.layer_num):
            if rank == 0:
                print(f"Processing layer {layer_idx}")

            self.init_send_recv(layer_idx)

            send_split_sizes = [self.nodes[self.rank].send_state[dst]["count"] for dst in range(self.world_size)]
            recv_split_sizes = [self.nodes[self.rank].recv_state[src]["count"] for src in range(self.world_size)]
            
            print(f"Rank {self.rank} send_split_sizes: {send_split_sizes}, recv_split_sizes: {recv_split_sizes}")
            send_tensor = torch.zeros((sum(send_split_sizes), self.hidden_size), dtype=self.dtype, device=self.device)
            recv_tensor = torch.zeros((sum(recv_split_sizes), self.hidden_size), dtype=self.dtype, device=self.device)

            dist.barrier()
            torch.cuda.synchronize()
            start_time = time.time()
            dist.all_to_all_single(recv_tensor, send_tensor, recv_split_sizes, send_split_sizes)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            result = {
                "layer": layer_idx,
                "rank": self.rank,
                "send_counts": send_split_sizes,
                "recv_counts": recv_split_sizes,
                "time_ms": elapsed_time,
            }
            if rank == 0:
                outfile.write(json.dumps(result) + "\n")
                outfile.flush()

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
    
    profiler = Profiler(rank, world_size, device)
    profiler.warmup()

    if rank == 0 or rank == 2:
        print(f"Distributed environment initialized. World size: {world_size}, warmup done.")
    
    profiler.profile()

    if rank == 0 or rank == 2:
        print("Profiling complete.")

    dist.destroy_process_group()
