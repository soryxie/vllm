import os
import sys
import json
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load_expert_assignments(filepath):
    assignments = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if "topk_ids" in data:
                assignments.append(data["topk_ids"])
    return assignments

def warmup():
    send_tensor = torch.zeros((1024, 2048), dtype=torch.bfloat16).cuda()
    recv_tensor = torch.zeros((1024, 2048), dtype=torch.bfloat16).cuda()
    dist.all_to_all_single(recv_tensor, send_tensor)

def main(rank, world_size):
    warmup()
    hidden_size = 2048        
    dtype = torch.bfloat16    
    num_experts = 128         
    experts_per_rank = num_experts // world_size
    
    local_expert_start = rank * experts_per_rank
    local_expert_end = local_expert_start + experts_per_rank  # 不含end
    local_expert_ids = set(range(local_expert_start, local_expert_end))
    
    output_filename = f"moe_profile_rank{rank}.jsonl"
    outfile = open(output_filename, "w")
    
    assignments = load_expert_assignments("./profile_json/ep_states.jsonl")
    
    for layer_idx, topk_ids_list in enumerate(assignments):
        send_counts = [0] * world_size
        send_token_indices = {dst: [] for dst in range(world_size)} 
        total_tokens = len(topk_ids_list)
        
        my_tokens_indices = list(range(rank, total_tokens, world_size))
        
        for token_idx in my_tokens_indices:
            experts_for_token = topk_ids_list[token_idx]
            for expert_id in experts_for_token:
                target_rank = expert_id // experts_per_rank
                if target_rank == rank:
                    continue
                send_counts[target_rank] += 1
                send_token_indices[target_rank].append(token_idx)
        
        send_tensors = []
        for dst in range(world_size):
            if send_counts[dst] > 0:
                token_tensor = torch.zeros((send_counts[dst], hidden_size), dtype=dtype, device=torch.device('cuda', rank))
                send_tensors.append(token_tensor)
            else:
                send_tensors.append(torch.zeros((0, hidden_size), dtype=dtype, device=torch.device('cuda', rank)))
        if send_tensors:
            input_tensor = torch.cat(send_tensors, dim=0)
        else:
            input_tensor = torch.zeros((0, hidden_size), dtype=dtype, device=torch.device('cuda', rank))
        
        input_split_sizes = [send_counts[dst] for dst in range(world_size)]
        
        counts_tensor = torch.tensor(input_split_sizes, dtype=torch.int, device=torch.device('cuda', rank))
        gather_list = [torch.empty_like(counts_tensor) for _ in range(world_size)]
        dist.barrier() 
        torch.cuda.synchronize()
        start_time = time.time()
        dist.all_gather(gather_list, counts_tensor)
        output_split_sizes = [int(gather_list[src][rank].item()) for src in range(world_size)]
        total_recv = sum(output_split_sizes)
        output_tensor = torch.empty((total_recv, hidden_size), dtype=dtype, device=torch.device('cuda', rank))
        dist.all_to_all_single(output_tensor, input_tensor, output_split_sizes, input_split_sizes)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        result = {
            "layer": layer_idx,
            "rank": rank,
            "send_counts": send_counts,
            "recv_counts": output_split_sizes,
            "time_ms": elapsed_time,
        }
        outfile.write(json.dumps(result) + "\n")
        outfile.flush()
    
    outfile.close()
    dist.barrier()
    dist.destroy_process_group()

def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.set_device(rank)
    fn(rank, size)

if __name__ == "__main__":
    world_size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, main))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
