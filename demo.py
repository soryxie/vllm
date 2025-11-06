import os

import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

send_tensor = (torch.arange(2, dtype=torch.float32, device=device)
               + 2 * rank)
recv_tensor = torch.empty(2, dtype=torch.float32, device=device)

send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size)

reqs = dist.batch_isend_irecv([send_op, recv_op])
for r in reqs:
    r.wait()

torch.cuda.synchronize(device)
print(f"Rank {rank} received:", recv_tensor)

dist.destroy_process_group()