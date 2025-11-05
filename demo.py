import torch.distributed as dist
# import torch

dist.init_process_group(backend='nccl', world_size=4, rank=0)

default_pg = dist.distributed_c10d._get_default_group()
print(default_pg)

# 创建新的组， 只包含部分进程
new_pg = dist.new_group(ranks=[0, 1])

# 在新组中执行通信
tensor = torch.ones(2).cuda()
dist.all_reduce(tensor, group=new_pg)
print(tensor)

