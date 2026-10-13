# CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 gpus_test.py
# python -m gpus_test
import torch.distributed as dist
import torch
# print(f"dist available:{dist.is_available()}, dist init:{dist.is_initialized()}")
dist.init_process_group(backend="nccl")
# print(f"dist available:{dist.is_available()}, dist init:{dist.is_initialized()}")
# print(f"rank {dist.get_rank()}, world_size {dist.get_world_size()}")
rank=dist.get_rank()
world_size=dist.get_world_size()
if rank==0:
    a=torch.tensor([1.,2.,3.,4.],device=f"cuda:{rank}")
elif rank==1:
    a=torch.tensor([1.,3.,2.,5.],device=f"cuda:{rank}")
assert a.shape[0]%world_size==0
rank_size=a.shape[0]//world_size
b=torch.empty(rank_size,device=f"cuda:{rank}")
dist.reduce_scatter_single(b,a,dist.ReduceOp.AVG)
print(f"rank={rank}, b={b}")


dist.destroy_process_group()