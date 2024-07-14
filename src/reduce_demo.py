import os
import torch
import torch.distributed as dist


# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
MASTER_RANK = 0
backend = "nccl"


def example_reduce():
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    # Create a tensor
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    tensor = (torch.ones(5) * (LOCAL_RANK + 1)).to (device)
    print(f"Rank {LOCAL_RANK} has data: {tensor}")
    
    # Reduce the tensor
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    
    # Print the result on LOCAL_RANK 0
    if LOCAL_RANK == 0:
        print(f"Rank 0 has reduced data: {tensor}")
    
    # Clean up

if __name__ == "__main__":
    example_reduce()