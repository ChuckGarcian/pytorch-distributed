import os
import argparse

import torch
import torch.distributed as dist

# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
backend = "nccl"

def run(backend):
    """
    Test/Toy kernel to run on a node
    """
    print (f"Called run, backend={backend}")    
    if WORLD_RANK == 0:
        print(
            f"Hello world! This is worker {WORLD_RANK} speaking. I have {WORLD_SIZE - 1} siblings!"
        )
    
    if WORLD_RANK == 1:
        print(
            f"Hello world! This is worker {WORLD_RANK} speaking. I have {WORLD_SIZE - 1} siblings!"
        )

def init_processes(backend):
    print (f"Called init_processes, backend={backend}")
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    print("Exited init_process_group")
    run(backend)

if __name__ == "__main__":
    print(f"args.backend:{backend}")
    print("Local Rank: {}".format(LOCAL_RANK))
    print("World Rank: {}".format(WORLD_RANK))
    print("World Size: {}".format(WORLD_SIZE))
    init_processes(backend=backend)

