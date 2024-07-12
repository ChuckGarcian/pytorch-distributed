# Title: dist_demo.py
# Author: Charles "Chuck" Garcia

import os
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
    Test/Toy kernel to run on a node: Each node will say hi
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

def run2(backend):
    """
    Test/Toy kernel to run on a node: Node 0 will send node 1 a tensor
    """         
    tensor = torch.zeros(1)
    
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('worker_{} has received data from rank {}\n'.format(WORLD_RANK, 0))

def init_processes(backend):
    print (f"Called init_processes, backend={backend}")
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    print("Exited init_process_group")
    run2(backend)

if __name__ == "__main__":
    print(f"args.backend:{backend}")
    print("Local Rank: {}".format(LOCAL_RANK))
    print("World Rank: {}".format(WORLD_RANK))
    print("World Size: {}".format(WORLD_SIZE))
    init_processes(backend=backend)

