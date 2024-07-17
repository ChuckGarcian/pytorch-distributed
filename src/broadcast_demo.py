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


def example_broadcast():
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    device = torch.device("cuda:{}".format(LOCAL_RANK))
                
    
    # Broadcast if host and receive if worker
    if (LOCAL_RANK == 0):
      # Create matrix rxc=2x5 with elements from 0 to 9
      t1 = torch.arange (0, 10).reshape (2, 5).to (device)
      print ("Host, tensor broadcasting: {}".format(t1), end="\n\n")

      # Broadcast it
      x = dist.broadcast (t1, src=0, async_op=True)
    else:
      # Verify 
      received_tensor = torch.zeros (5, dtype=torch.int64).to (device)
      y = dist.broadcast (received_tensor, src=0, async_op=True)
      
      print ("Rank {}, received tensor: {}".format (LOCAL_RANK, received_tensor), end="\n\n")
        

if __name__ == "__main__":
    example_broadcast()