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

def example_scatter ():  
  dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
  device = torch.device("cuda:{}".format(LOCAL_RANK))
    
  tensor_size = 5
  output_tensor = torch.zeros(tensor_size, dtype=torch.int64).to(device)

  if dist.get_rank() == 0:
      t1 = torch.arange (0, WORLD_SIZE * tensor_size, dtype=torch.int64).reshape(WORLD_SIZE, tensor_size).to(device)  
      scatter_list = list(t1.unbind(dim=0))
      print ("List to scatter: {}".format (scatter_list))
  else:
      scatter_list = None

  dist.scatter(output_tensor, scatter_list, src=0)

  # Rank i gets scatter_list[i]. For example, on rank 1:
  print("Rank {}, received tensor: {}".format(LOCAL_RANK, output_tensor), end="\n\n")

if __name__ == "__main__":
    example_scatter()

