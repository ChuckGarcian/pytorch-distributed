import os
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
MASTER_RANK = 0
backend = "nccl"


class DistributedReconstructor:
  def __init__(self, _smart_order, _tensor_size):
      self.smart_order = _smart_order
      self.tensor_size = _tensor_size

  def get_paulibase_probability (self):
    '''
    Emulates the subcircuit outs measured in a paulibase - the returned list is 
    to be kroneckerd
    '''
    kronecker_components = []
    
    for _ in range (self.smart_order):
      kronecker_components.append (torch.rand (self.tensor_size, device = 'cpu'))
    
    print ("Generated Component list Length: {}".format(len(kronecker_components)))
    return kronecker_components

  def compute(self):
      dataset_size = 8
      summation_terms_sequence = []
      
      # Build up Sequence of uncomputed kronecker pruducts
      for _ in range(dataset_size):
          summation_term = self.get_paulibase_probability()
          summation_terms_sequence.append(torch.stack(summation_term, dim=0))
    
      dataset = TensorDataset(torch.stack(summation_terms_sequence))      
      # dist_sampler = DistributedSampler (dataset, num_replicas=WORLD_SIZE, rank=WORLD_RANK)
      data_loader = DataLoader (dataset, batch_size=dataset_size//(WORLD_SIZE - 1), shuffle=False)
    
      siz = 0
      for siz, batch in enumerate (data_loader):
         # Prepare to send
         batch_as_tensor = torch.stack(batch, dim=0)
         shape_data = batch_as_tensor.shape
         
         # Send
         print ("Rank 0: {}".format (batch_as_tensor))
         dist.send (torch.tensor(shape_data).cuda(), dst=1) 
         dist.send (batch_as_tensor.cuda (), dst=1) 
        
                        
      print (siz)
      return dataset


def main (backend):
  print ("Creating Distributed Reconstructor")
  print(f"Hello world! This is worker {WORLD_RANK} speaking. I have {WORLD_SIZE - 1} siblings!")
  
  # Set Device 
  device = torch.device("cuda:{}".format(LOCAL_RANK))
  torch.cuda.device(device)
  
  # Master Process collect all that needs to be computed
  if (WORLD_RANK == MASTER_RANK):
    dr = DistributedReconstructor (3, 5)
    print ("Calling compute")
    dr.compute ()
  else:
    
    # Get shape of the batch we are receiving 
    shape_tensor = torch.zeros([4], dtype=torch.int64, device=device) 
    dist.recv (tensor=shape_tensor, src = MASTER_RANK) 
    
    # Create and empty batch tensor and recieve it's data
    print ("Rank 1, shapetuple = {}".format(shape_tensor))
    batch_recieved = torch.empty (tuple(shape_tensor), device=device) 
    dist.recv (tensor=batch_recieved, src = MASTER_RANK)    
    print (batch_recieved)

def init_processes(backend):
    print (f"Called init_processes, backend={backend}")
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    print("Exited init_process_group")
    main (backend)

if __name__ == "__main__":
    print(f"args.backend:{backend}")
    print("Local Rank: {}".format(LOCAL_RANK))
    print("World Rank: {}".format(WORLD_RANK))
    print("World Size: {}".format(WORLD_SIZE))
    print ("GPUS-Avail: {}".format (gpus_per_node))
    init_processes(backend=backend)

