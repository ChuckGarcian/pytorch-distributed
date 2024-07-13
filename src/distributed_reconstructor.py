import os
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
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
      kronecker_components.append (torch.rand (self.tensor_size))
    
    print ("Generated Component list Length: {}".format(len(kronecker_components)))
    return kronecker_components

  def compute(self):
      dataset_size = 8
      summation_terms_sequence = []
      
      for _ in range(dataset_size):
          summation_term = self.get_paulibase_probability()
          summation_terms_sequence.append(torch.stack(summation_term, dim=0))
    
      dataset = TensorDataset(torch.stack(summation_terms_sequence))
      
      print("Dataset: {}".format(dataset))
      print ("First Element type, dataset[0]): {}".format((dataset[0])))
      
      dist_sampler = DistributedSampler (dataset, num_replicas=WORLD_SIZE, rank=WORLD_RANK)
      data_loader = DataLoader (dataset, sampler=dist_sampler, shuffle=False)
    
      siz = 0
      for x in data_loader:
         siz += 1
         print ("Printing Tensor X.shape: {}".format(x), end="\n\n")

      print (siz)
      return dataset

def main ():
  print ("Creating Distributed Reconstructor")
  if (WORLD_RANK == 0):
    dr = DistributedReconstructor (3, 5)
    print ("Calling compute")
    dr.compute ()
  else 
  


main ()