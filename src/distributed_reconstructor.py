# Title: distributed_reconstructor.py
# Author: Charles "Chuck" Garcia

import os
from typing import List
import torch
import torch.distributed as dist
import random

# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
MASTER_RANK = 0
backend = "nccl"

def get_difference (expected, actual):
    diff = torch.abs (expected - actual)
    return torch.min (diff)

class DistributedReconstructor:
  def __init__(self, _smart_order, _tensor_size):
      self.subcircuit_entry_lengths = [2, 2, 4]
      self.tensor_size = _tensor_size
      self.dyn_size = 0
      self.refference = torch.zeros (self.tensor_size**self.smart_order)

  def get_paulibase_probability (self):
    '''
    Psuedo subcircuit probability measured in a paulibase - the returned list is 
    to be kroneckerd
    '''
    kronecker_components = []
    size_components = []
    ref_comp = None
    
    # add to result list and compute refference
    for size in self.subcircuit_entry_lengths:
      new_comp = torch.rand (size, device = 'cpu') * self.dyn_size
      
      if (ref_comp == None): ref_comp = new_comp
      else : ref_comp = torch.kron (ref_comp, new_comp)
      
      kronecker_components.append (new_comp)
      self.dyn_size +=1
    
    self.refference += ref_comp
    print ("Generated Component list Length: {}".format(len(kronecker_components)))
    return kronecker_components, size_components

  def compute(self):
      dataset_size = 8
      summation_terms_sequence = []
      
      # Build up Sequence of uncomputed kronecker product tuples
      for _ in range(dataset_size):
          summation_term = self.get_paulibase_probability()
          summation_terms_sequence.append(torch.nested.nested(summation_term, dim=0))
    
      # Batch all uncomputed product tuples into batches
      batches = torch.stack(summation_terms_sequence).chunk (chunks=(WORLD_SIZE - 1))
      
      # Send off to nodes for compute
      for dst_rank, batch in enumerate(batches):
         shape_data = batch.shape
         dist.send (torch.tensor(shape_data).cuda(), dst=dst_rank+1) 
         dist.send (batch.cuda (),  dst=dst_rank+1) 

      buff = torch.zeros (self.tensor_size**self.smart_order).cuda ()
      dist.reduce(buff, dst=MASTER_RANK, op=dist.ReduceOp.SUM)
      
      print ("Difference: {}".format(get_difference(buff, self.refference.cuda ())))
                                
@torch.jit.script
def vec_kronecker(components) -> torch.Tensor:
    res = components[0]
    for kron_prod in components[1:]:
        res = torch.kron(res, kron_prod)
    return res

def single_node (device):
    # -- Represents Computation On a SINGLE node --

    # Get shape of the batch we are receiving 
    shape_tensor = torch.zeros([3], dtype=torch.int64, device=device) 
    dist.recv (tensor=shape_tensor, src = MASTER_RANK) 
    
    # Create and empty batch tensor and recieve it's data
    print ("Rank 1, shapetuple = {}".format(shape_tensor))
    batch_recieved = torch.empty (tuple(shape_tensor), device=device) 
    dist.recv (tensor=batch_recieved, src = MASTER_RANK)    
  
    # Call vectorized kronecker and do sum reduce on this node
    res = torch.func.vmap (vec_kronecker) (batch_recieved)
    res = res.sum (dim=0)
    print ("Res: {}".format (res.shape)) 
    
    # Send Back to master
    dist.reduce(res, dst=MASTER_RANK, op=dist.ReduceOp.SUM)
    

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
    single_node (device)
    
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

