# Title: distributed_reconstructor.py
# Author: Charles "Chuck" Garcia

import os
import torch
import torch.distributed as dist
import numpy as np
from typing import List
import copy



def MSE(target, obs):
    """
    Mean Square Error
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    if isinstance(target, dict):
        se = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    elif isinstance(target, np.ndarray) and isinstance(obs, np.ndarray):
        target = target.reshape(-1, 1)
        obs = obs.reshape(-1, 1)
        squared_diff = (target - obs) ** 2
        se = np.sum(squared_diff)
        mse = np.mean(squared_diff)
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        se = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    else:
        raise Exception("target type : %s" % type(target))
    return mse

# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
MASTER_RANK = 0
backend = "nccl"

def get_difference (actual, expected):
    diff = torch.abs (expected - actual)
    return torch.min (diff)

class DistributedReconstructor:
  def __init__(self, tensor_sizes=[3, 4, 5]):
      self.dyn_size = 0
      self.tensor_sizes = tensor_sizes
      self.max_effective = np.max (self.tensor_sizes) # Used to pad
      self.result_size = np.prod (self.tensor_sizes)  # Result Tensor Size
      self.reference = torch.zeros (self.result_size) # Compared to actual result
      
  def get_paulibase_probability (self):
    '''
    Emulates the subcircuit outs measured in a paulibase - the returned list is 
    to be kroneckerd
    '''
    kronecker_components = []
    ref_comp = None
    
    # add to result list and compute refference
    for size in self.tensor_sizes:
      new_comp = torch.ones (size, dtype=torch.int64, device = 'cpu') * self.dyn_size

      if (ref_comp == None):
        ref_comp = new_comp
      else :
         ref_comp = torch.kron (ref_comp, new_comp)

      pad_amount = self.max_effective - size 
      new_comp = torch.nn.functional.pad (new_comp, (0, pad_amount)) 
      print (new_comp[0:size] , end='\n\n')      
  

      kronecker_components.append (new_comp)
      self.dyn_size +=1
    
    self.reference += ref_comp
    return kronecker_components

  def compute(self):
      dataset_size = 8
      summation_terms_sequence = []
      
      # Build up Sequence of uncomputed kronecker product tuples
      for _ in range(dataset_size):
          summation_term = self.get_paulibase_probability()
          summation_terms_sequence.append(torch.stack(summation_term, dim=0))
    
      # Batch all uncomputed product tuples into batches
      batches = torch.stack(summation_terms_sequence).chunk (chunks=(WORLD_SIZE - 1))
      tensor_sizes_data = torch.tensor(self.tensor_sizes, dtype=torch.int64).cuda () # Used to strip zero padding 

      # Send off to nodes for compute
      for dst_rank, batch in enumerate(batches):
         shape_data = batch.shape
         tensor_sizes_shape = tensor_sizes_data.shape 
         dist.send (torch.tensor(tensor_sizes_shape, dtype=torch.int64).cuda(), dst=dst_rank+1) 
         dist.send (tensor_sizes_data, dst=dst_rank+1)
         dist.send (torch.tensor(shape_data).cuda(), dst=dst_rank+1) 
         dist.send (batch.cuda (),  dst=dst_rank+1) 
      
      print ("About to reduce!")
      buff = torch.zeros (self.result_size, dtype=torch.int64).cuda ()
      dist.reduce(buff, dst=MASTER_RANK, op=dist.ReduceOp.SUM)


      print ("Difference MSE: {}".format(MSE (self.reference.cpu ().numpy(), buff.cpu().numpy())), flush=True)
      print ("Difference diff: {}".format(get_difference (self.reference.cpu(), buff.cpu())), flush=True)
                                
def vec_kronecker (components, tensor_sizes):
  '''
  Vectorized kronecker product of the tensor in components
  '''  
  components = torch.unbind (components, dim=0)
  
  val = tensor_sizes [0]
  res = (components [0]) [0:val] 
  
  i = 1
  for kron_prod in components[1:]:
    idx = tensor_sizes[i]
    res = torch.kron (res, kron_prod[0:idx])
    i += 1

  return res

def single_node (device):
    # -- Represents Computation On a SINGLE node --
    
    # Receive Tensor list information
    tensor_sizes_shape = torch.zeros([1], dtype=torch.int64, device=device) 
    dist.recv (tensor=tensor_sizes_shape, src = MASTER_RANK)     
    tensor_sizes = torch.zeros (tuple(tensor_sizes_shape), dtype=torch.int64, device=device) 
    dist.recv (tensor=tensor_sizes, src = MASTER_RANK)    
    print ("tensor_sizes: {}".format(tensor_sizes))

    # Get shape of the batch we are receiving 
    shape_tensor = torch.zeros([3], dtype=torch.int64, device=device) 
    dist.recv (tensor=shape_tensor, src = MASTER_RANK) 
    print ("Rank 1, shapetuple = {}".format(shape_tensor))
    
    # Create and empty batch tensor and recieve it's data
    batch_recieved = torch.empty (tuple(shape_tensor), dtype=torch.int64, device=device) 
    dist.recv (tensor=batch_recieved, src = MASTER_RANK)    
    
    # Call vectorized kronecker and do sum reduce on this node
    lambda_fn = lambda x: vec_kronecker (x, tensor_sizes)
    vec_fn = torch.func.vmap (lambda_fn)
    res = vec_fn (batch_recieved)    
    res = res.sum (dim=0)

    # Send Back to master
    dist.reduce (res, dst=MASTER_RANK, op=dist.ReduceOp.SUM)
    

def main (backend):
  print ("Creating Distributed Reconstructor")
  print(f"Hello world! This is worker {WORLD_RANK} speaking. I have {WORLD_SIZE - 1} siblings!")
  
  # Set Device 
  device = torch.device("cuda:{}".format(LOCAL_RANK))
  torch.cuda.device(device)
  
  # Master Process collect all that needs to be computed
  if (WORLD_RANK == MASTER_RANK):
    dr = DistributedReconstructor ()
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
