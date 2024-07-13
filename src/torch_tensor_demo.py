import torch
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

def tensor_demo ():  
  '''
  Testing how pytorch Tensors and datasets work
  '''
  cgl = []
  t1 = torch.arange (0, 10)
  t2 = torch.arange (10, 20)
  t3 = torch.arange (20, 30)
  
  cgl.append (t1)
  cgl.append (t2)
  cgl.append (t3)
  
  # Stacking 1D tensors into a matrix
  stacked_tensors = torch.stack (cgl, dim=0)
  print (TensorDataset (stacked_tensors))
  print (stacked_tensors.shape)
  
  # Retrieving back the 1D tensors
  unstacked_tensor_list = torch.unbind (stacked_tensors, dim=0)
  print (unstacked_tensor_list)
  print (cgl)
  
tensor_demo ()