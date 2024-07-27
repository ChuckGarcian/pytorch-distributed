import torch


def main ():
  # Create a 1D tensor (vector)
  t1 = torch.arange (0, 10)
  t1 = torch.nn.functional.pad (t1, (0, 4))
  print ("t1: {}".format (t1))
  
  # Select with index s.t. its equivelent to t1[0:4]
  print ("tensor_select [0:2] ", t1[0:4])
  print ("Tensor_select (t1, 0, 2): {}".format(torch.select(t1, 0, 4)))

main ()