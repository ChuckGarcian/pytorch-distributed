import torch


def main ():
  t1 = torch.tensor ([1, 0])
  t2 = torch.tensor ([1, 0])  
  print ("t1: {}\nt2: {}".format (t1, t2))

  # t1 ⊗ t2 
  res_dot  = torch.tensordot (t1, t2, dims=0).reshape (-1)
  res_kron = torch.kron (t1, t2)
  print ("tensor_dot t1 ⊗ t2: {}".format (res_dot))
  print ("kron t1 ⊗ t2: {}".format (res_kron))

main ()