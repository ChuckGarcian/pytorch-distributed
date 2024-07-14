import torch

def fn (v):
  # print (v) 
  return torch.sum (v)

def main ():
  # Create matrix rxc=2x5 with elements from 0 to 9
  t1 = torch.arange (0, 10).reshape (2, 5)
  print ("t1: {}".format(t1))
  
  # Vectorize Fn
  vec_fn = torch.func.vmap (fn, in_dims=0)
  res = vec_fn (t1)
  print ("vec_fn (t1): {}".format(res))
  

if __name__ == "__main__":
  main ()