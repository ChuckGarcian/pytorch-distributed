import torch
import numpy as np

def fn (v, given_list):
  # print (v)
  # print (f"print! {given_list}")
  return torch.sum (v)

def main ():
  # Create matrix rxc=2x5 with elements from 0 to 9
  t1 = torch.arange (0, 10).reshape (2, 5)
  print ("t1: {}".format(t1))
  
  # Vectorize Fn
  list_val = np.array ([0, 1, 2]) # List to pass to vectorized func
  lambda_fn = lambda x: fn (x, list_val)
  vec_fn = torch.func.vmap (lambda_fn)
  res = vec_fn (t1)
  print ("vec_fn (t1): {}".format(res))
  

if __name__ == "__main__":
  main ()