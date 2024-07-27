import torch
import numpy as np

def main ():
  # Create matrix rxc=2x5 with elements from 0 to 9
  t1 = torch.arange (1, 16*3*4 + 1).reshape (16, 3, 4)
  print ("t1.shape: {}".format(t1.shape))

  t1_chunk = t1.chunk (chunks=7)
  t1_split = t1.tensor_split (7)
  print (f"t1.chunk(7): {t1_chunk}")  
  print (f"t1.chunk(7) length: {len(t1_chunk)}")  
  
  print (f"t1.tensor_split (7): {t1_split}")
  print (f"t1.tensor_split (7) length: {len(t1_split)}")

if __name__ == "__main__":
  main ()