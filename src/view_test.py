import torch

sizes = [3, 3]
x = torch.arange (0, 15).reshape (5, 3)
print (x.shape)

shape = x.shape 
flattened = x.view(-1)  # functionally equivalent to x.flatten() in this case (contiguous tensor)

print (f'split {torch.split(flattened, [5, 4, 3, 3])}')
first = flattened[:sizes[0]]
flattened = flattened[sizes[0]:] # rest of the tensor
print (first)
print (flattened)


res = flattened[:sizes[0]]
flattened = flattened[sizes[0]:]

for size in sizes:
  next = flattened[:size]
  res = torch.kron (res, next)
  flattened =flattened[size:]

print (res)

