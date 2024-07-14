# Title: batched_kron_demo.py
# Author: Charles "Chuck" Garcia
# Description: Demonstration of batching a kronecker op

import torch
from torch.utils.data import TensorDataset, DataLoader

dataset_size = 8
smart_order = 3
tensor_size = 5
dyn_size = 0

def vec_kronecker(components):
    '''
    Vectorized kronecker product of the tensor in components
    '''
    components = torch.unbind(components, dim=0)
    res = components[0]
    
    for kron_prod in components[1:]:
        res = torch.kron(res, kron_prod)
    
    return res

def list_of_vectors():
    '''
    Returns list of random tensor/vectors
    '''
    global dyn_size
    res = []
    
    for _ in range(smart_order):
        res.append(torch.ones(tensor_size, device='cpu') * dyn_size)
        dyn_size += 1
    
    print("Generated tensor list Length: {}".format(len(res)))
    return res

def compute():
    summation_terms_sequence = []
    
    # Build up Sequence of uncomputed kronecker pruducts
    for _ in range(dataset_size):
        summation_term = list_of_vectors()
        summation_terms_sequence.append(torch.stack(summation_term, dim=0))
    
    res = torch.func.vmap(vec_kronecker)(torch.stack(summation_terms_sequence))
    print("Res: {}".format(res.shape))


compute()