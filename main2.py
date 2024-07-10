import torch
import torch.distributed as dist
import os

def get_mpi_rank():
    return int(os.environ['RANK'])

def get_mpi_size():
    return int(os.environ.get('WORLD_SIZE', '1'))

rank = get_mpi_rank()
world_size = get_mpi_size()

init_param={'backend': 'nccl',
        'init_method': 'env://',
        'rank': rank,
        'world_size': world_size}

from pprint import pformat
print('before {} - {}\n'.format(rank,
    pformat(init_param)))
dist.init_process_group(**init_param)
print('after {}'.format(rank))