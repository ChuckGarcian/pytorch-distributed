import os
import torch
import torch.distributed as dist

# Environment variables set by slurm script
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
MASTER_RANK = 0
backend = "gloo"

# Host (Rank 0) code
def host_distribute_data():
    with torch.no_grad():
        # Prepare data
        batches = torch.arange(0, 10, dtype=torch.int64).tensor_split(WORLD_SIZE - 1)
        tensor_sizes = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        tensor_sizes_shape = torch.tensor(tensor_sizes.shape, dtype=torch.int64)
        
        print(f"Host: tensor_sizes_data.shape: {tensor_sizes_shape}", flush=True)
        
        op_list = []
        # List of sending objects
        for dst, batch in enumerate(batches, start=1):
            print(f"Host: Sending to dst={dst}", flush=True)
            print(f"Host: batch={batch}", flush=True)
            op_list.extend([
                dist.P2POp(dist.isend, tensor_sizes_shape, dst),
                dist.P2POp(dist.isend, tensor_sizes, dst),
                dist.P2POp(dist.isend, torch.tensor(batch.shape), dst),
                dist.P2POp(dist.isend, batch, dst),
            ])
        
        handles = dist.batch_isend_irecv(op_list)
        dist.barrier()
        print("Host: Finished", flush=True)

# Worker (Rank > 0) code
def worker_receive_data():
    with torch.no_grad():
        tensor_sizes_shape = torch.zeros([1], dtype=torch.int64)
        dist.recv(tensor=tensor_sizes_shape, src=0)
        print(f"Worker {dist.get_rank()}: tensor_sizes_shape={tensor_sizes_shape}", flush=True)

        tensor_sizes = torch.zeros(tensor_sizes_shape, dtype=torch.int64)
        dist.recv(tensor=tensor_sizes, src=0)
        print(f"Worker {dist.get_rank()}: tensor_sizes={tensor_sizes}", flush=True)

        # Receive batch shape and data
        batch_shape = torch.zeros_like(tensor_sizes_shape)
        dist.recv(tensor=batch_shape, src=0)
        print(f"Worker {dist.get_rank()}: batch_shape={batch_shape}", flush=True)

        batch = torch.zeros(batch_shape, dtype=torch.int64)
        dist.recv(tensor=batch, src=0)
        print(f"Worker {dist.get_rank()}: batch={batch}", flush=True)
        print(f"Worker {dist.get_rank()}: Done!", flush=True)
        
        dist.barrier()

# Main process
def main():
    # Initialize process group
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    
    # Distribute
    if dist.get_rank() == 0:
        host_distribute_data()
    else:
        worker_receive_data()

if __name__ == "__main__":
    main()

