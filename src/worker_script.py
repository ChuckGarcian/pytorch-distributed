import os
# gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

# LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)

# print("Local Rank: {}".format(LOCAL_RANK))
print("World Rank: {}".format(WORLD_RANK))
print("World Size: {}".format(WORLD_SIZE))
