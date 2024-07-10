#!/bin/bash
#SBATCH --output=_output/%x.%j.out
#SBATCH --time=00:2:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100

module purge
module load anaconda3/2024.2
conda activate cutqc



### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  --master_addr=127.0.0.1 --master_port=0 main.py --backend=nccl

# echo Start
# valgrind --tool=helgrind python3 example.py
# echo Done

# SBATCH --nodes=1                    # Node Count
# SBATCH --ntasks=1                   # Total number of tasks across all nodes
# SBATCH --mail-user=cg1509@princeton.edu # Output Email
# SBATCH --cpus-per-task=2            # CPU-cores per task (>1 if multi-threaded tasks)
# SBATCH --mem=32G                    # Memory per cpu-core (4G is default)
# SBATCH --job-name=pytorch-dist      # Job Name
