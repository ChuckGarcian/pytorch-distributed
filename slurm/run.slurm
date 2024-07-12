#!/bin/bash
#SBATCH --job-name=pytorch-dist    
#SBATCH --mail-user=cg1509@princeton.edu  
#SBATCH --output=../_output/%x.%j.out
#SBATCH --mail-type=begin         # send email when job begins
#SBATCH --mail-type=end           # send email when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1       # total number of tasks across all nodes
#SBATCH --cpus-per-task=1         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1G                  # memory per cpu-core (4G is default)
#SBATCH --time=00:00:05           # total run time limit (HH:MM:SS)

# Setup for Multi-node Workload
export MASTER_PORT=$(get_free_port)  # Get a free Port
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) # Master node shall be the first hostname
export MASTER_ADDR=$master_addr

# Sanity Print
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

# Load Modules
module purge
module load anaconda3/2024.2
conda activate cutqc

# Run Job
script_name=$1
srun python $script_name
