# module load anaconda3/2024.2
# conda activate cutqc

ip_addr=$(hostname -i)

torchrun \
--nproc_per_node=1 --nnodes=1 --node_rank=0 \
--master_addr=$ip_addr --master_port=0 \
main.py \
--backend=nccl 