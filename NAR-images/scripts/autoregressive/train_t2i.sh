# !/bin/bash
set -x


MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NNODES=$SLURM_JOB_NUM_NODES
NPROC_PER_NODE=8
NODE_RANK=$SLURM_NODEID

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  autoregressive/train/train_t2i_webdata.py \
  "$@"
