# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
autoregressive/train/train_c2i.py \
"$@"
