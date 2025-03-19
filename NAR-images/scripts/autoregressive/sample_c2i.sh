# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
autoregressive/sample/sample_c2i_ddp.py \
--vq-ckpt /path/to/vq_ckpt \
--gpt-ckpt /path/to/model_ckpt \
--gpt-model GPT-B \
--image-size 384 \
--image-size-eval 256 \
--cfg-scale 2.0 \
"$@"
