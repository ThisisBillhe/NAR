# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
autoregressive/sample/sample_c2i_ddp.py \
--vq-ckpt /root/autodl-tmp/LlamaGen_R/pretrained_models/llamagen_c2i/vq_ds16_c2i.pt \
--gpt-ckpt /root/autodl-tmp/LlamaGen/pretrained_model/2024-12-19-16-22-23/001-GPT-B/checkpoints/0655000.pt \
--gpt-model GPT-B \
--image-size 384 \
--image-size-eval 256 \
--cfg-scale 2.0 \
"$@"
