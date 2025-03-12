# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12335 \
autoregressive/train/extract_codes_c2i.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
--data-path /path/to/imagenet/train \
--code-path /path/to/imagenet_code_c2i_flip_ten_crop \
--ten-crop \
--crop-range 1.1 \
--image-size 256 \
"$@"
