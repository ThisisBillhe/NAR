## Getting Started
### Requirements
- Linux with Python ≥ 3.7
- PyTorch ≥ 2.1
- A100 GPUs

## 🦄 Class-conditional image generation on ImageNet
### Pre-extract discrete codes of training images
```
bash scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --data-path /path/to/imagenet/train --code-path /path/to/imagenet_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 256
```

### Train AR models with DDP
```
bash scripts/autoregressive/train_c2i.sh --cloud-save-path /path/to/cloud_disk --no-local-save --code-path /path/to/imagenet_code_c2i_flip_ten_crop --image-size 256 --gpt-model GPT-B  --epochs 300
```


### Train AR models with FSDP
```
bash scripts/autoregressive/train_c2i_fsdp.sh --cloud-save-path /path/to/cloud_disk --no-local-save --code-path /path/to/imagenet_code_c2i_flip_ten_crop --image-size 256 --gpt-model GPT-XXL --epochs 300 --no-wandb
```


### Sampling
```
bash scripts/autoregressive/sample_c2i.sh --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_B.pt --gpt-model GPT-B --image-size 256 --image-size-eval 256 --cfg-scale 2.0
```


### Evaluation
Before evaluation, please refer [evaluation readme](evaluations/c2i/README.md) to install required packages. 
```
python3 evaluations/c2i/evaluator.py VIRTUAL_imagenet256_labeled.npz samples/GPT-B-c2i_B-size-384-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0.npz
```

## 🚀 Text-conditional image generation
### Prepare dataset
For the first stage, we use a [LAION COCO Subset(4M)](https://huggingface.co/datasets/guangyil/laion-coco-aesthetic) datset. For the second stage, we use a [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) high-resolution dataset. For both training stages, we use [webdataset](https://github.com/webdataset/webdataset) format, where all the data is stored in several .tar files. You can use [img2dataset] library to download the [LAION COCO Subset(4M)](https://huggingface.co/datasets/guangyil/laion-coco-aesthetic) dataset and save it in webdatsaet format. For [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M), it is already organized in webdataset format.

After downloading and organizing a dataset, use the following script to obtain a .json file for this datset. Remember to modify the directory and output_path in it.
```
python3 scripts\analyze_tar.py
```

### Train first stage model
```
bash scripts/autoregressive/train_t2i.sh --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --cloud-save-path /path/to/cloud_disk  --no-local-save --gpt-model GPT-XL --image-size 256 --epochs 60 --no-compile --data-path path_to_data.json
```


### Train second stage model
Resume from the first stage checkpoint.
```
bash scripts/autoregressive/train_t2i.sh --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --cloud-save-path /path/to/cloud_disk  --no-local-save --gpt-model GPT-XL --stage1-ckpt path_to_first_stage_ckpt.pt --image-size 512 --epochs 40 --no-compile --data-path path_to_data.json
```