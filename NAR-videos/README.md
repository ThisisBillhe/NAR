## Overview

NAR's experiments for UCF-101 video generation is built upon [LARP](https://github.com/hywang66/LARP/), a novel video tokenizer inherently aligned with autoregressive (AR) generative models. NAR's pretrained models can be downloaded from [here](https://huggingface.co/collections/chenfeng1271/nar-67d13fa93fe913b2e187ee1f).

## Get Started

1. Install pytorch 2.4.0 and torchvision 0.19.0
   ```
   pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
   ```

2. Install other dependencies
   ```
   pip install -r requirements.txt
   ```
   
3. Set up the datasets using `set_datasets.sh`
    <details>
    <summary> This script sets up datast for UCF101 and Kinetics-600 datasets. </summary>

    You need to download the datasets you want to use and set the paths in the script. This script will create the necessary symbolic links so that the code can find the data.

    After setting up the datasets, verify that all paths in the CSV files located in data/metadata are accessible.


## Pretrained Models

We provide pretrained models for LARP tokenizer, LARP AR model, and LARP AR frame prediction model.
| Model                 | #params | FVD         |                         🤗 HuggingFace                         |
| --------------------- | :-----: | ----------- | :-----------------------------------------------------------: |
| LARP-L-Long-tokenizer |  173M   | 20 (recon.) | [link](https://huggingface.co/hywang66/LARP-L-long-tokenizer) |

Please refer to the **sampling and evaluation** section for details on how to use these models.
   

## Training

### Training LARP AR model on UCF101 dataset
```bash
bash scripts/train_larp_ar.sh
```

### Training LARP AR frame prediction model on Kinetics-600 dataset
```bash
bash scripts/train_larp_ar_fp.sh
```

### Reproducing the Pretrained Models
To reproduce the pretrained models released on HuggingFace, refer to the following training scripts:
```bash
scripts/train_larp_tokenizer_reproduce.sh
scripts/train_larp_ar_reproduce.sh
scripts/train_larp_ar_fp_reproduce.sh
```


## Sampling and Evaluation

The `sample.py` script can be used to sample videos from the LARP AR model and LARP AR frame prediction model. It also computes the Frechet Video Distance (FVD) score with the real videos. 
The `eval/eval_larp_tokenizer.py` script can be used to evaluate reconstruction performance of the LARP tokenizer.
Unless specified, all commands in this section are supposed to be run on an single GPU machine.


### UCF101 Class-conditional Generation

The following command samples 10,000 videos from the LARP AR model trained on UCF101 dataset and compute the FVD score with the real videos.
The videos are generated class-conditionally, i.e., each video is generated from a single class. 
Note that the UCF101 dataset is required to run this run this script. 

This command can reproduce the UCF101 generation FVD results reported in the Table 1 of the paper. 

```bash
python3 sample.py \
    --ar_model hywang66/LARP-L-long-AR \
    --tokenizer hywang66/LARP-L-long-tokenizer \
    --output_dir samples/ucf_reproduce \
    --num_samples 10000 \
    --sample_batch_size 64 \
    --cfg_scale 1.25 \
    --dtype bfloat16 \
    --dataset_csv ucf101_train.csv \
    --dataset_split_seed 42
```

The FVD score will be displayed at the end of the script and also appended to the fvd_report.csv file in the project directory.


### Parallel Sampling and Evaluation

When multiple GPUs are available, `sample.py` can be run in parallel to accelerate the sampling process. Set the `--num_samples` argument to specify the **per-GPU** number of samples, and use the `--num_samples_total` argument to define the total number of samples. Importantly, set the `--starting_index` argument to specify the starting index for this process, ensuring that it samples videos from `--starting_index` to `--starting_index + --num_samples` (exclusive). 

Example commands:
```bash
python3 sample.py \
    --ar_model hywang66/LARP-L-long-AR \
    --tokenizer hywang66/LARP-L-long-tokenizer \
    --output_dir samples/ucf_reproduce \
    --num_samples 128 \
    --num_samples_total 10000 \
    --starting_index 0 \
    --sample_batch_size 64 \
    --cfg_scale 1.25 \
    --dtype bfloat16 \
    --dataset_csv ucf101_train.csv \
    --dataset_split_seed 42

python3 sample.py \
    --ar_model hywang66/LARP-L-long-AR \
    --tokenizer hywang66/LARP-L-long-tokenizer \
    --output_dir samples/ucf_reproduce \
    --num_samples 128 \
    --num_samples_total 10000 \
    --starting_index 32 \
    --sample_batch_size 64 \
    --cfg_scale 1.25 \
    --dtype bfloat16 \
    --dataset_csv ucf101_train.csv \
    --dataset_split_seed 42

......

python3 sample.py \
    --ar_model hywang66/LARP-L-long-AR \
    --tokenizer hywang66/LARP-L-long-tokenizer \
    --output_dir samples/ucf_reproduce \
    --num_samples 16 \
    --num_samples_total 10000 \
    --starting_index 9984 \
    --sample_batch_size 64 \
    --cfg_scale 1.25 \
    --dtype bfloat16 \
    --dataset_csv ucf101_train.csv \
    --dataset_split_seed 42

```

Ensure there is no overlap in sample indices across processes, and assign each process to a different GPU. Once all processes have completed (in any order), the FVD score will be automatically calculated and appended to the `fvd_report.csv` file in the project directory.