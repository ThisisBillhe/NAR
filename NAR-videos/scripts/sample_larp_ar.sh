
export CUDA_VISIBLE_DEVICES=0
python3 sample.py \
    --ar_model /path/to/ar_model \
    --tokenizer /path/to/tokenizer \
    --model_type llama-abs-xxx \
    --output_dir samples/ours_ucf_reproduce \
    --num_samples 10000 \
    --sample_batch_size 96 \
    --cfg_scale 2.5 \
    --dtype bfloat16 \
    --dataset_csv ucf101_train.csv \
    --dataset_split_seed 42
    