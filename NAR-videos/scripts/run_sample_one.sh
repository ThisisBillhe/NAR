export CUDA_VISIBLE_DEVICES="0"
python3 sample_one.py \
    --ar_model /path/to/ar_model \
    --tokenizer /path/to/tokenizer \
    --model_type llama-abs-xxx \
    --output_dir samples/run_sample_one \
    --num_samples 1 \
    --sample_batch_size 1 \
    --cfg_scale 1.25 \
    --dtype bfloat16 \
    --dataset_csv ucf101_train.csv \
    --dataset_split_seed 42