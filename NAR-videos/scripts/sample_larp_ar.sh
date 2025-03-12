
export CUDA_VISIBLE_DEVICES=0
python3 sample.py \
    --ar_model /root/autodl-tmp/LARP/save/larp_ar/larp_ar/lr0.0006_wd0.05_llama-abs-L__single_gpu/epoch-1000.pth \
    --tokenizer ./pretrained_models/LARP-L-long-tokenizer/ \
    --model_type llama-abs-L \
    --output_dir samples/ours_ucf_reproduce \
    --num_samples 10000 \
    --sample_batch_size 96 \
    --cfg_scale 2.5 \
    --dtype bfloat16 \
    --dataset_csv ucf101_train.csv \
    --dataset_split_seed 42
    