# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT
#   nanoGPT: https://github.com/karpathy/nanoGPT
import sys
sys.path.append('place the absolute path of NAR-images here')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from glob import glob
import time
import argparse
import os
import wids

from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from dataset.augmentation import center_crop_arr
from autoregressive.train.train_c2i import creat_optimizer
from autoregressive.models.gpt import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder

def make_dataset_train(trainset_url, transform):
    def make_sample(sample):
        # print(sample)
        image = sample[".jpg"]
        label = sample[".json"]["prompt"]

        return transform(image), label

    trainset = wids.ShardListDataset(trainset_url, keep=True)
    trainset = trainset.add_transform(make_sample)

    return trainset

def pad_caption_embs(args, caption_embs, emb_masks):
    t5_feat_padding = torch.zeros((caption_embs.shape[0], args.t5_feature_max_len, args.t5_feature_dim))
    valid_caption_embs = caption_embs[:, :emb_masks.sum()]
    t5_feat = valid_caption_embs
    t5_feat_len = t5_feat.shape[1] 
    feat_len = min(args.t5_feature_max_len, t5_feat_len)
    t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
    emb_mask = torch.zeros((args.t5_feature_max_len,))
    emb_mask[-feat_len:] = 1
    attn_mask = torch.tril(torch.ones(args.max_seq_length, args.max_seq_length))
    T = args.t5_feature_max_len
    attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
    eye_matrix = torch.eye(args.max_seq_length, args.max_seq_length)
    attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
    attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
    valid = 1
    
    return t5_feat_padding, attn_mask, torch.tensor(valid)

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-") 
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup model
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    args.max_seq_length = args.t5_feature_max_len + latent_size ** 2
    
    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Setup data:
    if args.dataset == 't2i_caption':     # create and load model
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim)
        vq_model.to(device)
        vq_model.eval()
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint   

        precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
        assert os.path.exists(args.t5_model_path)
        t5_xxl = T5Embedder(
            device=device, 
            local_cache=True, 
            cache_dir=args.t5_model_path, 
            dir_or_name=args.t5_model_type,
            torch_dtype=precision
        )     
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1).clone()),  # Convert grayscale to RGB
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    local_batch_size =int(args.global_batch_size // dist.get_world_size())
    data_path = args.data_path
    assert data_path is not None
    dataset = make_dataset_train(data_path, transform)
    sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=True)
    loader = DataLoader(
        dataset, 
        batch_size=local_batch_size, 
        sampler=sampler, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    try:
        logger.info(f"Dataset contains {len(dataset):,} images")
    except:
        pass

    # Prepare models for training:
    if args.stage1_ckpt:
        checkpoint = torch.load(args.stage1_ckpt, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        del checkpoint
    else:
        logger.info(f"Stage 1 checkpoint is not provided! Ensure you have gpt_ckpt provided")

    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
    
    max_seq_length = model.cls_token_num + model.block_size
    diagonal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool))
    model.setup_diagonal_mask(
        diagonal_mask[-model.block_size:, -model.block_size:], 
        model.block_size
    )
    diagonal_mask = diagonal_mask.unsqueeze(0).repeat(args.global_batch_size // dist.get_world_size(), 1, 1)

    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    eye_matrix = torch.eye(args.max_seq_length, args.max_seq_length)
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, caption in loader:
            x = x.to(device, non_blocking=True)

            # get t5 features
            y = torch.zeros((x.shape[0], args.t5_feature_max_len, args.t5_feature_dim))
            attn_mask = torch.tril(torch.ones(args.max_seq_length, args.max_seq_length)).unsqueeze(0).repeat(x.shape[0],1,1)
            caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
            for i in range(caption_embs.shape[0]):
                t5_feat_len = emb_masks[i].sum()
                valid_caption_embs = caption_embs[i, :t5_feat_len].unsqueeze(0)
                y[i, -t5_feat_len:] = valid_caption_embs[:, :t5_feat_len]
                i_emb_mask = torch.zeros((args.t5_feature_max_len,))
                i_emb_mask[-t5_feat_len:] = 1
                attn_mask[i, :, :args.t5_feature_max_len] = attn_mask[i, :, :args.t5_feature_max_len] * i_emb_mask.unsqueeze(0)
                attn_mask[i] = attn_mask[i] * (1 - eye_matrix) + eye_matrix
            y = y.to(device, non_blocking=True)

            ## extract code
            img = x
            with torch.no_grad():
                _, _, [_, _, indices] = vq_model.encode(img)
            x = indices.reshape(img.shape[0], -1)

            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(y.shape[0], y.shape[-2], y.shape[-1])
            assert z_indices.shape[0] == c_indices.shape[0]
            attn_mask = attn_mask.reshape(attn_mask.shape[0], attn_mask.shape[-2], attn_mask.shape[-1])
            tmp_mask = diagonal_mask.clone()
            tmp_mask[:, :, :120] = attn_mask[:, :, :120]
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                # _, loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices, mask=attn_mask[:, :, :-1,:-1], valid=valid)
                _, loss = model(cond_idx=c_indices, idx=z_indices, targets=z_indices, mask=tmp_mask)
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, lr: {scheduler.get_last_lr()[0]:.6f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()
        
        if rank == 0:
            if not args.no_compile:
                model_weight = model.module._orig_mod.state_dict()
            else:
                model_weight = model.module.state_dict()  
            checkpoint = {
                "model": model_weight,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "steps": train_steps,
                "args": args
            }
            if not args.no_local_save:
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # cloud_checkpoint_path = f"{cloud_checkpoint_dir}/epoch{epoch}.pt"
            cloud_checkpoint_path = f"/mnt/petrelfs/heyefei/ZipAR-X/experiments-t2i-XL-stage2/epoch{epoch}.pt"
            torch.save(checkpoint, cloud_checkpoint_path)
            logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
        dist.barrier()

        scheduler.step()
    # save the last model weights
    if rank == 0:
        if not args.no_compile:
            model_weight = model.module._orig_mod.state_dict()
        else:
            model_weight = model.module.state_dict()  
        checkpoint = {
            "model": model_weight,
        }

        cloud_checkpoint_path = f"{cloud_checkpoint_dir}/last_version.pt"
        torch.save(checkpoint, cloud_checkpoint_path)
        logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-model-path", type=str, default='./pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120, help="max length of t5 feature")
    parser.add_argument("--t5-feature-dim", type=int, default=2048, help="dimension of t5 feature")
    parser.add_argument("--short-t5-feat-path", type=str, default=None, help="short caption of t5_feat_path")
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--stage1-ckpt", type=str, default=None, help="ckpt path for start stage2 training")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='t2i_caption')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=12000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--data-path", type=str, default=None, help="ckpt path for start stage2 training")
    args = parser.parse_args()
    main(args)