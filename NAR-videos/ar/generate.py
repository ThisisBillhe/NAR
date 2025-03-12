# Modified from:
#   llamagen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/generate.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py


import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(all_logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    accept_token_num = all_logits.shape[1]
    all_idx = []
    for i in range(accept_token_num):
        logits = all_logits[:, i, :] / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample_logits:
            idx = torch.multinomial(probs, num_samples=1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
        all_idx.append(idx.view(-1, 1))
    all_idx = torch.cat(all_idx, dim=-1)

    return all_idx


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos=input_pos)

    return sample(logits, **sampling_kwargs)


def decode_one_token(
    model, x: torch.Tensor, coordinate: list, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool,
    next_coordinate: list, **sampling_kwargs):
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos, coordinate=coordinate, next_coordinate=next_coordinate)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos, coordinate=coordinate, next_coordinate=next_coordinate)
    return sample(logits, **sampling_kwargs)


def generate_position(c, latent_shape, cond_len):
    coordinate, position_ids = [], []
    T, H, W = latent_shape
    for t in range(T):
        for h in range(H):
            w = c - t - h
            if 0 <= w < W:
                coordinate.append([t, h, w])
                position_ids.append(cond_len + (t * (H * W) + h * W + w))
    return coordinate, torch.tensor(position_ids)


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int, cond_len: int,
    **sampling_kwargs):
    bsz = cur_token.shape[0]
    latent_shape = model.config.latent_shape
    T, H, W = latent_shape
    token_container = [[[0] * W for _ in range(H)] for _ in range(T)]
    token_container[0][0][0] = cur_token
    
    cfg_flag = True
    generated_token_num = 1
    coordinate = [[0, 0, 0]]

    for plane in tqdm(range(0, T + H + W - 3)):
        with sdpa_kernel(SDPBackend.MATH):
            if cfg_interval > -1 and generated_token_num > cfg_interval:
                cfg_flag = False

            next_coordinate, next_input_pos = generate_position(plane + 1, latent_shape, cond_len)
            next_token = decode_one_token(
                model, cur_token, coordinate, input_pos, cfg_scale, cfg_flag,
                next_coordinate, **sampling_kwargs
            )
            cur_token = next_token
            coordinate, input_pos = next_coordinate, next_input_pos
            
            for i, co in enumerate(next_coordinate):
                t, h, w = co[0], co[1], co[2]
                token_container[t][h][w] = next_token[:, i].view(-1, 1)
            generated_token_num += len(next_coordinate)
    tokens = torch.cat([torch.cat([torch.cat(h_tokens, dim=-1) for h_tokens in t_tokens], dim=-1) for t_tokens in token_container], dim=-1)
    # return torch.tensor(token_container).reshape(bsz, -1)[:, 1:]
    return tokens[:, 1:]


@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    if model.frame_prediction:
        assert cfg_scale == 1.0, "frame prediction requires cfg_scale=1.0 (no classifier-free guidance)"
        cond_combined = cond
        T = cond.shape[1]
    elif model.model_type == 'class_cond':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1 
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, T, **sampling_kwargs)
    seq[:, T+1:] = generated_tokens

    return seq[:, T:]
