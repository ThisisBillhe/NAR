## 🦄 Class-conditional image generation on ImageNet
### VQ-VAE models
We use a open-sourced image tokenizer introduced [here](https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt).

### AR models
All pretrained models can be downloaded from [here](https://huggingface.co/collections/chenfeng1271/nar-67d13fa93fe913b2e187ee1f). Please download models, rename them and put them in the folder `./pretrained_models` if you want to sample images.


### Sample
Please download models, put them in the folder `./pretrained_models`, and run
```
python3 autoregressive/sample/sample_c2i.py --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt --gpt-ckpt ./pretrained_models/c2i_L_384.pt --gpt-model GPT-L --image-size 256
```
The generated images will be saved to `sample_c2i.png`.

## 🚀 Text-conditional image generation
### VQ-VAE models
We use a open-sourced image tokenizer introduced [here](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt).

### AR models
Method | params | tokens | data | weight 
--- |:---:|:---:|:---:|:---:
NAR-XL  | 816M | 16x16 | [LAION COCO Subset(4M)](https://huggingface.co/datasets/guangyil/laion-coco-aesthetic) | [t2i_XL_stage1_256.pt](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/t2i_XL_stage1_256.pt)
NAR-XL  | 816M | 32x32 | [text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) | [t2i_XL_stage2_512.pt](https://huggingface.co/yefly/NAR-XL-t2i-stage2)

### Demo
Before running demo, please refer to [language readme](language/README.md) to install the required packages and language models.  

Please download models, rename them and put them in the folder `./pretrained_models`, and run
```
python3 autoregressive/sample/sample_t2i.py --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --gpt-ckpt ./pretrained_models/t2i_XL_stage1_256.pt --gpt-model GPT-XL --image-size 256
# or
python3 autoregressive/sample/sample_t2i.py --vq-ckpt ./pretrained_models/vq_ds16_t2i.pt --gpt-ckpt ./pretrained_models/t2i_XL_stage2_512.pt --gpt-model GPT-XL --image-size 512
```
The generated images will be saved to `sample_t2i.png`.


## Training
See [Getting Started](GETTING_STARTED.md) for installation, training and evaluation.

## BibTeX
```bibtex
@article{sun2024autoregressive,
  title={Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation},
  author={Sun, Peize and Jiang, Yi and Chen, Shoufa and Zhang, Shilong and Peng, Bingyue and Luo, Ping and Yuan, Zehuan},
  journal={arXiv preprint arXiv:2406.06525},
  year={2024}
}
```