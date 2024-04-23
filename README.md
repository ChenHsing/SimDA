

#  SimDA: Simple Diffusion Adapter for Efficient Video Generation

This is the official repo of the paper [SimDA: Simple Diffusion Adapter for Efficient Video Generation](https://arxiv.org/abs/2308.09710). 

The project website is [here](https://chenhsing.github.io/SimDA/).





[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://chenhsing.github.io/SimDA/)
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)](https://arxiv.org/abs/2308.09710)





## Setup

### Requirements

```shell
sh env.sh
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).

### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)). You can also use fine-tuned Stable Diffusion models trained on different styles (e.g, [Modern Disney](https://huggingface.co/nitrosocke/mo-di-diffusion), [Anything V4.0](https://huggingface.co/andite/anything-v4.0), [Redshift](https://huggingface.co/nitrosocke/redshift-diffusion), etc.).




## Usage

### Training

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```bash
sh train.sh
```

Note: Tuning a 24-frame video usually takes `200~500` steps, about `5~10` minutes using one A100 GPU. 
Reduce `n_sample_frames` if your GPU memory is limited.

### Inference

Once the training is done, run inference:

```python
from simda.pipelines.pipeline_simda import SimDAPipeline
from simda.models.unet import UNet3DConditionModel
from simda.util import save_videos_grid
import torch

pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
my_model_path = "./outputs/car-turn"
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = SimDAipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

prompt = "spider man is skiing"
ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)
video = pipe(prompt, latents=ddim_inv_latent, video_length=24, height=512, width=512, num_inference_steps=50, guidance_scale=12.5).videos

save_videos_grid(video, f"./{prompt}.gif")
```



## Citation
If you make use of our work, please cite our paper.
```
@inproceedings{xing2023simda,
  title={SimDA: Simple Diffusion Adapter for Efficient Video Generation},
  author={Xing, Zhen and Dai, Qi and Hu, Han and Wu, Zuxuan and Jiang, Yu-Gang},
  booktitle={CVPR},
  year={2024}
}
```

## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers) and [Tune-A-Video](https://github.com/showlab/Tune-A-Video). Thanks for open-sourcing!
