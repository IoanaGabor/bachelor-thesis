import torch
import torch.nn as nn
from functools import partial
import tqdm
import utils.data as data
import os
data_path="./data/natural-scenes-dataset"
subject=1
model_path = "./logs/model/client{subject}_best.pth".format(subject=subject)
batch_size=1
num_workers=1
seed=42
pool_type='max'
pool_num=8192
length=982
train_path = "{}/webdataset_avg_split/train/train_subj0{}".format(data_path, subject)
val_path = "{}/webdataset_avg_split/val/val_subj0{}".format(data_path, subject)
test_path = "{}/webdataset_avg_split/test/test_subj0{}".format(data_path, subject)
description = "alpha_value_60000"
original_output_dir="./outputs_original/test/test_subj0{}".format(subject)
recreated_output_dir="./outputs_recreated/{}/test/test_subj0{}".format(description, subject)
os.makedirs(os.path.dirname(recreated_output_dir), exist_ok=True)
os.makedirs(recreated_output_dir, exist_ok=True)
extensions = ['nsdgeneral.npy', "jpg"]


from PIL import Image
import torch
import numpy as np

def tensor_to_pil(image_tensor):
    image_tensor = image_tensor.detach().cpu()
    image_np = image_tensor.numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    return image_pil

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapterPlus
from tqdm import tqdm

base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "IP-Adapter/models/image_encoder"
ip_ckpt = "IP-Adapter/models/ip-adapter-plus_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

clip_emb_dim = 768
hidden_dim = 2048

import pickle

multi_voxel_dims = {1:15724, 2:14278, 5:13039}
subject_input_size = multi_voxel_dims[subject]

plain_image = Image.open("./black.png")
_, unconditional_embeddings = image_embeds=ip_model.get_image_embeds(pil_image=plain_image)

test_dl = data.get_dataloader(
    test_path,
    batch_size=batch_size,
    num_workers=num_workers,
    seed=seed,
    extensions=extensions,
    pool_type=pool_type,
    pool_num=pool_num,
    is_shuffle=True,
    length=length,
)

negative_prompt_embeds_default = torch.stack([torch.stack([unconditional_embeddings[0]])])
cnt = 0
with torch.no_grad():
    alpha_value = 60000
    pred_clip_path = f'regression_weights_{alpha_value}/subj{subject}/pred_clip.npy'
    if os.path.exists(pred_clip_path):
        pred_clip = np.load(pred_clip_path)
        print(f"Loaded pred_clip from {pred_clip_path}, shape: {pred_clip.shape}")
    else:
        raise FileNotFoundError(f"pred_clip file not found at {pred_clip_path}")

    pred_clip_tensor = torch.tensor(pred_clip, dtype=torch.float32).to(device)
    for idx in range(pred_clip_tensor.shape[0]):
        prompt_embeds = pred_clip_tensor[idx].half()
        positive_prompt_embeds = torch.stack([prompt_embeds])
        print(positive_prompt_embeds.shape)
        images = pipe(
            prompt_embeds=positive_prompt_embeds[0],
            negative_prompt_embeds=negative_prompt_embeds_default[0],
            num_inference_steps=100
        ).images[0]
        recreated_path = os.path.join(recreated_output_dir, f"{idx}.recreated.jpg")
        images.save(recreated_path)