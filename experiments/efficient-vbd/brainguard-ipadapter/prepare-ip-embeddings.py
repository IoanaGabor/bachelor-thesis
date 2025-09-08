
import torch
import os
import argparse
from PIL import Image
from tqdm import tqdm
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


import utils.data as data

data_path="./data/natural-scenes-dataset"
subject=1
batch_size=1
num_workers=1
seed=42
pool_type='max'
pool_num=8192
length=8859
train_path = "{}/webdataset_avg_split/train/train_subj0{}".format(data_path, subject)
val_path = "{}/webdataset_avg_split/val/val_subj0{}".format(data_path, subject)
extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy', 'ip']

train_dl = data.get_dataloader(
    val_path,
    batch_size=batch_size,
    num_workers=num_workers,
    seed=seed,
    extensions=extensions,
    pool_type=pool_type,
    pool_num=pool_num,
    is_shuffle=True,
    length=length,
)

def extract_features(img_path):
    img = Image.open(img_path)
    dir_path, filename = os.path.split(img_path)
    base_name, ext = os.path.splitext(filename)
    image_embeds=ip_model.get_image_embeds(pil_image=img)
    image_embeds=torch.stack([image_embeds[0].cpu(), image_embeds[1].cpu()])
    ip_path = os.path.join(dir_path, f"{base_name}.ip")
    torch.save(image_embeds, ip_path)
with torch.no_grad():
    for train_i, data_i in tqdm(enumerate(train_dl)):
        voxel, image, coco = data_i
        extract_features(image[0])
