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
description = ""
original_output_dir="./outputs_original/test/test_subj0{}".format(subject)
recreated_output_dir="./outputs_recreated/{}/test/test_subj0{}".format(description, subject)
os.makedirs(os.path.dirname(recreated_output_dir), exist_ok=True)
os.makedirs(recreated_output_dir, exist_ok=True)
extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy']


class BrainGuardModule(nn.Module):
    def __init__(self):
        super(BrainGuardModule, self).__init__()
    def forward(self, x):
        return x
    
class RidgeRegression(torch.nn.Module):
    def __init__(self, input_size, out_features): 
        super(RidgeRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, out_features)
    def forward(self, x):
        x = self.linear(x)
        return x

class BrainNetwork(nn.Module):
  def __init__(self, out_dim_image=768, in_dim=15724, latent_size=768, h=2048, n_blocks=4, norm_type='ln', use_projector=True, act_first=False, drop1=.5, drop2=.15, train_type='vision'):
    super().__init__()
    norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
    act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
    act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
    self.mlp = nn.ModuleList([
        nn.Sequential(
            nn.Linear(h, h),
            *[item() for item in act_and_norm],
            nn.Dropout(drop2)
        ) for _ in range(n_blocks)
    ])
    self.head_image = nn.Linear(h, out_dim_image, bias=True)
    self.n_blocks = n_blocks
    self.latent_size = latent_size
    self.use_projector = use_projector
    self.train_type = train_type
    if use_projector:
        self.projector_image = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.GELU(),
            nn.Linear(self.latent_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, self.latent_size)
        )
        
  def forward(self, x):
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x_image = self.head_image(x) 
        if self.use_projector: 
            return self.projector_image(x_image.reshape(len(x_image), -1, self.latent_size))
        return x


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

model = BrainGuardModule()
multi_voxel_dims = {1:15724, 2:14278, 5:13039}
subject_input_size = multi_voxel_dims[subject]
model.ridge = RidgeRegression(input_size=subject_input_size, out_features=hidden_dim)
model.backbone = BrainNetwork(in_dim=hidden_dim, latent_size=clip_emb_dim, out_dim_image=16*768, use_projector=True, train_type='vision')   

plain_image = Image.open("./black.png")
_, unconditional_embeddings = image_embeds=ip_model.get_image_embeds(pil_image=plain_image)

try:
    state_dict = torch.load(f'./logs/{model_path}')
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
except RuntimeError as e:
    print(" Error loading model state_dict:", e)
except FileNotFoundError:
    print("File not found")

model.eval()
model.ridge.eval()
model.backbone.eval()

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
    for test_i, data_i in tqdm(enumerate(test_dl)):
        voxel, image, coco = data_i
        image_pil = tensor_to_pil(image[0])
        voxel = torch.mean(voxel,axis=1).float()
        ridge_out = model.ridge(voxel)
        results = model.backbone(ridge_out)
        images = pipe(
            prompt_embeds=torch.stack([results[0][0:16]]),
            negative_prompt_embeds=negative_prompt_embeds_default,
            num_inference_steps=100
        ).images[0]
        recreated_path = os.path.join(recreated_output_dir, f"{cnt}.recreated.jpg")
        original_path = os.path.join(original_output_dir, f"{cnt}.original.jpg") 
        cnt +=1
        images.save(recreated_path)
        image_pil.save(original_path)
