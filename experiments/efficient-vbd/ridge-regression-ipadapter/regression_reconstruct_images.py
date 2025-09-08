import torch
import argparse
import numpy as np
import os
from tqdm import trange
from PIL import Image

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapterPlus

# parameters taken from https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter-plus_demo.ipynb
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "ip-adapter/models/image_encoder"
ip_ckpt = "ip-adapter/models/ip-adapter-plus_sd15.bin"
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

print("Loading Stable Diffusion pipeline with IP-Adapter...")


torch.cuda.empty_cache()
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None,
    device="cuda"
)



generator = torch.Generator(device="cuda").manual_seed(0)
ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

plain_image = Image.open("./black.png")
_, unconditional_embeddings = image_embeds=ip_model.get_image_embeds(pil_image=plain_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindFormer Prediction")
    parser.add_argument("--sub", type=int, choices=[1, 2, 5, 7], required=True, help="Subject Number")
    args = parser.parse_args()
    torch.cuda.empty_cache()

    image_embeds = np.load('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral_other.npy'.format(args.sub))
    generator = torch.Generator(device="cuda").manual_seed(0)
    num_images=len(image_embeds)
    batch_size=1
    for start in trange(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch = image_embeds[start:end]

        prompt_embeds = torch.stack([torch.stack([torch.tensor(batch[i, :16,:])]) for i in range(batch_size)])
        print(prompt_embeds.shape)

        images = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=unconditional_embeddings,
            num_inference_steps=100,
            generator=generator
        ).images
        descr="ridge-regression"
        output_dir = f"generated_images_{descr}"
        os.makedirs(output_dir, exist_ok=True) 
        for i, img in enumerate(images):
            img.save(f"{output_dir}/{start + i}.png")
    print(f"Generated images saved in {output_dir}")
