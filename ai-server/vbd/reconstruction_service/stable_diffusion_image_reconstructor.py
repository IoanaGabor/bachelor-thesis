from reconstruction_service.abstract_image_embedding_reconstructor import AbstractImageReconstructor
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from reconstruction_service.constants import (
    BASE_MODEL_PATH,
    VAE_MODEL_PATH,
    UNCONDITIONAL_EMBEDDINGS_PATH,
    DDIM_CONFIG,
    INFERENCE_CONFIG,
    IP_CKPT_PATH,
)
from ip_adapter import IPAdapterPlus


from reconstruction_service.logger_config import logger
from huggingface_hub import hf_hub_download, snapshot_download

class StableDiffusionReconstructor(AbstractImageReconstructor):

    def __init__(self):
        self.base_model_path = BASE_MODEL_PATH
        self.vae_model_path = VAE_MODEL_PATH
        logger.info(f"Initializing StableDiffusionReconstructor with base model: {self.base_model_path}")
        logger.info("Initializing DDIM scheduler")
        noise_scheduler = DDIMScheduler(**DDIM_CONFIG)
        
        logger.info(f"Loading VAE model from {self.vae_model_path}")
        vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16)

        logger.info("Clearing CUDA cache")
        torch.cuda.empty_cache()
        
        logger.info(f"Loading Stable Diffusion pipeline from {self.base_model_path}")
        image_encoder_path = snapshot_download(
            repo_id="h94/IP-Adapter",
            repo_type="model",
            allow_patterns=["models/image_encoder/*"]
        ) + "/models/image_encoder"
        
        ip_ckpt_path = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename=IP_CKPT_PATH,
            repo_type="model"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            device="cuda"
        )

        self.ip_model = IPAdapterPlus(self.pipe, image_encoder_path, ip_ckpt_path, "cuda", num_tokens=16)
        
    def reconstruct(self, embeddings: np.ndarray, number_of_steps: int = 100) -> np.ndarray:
        logger.info("Starting image reconstruction process")
        logger.debug(f"Input embeddings shape: {embeddings.shape}")
        logger.info(f"embeddings {embeddings}")
        
        try:            
            logger.info(f"Loading unconditional embeddings from {UNCONDITIONAL_EMBEDDINGS_PATH}")
            unconditional_embeddings_path = hf_hub_download(
                repo_id="ig16/visual-brain-decoding-1",
                filename="unconditional_embeddings.pt",
                repo_type="model"
            )
            unconditional_embeddings = torch.load(unconditional_embeddings_path)

            logger.info(f"Running inference with {number_of_steps} steps")
            images = self.pipe(
                prompt_embeds=torch.stack([torch.tensor(embeddings[0:INFERENCE_CONFIG["num_embeddings"]])]),
                negative_prompt_embeds=unconditional_embeddings,
                num_inference_steps=number_of_steps
            ).images
            
            logger.info("Image reconstruction completed successfully")
            return images[0]
            
        except Exception as e:
            logger.error(f"Error during image reconstruction: {str(e)}", exc_info=True)
            raise
