import numpy as np
from reconstruction_service.abstract_voxel_embedding_extractor import AbstractVoxelEmbeddingExtractor
import torch
from huggingface_hub import hf_hub_download
from reconstruction_service.model import BrainGuardModule, BrainNetwork, RidgeRegression
from reconstruction_service.logger_config import logger

class BrainGuardVoxelEmbeddingExtractor(AbstractVoxelEmbeddingExtractor):
    def __init__(self, hidden_dim=2048, clip_emb_dim=768):
        logger.info("Initializing BrainGuardVoxelEmbeddingExtractor (subject agnostic)")
        self.hidden_dim = hidden_dim
        self.clip_emb_dim = clip_emb_dim
        self.multi_voxel_dims = {"1":15724, "2":14278, "5":13039}
        self.subject_models = {}
        self.model_paths = {}

    def _load_model_for_subject(self, subject: str):
        if subject in self.subject_models:
            logger.debug(f"Model for subject {subject} already loaded in dict.")
            return 
        if subject not in self.multi_voxel_dims:
            logger.error(f"Subject {subject} not supported. Available: {list(self.multi_voxel_dims.keys())}")
            raise ValueError(f"Subject {subject} not supported. Available: {list(self.multi_voxel_dims.keys())}")
        logger.info(f"Loading BrainGuard model for subject {subject}")
        model_path = hf_hub_download(
            repo_id="ig16/visual-brain-decoding-1",
            filename="model.pth",
            repo_type="model"
        )
        logger.info(f"Model path downloaded: {model_path}")
        self.model_paths[subject] = model_path
        subject_input_size = self.multi_voxel_dims[subject]
        model = BrainGuardModule()
        model.ridge = RidgeRegression(input_size=subject_input_size, out_features=1024)
        model.backbone = BrainNetwork(
            in_dim=1024, h=1024, latent_size=768, out_dim_image=16*768,
            use_projector=True, train_type='vision'
        )

        try:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            model.eval()
            model.ridge.eval()
            model.backbone.eval()
            logger.info(f"Model loaded successfully for subject {subject} in BrainGuardVoxelEmbeddingExtractor.")
        except RuntimeError as e:
            logger.error(f"Error loading model state_dict in BrainGuardVoxelEmbeddingExtractor: {e}")
            raise
        except FileNotFoundError:
            logger.error("File not found in BrainGuardVoxelEmbeddingExtractor. Check the path and subj_id.")
            raise

        self.subject_models[subject] = model

    def extract(self, voxel_data, person_id: str) -> np.ndarray:
        logger.debug(f"Extract called for person_id={person_id}")
        self._load_model_for_subject(person_id)
        model = self.subject_models[person_id]
        voxel_data = np.array(voxel_data, dtype=np.float32)
        logger.debug(f"Voxel data shape: {voxel_data.shape}")
        if voxel_data.ndim == 1:
            voxel_tensor = torch.from_numpy(voxel_data).unsqueeze(0)
        else:
            voxel_tensor = torch.from_numpy(voxel_data)

        with torch.no_grad():
            ridge_out = model.ridge(voxel_tensor)
            logger.info(f"Ridge output shape: {ridge_out.shape}")
            image_embeds = model.backbone(ridge_out)
            if isinstance(image_embeds, tuple):
                image_embeds = image_embeds[0]
            image_embeds = image_embeds.cpu().numpy()
            logger.debug(f"Image embeds shape after backbone: {image_embeds.shape}")

        if image_embeds.shape[0] == 1:
            image_embeds = image_embeds[0]
            logger.debug(f"Squeezed image embeds to shape: {image_embeds.shape}")

        logger.info(f"Extraction complete for person_id={person_id}")
        return image_embeds