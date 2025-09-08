import torch
import numpy as np
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from metrics.image_similarity_metric import ImageSimilarityMetric

class SSIMMetric(ImageSimilarityMetric):
    def __init__(self):
        self.name="SSIM"

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        original = original.cpu().permute(1, 2, 0).numpy()
        reconstructed = reconstructed.cpu().permute(1, 2, 0).numpy()
        original_gray = rgb2gray(original)
        reconstructed_gray = rgb2gray(reconstructed)
        return float(ssim(
            reconstructed_gray, original_gray,
            channel_axis=None,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0
        ))
