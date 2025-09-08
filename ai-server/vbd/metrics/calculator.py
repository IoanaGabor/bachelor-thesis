import torch
from metrics.ssim_metric import SSIMMetric
from metrics.pixcorr_metric import PixCorrMetric
import numpy as np

class MetricsCalculator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.metrics = [SSIMMetric(), PixCorrMetric()]

    def compute_all(self, pil_img1, pil_img2):
        img1 = torch.from_numpy(np.array(pil_img1)).float().to(self.device) / 255.0
        img2 = torch.from_numpy(np.array(pil_img2)).float().to(self.device) / 255.0
        results = {
            metric.name: metric.compute(img1, img2)
            for metric in self.metrics
        }
        return results