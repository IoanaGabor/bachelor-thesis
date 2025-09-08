from abc import ABC, abstractmethod
import torch

class ImageSimilarityMetric(ABC):
    @abstractmethod
    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        pass