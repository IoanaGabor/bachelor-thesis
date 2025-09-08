import numpy as np
from metrics.image_similarity_metric import ImageSimilarityMetric

class PixCorrMetric(ImageSimilarityMetric):
    def __init__(self):
        self.name = "PixCorr"

    def compute(self, original, reconstructed) -> float:
        original = original.reshape(1, -1).cpu().numpy()
        reconstructed = reconstructed.reshape(1, -1).cpu().numpy()
        return float(np.corrcoef(original[0], reconstructed[0])[0, 1])
