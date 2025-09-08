from abc import ABC, abstractmethod
from reconstruction_service.abstract_voxel_embedding_extractor import AbstractVoxelEmbeddingExtractor
from reconstruction_service.abstract_image_embedding_reconstructor import AbstractImageReconstructor
import numpy as np

class AbstractReconstructionPipeline(ABC):
    
    def __init__(self):
        self.embedding_extractor = self.create_extractor()
        self.image_reconstructor = self.create_reconstructor()
    
    @abstractmethod
    def create_extractor(self) -> AbstractVoxelEmbeddingExtractor:
        pass
    
    @abstractmethod
    def create_reconstructor(self) -> AbstractImageReconstructor:
        pass
    
    def run(self, voxel_data: np.ndarray, person_id: str, number_of_steps: int = 100) -> np.ndarray:
        embeddings = self.embedding_extractor.extract(voxel_data, person_id)
        image = self.image_reconstructor.reconstruct(embeddings, number_of_steps)

        return image