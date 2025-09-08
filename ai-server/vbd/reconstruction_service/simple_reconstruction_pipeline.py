from reconstruction_service.abstract_reconstruction_pipeline import AbstractReconstructionPipeline
from reconstruction_service.abstract_image_embedding_reconstructor import AbstractImageReconstructor
from reconstruction_service.abstract_voxel_embedding_extractor import AbstractVoxelEmbeddingExtractor
from reconstruction_service.ridge_regression_embedding_extractor import RidgeRegressionVoxelEmbeddingExtractor
from reconstruction_service.stable_diffusion_image_reconstructor import StableDiffusionReconstructor
from reconstruction_service.brain_guard_embedding_extractor import BrainGuardVoxelEmbeddingExtractor

class BrainScanReconstructionPipeline(AbstractReconstructionPipeline):

    _instance = None  
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BrainScanReconstructionPipeline, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def create_extractor(self) -> AbstractVoxelEmbeddingExtractor:
        return BrainGuardVoxelEmbeddingExtractor()
    
    def create_reconstructor(self) -> AbstractImageReconstructor:
        return StableDiffusionReconstructor()