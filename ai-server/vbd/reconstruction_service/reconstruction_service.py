from typing import List
from reconstruction_service.simple_reconstruction_pipeline import BrainScanReconstructionPipeline
from reconstruction_service.logger_config import logger

class ReconstructionService:
    def __init__(self):
        logger.info("Initializing ReconstructionService and reconstruction pipeline")
        self.pipe = BrainScanReconstructionPipeline()

    def reconstruct_image(self, person_id: str, voxels: List[int], number_of_steps: int = 100):
        logger.info(f"Starting image reconstruction for person_id: {person_id}")
        logger.debug(f"Input voxels length: {len(voxels)}")
        
        if not voxels:
            logger.error("Empty voxel array provided")
            raise ValueError("Voxel array cannot be empty")
            
        try:
            logger.info("Running reconstruction pipeline")
            image = self.pipe.run(voxels, person_id, number_of_steps)
            
            return image
            
        except Exception as e:
            logger.error(f"Error during image reconstruction: {str(e)}")
            raise
