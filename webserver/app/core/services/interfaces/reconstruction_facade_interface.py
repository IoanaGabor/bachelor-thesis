from abc import ABC, abstractmethod

class IReconstructionSQSFacade(ABC):
    @abstractmethod
    async def reconstruct_image(self, person_id, brain_recording_id, voxels, number_of_steps):
        pass