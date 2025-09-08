from abc import ABC, abstractmethod

class IReconstructionRepository(ABC):
    @abstractmethod
    async def get_all_recordings(self):
        pass

    @abstractmethod
    async def get_reconstruction_by_id(self, reconstruction_id: int):
        pass

    @abstractmethod
    async def get_reconstructions_by_brain_recording_id(self, brain_recording_id: int):
        pass

    @abstractmethod
    async def create_reconstruction(self, brain_recording_id: int, reconstruction_file: str, metrics_json: str, number_of_steps: int):
        pass