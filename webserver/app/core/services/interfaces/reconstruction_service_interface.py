from abc import ABC, abstractmethod

class IReconstructionService(ABC):
    @abstractmethod
    async def get_reconstruction_by_id(self, reconstruction_id):
        pass

    @abstractmethod
    async def get_reconstructions_by_brain_recording_id(self, brain_recording_id: int):
        pass

    @abstractmethod
    async def request_reconstruction(self, brain_recording_id: str, user_id: str, number_of_steps: int):
        pass

    @abstractmethod
    async def handle_reconstruction_webhook(self, person_id: str, brain_recording_id: str, image_b64: str, message_id: str, status: str):
        pass