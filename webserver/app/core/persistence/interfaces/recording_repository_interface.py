from abc import ABC, abstractmethod

class IRecordingRepository(ABC):
    @abstractmethod
    async def get_all_recordings(self, nsd_id: str):
        pass

    @abstractmethod
    async def get_recording_by_id(self, recording_id: int):
        pass

    @abstractmethod
    async def create_recording(self, voxels_file: str, png_file: str, description: str, user_id: int):
        pass