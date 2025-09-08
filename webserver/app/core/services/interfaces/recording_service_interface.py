from abc import ABC, abstractmethod
from fastapi import UploadFile

class IRecordingService(ABC):
    @abstractmethod
    async def get_all_recordings(self, nsd_id: str):
        pass

    @abstractmethod
    async def get_recording_by_id(self, recording_id):
        pass

    @abstractmethod
    async def create_recording(self, nifti_file: UploadFile, png_file: UploadFile, description: str, user_id: str):
        pass