from fastapi import Depends, UploadFile
from app.core.persistence.recording_repository import RecordingRepository
from app.utils.file_storage import save_file, FileType
from datetime import datetime
from app.core.services.interfaces.recording_service_interface import IRecordingService

class RecordingService(IRecordingService):
    def __init__(self, repository: RecordingRepository):
        self.__repository = repository

    async def get_all_recordings(self, nsd_id: str):
        return await self.__repository.get_all_recordings(nsd_id)
    
    async def get_recording_by_id(self, recording_id):
        return await self.__repository.get_recording_by_id(recording_id)

    async def create_recording(self, nifti_file: UploadFile, png_file: UploadFile, description: str, user_id:str):
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        voxels_filename = f"{user_id}_{timestamp}.npy"
        png_filename = f"{user_id}_{timestamp}.png"
        nifti_path = save_file(nifti_file, voxels_filename, FileType.NPY)
        png_path = save_file(png_file, png_filename, FileType.PNG)
        
        return await self.__repository.create_recording(nifti_path, png_path, description, user_id=int(user_id))

    async def delete_recording(self, recording_id: int):
        return await self.__repository.delete_recording(recording_id)
