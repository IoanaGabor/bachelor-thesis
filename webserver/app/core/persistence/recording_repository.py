from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.models.brain_recording import BrainRecording
from app.core.persistence.interfaces.recording_repository_interface import IRecordingRepository
from app.core.logger_config import logger
class RecordingRepository(IRecordingRepository):

    def __init__(self, database: AsyncSession):
        self.__database = database

    async def get_all_recordings(self, nsd_id: str):
        user_id = int(nsd_id)
        result = await self.__database.execute(select(BrainRecording).where(BrainRecording.user_id == user_id))
        return result.scalars().all()
    
    async def get_recording_by_id(self, recording_id: int):
        result = await self.__database.execute(
            select(BrainRecording).where(BrainRecording.id == recording_id)
        )
        return result.scalar_one_or_none()

    async def create_recording(self, voxels_file: str, png_file: str, description: str, user_id: int):
        recording = BrainRecording(
            user_id=user_id,
            voxels_file=voxels_file,
            png_file=png_file,
            description=description,
        )
        self.__database.add(recording)
        await self.__database.commit()
        await self.__database.refresh(recording)
        return recording

    async def remove_recording(self, recording_id: int):
        recording = await self.get_recording_by_id(recording_id)
        if recording:
            await self.__database.delete(recording)
            await self.__database.commit()
            return True
        return False
    
    async def delete_recording(self, recording_id: int):
        recording = await self.get_recording_by_id(recording_id)
        if recording:
            await self.__database.delete(recording)
            await self.__database.commit()
            return True
        return False