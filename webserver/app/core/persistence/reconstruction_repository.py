from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.models.reconstruction import Reconstruction
from app.core.persistence.interfaces.reconstruction_repository_interface import IReconstructionRepository
from app.core.models.brain_recording import BrainRecording
from app.core.logger_config import logger

class ReconstructionRepository(IReconstructionRepository):
    
    def __init__(self, database: AsyncSession):
        self.__database = database

    async def get_all_recordings(self):
        result = await self.__database.execute(select(Reconstruction))
        return result.scalars().all()
    
    async def get_reconstruction_by_id(self, reconstruction_id: int):
        result = await self.__database.execute(
            select(Reconstruction).where(Reconstruction.id == reconstruction_id)
        )
        return result.scalar_one_or_none()
    
    async def get_reconstructions_by_brain_recording_id(self, brain_recording_id: int):
        result = await self.__database.execute(
            select(Reconstruction).where(Reconstruction.brain_recording_id == brain_recording_id)
        )
        return result.scalars().all()

    async def create_reconstruction(self, brain_recording_id: int, status: str, number_of_steps: int):
        logger.info(f"Creating reconstruction for brain recording id {brain_recording_id} with status {status} and number of steps {number_of_steps}")
        recording = Reconstruction(
            brain_recording_id=brain_recording_id,
            status=status,
            number_of_steps=number_of_steps
        )
        self.__database.add(recording)
        await self.__database.commit()
        await self.__database.refresh(recording)
        return recording


    async def update_reconstruction(self, reconstruction_id: int, **kwargs):
        result = await self.__database.execute(
            select(Reconstruction).where(Reconstruction.id == reconstruction_id)
        )
        reconstruction = result.scalar_one_or_none()
        if not reconstruction:
            return None
        for key, value in kwargs.items():
            if hasattr(reconstruction, key):
                setattr(reconstruction, key, value)
        await self.__database.commit()
        await self.__database.refresh(reconstruction)
        return reconstruction
    
    async def get_reconstruction_by_message_id(self, message_id: str):
        result = await self.__database.execute(
            select(Reconstruction).where(Reconstruction.message_id == message_id)
        )
        return result.scalar_one_or_none()


    async def get_metrics_by_subject_id(self, subject_id: str):
        stmt = (
            select(Reconstruction)
            .join(BrainRecording, Reconstruction.brain_recording_id == BrainRecording.id)
            .where(BrainRecording.user_id == int(subject_id))
        )
        result = await self.__database.execute(stmt)
        return result.scalars().all()