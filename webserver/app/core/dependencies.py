from app.core.services.interfaces.reconstruction_facade_interface import IReconstructionSQSFacade
from app.core.services.interfaces.recording_service_interface import IRecordingService
from fastapi import Depends
import os
from dotenv import load_dotenv
from app.core.services.reconstruction_service import ReconstructionService
from app.core.services.recording_service import RecordingService
from app.core.services.user_service import UserService
from app.core.persistence.recording_repository import RecordingRepository
from app.core.persistence.reconstruction_repository import ReconstructionRepository
from app.core.services.reconstruction_facade import ReconstructionFacade
from app.core.services.cognito_facade import CognitoFacade
from app.core.services.interfaces.cognito_interface import ICognitoFacade
from app.core.services.interfaces.reconstruction_service_interface import IReconstructionService
from app.core.persistence.database import get_async_session
from app.core.persistence.interfaces.recording_repository_interface import IRecordingRepository
from app.core.persistence.interfaces.reconstruction_repository_interface import IReconstructionRepository
from app.core.websocket_manager import ConnectionManager
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

async def get_connection_manager() -> ConnectionManager:
    return ConnectionManager()

async def get_recording_repository(db_session: AsyncSession = Depends(get_async_session)) -> IRecordingRepository:
    return RecordingRepository(db_session)

async def get_reconstruction_repository(db_session: AsyncSession = Depends(get_async_session)) -> IReconstructionRepository:
    return ReconstructionRepository(db_session)

async def get_cognito_facade() -> ICognitoFacade:
    user_pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    region_name = os.environ.get("AWS_REGION")
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    print(user_pool_id, region_name, aws_access_key_id, aws_secret_access_key)
    return CognitoFacade(
        user_pool_id=user_pool_id,
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

async def get_reconstruction_api_accessor() -> IReconstructionSQSFacade:
    return ReconstructionFacade()

async def get_recording_service(
    recording_repository: RecordingRepository = Depends(get_recording_repository)
) -> RecordingService:
    return RecordingService(recording_repository)

async def get_user_service(
    cognito_facade: ICognitoFacade = Depends(get_cognito_facade)
) -> UserService:
    return UserService(cognito_facade)

async def get_reconstruction_service(
    reconstruction_repository: IReconstructionRepository = Depends(get_reconstruction_repository),
    reconstruction_api_accessor: IReconstructionSQSFacade = Depends(get_reconstruction_api_accessor),
    recording_service: IRecordingService = Depends(get_recording_service),
    connection_manager: ConnectionManager = Depends(get_connection_manager)
) -> IReconstructionService:
    return ReconstructionService(
        reconstruction_repository,
        reconstruction_api_accessor,
        recording_service,
        connection_manager
    )
