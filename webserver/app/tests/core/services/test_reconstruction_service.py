import pytest
from unittest.mock import AsyncMock
from app.core.services.reconstruction_service import ReconstructionService
from app.core.services.interfaces.websocket_manager_interface import IConnectionManager
from app.core.services.interfaces.reconstruction_facade_interface import IReconstructionSQSFacade
from app.core.persistence.interfaces.reconstruction_repository_interface import IReconstructionRepository
from app.core.services.interfaces.recording_service_interface import IRecordingService

@pytest.fixture
def mock_reconstruction_repository():
    return AsyncMock(spec=IReconstructionRepository)

@pytest.fixture
def mock_recording_service():
    return AsyncMock(spec=IRecordingService)

@pytest.fixture
def mock_reconstruction_facade():
    return AsyncMock(spec=IReconstructionSQSFacade)

@pytest.fixture
def mock_connection_manager():
    return AsyncMock(spec=IConnectionManager)

@pytest.fixture
def reconstruction_service(mock_reconstruction_repository, mock_reconstruction_facade, mock_recording_service, mock_connection_manager):
    return ReconstructionService(
        mock_reconstruction_repository,
        mock_reconstruction_facade,
        mock_recording_service,
        mock_connection_manager
    )

@pytest.mark.asyncio
async def test_request_reconstruction_recording_not_found(
    reconstruction_service,
    mock_recording_service
):
    mock_recording_service.get_recording_by_id.side_effect = ValueError("Recording not found")
    with pytest.raises(ValueError, match="Recording not found"):
        await reconstruction_service.request_reconstruction(999, "user", 5)


@pytest.mark.asyncio
async def test_get_reconstruction_by_id(
    reconstruction_service,
    mock_reconstruction_repository
):
    mock_reconstruction = {"id": 1, "brain_recording_id": 2}
    mock_reconstruction_repository.get_reconstruction_by_id.return_value = mock_reconstruction
    result = await reconstruction_service.get_reconstruction_by_id(1)
    assert result == mock_reconstruction
    mock_reconstruction_repository.get_reconstruction_by_id.assert_awaited_once_with(1)
