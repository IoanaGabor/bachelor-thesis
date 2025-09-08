import pytest
from unittest.mock import AsyncMock
from app.core.persistence.interfaces.recording_repository_interface import IRecordingRepository
from app.core.services.recording_service import RecordingService

@pytest.fixture
def mock_repository():
    repository = AsyncMock(spec=IRecordingRepository)
    return repository

@pytest.fixture
def recording_service(mock_repository):
    return RecordingService(repository=mock_repository)

@pytest.mark.asyncio
async def test_get_all_recordings(recording_service, mock_repository):
    nsd_id = "test_nsd_id"
    mock_recordings = [
        {"id": 1, "description": "Test 1"},
        {"id": 2, "description": "Test 2"}
    ]
    mock_repository.get_all_recordings.return_value = mock_recordings
    
    result = await recording_service.get_all_recordings(nsd_id)
    
    assert result == mock_recordings
    mock_repository.get_all_recordings.assert_called_once_with(nsd_id)

@pytest.mark.asyncio
async def test_get_recording_by_id(recording_service, mock_repository):
    recording_id = 1
    mock_recording = {"id": recording_id, "description": "Test recording"}
    mock_repository.get_recording_by_id.return_value = mock_recording
    
    result = await recording_service.get_recording_by_id(recording_id)
    
    assert result == mock_recording
    mock_repository.get_recording_by_id.assert_called_once_with(recording_id)

@pytest.mark.asyncio
async def test_get_recording_by_id_not_found(recording_service, mock_repository):
    recording_id = 999
    mock_repository.get_recording_by_id.return_value = None
    
    result = await recording_service.get_recording_by_id(recording_id)
    
    assert result is None
    mock_repository.get_recording_by_id.assert_called_once_with(recording_id)