import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.endpoints.recordings import router
from app.core.services.recording_service import RecordingService
from app.core.dependencies import get_recording_service
from app.core.services.user_service import UserService
from app.core.dependencies import get_user_service
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

class MockCognitoUserMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, cognito_user):
        super().__init__(app)
        self.cognito_user = cognito_user
    async def dispatch(self, request, call_next):
        request.state.cognito_user = self.cognito_user
        response = await call_next(request)
        return response

@pytest.fixture
def mock_cognito_user():
    return {
        "sub": "test-user-id",
        "email": "test@example.com",
        "username": "testuser"
    }

@pytest.fixture
def test_app(mock_cognito_user):
    app = FastAPI()
    app.add_middleware(MockCognitoUserMiddleware, cognito_user=mock_cognito_user)
    app.include_router(router)
    return app

@pytest.fixture
def test_client(test_app: FastAPI):
    return TestClient(test_app)

@pytest.fixture
def mock_recording_service():
    return MagicMock(spec=RecordingService)

@pytest.fixture
def mock_user_service():
    return MagicMock(spec=UserService)

def test_get_recordings_success(test_app, test_client, mock_recording_service, mock_user_service):
    now = datetime.utcnow()
    mock_recording_service.get_all_recordings.return_value = [
        {
            "id": 1,
            "description": "Test recording 1",
            "voxels_file": "recording1.npy",
            "png_file": "recording1.png",
            "uploaded_at": now
        },
        {
            "id": 2,
            "description": "Test recording 2",
            "voxels_file": "recording2.npy",
            "png_file": "recording2.png",
            "uploaded_at": now
        }
    ]
    test_app.dependency_overrides[get_recording_service] = lambda: mock_recording_service
    test_app.dependency_overrides[get_user_service] = lambda: mock_user_service
    mock_user_service.get_nsd_id.return_value = "test-nsd-id"
    response = test_client.get("/recordings")
    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 1,
            "description": "Test recording 1",
            "voxels_file": "recording1.npy",
            "png_file": "recording1.png",
            "uploaded_at": now.isoformat()
        },
        {
            "id": 2,
            "description": "Test recording 2",
            "voxels_file": "recording2.npy",
            "png_file": "recording2.png",
            "uploaded_at": now.isoformat()
        }
    ]

def test_get_recording_by_id_success(test_app, test_client, mock_recording_service, mock_user_service):
    now = datetime.utcnow()
    mock_recording_service.get_recording_by_id.return_value = {
        "id": 1,
        "description": "Test recording",
        "voxels_file": "recording1.npy",
        "png_file": "recording1.png",
        "uploaded_at": now
    }
    test_app.dependency_overrides[get_recording_service] = lambda: mock_recording_service
    test_app.dependency_overrides[get_user_service] = lambda: mock_user_service
    response = test_client.get("/recordings/1")
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "description": "Test recording",
        "voxels_file": "recording1.npy",
        "png_file": "recording1.png",
        "uploaded_at": now.isoformat()
    }
