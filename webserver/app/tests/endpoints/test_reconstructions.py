import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from app.core.services.interfaces.reconstruction_service_interface import IReconstructionService
from app.core.dependencies import get_reconstruction_service
from app.core.services.interfaces.user_service_interfaces import IUserService
from app.core.dependencies import get_user_service
from starlette.middleware.base import BaseHTTPMiddleware
from app.endpoints.reconstructions import router

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
def test_client(test_app):
    return TestClient(test_app)

@pytest.fixture
def mock_reconstruction_service():
    return MagicMock(spec=IReconstructionService)

@pytest.fixture
def mock_user_service():
    return MagicMock(spec=IUserService)

def test_reconstruct_id_success(test_app, test_client, mock_reconstruction_service, mock_user_service):
    mock_reconstruction_service.request_reconstruction.return_value = {
        "image_filename": "test.png",
        "metrics": {"accuracy": 0.99}
    }
    test_app.dependency_overrides[get_reconstruction_service] = lambda: mock_reconstruction_service
    test_app.dependency_overrides[get_user_service] = lambda: mock_user_service
    mock_user_service.get_nsd_id.return_value = "test-nsd-id"
    response = test_client.post("/reconstructions/1", json={"number_of_steps": 5})
    assert response.status_code == 200
    assert response.json() == {
        "message": "Reconstruction requested successfully"
    }

