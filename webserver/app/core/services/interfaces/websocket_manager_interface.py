from abc import ABC, abstractmethod
from fastapi import WebSocket

class IConnectionManager(ABC):
    @abstractmethod
    async def connect(self, person_id: str, websocket: WebSocket):
        pass

    @abstractmethod
    def disconnect(self, person_id: str, websocket: WebSocket):
        pass

    @abstractmethod
    async def send_reconstruction_notification(self, person_id: str, reconstruction_id: str, brain_recording_id: str, number_of_steps: int):
        pass