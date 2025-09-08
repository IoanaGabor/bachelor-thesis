from fastapi import WebSocket
from typing import Dict, Set
from app.core.services.interfaces.websocket_manager_interface import IConnectionManager
from app.core.logger_config import logger

class ConnectionManager(IConnectionManager):
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self.__class__._initialized:
            self.active_connections: Dict[str, Set[WebSocket]] = {}
            self.__class__._initialized = True

    async def connect(self, person_id: str, websocket: WebSocket):
        if person_id not in self.active_connections:
            self.active_connections[person_id] = set()
        self.active_connections[person_id].add(websocket)
        logger.info(f"Connected to websocket for person {person_id} and websocket {websocket}")
        logger.info(f"Active connections: {self.active_connections}")

    def disconnect(self, person_id: str, websocket: WebSocket):
        if person_id in self.active_connections:
            self.active_connections[person_id].discard(websocket)
            if not self.active_connections[person_id]:
                del self.active_connections[person_id]
        logger.info(f"Disconnected from websocket for person {person_id} and websocket {websocket}")
        logger.info(f"Active connections: {self.active_connections}")

    async def send_reconstruction_notification(self, person_id: str, reconstruction_id: str, brain_recording_id: str, status: str):
        logger.info(f"Sending reconstruction notification to {person_id}")
        logger.info(f"Active connections: {self.active_connections}")
        if person_id in self.active_connections:
            message = {
                "type": "reconstruction_notification",
                "reconstruction_id": reconstruction_id,
                "brain_recording_id": brain_recording_id,
                "status": status
            }
            print(f"Sending reconstruction notification to {person_id}")

            connections = set(self.active_connections[person_id])
            for connection in connections:
                try:
                    await connection.send_json(message)
                    print(f"Sent reconstruction notification to {person_id}")
                except Exception:
                    self.disconnect(person_id, connection)

            return True
        return False
