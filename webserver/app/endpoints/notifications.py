import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from app.core.websocket_manager import ConnectionManager
from app.core.dependencies import get_connection_manager

router = APIRouter(
    prefix="/notifications",
    tags=["Notifications"],
)

@router.websocket("/ws/{person_id}")
async def websocket_endpoint(websocket: WebSocket, person_id: str, connection_manager: ConnectionManager = Depends(get_connection_manager)):
    print(f"Connecting to websocket for person {person_id}")
    await websocket.accept()
    await connection_manager.connect(person_id, websocket)

    try:
        while True:
            message = {
                "type": "heartbeat",
                "msg": f"Hello {person_id}",
            }
            await websocket.send_json(message)
            await connection_manager.connect(person_id, websocket)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        connection_manager.disconnect(person_id, websocket)
    print(f"Disconnected from websocket for person {person_id}")