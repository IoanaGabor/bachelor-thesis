from fastapi import APIRouter, Depends, Request as FastAPIRequest
from app.core.services.reconstruction_service import ReconstructionService
from app.core.dependencies import get_reconstruction_service
from app.core.logger_config import logger

router = APIRouter(
    prefix="/webhooks",
    tags=["Webhooks"],
)

@router.post("/reconstruction")
async def receive_reconstruction_webhook(
    request: FastAPIRequest, 
    reconstruction_service: ReconstructionService = Depends(get_reconstruction_service)
):
    data = await request.json()
    
    person_id, brain_recording_id, image_b64, reconstruction_id, status = (
        data.get("person_id"),
        data.get("brain_recording_id"),
        data.get("reconstructed_image"),
        data.get("reconstruction_id"),
        data.get("status")
    )
    logger.info(f"Received reconstruction webhook for person {person_id}, brain recording {brain_recording_id}, reconstruction id {reconstruction_id}, status {status}")

    await reconstruction_service.handle_reconstruction_webhook(person_id, brain_recording_id, image_b64, reconstruction_id, status)
    return {"status": "received"}
