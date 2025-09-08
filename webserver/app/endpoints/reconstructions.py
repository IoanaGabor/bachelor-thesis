from fastapi import APIRouter, Depends, HTTPException
from app.core.dependencies import get_reconstruction_service, get_user_service
from app.core.services.reconstruction_service import ReconstructionService
from app.core.services.user_service import UserService
from pydantic import BaseModel
from typing import Dict
from fastapi import Request

router = APIRouter(prefix="/reconstructions", tags=["Reconstructions"])

class ReconstructionResponse(BaseModel):
    image_filename: str
    metrics: Dict[str, float]  


@router.post("/{id}")
async def reconstruct_id(
    request: Request,
    id: int,
    reconstruction_service: ReconstructionService = Depends(get_reconstruction_service),
    user_service: UserService = Depends(get_user_service),
):
    body = await request.json()
    number_of_steps = body.get("number_of_steps")
    if number_of_steps is None:
        raise HTTPException(status_code=400, detail="number_of_steps is required in the request body")
    nsd_id = await user_service.get_nsd_id(request.state.cognito_user.get("sub"))
    await reconstruction_service.request_reconstruction(id, nsd_id, number_of_steps)
    return {"message": "Reconstruction requested successfully"}


@router.get("/statistics", summary="Get reconstruction statistics")
async def get_reconstruction_statistics(
    request: Request,
    reconstruction_service: ReconstructionService = Depends(get_reconstruction_service),
    user_service: UserService = Depends(get_user_service),
):
    nsd_id = await user_service.get_nsd_id(request.state.cognito_user.get("sub"))
    try:
        stats = await reconstruction_service.get_metrics_by_subject_id(nsd_id)
        if stats is None:
            raise HTTPException(status_code=404, detail="Statistics not found")
        return stats
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
