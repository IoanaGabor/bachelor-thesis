import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from reconstruction_service.reconstruction_service import ReconstructionService

class VoxelReconstructionRequest(BaseModel):
    person_id: str
    voxels: List[float]  

def get_reconstruction_service():
    return ReconstructionService()

router = APIRouter(prefix="/reconstruct", tags=["Reconstructions"])

@router.post("/")
def reconstruct_image(
    request: VoxelReconstructionRequest,
    service: ReconstructionService = Depends(get_reconstruction_service),
):
    try:
        result = service.reconstruct_image(request.person_id, request.voxels)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
