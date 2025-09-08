from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from app.core.schemas.schemas import BrainRecordingSchema
from app.core.dependencies import get_recording_service, get_reconstruction_service
from app.core.services.recording_service import RecordingService
from app.core.services.reconstruction_service import ReconstructionService
from app.core.schemas.schemas import ReconstructionSchema
from app.core.services.user_service import UserService
from app.core.dependencies import get_user_service

router = APIRouter(prefix="/recordings", tags=["Recordings"])

@router.get("/", response_model=List[BrainRecordingSchema])
async def get_recordings(
    request: Request,
    recording_service: RecordingService = Depends(get_recording_service),
    user_service: UserService = Depends(get_user_service), 
):
    nsd_id = await user_service.get_nsd_id(request.state.cognito_user.get("sub"))
    return await recording_service.get_all_recordings(nsd_id)


@router.get("/{recording_id}", response_model=BrainRecordingSchema)
async def get_recording(
    recording_id: int,
    recording_service: RecordingService = Depends(get_recording_service) 
):
    recording = await recording_service.get_recording_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    return recording


@router.post("/")
async def create_recording(
    request: Request,
    nifti_file: UploadFile = File(...),
    png_file: UploadFile = File(...),
    description: str = Form(""),
    recording_service: RecordingService = Depends(get_recording_service),
    user_service: UserService = Depends(get_user_service), 
):
    nsd_id = await user_service.get_nsd_id(request.state.cognito_user.get("sub"))
    return await recording_service.create_recording(nifti_file, png_file, description, nsd_id)


@router.get("/{id}/reconstructions", response_model=List[ReconstructionSchema])
async def get_reconstructions_for_brain_recording(
    id: int,
    reconstruction_service: ReconstructionService = Depends(get_reconstruction_service),
):
    reconstructions = await reconstruction_service.get_reconstructions_by_brain_recording_id(id)
    return reconstructions


@router.delete("/{recording_id}")
async def delete_recording(
    recording_id: int,
    recording_service: RecordingService = Depends(get_recording_service)
):
    try:
        await recording_service.delete_recording(recording_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Recording deleted successfully"}
