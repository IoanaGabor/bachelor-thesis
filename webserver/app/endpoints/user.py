from app.core.dependencies import get_user_service
from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.core.services.cognito_facade import CognitoFacade
from app.core.services.user_service import UserService
import os

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/attributes", summary="Get user attributes")
async def get_user_attributes(request: Request,user_service: UserService = Depends(get_user_service)):
    try:
        print(request.state.cognito_user)
        sub = request.state.cognito_user.get("sub")
        print(sub)
        attributes = await user_service.get_user_attributes(sub)
        if attributes is None:
            raise HTTPException(status_code=404, detail="User not found")
        return attributes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
