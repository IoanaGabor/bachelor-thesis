from pydantic import BaseModel
from datetime import datetime

class BrainRecordingSchema(BaseModel):
    id: int
    voxels_file: str | None
    png_file: str | None
    description: str | None
    uploaded_at: datetime

    class Config:
        from_attributes = True


class ReconstructionSchema(BaseModel):
    id: int
    brain_recording_id: int | None
    reconstruction_png_path: str | None
    metrics_json: str | None
    number_of_steps: int | None
    status: str | None
    uploaded_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str
