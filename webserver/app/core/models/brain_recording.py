from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.core.persistence.database import Base

class BrainRecording(Base):
    __tablename__ = "brain_recordings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    voxels_file = Column(String, nullable=False)
    png_file = Column(String, nullable=False)
    description = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
