from sqlalchemy import Column, Integer, String, DateTime
from app.core.persistence.database import Base
from datetime import datetime

class Reconstruction(Base):
    __tablename__ = "reconstruction"
    id = Column(Integer, primary_key=True, index=True)
    brain_recording_id = Column(Integer, nullable=False)
    reconstruction_png_path = Column(String, nullable=True)
    metrics_json = Column(String, nullable=True)
    number_of_steps = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=True)
    status = Column(String, nullable=True)
