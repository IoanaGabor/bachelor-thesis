import os
from pathlib import Path
import shutil
from sqlalchemy import Enum

class FileType(str, Enum):
    NPY = "npy"
    PNG = "png"


UPLOAD_DIR = Path("uploads")

def save_file(file, new_filename: str, file_type: FileType) -> str:
    actual_file = file.file if hasattr(file, "file") else file

    subdir = UPLOAD_DIR / file_type
    os.makedirs(subdir, exist_ok=True)
    path = subdir / new_filename

    with open(path, "wb") as buffer:
        shutil.copyfileobj(actual_file, buffer)

    return str(path)