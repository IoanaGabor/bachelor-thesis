import io
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from torchvision import transforms
from PIL import Image
from metrics.calculator import MetricsCalculator
from reconstruction_service.logger_config import logger

router = APIRouter(prefix="/metrics", tags=["Metrics"])

def get_metrics_calculator():
    return MetricsCalculator()

def read_image(file: UploadFile):
    if file.content_type != "image/png":
        raise HTTPException(status_code=400, detail="Not a PNG image.")

    image_data = file.file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((425, 425))
    transform = transforms.ToTensor()
    return transform(image)

@router.post("/")
async def compute_metrics(
    original: UploadFile = File(...),
    reconstructed: UploadFile = File(...),
    calculator: MetricsCalculator = Depends(get_metrics_calculator)
):
    logger.info("Computing metrics")
    img1 = read_image(original)
    img2 = read_image(reconstructed)

    metrics = calculator.compute_all(img1, img2)
    logger.info(f"Metrics: {metrics}")
    return metrics
