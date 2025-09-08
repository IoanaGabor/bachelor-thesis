from app.core.persistence.reconstruction_repository import ReconstructionRepository
from app.utils.file_storage import save_file, FileType
from datetime import datetime
from app.core.services.reconstruction_facade import ReconstructionFacade
from app.core.services.recording_service import RecordingService
from app.core.websocket_manager import ConnectionManager
import numpy as np
import io
import base64
import json
from app.core.services.interfaces.reconstruction_service_interface import IReconstructionService
from app.core.logger_config import logger

class ReconstructionService(IReconstructionService):
    def __init__(self, repository: ReconstructionRepository, reconstruction_facade: ReconstructionFacade, recording_service: RecordingService, connection_manager: ConnectionManager):
        self.__repository = repository
        self.__reconstruction_facade = reconstruction_facade
        self.__recording_service = recording_service
        self.__connection_manager = connection_manager

    async def get_reconstruction_by_id(self, reconstruction_id: int):
        reconstruction = await self.__repository.get_reconstruction_by_id(reconstruction_id)
        if reconstruction is None:
            raise ValueError(f"No reconstruction found with id {reconstruction_id}")
        return reconstruction

    async def get_reconstructions_by_brain_recording_id(self, brain_recording_id: int):
        return await self.__repository.get_reconstructions_by_brain_recording_id(brain_recording_id)

    async def request_reconstruction(self, brain_recording_id: str, user_id: str, number_of_steps: int):
        recording = await self.__recording_service.get_recording_by_id(brain_recording_id)
        voxels_path = f"{recording.voxels_file}"
        data = np.load(voxels_path)
        reconstruction = await self.__repository.create_reconstruction(
            brain_recording_id=brain_recording_id,
            number_of_steps=number_of_steps,
            status="pending"
        )
        result = await self.__reconstruction_facade.reconstruct_image(user_id, brain_recording_id, data.tolist(), number_of_steps, reconstruction.id)

        logger.info(f"Reconstruction created with id {reconstruction.id} for message id {result['message_id']}")
        return result

    async def handle_reconstruction_webhook(self, person_id: str, brain_recording_id: str, image_b64: str,  reconstruction_id: int, status: str):
        if status == "completed":
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            png_filename = f"{person_id}_{brain_recording_id}_{timestamp}.png"
            image_file = io.BytesIO(base64.b64decode(image_b64)); image_file.name = png_filename
            reconstruction_png_path = save_file(image_file, png_filename, FileType.PNG)
            recording = await self.__recording_service.get_recording_by_id(brain_recording_id)
            original_image = recording.png_file

            with open(recording.png_file, "rb") as f:
                original_image_file = io.BytesIO(f.read()); original_image_file.name = original_image
            image_file.seek(0); original_image_file.seek(0)
            metrics = await self.__reconstruction_facade.get_metrics_for_reconstruction(original_image_file, image_file)
            metrics_json = json.dumps(metrics)
            reconstruction = await self.__repository.get_reconstruction_by_id(reconstruction_id)

            await self.__repository.update_reconstruction(
                reconstruction_id=reconstruction.id,
                reconstruction_png_path=reconstruction_png_path,
                metrics_json=metrics_json,
                uploaded_at=datetime.utcnow(),
                status=status
            )
        else:
            await self.__repository.update_reconstruction(
                reconstruction_id=reconstruction.id,
                status=status
            )
        await self.__connection_manager.send_reconstruction_notification(
            person_id=person_id,
            reconstruction_id=reconstruction.id,
            brain_recording_id=brain_recording_id,
            status=status
        )


    async def get_metrics_by_subject_id(self, subject_id: str):
        metrics = await self.__repository.get_metrics_by_subject_id(subject_id)
        if not metrics:
            return {}
        metric_sums = {}
        metric_counts = {}

        for reconstruction in metrics:
            if reconstruction.metrics_json:
                try:
                    metric_dict = json.loads(reconstruction.metrics_json)
                    logger.info(metric_dict)
                    for k, v in metric_dict.items():
                        try:
                            v = float(v)
                        except Exception:
                            continue
                        if k in metric_sums:
                            metric_sums[k] += v
                            metric_counts[k] += 1
                        else:
                            metric_sums[k] = v
                            metric_counts[k] = 1
                except Exception:
                    continue

        averages = {}
        for k in metric_sums:
            if metric_counts[k] > 0:
                averages[k] = metric_sums[k] / metric_counts[k]

        return averages
        return metrics

