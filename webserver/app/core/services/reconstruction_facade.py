import aioboto3
import json
import os
import aiohttp
from app.core.services.interfaces.reconstruction_facade_interface import IReconstructionSQSFacade

class ReconstructionFacade(IReconstructionSQSFacade):
    def __init__(self):
        self.queue_name = "reconstruction-jobs"
        self.aws_region = os.getenv("AWS_REGION", "eu-north-1")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.session = aioboto3.Session(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )

    async def reconstruct_image(self, person_id, brain_recording_id, voxels, number_of_steps, reconstruction_id):
        async with self.session.client("sqs") as sqs:
            queue_url = (await sqs.get_queue_url(QueueName=self.queue_name))["QueueUrl"]
            message_body = {
                "action": "reconstruct_image",
                "person_id": person_id,
                "voxels": voxels,
                "brain_recording_id": brain_recording_id,
                "number_of_steps": number_of_steps,
                "reconstruction_id": reconstruction_id
            }
            send_response = await sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message_body)
            )
            return {"message_id": send_response["MessageId"]}

    async def get_metrics_for_reconstruction(self, original_image, reconstructed_image):
        url = os.getenv("RECONSTRUCTION_SERVER_URL")
        token = os.getenv("RECONSTRUCTION_SERVER_TOKEN")
        endpoint = f"{url}metrics/"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        data = aiohttp.FormData()
        data.add_field("original", original_image, filename="original.png", content_type="image/png")
        data.add_field("reconstructed", reconstructed_image, filename="reconstructed.png", content_type="image/png")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, data=data, headers=headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            return {}
                