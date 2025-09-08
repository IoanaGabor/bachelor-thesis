import aioboto3
import json
import os
from app.core.services.interfaces.reconstruction_facade_interface import IReconstructionSQSFacade

class ReconstructionSQSFacade(IReconstructionSQSFacade):
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

    async def reconstruct_image(self, person_id, brain_recording_id, voxels, number_of_steps):
        async with self.session.client("sqs") as sqs:
            queue_url = (await sqs.get_queue_url(QueueName=self.queue_name))["QueueUrl"]
            message_body = {
                "action": "reconstruct_image",
                "person_id": person_id,
                "voxels": voxels,
                "brain_recording_id": brain_recording_id,
                "number_of_steps": number_of_steps
            }
            send_response = await sqs.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message_body)
            )
            return {"message_id": send_response["MessageId"]}
