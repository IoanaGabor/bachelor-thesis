import io
import boto3
import os
import time
import json
from reconstruction_service.reconstruction_service import ReconstructionService
import base64
import requests
from reconstruction_service.logger_config import logger

def listen_to_sqs(service: ReconstructionService):
    queue_name = os.getenv("SQS_QUEUE_NAME")
    aws_region = os.getenv("AWS_REGION")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    webhook_url = os.getenv("WEBHOOK_URL")

    if not all([queue_name, aws_region, aws_access_key_id, aws_secret_access_key]):
        logger.error("Missing required AWS configuration")
        raise ValueError("Missing required AWS configuration")

    logger.info(f"Initializing SQS client for queue: {queue_name}")
    sqs = boto3.client(
        "sqs",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    try:
        response = sqs.get_queue_url(QueueName=queue_name)
        queue_url = response["QueueUrl"]
        logger.info(f"Connected to queue: {queue_url}")
    except Exception as e:
        logger.error(f"Failed to get queue URL: {str(e)}", exc_info=True)
        raise

    while True:
        try:
            receive_response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=10  
            )

            messages = receive_response.get("Messages", [])

            if not messages:
                continue

            for message in messages:                
                receipt_handle = message["ReceiptHandle"]
                body = json.loads(message["Body"])
                logger.info(f"Processing message for person_id: {body['person_id']}")
                
                try:
                    result = service.reconstruct_image(body["person_id"], body["voxels"], body["number_of_steps"])

                    logger.info("Converting image to response format")
                    img_io = io.BytesIO()
                    result.save(img_io, format="PNG")
                    img_bytes = img_io.getvalue()
                    logger.info("Image reconstruction completed successfully")
                    img_bytes = base64.b64encode(img_bytes).decode("utf-8")
                    status = "completed"
                except Exception as e:
                    logger.warning("No result generated for reconstruction")
                    img_bytes = None
                    status = "failed"

                if webhook_url:
                    payload = {
                        "person_id": body.get("person_id"),
                        "brain_recording_id": body.get("brain_recording_id"),
                        "reconstructed_image": img_bytes,
                        "reconstruction_id": body.get("reconstruction_id"),
                        "status": status,
                    }
                    try:
                        response = requests.post(webhook_url, json=payload)
                        logger.info(f"Webhook POST status: {response.status_code}")
                    except Exception as e:
                        logger.error(f"Failed to call webhook: {str(e)}", exc_info=True)
                else:
                    logger.warning("WEBHOOK_URL environment variable not set. Skipping webhook call.")

                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle
                )
                logger.info("Deleted message from queue")

        except Exception as e:
            logger.error(f"Error in SQS processing loop: {str(e)}", exc_info=True)
            time.sleep(5) 
