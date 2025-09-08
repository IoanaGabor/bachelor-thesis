from fastapi import FastAPI
from endpoints import reconstructions
from endpoints import metrics
from fastapi.middleware.cors import CORSMiddleware
from background_worker.sqs import listen_to_sqs
import uvicorn
import threading
from reconstruction_service.reconstruction_service import ReconstructionService
from reconstruction_service.logger_config import logger
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],           
    allow_headers=["*"],           
)

app.include_router(metrics.router)
app.include_router(reconstructions.router)

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello, FastAPI!"}

def start_background_workers(app):
    logger.info("Starting background workers")
    service = ReconstructionService()
    sqs_thread = threading.Thread(target=listen_to_sqs, args=(service,), daemon=True)
    sqs_thread.start()
    app.state.sqs_thread = sqs_thread
    app.state.reconstruction_service = service
    logger.info("Background workers started successfully")

@app.on_event("startup")
def on_startup():
    logger.info("Application startup initiated")
    start_background_workers(app)
    logger.info("Application startup completed")

@app.on_event("shutdown")
def on_shutdown():
    logger.info("Application shutdown completed")

if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
