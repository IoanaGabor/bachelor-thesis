import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.persistence.database import Base
from app.endpoints import auth_middleware, recordings, reconstructions, webhooks, notifications, user
from dotenv import load_dotenv
from app.core.persistence.database import DatabaseEngine
from contextlib import asynccontextmanager
from app.endpoints.auth_middleware import cognito_token_middleware
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with DatabaseEngine().engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(lifespan=lifespan)
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recordings.router)
app.include_router(auth_middleware.router)
app.include_router(reconstructions.router)
app.include_router(webhooks.router)
app.include_router(notifications.router)
app.include_router(user.router)
app.middleware("http")(cognito_token_middleware)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
