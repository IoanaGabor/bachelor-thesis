from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession
from app.core.config import Settings
import asyncio
from sqlalchemy.exc import OperationalError

Base = declarative_base()

class DatabaseEngine:
    __instance = None
    __engine = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(DatabaseEngine, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        if self.__engine is None:
            settings = Settings()
            self.__engine = create_async_engine(
                settings.database_url,
                pool_size=5,
                max_overflow=10,
                pool_timeout=60,
                pool_recycle=1800,
                connect_args={
                    "timeout": 60,
                    "command_timeout": 60,
                    "server_settings": {
                        "statement_timeout": "60000",
                        "lock_timeout": "60000"
                    }
                }
            )

    @property
    def engine(self):
        return self.__engine

async def get_async_session():
    db_engine = DatabaseEngine()
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            async with AsyncSession(db_engine.engine) as session:
                yield session
                break
        except OperationalError as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(retry_delay)