from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    reconstruction_server_api: str
    nextauth_url: str
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    aws_region: str
    cognito_user_pool_id: str
    cognito_client_id: str
    cognito_jwks_url: str
    class ConfigDict:
        env_file = ".env"

def get_settings():
    return Settings()
