from pydantic_settings import BaseSettings
import secrets


class Settings(BaseSettings):

    APP_NAME: str = "Career Guidance Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    SECRET_KEY: str = secrets.token_hex(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
