"""Settings from environment only (TRD 17.4). Missing required secrets fail fast at startup."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "development"
    database_url: str = Field(..., description="postgresql+psycopg://user:pass@host/db")
    redis_url: str = "redis://redis:6379/0"
    jwt_secret: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    business_timezone: str = "Asia/Karachi"
    max_import_bytes: int = 5_000_000
    max_import_rows: int = 50_000
    login_rate_limit_per_minute: int = 10
    cors_origins: list[str] = ["http://localhost:3000"]
    use_celery: bool = False          # True in docker-compose (worker container)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # raises ValidationError if DATABASE_URL / JWT_SECRET are missing
