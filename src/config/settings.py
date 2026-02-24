import functools

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-5"

    # Supabase
    supabase_url: str
    supabase_service_key: str

    # Web search
    web_search_api_key: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    debug: bool = False
    cors_allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Cache TTLs (hours)
    company_cache_ttl_hours: int = 4
    query_cache_ttl_hours: int = 2

    # Data export
    export_supabase_data: bool = False
    supabase_export_dir: str = "data_exports"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_version: str = "1.0.0"

    def get_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


@functools.cache
def get_settings() -> Settings:
    return Settings()
