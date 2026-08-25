from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret_key: str = "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Docker sets this to PostgreSQL; SQLite keeps local tests self-contained.
    database_url: str = "sqlite:///data/hr_policy.db"

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "data/qdrant"
    qdrant_collection_prefix: str = "company_"

    rate_limit_per_minute: int = 20
    login_rate_limit_per_minute: int = 5
    # Kill switch for rate limiting. Defaults to ON (fail-closed). Set to false
    # for local load-testing / unit tests that don't provision a Redis.
    rate_limit_enabled: bool = True

    # The Streamlit client runs on 8501. Credentials-enabled CORS must use an
    # explicit origin allowlist -- ["*"] + allow_credentials is not allowed.
    cors_allow_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]


settings = Settings()
