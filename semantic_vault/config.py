from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_synthesis_model: str = "gemini-3-flash-preview"
    gemini_embed_model: str = "gemini-embedding-2"
    gemini_embed_dim: int = 3072

    qdrant_url: str = "./qdrant_db"
    db_path: str = "semantic_vault.db"

    chunk_size_chars: int = 1600
    chunk_overlap_chars: int = 200

    top_k: int = 8


settings = Settings()
