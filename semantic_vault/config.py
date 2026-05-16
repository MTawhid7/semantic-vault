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

    # Phase 2 — Neo4j knowledge graph (leave blank to disable graph features)
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Entity resolution thresholds
    er_high_threshold: float = 0.85   # cosine ≥ this → definite match
    er_low_threshold: float = 0.65    # cosine < this → definitely new entity

    # Phase 3 — agentic retrieval
    enable_query_planner: bool = False   # decompose queries with Gemini
    enable_reranking: bool = False       # LLM-as-judge reranking
    rerank_top_n: int = 20               # candidates sent to reranker
    enable_authority: bool = False       # source authority weighting
    ontology_evolution_interval: int = 10  # run evolution every N ingestions


settings = Settings()
