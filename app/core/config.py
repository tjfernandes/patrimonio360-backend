from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_csv(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or ["*"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    APP_ENV: str = "development"
    API_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    LOG_JSON_PRETTY: bool = True
    LOG_JSON_INDENT: int = 2
    LOG_CHAT_MESSAGES: bool = False
    LOG_CHAT_STATE_HISTORY: bool = False

    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ALLOW_METHODS: str = "*"
    CORS_ALLOW_HEADERS: str = "*"

    OPENSEARCH_HOST: str | None = None
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_USERNAME: str | None = None
    OPENSEARCH_PASSWORD: str | None = None
    OPENSEARCH_SCHEME: str = "https"
    OPENSEARCH_USE_SSL: bool | None = None
    OPENSEARCH_VERIFY_CERTS: bool = False
    OPENSEARCH_SSL_SHOW_WARN: bool = False

    OPENSEARCH_INDEX_ARTIFACT: str = "cultural_heritage_artifacts_v1"
    OPENSEARCH_INDEX_IMAGE: str = "cultural_heritage_images"
    OPENSEARCH_INDEX_MUSEUM: str = "cultural_heritage_museums"

    ARTIFACT_TEXT_EMBEDDING_DIMENSION: int = 2560
    ARTIFACT_MULTIMODAL_EMBEDDING_DIMENSION: int = 2048
    MUSEUM_TEXT_EMBEDDING_DIMENSION: int = 2560
    IMAGE_MULTIMODAL_EMBEDDING_DIMENSION: int = 2048

    QWEN_TEXT_EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-Embedding-4B"
    QWEN_MULTIMODAL_EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-VL-Embedding-2B"
    # Backward compatibility with previous backend env naming.
    TEXT_EMBEDDING_MODEL: str | None = None
    MULTIMODAL_EMBEDDING_MODEL: str | None = None
    EMBEDDING_PREFER_BF16: bool = True
    EMBEDDING_MAX_LENGTH: int = 2048
    TEXT_EMBEDDING_BATCH_SIZE: int = 2
    MULTIMODAL_TEXT_EMBEDDING_BATCH_SIZE: int = 8
    MULTIMODAL_IMAGE_EMBEDDING_BATCH_SIZE: int = 4

    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str = "https://amalia.novasearch.org/vlm/chat/completions"
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "carminho/AMALIA-9B-50-DPO"
    # Backward-compatible legacy settings; ignored when LLM_MODEL is set.
    LLM_MODEL_TEXT: str | None = None
    LLM_MODEL_JSON: str | None = None
    LLM_TEMPERATURE_TEXT: float = 0.4
    LLM_TEMPERATURE_JSON: float = 0.0
    # 0 means no server-side max token cap from this API layer.
    LLM_MAX_TOKENS: int = 0
    LLM_MAX_TOKENS_TEXT: int | None = None
    LLM_MAX_TOKENS_JSON: int | None = None
    LLM_TIMEOUT_SECONDS: float = 45.0

    CHAT_HISTORY_WINDOW: int = 8
    CHAT_SESSION_TTL_SECONDS: int = 3600
    CHAT_ROLLING_SUMMARY_MAX_CHARS: int = 600
    CHAT_ENABLE_RAG: bool = True
    CHAT_ENABLE_LLM_LEXICAL_QUERY: bool = True
    CHAT_ENABLE_STRUCTURED_QUERY_PLANNING: bool = True
    CHAT_ANALYTICS_PLANNER_MIN_CONFIDENCE: float = 0.55
    CHAT_ANALYTICS_LIST_TOP_K: int = 10
    CHAT_PREWARM_ON_STARTUP: bool = False
    CHAT_PREWARM_INCLUDE_MULTIMODAL: bool = True
    CHAT_PREWARM_INCLUDE_RERANKER: bool = False
    CHAT_PREWARM_INCLUDE_MULTIVIEW_WORKER: bool = False
    CHAT_USE_QUERY_EMBEDDINGS: bool = True
    CHAT_RETRIEVAL_EMBEDDING_ONLY: bool = False
    CHAT_RETRIEVAL_CANDIDATES: int = 15
    CHAT_RETRIEVAL_TOP_K: int = 5
    CHAT_IN_TOUR_BOOST: float = 1.75
    CHAT_ENABLE_RERANKING: bool = False
    RERANKER_MODEL_ID: str = "Qwen/Qwen3-Reranker-4B"
    RERANKER_INSTRUCTION: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    RERANKER_PREFER_BF16: bool = True
    RERANKER_MAX_LENGTH: int = 1024
    RERANKER_BATCH_SIZE: int = 4
    CHAT_IMAGE_RETRIEVAL_TOP_K: int = 6
    CHAT_IMAGE_ARTIFACT_TOP_K: int = 5
    CHAT_IMAGE_DEFAULT_MESSAGE: str = "Analisa a imagem e identifica a peça mais provável no museu."
    CHAT_MODEL_DEFAULT_MESSAGE: str = "Analisa este modelo 3D e identifica a peça mais provável no museu."
    CHAT_MODEL_FIRST_PASS_VIEWS: int = 3
    CHAT_MODEL_TOTAL_VIEWS: int = 5
    CHAT_MODEL_LOW_CONFIDENCE_SCORE_THRESHOLD: float = 0.35
    CHAT_MODEL_CACHE_SIZE: int = 12
    IMAGE_ASSET_ROOT: str | None = None
    POI_TOURS_DIR: str | None = None
    MULTIVIEW_WORKER_HOST: str = "127.0.0.1"
    MULTIVIEW_WORKER_PORT: int = 3101
    MULTIVIEW_WORKER_START_TIMEOUT_SECONDS: float = 30.0
    MULTIVIEW_RENDER_SIZE: int = 512
    MULTIVIEW_RENDER_BACKGROUND: str = "#FFFFFF"
    MULTIVIEW_RENDER_FOV: int = 35
    MULTIVIEW_RENDER_DPR: float = 1.0
    MULTIVIEW_RENDER_STRATEGY: str = "adaptive"
    MULTIVIEW_RENDER_OVERSAMPLE: int = 320
    MULTIVIEW_RENDER_ORBIT_MARGIN: float = 1.35
    MULTIVIEW_RENDER_ENSURE_TOP: bool = True
    MULTIVIEW_RENDER_DELAY_MS: int = 8
    MULTIVIEW_SAVE_LAST_VIEWS: bool = True
    MULTIVIEW_LAST_VIEWS_DIR: str = "tmp/multiview_last_views"

    @property
    def cors_allow_origins_list(self) -> list[str]:
        return _parse_csv(self.CORS_ALLOW_ORIGINS)

    @property
    def cors_allow_methods_list(self) -> list[str]:
        return _parse_csv(self.CORS_ALLOW_METHODS)

    @property
    def cors_allow_headers_list(self) -> list[str]:
        return _parse_csv(self.CORS_ALLOW_HEADERS)

    @property
    def opensearch_use_ssl_resolved(self) -> bool:
        if self.OPENSEARCH_USE_SSL is not None:
            return self.OPENSEARCH_USE_SSL
        return bool(self.OPENSEARCH_USERNAME and self.OPENSEARCH_PASSWORD)

    @property
    def llm_base_url_resolved(self) -> str:
        return self.LLM_BASE_URL.rstrip("/")

    @property
    def llm_openai_base_url_resolved(self) -> str:
        """Accept full chat-completions URL and convert it to OpenAI client base_url."""
        base = self.llm_base_url_resolved
        suffixes = ("/chat/completions",)
        for suffix in suffixes:
            if base.endswith(suffix):
                return base[: -len(suffix)].rstrip("/")
        return base

    @property
    def llm_model_resolved(self) -> str:
        if self.LLM_MODEL and self.LLM_MODEL.strip():
            return self.LLM_MODEL.strip()
        if self.LLM_MODEL_TEXT and self.LLM_MODEL_TEXT.strip():
            return self.LLM_MODEL_TEXT.strip()
        if self.LLM_MODEL_JSON and self.LLM_MODEL_JSON.strip():
            return self.LLM_MODEL_JSON.strip()
        return "carminho/AMALIA-9B-50-DPO"

    @property
    def llm_auth_header(self) -> dict[str, str]:
        if not self.LLM_API_KEY:
            return {}
        return {"Authorization": f"Bearer {self.LLM_API_KEY}"}

    @property
    def text_embedding_model_resolved(self) -> str:
        if self.TEXT_EMBEDDING_MODEL and self.TEXT_EMBEDDING_MODEL.strip():
            return self.TEXT_EMBEDDING_MODEL.strip()
        return self.QWEN_TEXT_EMBEDDING_MODEL_ID.strip()

    @property
    def multimodal_embedding_model_resolved(self) -> str:
        if self.MULTIMODAL_EMBEDDING_MODEL and self.MULTIMODAL_EMBEDDING_MODEL.strip():
            return self.MULTIMODAL_EMBEDDING_MODEL.strip()
        return self.QWEN_MULTIMODAL_EMBEDDING_MODEL_ID.strip()

    @property
    def reranker_model_resolved(self) -> str:
        model_id = (self.RERANKER_MODEL_ID or "").strip()
        return model_id or "Qwen/Qwen3-Reranker-4B"

    @property
    def image_asset_root_resolved(self) -> Path | None:
        raw = (self.IMAGE_ASSET_ROOT or "").strip()
        if not raw:
            return None
        return Path(raw).expanduser().resolve()

    @property
    def poi_tours_dir_resolved(self) -> Path:
        raw = (self.POI_TOURS_DIR or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path(__file__).resolve().parents[2] / "poi_tours").resolve()

    @property
    def multiview_last_views_dir_resolved(self) -> Path:
        raw = (self.MULTIVIEW_LAST_VIEWS_DIR or "").strip()
        if not raw:
            raw = "tmp/multiview_last_views"
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
