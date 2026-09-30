from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
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
    # Post-submission deployment candidates are opt-in; the current stack stays intact.
    RETRIEVAL_PROFILE_PATH: str = ""
    CHAT_ENABLE_EFFICIENCY_GUARDS: bool = False
    CHAT_BOUNDED_GPU_EXECUTION: bool = False
    CHAT_SERIALIZE_CONVERSATIONS: bool = False
    QUERY_LOG_COMPACT_VECTORS: bool = False
    # Quick wins pós-submissão (branch feature/post-submission-quick-wins).
    # Todos opt-in; desligados reproduzem byte a byte o comportamento do estudo.
    CHAT_DETERMINISTIC_SMALLTALK: bool = False      # saudação pura sem router nem LLM
    CHAT_TEMPLATE_RESULTS_PAGE_REPLY: bool = False  # «ver mais» sem chamada LLM
    CHAT_STREAM_FINAL_ANSWER: bool = False          # tokens da resposta final por SSE
    CHAT_COMPACT_ANSWER_CONTEXT: bool = False       # cartões compactos + prompt final PT v2
    CHAT_INVENTORY_GRAMMAR_V2: bool = False         # «793 Mob MNSR» reconhecido como inventário
    # off | log — verifica títulos/inventários citados | repair — log + repõe
    # inventários abreviados que correspondem a um único cartão («71 CMP/MNSR»)
    CHAT_GROUNDING_CHECK: str = "off"
    # Router: "llm" = chamada JSON ao 9B em todos os turnos (2,7–2,9 s, estudo);
    # "deterministic" = política de contexto decide (rag salvo small talk;
    # refine se follow-up; filtros só por extração determinística). Nos 146
    # turnos de texto do estudo o router LLM nunca produziu filters_delta e as
    # 3 vezes em que escolheu llm_only fora de saudações estava errado.
    CHAT_ROUTER_MODE: str = "llm"
    # «Onde estou?» / «o que há nesta sala?»: resposta determinística a partir
    # da posição na visita (metadata.tour_location enviado pelo widget) e dos
    # ficheiros poi_tours, sem pesquisa nem LLM. Desligado = vai ao router.
    CHAT_TOUR_LOCATION_ANSWERS: bool = False
    # Peça selecionada («Perguntar», hotspot aberto): um pedido que não é sobre ela
    # (nova pesquisa, «onde estou?») liberta-a em vez de a forçar como contexto.
    CHAT_SELECTED_ARTIFACT_AUTO_RELEASE: bool = False
    # Peça implícita: sem botões nem etiquetas, a pergunta resolve-se para a peça de que
    # o visitante fala (ordinal, n.º de inventário ou nome de um cartão visível; senão o
    # hotspot aberto ou o primeiro resultado) quando não traz assunto próprio.
    CHAT_IMPLICIT_ARTIFACT_CONTEXT: bool = False
    # Com o anterior: a frase é redigida pelo LLM a partir dos factos da
    # visita (sala e peças fixas; cartões trancados); template se falhar.
    CHAT_TOUR_CONTEXT_LLM_WORDING: bool = False
    # Intenção explícita do frontend: ao carregar em «Ver na tour» num cartão, o
    # pedido traz metadata.intent = "describe_artifact" e a peça em
    # metadata.selected_artifact. Sem pesquisa: a ficha vai buscar-se pelo
    # artifact_id e a resposta é uma apresentação curta da peça, só dela.
    CHAT_DESCRIBE_ARTIFACT_INTENT: bool = False
    # O modelo escolhe os cartões: termina a resposta com «[[cartões: doc_1, doc_3]]»
    # (retirado do texto) e só essas peças, pela ordem dele, ficam em
    # artifact_results, image_matches e navigation_targets. Um pedido por nome
    # ou n.º de inventário mostra só as peças que correspondem («egito» ≈
    # «Egipto»), sem «ver mais». Sem bloco, os cartões ficam como hoje.
    CHAT_MODEL_SELECTS_CARDS: bool = False
    # EXPERIMENTAL, manter desligado: medido em 2026-09-15 sem ganho de tempo
    # (o 9B enche filters_delta com filtros inventados). Ver PLANO_MELHORIAS §3.
    CHAT_ROUTER_COMPACT_SCHEMA: bool = False        # router pede 5 campos em vez de 12
    API_PREFIX: str = "/api/v1"

    BACKEND_LOG_ENABLED: bool = True
    BACKEND_LOG_LEVEL: str = "INFO"
    BACKEND_ACCESS_LOG_ENABLED: bool = True
    BACKEND_LOG_HEALTHCHECKS: bool = False
    BACKEND_RAG_DEBUG_ENABLED: bool = True
    BACKEND_RAG_DEBUG_MAX_CHARS: int = 40000

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

    # Defaults reflect the live production indexes; .env still overrides.
    OPENSEARCH_INDEX_ARTIFACT: str = "cultural_heritage_artifacts"
    OPENSEARCH_INDEX_IMAGE: str = "cultural_heritage_images"
    OPENSEARCH_INDEX_MUSEUM: str = "cultural_heritage_museums"
    # Indices das entidades relacionais (autores, conjuntos, exposicoes).
    # Geridos pelo patrimonio360_indexer; usados para enriquecer o modal de
    # artefacto com a info da entidade + lista de outros objetos relacionados.
    OPENSEARCH_INDEX_AUTOR: str = "cultural_heritage_authors"
    OPENSEARCH_INDEX_CONJUNTO: str = "cultural_heritage_sets"
    OPENSEARCH_INDEX_EXPOSICAO: str = "cultural_heritage_exhibitions"
    # Top N de artefactos por conjunto / exposicao a mostrar no modal.
    CHAT_RELATED_ARTIFACTS_TOP_K: int = 24

    # Defaults reflect the production embedding stack (Qwen3-Embedding-4B /
    # Qwen3-VL-Embedding-2B); .env still overrides.
    ARTIFACT_TEXT_EMBEDDING_DIMENSION: int = 2560
    ARTIFACT_MULTIMODAL_EMBEDDING_DIMENSION: int = 2048
    MUSEUM_TEXT_EMBEDDING_DIMENSION: int = 2560
    IMAGE_MULTIMODAL_EMBEDDING_DIMENSION: int = 2048

    QWEN_TEXT_EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-Embedding-4B"
    QWEN_MULTIMODAL_EMBEDDING_MODEL_ID: str = "Qwen/Qwen3-VL-Embedding-2B"
    # Optional pinned HF revisions (snapshot hashes). Empty = unpinned, which
    # preserves the pre-existing loading behavior; the v4/8B deploy pins these
    # to the snapshots used by the index rebuild (see indexer .env.example).
    QWEN_TEXT_EMBEDDING_MODEL_REVISION: str = ""
    QWEN_MULTIMODAL_EMBEDDING_MODEL_REVISION: str = ""
    USE_OPENROUTER_BGE_M3: bool = False
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_BGE_MODEL: str = "baai/bge-m3"
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
    # LLM local (vLLM, deploy/llm/amalia_llm.sh) com a API remota como reserva:
    # se o servidor primário recusar a ligação (em baixo, a arrancar) o mesmo
    # pedido segue para o fallback. Vazio = sem fallback (comportamento antigo).
    LLM_CONNECT_TIMEOUT_SECONDS: float = 5.0
    LLM_FALLBACK_BASE_URL: str = ""
    LLM_FALLBACK_MODEL: str = ""
    LLM_FALLBACK_API_KEY: str | None = None

    CHAT_HISTORY_WINDOW: int = 8
    CHAT_SESSION_TTL_SECONDS: int = 3600
    CHAT_ROLLING_SUMMARY_MAX_CHARS: int = 600
    CHAT_ENABLE_RAG: bool = True
    CHAT_ENABLE_LLM_LEXICAL_QUERY: bool = True
    CHAT_PREWARM_ON_STARTUP: bool = False
    CHAT_PREWARM_INCLUDE_MULTIMODAL: bool = True
    CHAT_PREWARM_INCLUDE_MULTIVIEW_WORKER: bool = False
    CHAT_USE_QUERY_EMBEDDINGS: bool = True
    CHAT_RETRIEVAL_EMBEDDING_ONLY: bool = False
    CHAT_RETRIEVAL_CANDIDATES: int = 15
    CHAT_RETRIEVAL_TOP_K: int = 5
    CHAT_RETRIEVAL_RESULTS_PAGE_SIZE: int = 10
    CHAT_INCLUDE_ORIGIN_HISTORY_IN_LLM_CONTEXT: bool = True
    CHAT_RELATED_ARTIFACTS_PAGE_SIZE: int = 10
    CHAT_RETRIEVAL_PAGINATION_WINDOW: int = 150
    CHAT_IN_TOUR_BOOST: float = 5
    # Legacy (sem efeito desde a Fase 3/E7A): as queries visuais deixaram de
    # transportar preferencia in_tour; a preferencia passa a ser pos-fusao
    # (MULTIMODAL_IN_TOUR_MARGIN, Etapa 10). Declarado para paridade de .env.
    IMAGE_IN_TOUR_BOOST: float = 5
    CHAT_IMAGE_RETRIEVAL_TOP_K: int = 6
    CHAT_IMAGE_ARTIFACT_TOP_K: int = 5
    CHAT_IMAGE_RETRIEVAL_PAGINATION_WINDOW: int = 150
    CHAT_IMAGE_DEFAULT_MESSAGE: str = "Analisa a imagem e identifica a peça mais provável no museu."
    CHAT_MODEL_DEFAULT_MESSAGE: str = "Analisa este modelo 3D e identifica a peça mais provável no museu."
    CHAT_MODEL_FIRST_PASS_VIEWS: int = 3
    CHAT_MODEL_TOTAL_VIEWS: int = 5
    CHAT_MODEL_LOW_CONFIDENCE_SCORE_THRESHOLD: float = 0.35
    CHAT_MODEL_CACHE_SIZE: int = 12

    # --- Fase 3: retrieval multimodal de produção (tudo atrás de flags) ---
    # off    -> comportamento atual, sem ramo visual nem alteração de ranking.
    # intent -> ramo texto→imagem só quando a intenção visual é identificada.
    # always -> ramo texto→imagem em todas as pesquisas textuais (avaliação).
    MULTIMODAL_RETRIEVAL_MODE: str = "off"
    MULTIMODAL_RRF_K: int = 60
    MULTIMODAL_ARTIFACT_WEIGHT: float = 1.0
    MULTIMODAL_IMAGE_WEIGHT: float = 0.7
    # Floor de score do ramo visual antes da fusão. Calibrado 2026-07-13 com
    # dados reais (12 positivos com alvo + 12 queries factuais como negativos):
    # positivos observados >= 0.6677; amostra visual >= 0.645; negativos top-1
    # mediana 0.682/max 0.712 — o espaço cross-modal é estreito e o score tem
    # fraco poder separador, por isso o floor corta apenas a cauda fraca
    # (< 0.64) e a qualidade é garantida pelo router + pesos RRF por rank.
    MULTIMODAL_MIN_IMAGE_SCORE: float = 0.64
    MULTIMODAL_IMAGE_TOP_K: int = 30
    MULTIMODAL_DEBUG: bool = False
    # Pesos da fusão imagem+texto (Etapa 8): a semelhança visual (i2i) é o
    # ramo principal; o texto refina via t2i (MULTIMODAL_IMAGE_WEIGHT) e/ou
    # pesquisa documental de artefactos com peso reduzido.
    MULTIMODAL_I2I_WEIGHT: float = 1.0
    MULTIMODAL_IMAGE_TEXT_ARTIFACT_WEIGHT: float = 0.5
    # E9: ramos múltiplos numa mensagem usam _msearch (erros por ramo isolados;
    # fallback automático para queries separadas em falha transport-level).
    MULTIMODAL_USE_MSEARCH: bool = True
    # E10: preferência in_tour APÓS a fusão — um resultado in_tour sobe no
    # máximo uma posição quando a diferença de fusion_score é <= margem.
    # 0.0 desliga a política. Nunca altera scores nem introduz candidatos.
    MULTIMODAL_IN_TOUR_MARGIN: float = 0.0
    # Ramo conjunto imagem+texto (imagem carregada + texto do utilizador num
    # ÚNICO embedding Qwen3-VL contra visual_embedding). Só corre quando a
    # mensagem tem imagem E texto não vazio; nunca substitui os ramos atuais.
    MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED: bool = False
    MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K: int = 30
    MULTIMODAL_JOINT_IMAGE_TEXT_WEIGHT: float = 1.0
    # Posição do rerank VL na rota de imagem:
    # local  -> só o pool i2i pré-fusão (comportamento histórico).
    # global -> pool fundido pós-RRF (todas as origens; recomendado).
    # both   -> ambos (ablação). off -> nenhum, mesmo com o flag VL ligado.
    # CHAT_ENABLE_VL_RERANKING=false continua a desligar tudo (master switch).
    MULTIMODAL_VL_RERANK_STAGE: str = "local"
    MULTIMODAL_GLOBAL_VL_RERANK_TOP_N: int = 16
    # Experiência das legendas (../patrimonio360_caption_indexer): nas perguntas
    # escritas com intenção visual, um ramo texto→legendas (kNN no campo
    # caption_embedding, o mesmo Qwen3-Embedding-8B do ramo de texto). Vazio
    # desliga. replace -> as legendas ocupam o lugar do ramo texto→imagem (a
    # variante A medida a 2026-09-30); add -> correm os dois. Se o ramo das
    # legendas falhar, fica o texto→imagem. As fotografias não mudam.
    CAPTION_TEXT_INDEX: str = ""
    CAPTION_TEXT_MODE: str = "replace"
    CAPTION_TEXT_WEIGHT: float = 0.7
    CAPTION_TEXT_TOP_K: int = 30
    # O índice das legendas tem todos os museus e o filtro de museu vem depois
    # do kNN, por isso o k é largo (como na avaliação: --k 1000).
    CAPTION_TEXT_KNN_K: int = 1000

    @field_validator("CAPTION_TEXT_MODE")
    @classmethod
    def _validate_caption_text_mode(cls, value: str) -> str:
        normalized = (value or "").strip().lower() or "replace"
        if normalized not in {"replace", "add"}:
            raise ValueError(f"CAPTION_TEXT_MODE must be one of ['add', 'replace']; got {value!r}")
        return normalized

    @field_validator("CHAT_ROUTER_MODE")
    @classmethod
    def _validate_router_mode(cls, value: str) -> str:
        normalized = (value or "").strip().lower() or "llm"
        if normalized not in {"llm", "deterministic"}:
            raise ValueError(
                f"CHAT_ROUTER_MODE must be one of ['deterministic', 'llm']; got {value!r}"
            )
        return normalized

    @field_validator("OPENSEARCH_USE_SSL", mode="before")
    @classmethod
    def _blank_use_ssl_is_auto(cls, value: object) -> object:
        # O .env.example documenta «vazio = automático», mas o pydantic rejeitava
        # a string vazia e um .env copiado do exemplo não arrancava.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("CHAT_GROUNDING_CHECK")
    @classmethod
    def _validate_grounding_check(cls, value: str) -> str:
        normalized = (value or "").strip().lower() or "off"
        if normalized not in {"off", "log", "repair"}:
            raise ValueError(
                f"CHAT_GROUNDING_CHECK must be one of ['log', 'off', 'repair']; got {value!r}"
            )
        return normalized

    @field_validator("MULTIMODAL_VL_RERANK_STAGE")
    @classmethod
    def _validate_multimodal_vl_rerank_stage(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        allowed = {"off", "local", "global", "both"}
        if normalized not in allowed:
            raise ValueError(
                f"MULTIMODAL_VL_RERANK_STAGE must be one of {sorted(allowed)}; got {value!r}"
            )
        return normalized

    @field_validator("MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K")
    @classmethod
    def _validate_multimodal_joint_top_k(cls, value: int) -> int:
        if not 1 <= value <= 500:
            raise ValueError(
                f"MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K must be within [1, 500]; got {value}"
            )
        return value

    @field_validator("MULTIMODAL_GLOBAL_VL_RERANK_TOP_N")
    @classmethod
    def _validate_multimodal_global_rerank_top_n(cls, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError(
                f"MULTIMODAL_GLOBAL_VL_RERANK_TOP_N must be within [1, 100]; got {value}"
            )
        return value

    @field_validator("MULTIMODAL_IN_TOUR_MARGIN")
    @classmethod
    def _validate_multimodal_in_tour_margin(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"MULTIMODAL_IN_TOUR_MARGIN must be within [0, 1]; got {value}"
            )
        return value

    @field_validator("MULTIMODAL_RETRIEVAL_MODE")
    @classmethod
    def _validate_multimodal_mode(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        allowed = {"off", "intent", "always"}
        if normalized not in allowed:
            raise ValueError(
                f"MULTIMODAL_RETRIEVAL_MODE must be one of {sorted(allowed)}; got {value!r}"
            )
        return normalized

    @field_validator("MULTIMODAL_RRF_K")
    @classmethod
    def _validate_multimodal_rrf_k(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"MULTIMODAL_RRF_K must be >= 1; got {value}")
        return value

    @field_validator(
        "MULTIMODAL_ARTIFACT_WEIGHT",
        "MULTIMODAL_IMAGE_WEIGHT",
        "MULTIMODAL_I2I_WEIGHT",
        "MULTIMODAL_IMAGE_TEXT_ARTIFACT_WEIGHT",
        "MULTIMODAL_JOINT_IMAGE_TEXT_WEIGHT",
    )
    @classmethod
    def _validate_multimodal_weights(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"multimodal fusion weights must be >= 0; got {value}")
        return value

    @field_validator("MULTIMODAL_MIN_IMAGE_SCORE")
    @classmethod
    def _validate_multimodal_min_image_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"MULTIMODAL_MIN_IMAGE_SCORE must be within [0, 1]; got {value}"
            )
        return value

    @field_validator("MULTIMODAL_IMAGE_TOP_K")
    @classmethod
    def _validate_multimodal_image_top_k(cls, value: int) -> int:
        if not 1 <= value <= 500:
            raise ValueError(
                f"MULTIMODAL_IMAGE_TOP_K must be within [1, 500]; got {value}"
            )
        return value

    # Vocabulário leigo -> catálogo (app/config/lay_vocabulary.json): frases do
    # catálogo como alternativa no ramo lexical. Desligado = query histórica.
    CHAT_LAY_VOCABULARY_EXPANSION: bool = False
    IMAGE_ASSET_ROOT: str | None = None
    # Miniaturas `?w=` em /chat/images (ver app/services/image_thumbnails.py).
    # Desligado: ignora `w` e serve sempre o original, sem Cache-Control.
    CHAT_IMAGE_THUMBNAILS: bool = False
    IMAGE_THUMBNAIL_CACHE_DIR: str | None = None
    CHAT_IMAGE_CACHE_MAX_AGE_SECONDS: int = 86400
    POI_TOURS_DIR: str | None = None
    # Lugares das visitas 3DVista (scripts/build_tour_locations.py); vazio = app/data/tour_locations.json
    TOUR_LOCATIONS_PATH: str | None = None
    MULTIVIEW_WORKER_HOST: str = "127.0.0.1"
    MULTIVIEW_WORKER_PORT: int = 3101
    MULTIVIEW_WORKER_START_TIMEOUT_SECONDS: float = 30.0
    # Render HTTP timeout. Kept at the historical 60s floor by default so a
    # broken/hung render fails FAST and the worker recovers, rather than making
    # the user wait. Operators can raise it once GPU-accelerated rendering (or a
    # known-good Chromium) makes genuinely-slow renders viable.
    MULTIVIEW_RENDER_TIMEOUT_SECONDS: float = 60.0
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

    # Structured backend_query JSONL logging for offline evaluation/paper analysis.
    QUERY_LOG_ENABLED: bool = True
    QUERY_LOG_PATH: str = "logs/evaluation/backend_queries.jsonl"
    FRONTEND_EVENT_LOG_PATH: str = "logs/evaluation/frontend_events.jsonl"

    # Reranker (second-stage; disabled by default). These keys existed in .env but
    # were silently dropped by extra="ignore" until declared here — declaring them
    # is what makes PairwiseRerankerService constructible at all.
    CHAT_ENABLE_RERANKING: bool = False
    RERANKER_MODEL_ID: str = "Qwen/Qwen3-Reranker-4B"
    RERANKER_MODEL_REVISION: str | None = None
    RERANKER_MAX_LENGTH: int = 1024
    RERANKER_BATCH_SIZE: int = 10
    RERANKER_INSTRUCTION: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    RERANKER_PREFER_BF16: bool = True
    # Candidate pool reranked on the first page of text retrieval. When reranking
    # is enabled the OpenSearch request is widened to this size and truncated back
    # to the original page size after reordering.
    RERANKER_CANDIDATES: int = 24

    # Visual reranker (second-stage for image retrieval; disabled by default).
    # Scores (query image/text, candidate image+caption) pairs with Qwen3-VL-Reranker.
    CHAT_ENABLE_VL_RERANKING: bool = False
    VL_RERANKER_MODEL_ID: str = "Qwen/Qwen3-VL-Reranker-8B"
    VL_RERANKER_MODEL_REVISION: str | None = None
    VL_RERANKER_INSTRUCTION: str = (
        "Given a user query (image and/or text), retrieve museum artifact images "
        "that match the queried object"
    )
    VL_RERANKER_BATCH_SIZE: int = 4
    VL_RERANKER_CANDIDATES: int = 16
    # Candidate images are downscaled to this max side (px) before scoring.
    VL_RERANKER_MAX_IMAGE_SIDE: int = 1024

    # Declared for .env parity (currently without consumers in app code).
    CHAT_ENABLE_STRUCTURED_QUERY_PLANNING: bool = False
    CHAT_ANALYTICS_PLANNER_MIN_CONFIDENCE: float = 0.55
    CHAT_ANALYTICS_LIST_TOP_K: int = 8
    LOG_JSON: bool = False
    LOG_JSON_PRETTY: bool = False
    LOG_JSON_INDENT: int = 2
    LOG_LEVEL: str = "INFO"
    LOG_CHAT_MESSAGES: bool = False
    LOG_CHAT_STATE_HISTORY: bool = False
    DEBUG_EMBEDDINGS: bool = False

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
    def llm_fallback_openai_base_url_resolved(self) -> str | None:
        base = (self.LLM_FALLBACK_BASE_URL or "").strip().rstrip("/")
        if not base:
            return None
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")
        return base

    @property
    def llm_model_resolved(self) -> str:
        if self.LLM_MODEL and self.LLM_MODEL.strip():
            return self.LLM_MODEL.strip()
        if self.LLM_MODEL_TEXT and self.LLM_MODEL_TEXT.strip():
            return self.LLM_MODEL_TEXT.strip()
        if self.LLM_MODEL_JSON and self.LLM_MODEL_JSON.strip():
            return self.LLM_MODEL_JSON.strip()
        return "amalia-llm/AMALIA-9B-0626-DPO"

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
        raw = (self.RERANKER_MODEL_ID or "").strip()
        return raw or "Qwen/Qwen3-Reranker-4B"

    @property
    def reranker_model_revision_resolved(self) -> str | None:
        raw = (self.RERANKER_MODEL_REVISION or "").strip()
        return raw or None

    @property
    def vl_reranker_model_resolved(self) -> str:
        raw = (self.VL_RERANKER_MODEL_ID or "").strip()
        return raw or "Qwen/Qwen3-VL-Reranker-8B"

    @property
    def vl_reranker_model_revision_resolved(self) -> str | None:
        raw = (self.VL_RERANKER_MODEL_REVISION or "").strip()
        return raw or None

    @property
    def openrouter_base_url_resolved(self) -> str:
        return self.OPENROUTER_BASE_URL.strip().rstrip("/")

    @property
    def openrouter_bge_model_resolved(self) -> str:
        raw = self.OPENROUTER_BGE_MODEL.strip()
        if raw:
            return raw
        return "baai/bge-m3"

    @property
    def image_asset_root_resolved(self) -> Path | None:
        raw = (self.IMAGE_ASSET_ROOT or "").strip()
        if not raw:
            default_root = (Path(__file__).resolve().parents[2] / "Images").resolve()
            if default_root.exists():
                return default_root
            return None
        return Path(raw).expanduser().resolve()

    @property
    def image_thumbnail_cache_dir_resolved(self) -> Path:
        raw = (self.IMAGE_THUMBNAIL_CACHE_DIR or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path(__file__).resolve().parents[2] / "tmp" / "image_thumbs").resolve()

    @property
    def tour_locations_path_resolved(self) -> Path:
        raw = (self.TOUR_LOCATIONS_PATH or "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
        return (Path(__file__).resolve().parents[1] / "data" / "tour_locations.json").resolve()

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

    @property
    def query_log_path_resolved(self) -> Path:
        raw = (self.QUERY_LOG_PATH or "").strip()
        if not raw:
            raw = "logs/evaluation/backend_queries.jsonl"
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path

    @property
    def frontend_event_log_path_resolved(self) -> Path:
        raw = (self.FRONTEND_EVENT_LOG_PATH or "").strip()
        if not raw:
            raw = "logs/evaluation/frontend_events.jsonl"
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from app.services.retrieval_profile import apply_profile
    return apply_profile(Settings())
