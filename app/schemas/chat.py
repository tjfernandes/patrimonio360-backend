from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

ResponseFormatType = Literal["json_object", "text"]
LanguageCode = Literal["pt", "en"]


class ResponseFormatObject(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: ResponseFormatType


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    museum_slug: str = Field(
        ...,
        min_length=1,
        description="Museum slug",
    )
    museum_id: str | None = Field(
        default=None,
        min_length=1,
        description="Museum id used for strict retrieval filters.",
    )
    museum_name: str | None = Field(
        default=None,
        min_length=1,
        description="Museum display name for LLM context grounding.",
    )
    language: LanguageCode | None = Field(
        default=None,
        description="Preferred language for status messages and final assistant reply ('pt' or 'en').",
    )
    message: str = Field(..., min_length=1, description="User message text")
    conversation_id: str | None = Field(
        default=None,
        description="Conversation id from the frontend; if absent, backend generates one.",
    )
    response_format: ResponseFormatObject = Field(
        default_factory=lambda: ResponseFormatObject(type="text"),
        description="Desired LLM output format.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt for the model.",
    )
    model_override: str | None = Field(
        default=None,
        description="Optional model override.",
    )
    results_page: int = Field(
        default=1,
        ge=1,
        description="Requested results page number for retrieval cards.",
    )
    results_page_size: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Requested page size for retrieval cards.",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Optional payload metadata")


class ChatRegenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    museum_slug: str = Field(
        ...,
        min_length=1,
        description="Museum slug",
    )
    museum_id: str | None = Field(
        default=None,
        min_length=1,
        description="Museum id used for strict retrieval filters.",
    )
    museum_name: str | None = Field(
        default=None,
        min_length=1,
        description="Museum display name for LLM context grounding.",
    )
    language: LanguageCode | None = Field(
        default=None,
        description="Preferred language for status messages and final assistant reply ('pt' or 'en').",
    )
    conversation_id: str = Field(
        ...,
        min_length=1,
        description="Existing conversation id to regenerate the latest assistant reply.",
    )
    response_format: ResponseFormatObject = Field(
        default_factory=lambda: ResponseFormatObject(type="text"),
        description="Desired LLM output format.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt for the model.",
    )
    model_override: str | None = Field(
        default=None,
        description="Optional model override.",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Optional payload metadata")


class ChatImageMessageRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    museum_slug: str = Field(
        ...,
        min_length=1,
        description="Museum slug",
    )
    museum_id: str | None = Field(
        default=None,
        min_length=1,
        description="Museum id used for strict retrieval filters.",
    )
    museum_name: str | None = Field(
        default=None,
        min_length=1,
        description="Museum display name for LLM context grounding.",
    )
    language: LanguageCode | None = Field(
        default=None,
        description="Preferred language for status messages and final assistant reply ('pt' or 'en').",
    )
    message: str | None = Field(
        default=None,
        description="Optional user message text for image search.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation id from the frontend; if absent, backend generates one.",
    )
    response_format: ResponseFormatObject = Field(
        default_factory=lambda: ResponseFormatObject(type="text"),
        description="Desired LLM output format.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt for the model.",
    )
    model_override: str | None = Field(
        default=None,
        description="Optional model override.",
    )
    results_page: int = Field(
        default=1,
        ge=1,
        description="Requested results page number for retrieval cards.",
    )
    results_page_size: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Requested page size for retrieval cards.",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Optional payload metadata")


class ChatModelMessageRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    museum_slug: str = Field(
        ...,
        min_length=1,
        description="Museum slug",
    )
    museum_id: str | None = Field(
        default=None,
        min_length=1,
        description="Museum id used for strict retrieval filters.",
    )
    museum_name: str | None = Field(
        default=None,
        min_length=1,
        description="Museum display name for LLM context grounding.",
    )
    language: LanguageCode | None = Field(
        default=None,
        description="Preferred language for status messages and final assistant reply ('pt' or 'en').",
    )
    message: str | None = Field(
        default=None,
        description="Optional user message text for 3D model search.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Conversation id from the frontend; if absent, backend generates one.",
    )
    response_format: ResponseFormatObject = Field(
        default_factory=lambda: ResponseFormatObject(type="text"),
        description="Desired LLM output format.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt for the model.",
    )
    model_override: str | None = Field(
        default=None,
        description="Optional model override.",
    )
    results_page: int = Field(
        default=1,
        ge=1,
        description="Requested results page number for retrieval cards.",
    )
    results_page_size: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Requested page size for retrieval cards.",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Optional payload metadata")


class ChatResultsPageRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    museum_slug: str = Field(..., min_length=1, description="Museum slug.")
    museum_id: str | None = Field(
        default=None,
        min_length=1,
        description="Museum id used for strict retrieval filters.",
    )
    language: LanguageCode | None = Field(
        default=None,
        description="Preferred language for user-facing errors ('pt' or 'en').",
    )
    conversation_id: str = Field(..., min_length=1, description="Existing conversation id.")
    results_page: int = Field(default=1, ge=1, description="Requested page number.")
    results_page_size: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Requested page size.",
    )
    results_request_id: str | None = Field(
        default=None,
        min_length=1,
        description="Identifier of the retrieval request being paged.",
    )


class ImageMatchResult(BaseModel):
    original_image_name: str
    artifact_id: str | None = None
    score: float | None = None
    title: str | None = None
    inventory: str | None = None
    image_id: str | None = None
    local_path: str | None = None
    source_url: str | None = None
    artifact: dict[str, Any] | None = None
    navigation_target: dict[str, Any] | None = None


class ArtifactImageResult(BaseModel):
    original_image_name: str | None = None
    image_id: str | None = None
    image_order: int | None = None
    local_path: str | None = None
    source_url: str | None = None
    caption: str | None = None
    alt_text: str | None = None


class ArtifactResult(BaseModel):
    artifact_id: str
    tipo_inventario: str | None = None
    inventory_number: str | None = None
    title: str | None = None
    museum_id: str | None = None
    museum: str | None = None
    category: str | None = None
    super_category: str | None = None
    # Autores: novo array + string legada (primeiro nome, retrocompatibilidade).
    creator: str | None = None
    creators: list[str] = Field(default_factory=list)
    creator_ids: list[str] = Field(default_factory=list)
    date_or_period: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    support_or_material: str | None = None
    technique: str | None = None
    origin_history: str | None = None
    incorporation: str | None = None
    production_center: str | None = None
    description: str | None = None
    search_text: str | None = None
    detail_type: str | None = None
    detail_url: str | None = None
    in_tour: bool = False
    # Relacoes (export relacional do RAIZ).
    sets: list[str] = Field(default_factory=list)
    set_ids: list[str] = Field(default_factory=list)
    set_numbers: list[str] = Field(default_factory=list)
    exhibitions: list[str] = Field(default_factory=list)
    exhibition_ids: list[str] = Field(default_factory=list)
    exhibition_types: list[str] = Field(default_factory=list)
    exhibition_count: int | None = None
    bibliography: str | None = None
    bibliography_count: int | None = None
    image_count: int | None = None
    images: list[ArtifactImageResult] = Field(default_factory=list)


# ----- Schemas de entidades para o modal de detalhe ----- #


class AuthorEntity(BaseModel):
    entity_id: str
    name: str | None = None
    atividade: str | None = None
    data_nascimento: str | None = None
    data_obito: str | None = None
    local_nascimento: str | None = None
    local_obito: str | None = None
    biografia: str | None = None
    biography: str | None = None
    url: str | None = None
    n_objetos: int | None = None


class RelatedArtifactItem(BaseModel):
    """Versao leve de um artefacto para listas relacionadas (modal)."""
    artifact_id: str
    inventory_number: str | None = None
    title: str | None = None
    museum_id: str | None = None
    museum: str | None = None
    category: str | None = None
    creators: list[str] = Field(default_factory=list)
    date_or_period: str | None = None
    detail_type: str | None = None
    detail_url: str | None = None
    in_tour: bool = False
    image_count: int | None = None
    images: list[ArtifactImageResult] = Field(default_factory=list)
    navigation_target: "TourNavigationTarget | None" = None


class SetEntityWithArtifacts(BaseModel):
    entity_id: str
    name: str | None = None
    num_conjunto: str | None = None
    historial: str | None = None
    descricao: str | None = None
    url: str | None = None
    n_objetos: int | None = None
    artifacts: list[RelatedArtifactItem] = Field(default_factory=list)
    artifacts_returned: int = 0


class ExhibitionEntityWithArtifacts(BaseModel):
    entity_id: str
    name: str | None = None
    tipo_exposicao: str | None = None
    local: str | None = None
    ano_inicial: int | None = None
    ano_final: int | None = None
    texto: str | None = None
    ficha_tecnica: str | None = None
    url: str | None = None
    n_objetos: int | None = None
    artifacts: list[RelatedArtifactItem] = Field(default_factory=list)
    artifacts_returned: int = 0


class RelatedArtifactsPageResponse(BaseModel):
    status: Literal["ok"] = "ok"
    artifact_id: str
    tipo: Literal["conjunto", "exposicao"]
    entity_id: str
    artifacts: list[RelatedArtifactItem] = Field(default_factory=list)
    artifacts_offset: int = 0
    artifacts_limit: int = 10
    artifacts_total: int = 0
    artifacts_has_more: bool = False


class TourNavigationTarget(BaseModel):
    overlay_id: str
    panorama_key: str
    inventory_id: str
    location: str | None = None
    title: str | None = None


class ChatSearchScope(BaseModel):
    museum_id: str | None = None
    museum_slug: str
    museum_name: str | None = None
    is_cross_museum: bool = False


class ChatMessageResponse(BaseModel):
    status: Literal["ok"] = "ok"
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str | None = None
    response_format: ResponseFormatObject = Field(
        default_factory=lambda: ResponseFormatObject(type="text")
    )
    reply: str
    reply_json: dict[str, Any] | list[Any] | None = None
    model_hint: str | None = None
    image_matches: list[ImageMatchResult] = Field(default_factory=list)
    artifact_results: list[ArtifactResult] = Field(default_factory=list)
    navigation_targets: list[TourNavigationTarget] = Field(default_factory=list)
    results_page: int = 1
    results_page_size: int = 0
    results_total: int = 0
    results_has_more: bool = False
    results_request_id: str | None = None
    search_scope: ChatSearchScope | None = None


class ChatResultsPageResponse(BaseModel):
    status: Literal["ok"] = "ok"
    conversation_id: str
    reply: str | None = None
    image_matches: list[ImageMatchResult] = Field(default_factory=list)
    artifact_results: list[ArtifactResult] = Field(default_factory=list)
    navigation_targets: list[TourNavigationTarget] = Field(default_factory=list)
    results_page: int = 1
    results_page_size: int = 0
    results_total: int = 0
    results_has_more: bool = False
    results_request_id: str | None = None
    search_scope: ChatSearchScope | None = None


class ArtifactDetailContextResponse(BaseModel):
    """Resposta do endpoint /artifacts/{id}/detail-context para o modal."""
    status: Literal["ok"] = "ok"
    artifact_id: str
    authors: list[AuthorEntity] = Field(default_factory=list)
    sets: list[SetEntityWithArtifacts] = Field(default_factory=list)
    exhibitions: list[ExhibitionEntityWithArtifacts] = Field(default_factory=list)


class ChatHealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    backend_mode: Literal["llm_dev"] = "llm_dev"
    llm_provider: str
    llm_base_url: str
    llm_text_model: str
    llm_json_model: str
    text_embedding_model: str
    multimodal_embedding_model: str



# Resolve forward references (RelatedArtifactItem -> TourNavigationTarget).
RelatedArtifactItem.model_rebuild()
SetEntityWithArtifacts.model_rebuild()
ExhibitionEntityWithArtifacts.model_rebuild()
ArtifactDetailContextResponse.model_rebuild()
