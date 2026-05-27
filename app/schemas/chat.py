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
    local_path: str | None = None
    source_url: str | None = None
    caption: str | None = None
    alt_text: str | None = None


class ArtifactResult(BaseModel):
    artifact_id: str
    inventory_number: str | None = None
    title: str | None = None
    museum_id: str | None = None
    museum: str | None = None
    category: str | None = None
    super_category: str | None = None
    creator: str | None = None
    date_or_period: str | None = None
    support_or_material: str | None = None
    technique: str | None = None
    origin_history: str | None = None
    incorporation: str | None = None
    production_center: str | None = None
    description: str | None = None
    search_text: str | None = None
    detail_type: str | None = None
    detail_url: str | None = None
    image_count: int | None = None
    images: list[ArtifactImageResult] = Field(default_factory=list)


class TourNavigationTarget(BaseModel):
    overlay_id: str
    panorama_key: str
    inventory_id: str
    location: str | None = None
    title: str | None = None


class ChatMessageResponse(BaseModel):
    status: Literal["ok"] = "ok"
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
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


class ChatResultsPageResponse(BaseModel):
    status: Literal["ok"] = "ok"
    conversation_id: str
    image_matches: list[ImageMatchResult] = Field(default_factory=list)
    artifact_results: list[ArtifactResult] = Field(default_factory=list)
    navigation_targets: list[TourNavigationTarget] = Field(default_factory=list)
    results_page: int = 1
    results_page_size: int = 0
    results_total: int = 0
    results_has_more: bool = False


class ChatHealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    backend_mode: Literal["llm_dev"] = "llm_dev"
    llm_provider: str
    llm_base_url: str
    llm_text_model: str
    llm_json_model: str
    text_embedding_model: str
    multimodal_embedding_model: str
