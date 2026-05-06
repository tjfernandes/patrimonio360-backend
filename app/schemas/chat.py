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
        description="Museum id used for strict retrieval filter (in this project it matches museum_slug).",
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
        description="Museum id used for strict retrieval filter (in this project it matches museum_slug).",
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
        description="Museum id used for strict retrieval filter (in this project it matches museum_slug).",
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
        description="Museum id used for strict retrieval filter (in this project it matches museum_slug).",
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
    metadata: dict[str, Any] | None = Field(default=None, description="Optional payload metadata")


class ImageMatchResult(BaseModel):
    original_image_name: str
    score: float | None = None
    title: str | None = None
    inventory: str | None = None


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
    navigation_targets: list[TourNavigationTarget] = Field(default_factory=list)


class ChatHealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    backend_mode: Literal["llm_dev"] = "llm_dev"
    llm_provider: str
    llm_base_url: str
    llm_text_model: str
    llm_json_model: str
    text_embedding_model: str
    multimodal_embedding_model: str
    reranking_enabled: bool
    reranker_model: str
