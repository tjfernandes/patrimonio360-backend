import json
import logging
from pathlib import Path
import asyncio
import contextlib
from functools import lru_cache
import os
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import get_settings
from app.core.logging import log_event
from app.schemas.chat import (
    ArtifactDetailContextResponse,
    ArtifactResult,
    ChatHealthResponse,
    ChatImageMessageRequest,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatModelMessageRequest,
    ChatRegenerateRequest,
    ChatResultsPageRequest,
    ChatResultsPageResponse,
    RelatedArtifactsPageResponse,
    ResponseFormatObject,
)
from app.services.chat_service import ChatService, get_chat_service
from app.services.chat_i18n import translate

router = APIRouter()
SUPPORTED_MODEL_EXTENSIONS = {".glb", ".gltf", ".obj"}
logger = logging.getLogger(__name__)


@lru_cache(maxsize=4096)
def _find_legacy_image_by_basename(root_path: str, file_name: str) -> Path | None:
    safe_name = Path(file_name).name
    if not safe_name or safe_name != file_name:
        return None

    root = Path(root_path).resolve()
    for dir_path, _dir_names, file_names in os.walk(root):
        if safe_name not in file_names:
            continue
        candidate = (Path(dir_path) / safe_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _to_sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_streaming_response(
    run: Callable[[Callable[[dict[str, Any]], Awaitable[None]]], Awaitable[ChatMessageResponse]]
) -> StreamingResponse:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def status_cb(event: dict[str, Any]) -> None:
        await queue.put(event)

    async def runner() -> None:
        try:
            result = await run(status_cb)
            await queue.put({"type": "result", "payload": result.model_dump(mode="json")})
        except Exception as exc:
            log_event(logger, logging.ERROR, "chat.stream.error", error=exc)
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put({"type": "done"})

    async def event_stream() -> AsyncIterator[str]:
        task = asyncio.create_task(runner())
        try:
            while True:
                event = await queue.get()
                event_type = str(event.get("type", "status"))
                yield _to_sse(event_type, event)
                if event_type == "done":
                    break
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(Exception):
                await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _parse_metadata_field(
    metadata: str | None,
    *,
    language: str | None = None,
) -> dict[str, object] | None:
    if not metadata:
        return None
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=translate("error.metadata_invalid", language, error=str(exc)),
        ) from exc
    if isinstance(parsed, dict):
        return parsed
    return None


def _validate_response_format(value: str, *, language: str | None = None) -> None:
    if value not in {"text", "json_object"}:
        raise HTTPException(
            status_code=400,
            detail=translate("error.response_format_invalid", language),
        )


def _validate_model_filename(filename: str | None, *, language: str | None = None) -> str:
    file_name = (filename or "").strip()
    suffix = Path(file_name).suffix.lower()
    if not file_name or suffix not in SUPPORTED_MODEL_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_MODEL_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=translate("error.model_format_invalid", language, allowed=allowed),
        )
    return file_name


@router.get("/health", response_model=ChatHealthResponse)
async def chat_health(service: ChatService = Depends(get_chat_service)) -> ChatHealthResponse:
    return service.health()


@router.post("/messages", response_model=ChatMessageResponse)
async def post_chat_message(
    payload: ChatMessageRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatMessageResponse:
    return await service.handle_message(payload)


@router.post("/messages/results", response_model=ChatResultsPageResponse)
async def post_chat_results_page(
    payload: ChatResultsPageRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResultsPageResponse:
    return await service.get_results_page(payload)


@router.get("/artifacts/{artifact_id:path}/detail-context", response_model=ArtifactDetailContextResponse)
async def get_artifact_detail_context(
    artifact_id: str,
    museum_slug: str,
    museum_id: str | None = None,
    service: ChatService = Depends(get_chat_service),
) -> ArtifactDetailContextResponse:
    """Contexto expandido de um artefacto: autores, conjuntos e exposicoes,
    cada um com os outros artefactos relacionados (e navigation_target se
    existir na tour). Lazy-loaded pelo modal do frontend."""
    cleaned_id = (artifact_id or "").strip()
    if not cleaned_id:
        raise HTTPException(status_code=400, detail="artifact_id obrigatorio.")
    cleaned_slug = (museum_slug or "").strip()
    if not cleaned_slug:
        raise HTTPException(status_code=400, detail="museum_slug obrigatorio.")
    return await service.get_artifact_detail_context(
        museum_slug=cleaned_slug,
        museum_id=(museum_id or "").strip() or None,
        artifact_id=cleaned_id,
    )


@router.get("/artifacts/related", response_model=RelatedArtifactsPageResponse)
async def get_related_artifacts_page(
    artifact_id: str,
    tipo: str,
    entity_id: str,
    museum_slug: str,
    museum_id: str | None = None,
    offset: int = 0,
    limit: int = 10,
    service: ChatService = Depends(get_chat_service),
) -> RelatedArtifactsPageResponse:
    cleaned_slug = (museum_slug or "").strip()
    if not cleaned_slug:
        raise HTTPException(status_code=400, detail="museum_slug obrigatorio.")
    cleaned_artifact_id = (artifact_id or "").strip()
    cleaned_entity_id = (entity_id or "").strip()
    if not cleaned_artifact_id:
        raise HTTPException(status_code=400, detail="artifact_id obrigatorio.")
    if not cleaned_entity_id:
        raise HTTPException(status_code=400, detail="entity_id obrigatorio.")
    try:
        return await service.get_related_artifacts_page(
            museum_slug=cleaned_slug,
            museum_id=(museum_id or "").strip() or None,
            artifact_id=cleaned_artifact_id,
            tipo=(tipo or "").strip(),
            entity_id=cleaned_entity_id,
            offset=max(int(offset), 0),
            limit=max(1, min(int(limit), 50)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/artifacts/by-inventory/full", response_model=ArtifactResult)
async def get_artifact_full_by_inventory(
    inventory_number: str,
    museum_slug: str,
    museum_id: str | None = None,
    service: ChatService = Depends(get_chat_service),
) -> ArtifactResult:
    """Devolve um ArtifactResult completo por numero de inventario.
    Usado para converter eventos da tour em referencias de chat."""
    cleaned_inventory = (inventory_number or "").strip()
    if not cleaned_inventory:
        raise HTTPException(status_code=400, detail="inventory_number obrigatorio.")
    cleaned_slug = (museum_slug or "").strip()
    if not cleaned_slug:
        raise HTTPException(status_code=400, detail="museum_slug obrigatorio.")
    result = await service.get_artifact_full_by_inventory(
        museum_slug=cleaned_slug,
        museum_id=(museum_id or "").strip() or None,
        inventory_number=cleaned_inventory,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Artefacto nao encontrado.")
    return result


@router.get("/artifacts/{artifact_id:path}/full", response_model=ArtifactResult)
async def get_artifact_full(
    artifact_id: str,
    museum_slug: str,
    museum_id: str | None = None,
    service: ChatService = Depends(get_chat_service),
) -> ArtifactResult:
    """Devolve um ArtifactResult completo (com imagens) por artifact_id.
    Usado pelo modal quando se clica num artefacto relacionado."""
    cleaned_id = (artifact_id or "").strip()
    if not cleaned_id:
        raise HTTPException(status_code=400, detail="artifact_id obrigatorio.")
    cleaned_slug = (museum_slug or "").strip()
    if not cleaned_slug:
        raise HTTPException(status_code=400, detail="museum_slug obrigatorio.")
    result = await service.get_artifact_full(
        museum_slug=cleaned_slug,
        museum_id=(museum_id or "").strip() or None,
        artifact_id=cleaned_id,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Artefacto nao encontrado.")
    return result


@router.post("/messages/stream")
async def post_chat_message_stream(
    payload: ChatMessageRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    return _build_streaming_response(lambda status_cb: service.handle_message(payload, status_cb=status_cb))


@router.post("/messages/regenerate", response_model=ChatMessageResponse)
async def post_chat_regenerate_message(
    payload: ChatRegenerateRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatMessageResponse:
    try:
        return await service.regenerate_last_reply(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/messages/regenerate/stream")
async def post_chat_regenerate_message_stream(
    payload: ChatRegenerateRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    return _build_streaming_response(
        lambda status_cb: service.regenerate_last_reply(payload, status_cb=status_cb)
    )


@router.post("/messages/image", response_model=ChatMessageResponse)
async def post_chat_image_message(
    museum_slug: str = Form(...),
    museum_id: str | None = Form(default=None),
    museum_name: str | None = Form(default=None),
    language: str | None = Form(default=None),
    message: str | None = Form(default=None),
    conversation_id: str | None = Form(default=None),
    response_format: str = Form(default="text"),
    system_prompt: str | None = Form(default=None),
    model_override: str | None = Form(default=None),
    results_page: int = Form(default=1),
    results_page_size: int | None = Form(default=None),
    metadata: str | None = Form(default=None),
    image: UploadFile = File(...),
    service: ChatService = Depends(get_chat_service),
) -> ChatMessageResponse:
    content = await image.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail=translate("error.empty_image_upload", language),
        )
    _validate_response_format(response_format, language=language)
    metadata_payload = _parse_metadata_field(metadata, language=language)

    payload = ChatImageMessageRequest(
        museum_slug=museum_slug,
        museum_id=museum_id,
        museum_name=museum_name,
        language=language,
        message=message,
        conversation_id=conversation_id,
        response_format=ResponseFormatObject(type=response_format),
        system_prompt=system_prompt,
        model_override=model_override,
        results_page=results_page,
        results_page_size=results_page_size,
        metadata=metadata_payload,
    )
    return await service.handle_image_message(
        payload,
        image_bytes=content,
        image_filename=image.filename,
        image_content_type=image.content_type,
    )


@router.post("/messages/image/stream")
async def post_chat_image_message_stream(
    museum_slug: str = Form(...),
    museum_id: str | None = Form(default=None),
    museum_name: str | None = Form(default=None),
    language: str | None = Form(default=None),
    message: str | None = Form(default=None),
    conversation_id: str | None = Form(default=None),
    response_format: str = Form(default="text"),
    system_prompt: str | None = Form(default=None),
    model_override: str | None = Form(default=None),
    results_page: int = Form(default=1),
    results_page_size: int | None = Form(default=None),
    metadata: str | None = Form(default=None),
    image: UploadFile = File(...),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    content = await image.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail=translate("error.empty_image_upload", language),
        )
    _validate_response_format(response_format, language=language)
    metadata_payload = _parse_metadata_field(metadata, language=language)

    payload = ChatImageMessageRequest(
        museum_slug=museum_slug,
        museum_id=museum_id,
        museum_name=museum_name,
        language=language,
        message=message,
        conversation_id=conversation_id,
        response_format=ResponseFormatObject(type=response_format),
        system_prompt=system_prompt,
        model_override=model_override,
        results_page=results_page,
        results_page_size=results_page_size,
        metadata=metadata_payload,
    )

    return _build_streaming_response(
        lambda status_cb: service.handle_image_message(
            payload,
            image_bytes=content,
            image_filename=image.filename,
            image_content_type=image.content_type,
            status_cb=status_cb,
        )
    )


@router.post("/messages/model", response_model=ChatMessageResponse)
async def post_chat_model_message(
    museum_slug: str = Form(...),
    museum_id: str | None = Form(default=None),
    museum_name: str | None = Form(default=None),
    language: str | None = Form(default=None),
    message: str | None = Form(default=None),
    conversation_id: str | None = Form(default=None),
    response_format: str = Form(default="text"),
    system_prompt: str | None = Form(default=None),
    model_override: str | None = Form(default=None),
    results_page: int = Form(default=1),
    results_page_size: int | None = Form(default=None),
    metadata: str | None = Form(default=None),
    model_file: UploadFile = File(...),
    service: ChatService = Depends(get_chat_service),
) -> ChatMessageResponse:
    content = await model_file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail=translate("error.empty_model_upload", language),
        )
    _validate_response_format(response_format, language=language)
    validated_file_name = _validate_model_filename(model_file.filename, language=language)
    metadata_payload = _parse_metadata_field(metadata, language=language)

    payload = ChatModelMessageRequest(
        museum_slug=museum_slug,
        museum_id=museum_id,
        museum_name=museum_name,
        language=language,
        message=message,
        conversation_id=conversation_id,
        response_format=ResponseFormatObject(type=response_format),
        system_prompt=system_prompt,
        model_override=model_override,
        results_page=results_page,
        results_page_size=results_page_size,
        metadata=metadata_payload,
    )
    return await service.handle_model_message(
        payload,
        model_bytes=content,
        model_filename=validated_file_name,
        model_content_type=model_file.content_type,
    )


@router.post("/messages/model/stream")
async def post_chat_model_message_stream(
    museum_slug: str = Form(...),
    museum_id: str | None = Form(default=None),
    museum_name: str | None = Form(default=None),
    language: str | None = Form(default=None),
    message: str | None = Form(default=None),
    conversation_id: str | None = Form(default=None),
    response_format: str = Form(default="text"),
    system_prompt: str | None = Form(default=None),
    model_override: str | None = Form(default=None),
    results_page: int = Form(default=1),
    results_page_size: int | None = Form(default=None),
    metadata: str | None = Form(default=None),
    model_file: UploadFile = File(...),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    content = await model_file.read()
    if not content:
        raise HTTPException(
            status_code=400,
            detail=translate("error.empty_model_upload", language),
        )
    _validate_response_format(response_format, language=language)
    validated_file_name = _validate_model_filename(model_file.filename, language=language)
    metadata_payload = _parse_metadata_field(metadata, language=language)

    payload = ChatModelMessageRequest(
        museum_slug=museum_slug,
        museum_id=museum_id,
        museum_name=museum_name,
        language=language,
        message=message,
        conversation_id=conversation_id,
        response_format=ResponseFormatObject(type=response_format),
        system_prompt=system_prompt,
        model_override=model_override,
        results_page=results_page,
        results_page_size=results_page_size,
        metadata=metadata_payload,
    )
    return _build_streaming_response(
        lambda status_cb: service.handle_model_message(
            payload,
            model_bytes=content,
            model_filename=validated_file_name,
            model_content_type=model_file.content_type,
            status_cb=status_cb,
        )
    )


@router.get("/images/{image_ref:path}")
async def get_chat_image_asset(image_ref: str) -> FileResponse:
    settings = get_settings()
    root = settings.image_asset_root_resolved
    if root is None:
        raise HTTPException(status_code=404, detail="IMAGE_ASSET_ROOT nao configurado.")

    raw_ref = (image_ref or "").strip()
    if not raw_ref:
        raise HTTPException(status_code=400, detail="image_ref invalido.")

    normalized_ref = raw_ref.replace("\\", "/").strip("/")
    candidate = Path(normalized_ref)
    parts = [part for part in candidate.parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise HTTPException(status_code=400, detail="image_ref invalido.")
    if parts and parts[0].lower() == root.name.lower():
        parts = parts[1:]
    if not parts:
        raise HTTPException(status_code=400, detail="image_ref invalido.")

    target = (root / Path(*parts)).resolve()
    try:
        target.relative_to(root)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="image_ref invalido.") from exc

    if not target.exists() or not target.is_file():
        legacy_target = None
        if len(parts) == 1:
            legacy_target = _find_legacy_image_by_basename(str(root), parts[0])
        if legacy_target is None:
            log_event(logger, logging.WARNING, "chat.image_asset.not_found", image_ref=raw_ref)
            raise HTTPException(status_code=404, detail="Imagem nao encontrada.")
        target = legacy_target

    media_type = None
    suffix = target.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif suffix == ".png":
        media_type = "image/png"
    elif suffix == ".webp":
        media_type = "image/webp"

    log_event(logger, logging.INFO, "chat.image_asset.serve", image_ref=raw_ref, path=target)
    return FileResponse(path=Path(target), media_type=media_type)
