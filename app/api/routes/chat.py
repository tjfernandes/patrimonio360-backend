import json
from pathlib import Path
import asyncio
import contextlib
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.core.config import get_settings
from app.schemas.chat import (
    ChatHealthResponse,
    ChatImageMessageRequest,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatModelMessageRequest,
    ChatRegenerateRequest,
    ResponseFormatObject,
)
from app.services.chat_service import ChatService, get_chat_service

router = APIRouter()
SUPPORTED_MODEL_EXTENSIONS = {".glb", ".gltf", ".obj"}


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


def _parse_metadata_field(metadata: str | None) -> dict[str, object] | None:
    if not metadata:
        return None
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata invalido: {exc}") from exc
    if isinstance(parsed, dict):
        return parsed
    return None


def _validate_response_format(value: str) -> None:
    if value not in {"text", "json_object"}:
        raise HTTPException(status_code=400, detail="response_format invalido.")


def _validate_model_filename(filename: str | None) -> str:
    file_name = (filename or "").strip()
    suffix = Path(file_name).suffix.lower()
    if not file_name or suffix not in SUPPORTED_MODEL_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_MODEL_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Formato 3D invalido. Usa {allowed}.")
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
    metadata: str | None = Form(default=None),
    image: UploadFile = File(...),
    service: ChatService = Depends(get_chat_service),
) -> ChatMessageResponse:
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Imagem vazia no upload.")
    _validate_response_format(response_format)
    metadata_payload = _parse_metadata_field(metadata)

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
    metadata: str | None = Form(default=None),
    image: UploadFile = File(...),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Imagem vazia no upload.")
    _validate_response_format(response_format)
    metadata_payload = _parse_metadata_field(metadata)

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
    metadata: str | None = Form(default=None),
    model_file: UploadFile = File(...),
    service: ChatService = Depends(get_chat_service),
) -> ChatMessageResponse:
    content = await model_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Modelo 3D vazio no upload.")
    _validate_response_format(response_format)
    validated_file_name = _validate_model_filename(model_file.filename)
    metadata_payload = _parse_metadata_field(metadata)

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
    metadata: str | None = Form(default=None),
    model_file: UploadFile = File(...),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    content = await model_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Modelo 3D vazio no upload.")
    _validate_response_format(response_format)
    validated_file_name = _validate_model_filename(model_file.filename)
    metadata_payload = _parse_metadata_field(metadata)

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


@router.get("/images/{original_image_name}")
async def get_chat_image_asset(original_image_name: str) -> FileResponse:
    settings = get_settings()
    root = settings.image_asset_root_resolved
    if root is None:
        raise HTTPException(status_code=404, detail="IMAGE_ASSET_ROOT nao configurado.")

    file_name = Path(original_image_name).name
    if file_name != original_image_name:
        raise HTTPException(status_code=400, detail="original_image_name invalido.")

    target = (root / file_name).resolve()
    try:
        target.relative_to(root)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="original_image_name invalido.") from exc

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Imagem nao encontrada.")

    media_type = None
    suffix = target.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif suffix == ".png":
        media_type = "image/png"
    elif suffix == ".webp":
        media_type = "image/webp"

    return FileResponse(path=Path(target), media_type=media_type)
