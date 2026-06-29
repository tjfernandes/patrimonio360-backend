from __future__ import annotations

import logging
import time
from typing import Any

from app.core.logging import log_event
from app.services.chat_service import ChatService, get_chat_service

logger = logging.getLogger(__name__)


async def warmup_chat_stack(
    *,
    service: ChatService | None = None,
    include_multimodal: bool = True,
    include_multiview_worker: bool = False,
) -> dict[str, Any]:
    resolved_service = service or get_chat_service()
    started_at = time.perf_counter()
    log_event(
        logger,
        logging.INFO,
        "warmup.chat_stack.start",
        include_multimodal=include_multimodal,
        include_multiview_worker=include_multiview_worker,
    )
    results: dict[str, Any] = {
        "opensearch_ready": False,
        "text_embeddings_ready": False,
        "multimodal_ready": False,
        "multiview_worker_ready": False,
    }

    results["opensearch_ready"] = await resolved_service.opensearch_gateway.ensure_ready()
    await resolved_service.embedding_provider.warmup(include_multimodal=include_multimodal)
    results["text_embeddings_ready"] = True
    results["multimodal_ready"] = include_multimodal

    if include_multiview_worker:
        await resolved_service.model_retrieval_service.renderer.ensure_worker_running()
        results["multiview_worker_ready"] = True

    duration_ms = (time.perf_counter() - started_at) * 1000
    log_event(
        logger,
        logging.INFO,
        "warmup.chat_stack.finish",
        duration_ms=round(duration_ms, 1),
        **results,
    )
    return results
