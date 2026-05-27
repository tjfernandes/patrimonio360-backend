from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache
import threading
from typing import Any

from app.core.config import Settings, get_settings
from app.services.embeddings import EmbeddingProvider, get_embedding_provider
from app.services.multiview_renderer import (
    PersistentMultiviewRenderer,
    RenderedModelView,
    get_multiview_renderer,
)
from app.services.opensearch_client import OpenSearchGateway, get_opensearch_gateway

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, dict[str, object]], Awaitable[None]]

EVENT_LABELS: dict[str, str] = {
    "model.render.first_pass": "Primeiras vistas do modelo 3D geradas.",
    "model.render.second_pass": "Vistas adicionais do modelo 3D geradas.",
    "model.render.cache_hit": "Cache de multiview reutilizado para este modelo 3D.",
    "model.embedding.ready": "Embeddings multimodais gerados para as vistas do modelo 3D.",
    "model.retrieve.response": "Retrieval multimodal do modelo 3D concluido.",
}


@dataclass(slots=True)
class ModelRetrievalResult:
    image_hits: list[dict[str, Any]]
    artifact_docs: list[dict[str, Any]]
    image_embeddings: list[list[float]]
    image_hits_total: int
    retrieval_window_size: int
    extra_views_used: bool
    top_score: float | None


@dataclass(slots=True)
class _CachedModelEntry:
    file_name: str
    first_pass_views: list[RenderedModelView] = field(default_factory=list)
    first_pass_embeddings: list[list[float]] = field(default_factory=list)
    extra_views: list[RenderedModelView] = field(default_factory=list)
    extra_embeddings: list[list[float]] = field(default_factory=list)


class ModelRetrievalService:
    def __init__(
        self,
        settings: Settings,
        renderer: PersistentMultiviewRenderer,
        embedding_provider: EmbeddingProvider,
        opensearch_gateway: OpenSearchGateway,
    ) -> None:
        self.settings = settings
        self.renderer = renderer
        self.embedding_provider = embedding_provider
        self.opensearch_gateway = opensearch_gateway
        self._cache: OrderedDict[str, _CachedModelEntry] = OrderedDict()
        self._cache_lock = threading.Lock()

    async def _emit_status(
        self,
        progress_cb: ProgressCallback | None,
        message: str,
        **fields: object,
    ) -> None:
        if progress_cb is None:
            return
        try:
            await progress_cb(message, dict(fields))
        except Exception:
            logger.debug("model retrieval progress callback failed", exc_info=True)

    def _log(self, level: int, event: str, **fields: object) -> None:
        if self.settings.LOG_JSON:
            payload = {"event": event, **fields}
            prefix = EVENT_LABELS.get(event, event)
            if self.settings.LOG_JSON_PRETTY:
                logger.log(
                    level,
                    f"{prefix}\n"
                    + json.dumps(
                        payload,
                        ensure_ascii=False,
                        default=str,
                        indent=max(self.settings.LOG_JSON_INDENT, 0),
                    ),
                )
            else:
                logger.log(level, f"{prefix} " + json.dumps(payload, ensure_ascii=False, default=str))
            return

        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.log(level, f"{event} {details}".strip())

    def _cache_key(self, *, model_bytes: bytes, file_name: str) -> str:
        digest = hashlib.sha256()
        digest.update(file_name.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
        digest.update(model_bytes)
        return digest.hexdigest()

    def _touch_entry(self, cache_key: str, entry: _CachedModelEntry) -> _CachedModelEntry:
        with self._cache_lock:
            self._cache[cache_key] = entry
            self._cache.move_to_end(cache_key)
            while len(self._cache) > max(self.settings.CHAT_MODEL_CACHE_SIZE, 1):
                self._cache.popitem(last=False)
        return entry

    def _get_entry(self, cache_key: str, file_name: str) -> _CachedModelEntry:
        with self._cache_lock:
            entry = self._cache.get(cache_key)
        if entry is None:
            entry = _CachedModelEntry(file_name=file_name)
        return self._touch_entry(cache_key, entry)

    async def _ensure_first_pass(
        self,
        *,
        cache_key: str,
        entry: _CachedModelEntry,
        model_bytes: bytes,
        file_name: str,
    ) -> _CachedModelEntry:
        if entry.first_pass_views and entry.first_pass_embeddings:
            self._log(
                logging.INFO,
                "model.render.cache_hit",
                cache_key=cache_key,
                stage="first_pass",
                view_count=len(entry.first_pass_views),
            )
            return self._touch_entry(cache_key, entry)

        # Always generate exactly 5 views for deterministic model retrieval.
        total_views = 5
        view_count = 5
        entry.first_pass_views = await self.renderer.render_views(
            model_bytes=model_bytes,
            file_name=file_name,
            views=view_count,
            skip_views=0,
            target_view_count=total_views,
        )
        self._log(
            logging.INFO,
            "model.render.first_pass",
            cache_key=cache_key,
            file_name=file_name,
            view_count=len(entry.first_pass_views),
            target_view_count=total_views,
        )
        entry.first_pass_embeddings = await self.embedding_provider.embed_many_multimodal_image_bytes(
            image_bytes_values=[view.png_bytes for view in entry.first_pass_views],
            text=None,
        )
        self._log(
            logging.INFO,
            "model.embedding.ready",
            cache_key=cache_key,
            stage="first_pass",
            embedding_count=len(entry.first_pass_embeddings),
            embedding_dim=len(entry.first_pass_embeddings[0]) if entry.first_pass_embeddings else 0,
        )
        return self._touch_entry(cache_key, entry)

    async def _ensure_second_pass(
        self,
        *,
        cache_key: str,
        entry: _CachedModelEntry,
        model_bytes: bytes,
        file_name: str,
    ) -> _CachedModelEntry:
        if entry.extra_views and entry.extra_embeddings:
            self._log(
                logging.INFO,
                "model.render.cache_hit",
                cache_key=cache_key,
                stage="second_pass",
                view_count=len(entry.extra_views),
            )
            return self._touch_entry(cache_key, entry)

        total_views = max(self.settings.CHAT_MODEL_TOTAL_VIEWS, self.settings.CHAT_MODEL_FIRST_PASS_VIEWS)
        extra_views = max(0, total_views - self.settings.CHAT_MODEL_FIRST_PASS_VIEWS)
        if extra_views <= 0:
            return self._touch_entry(cache_key, entry)

        entry.extra_views = await self.renderer.render_views(
            model_bytes=model_bytes,
            file_name=file_name,
            views=extra_views,
            skip_views=self.settings.CHAT_MODEL_FIRST_PASS_VIEWS,
            target_view_count=total_views,
        )
        self._log(
            logging.INFO,
            "model.render.second_pass",
            cache_key=cache_key,
            file_name=file_name,
            view_count=len(entry.extra_views),
            target_view_count=total_views,
        )
        entry.extra_embeddings = await self.embedding_provider.embed_many_multimodal_image_bytes(
            image_bytes_values=[view.png_bytes for view in entry.extra_views],
            text=None,
        )
        self._log(
            logging.INFO,
            "model.embedding.ready",
            cache_key=cache_key,
            stage="second_pass",
            embedding_count=len(entry.extra_embeddings),
            embedding_dim=len(entry.extra_embeddings[0]) if entry.extra_embeddings else 0,
        )
        return self._touch_entry(cache_key, entry)

    def _needs_second_pass(self, image_hits: list[dict[str, Any]]) -> bool:
        if not image_hits:
            return True
        try:
            top_score = float(image_hits[0].get("score"))
        except (TypeError, ValueError):
            return True
        return top_score < float(self.settings.CHAT_MODEL_LOW_CONFIDENCE_SCORE_THRESHOLD)

    async def retrieve(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        model_bytes: bytes,
        file_name: str,
        artifact_museum_id: str | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> ModelRetrievalResult:
        cache_key = self._cache_key(model_bytes=model_bytes, file_name=file_name)
        entry = self._get_entry(cache_key, file_name)
        await self._emit_status(progress_cb, "status.generating_model_views", stage="render_first_pass")
        entry = await self._ensure_first_pass(
            cache_key=cache_key,
            entry=entry,
            model_bytes=model_bytes,
            file_name=file_name,
        )

        candidate_top_k = max(
            self.settings.CHAT_IMAGE_RETRIEVAL_TOP_K,
            self.settings.CHAT_IMAGE_ARTIFACT_TOP_K,
            self.settings.CHAT_RETRIEVAL_CANDIDATES,
            1,
        )
        retrieval_window_size = max(
            int(getattr(self.settings, "CHAT_IMAGE_RETRIEVAL_PAGINATION_WINDOW", 0) or 0),
            candidate_top_k,
            1,
        )
        await self._emit_status(progress_cb, "status.searching_collection", stage="search_first_pass")
        image_embeddings = list(entry.first_pass_embeddings)
        image_page = await self.opensearch_gateway.search_similar_images_multi_page(
            museum_slug=museum_slug,
            museum_id=museum_id,
            image_embeddings=image_embeddings,
            from_offset=0,
            page_size=candidate_top_k,
            retrieval_window_size=retrieval_window_size,
        )
        image_hits = image_page.results
        image_hits_total = min(max(int(image_page.total), 0), retrieval_window_size)

        extra_views_used = False

        inventory_numbers = [
            str(hit.get("inventory_number") or hit.get("inventory") or "").strip()
            for hit in image_hits
            if str(hit.get("inventory_number") or hit.get("inventory") or "").strip()
        ]
        artifact_docs = await self.opensearch_gateway.fetch_artifacts_by_inventory_numbers(
            museum_slug=museum_slug,
            museum_id=artifact_museum_id,
            inventory_numbers=inventory_numbers,
            top_k=candidate_top_k,
        )
        if not artifact_docs:
            artifact_ids = [
                str(hit.get("artifact_id") or "").strip()
                for hit in image_hits
                if str(hit.get("artifact_id") or "").strip()
            ]
            artifact_docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
                museum_slug=museum_slug,
                museum_id=museum_id,
                artifact_ids=artifact_ids,
                top_k=candidate_top_k,
            )

        top_score: float | None = None
        if image_hits:
            try:
                top_score = float(image_hits[0].get("score"))
            except (TypeError, ValueError):
                top_score = None
        self._log(
            logging.INFO,
            "model.retrieve.response",
            cache_key=cache_key,
            museum_slug=museum_slug,
            museum_id=museum_id,
            image_hits=len(image_hits),
            image_hits_total=image_hits_total,
            artifact_docs=len(artifact_docs),
            extra_views_used=extra_views_used,
            top_score=top_score,
        )
        return ModelRetrievalResult(
            image_hits=image_hits,
            artifact_docs=artifact_docs,
            image_embeddings=image_embeddings,
            image_hits_total=image_hits_total,
            retrieval_window_size=retrieval_window_size,
            extra_views_used=extra_views_used,
            top_score=top_score,
        )


@lru_cache(maxsize=1)
def get_model_retrieval_service() -> ModelRetrievalService:
    settings = get_settings()
    return ModelRetrievalService(
        settings=settings,
        renderer=get_multiview_renderer(),
        embedding_provider=get_embedding_provider(),
        opensearch_gateway=get_opensearch_gateway(),
    )
