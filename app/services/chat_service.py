from functools import lru_cache
import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.core.logging import log_event, log_json_event
from app.prompts import (
    build_final_answer_prompt,
    build_router_user_prompt,
    get_router_system_prompt,
)
from app.prompts.query_planner_prompts import (
    RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT,
    build_retrieval_query_rewrite_prompt,
)
from app.schemas.chat import (
    ArtifactImageResult,
    ArtifactResult,
    ChatHealthResponse,
    ChatImageMessageRequest,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatModelMessageRequest,
    ChatRegenerateRequest,
    ChatResultsPageRequest,
    ChatResultsPageResponse,
    ChatSearchScope,
    ImageMatchResult,
    ResponseFormatObject,
    TourNavigationTarget,
    ArtifactDetailContextResponse,
    AuthorEntity,
    SetEntityWithArtifacts,
    ExhibitionEntityWithArtifacts,
    RelatedArtifactItem,
    RelatedArtifactsPageResponse,
)
from app.services.embeddings import EmbeddingProvider, get_embedding_provider
from app.services.chat_session_store import (
    ChatSessionState,
    ChatSessionStore,
    get_chat_session_store,
)
from app.services.chat_i18n import normalize_language, translate
from app.services.llm_service import LLMService, LLMServiceError, get_llm_service
from app.services.opensearch_client import OpenSearchGateway, get_opensearch_gateway
from app.services.model_retrieval import ModelRetrievalService, get_model_retrieval_service
from app.services.query_logger import QueryLogger, get_query_logger, utc_timestamp
from app.services.tour_navigation import TourNavigationService, get_tour_navigation_service

StatusCallback = Callable[[dict[str, Any]], Awaitable[None]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TemporalQuery:
    start_year: int | None
    end_year: int | None
    expression: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SearchMuseum:
    museum_id: str
    museum_slug: str
    museum_name: str | None
    aliases: tuple[str, ...]


_SEARCH_MUSEUMS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "search_museums.json"


@lru_cache(maxsize=1)
def _load_search_museums() -> tuple[SearchMuseum, ...]:
    fallback = [
        {
            "museum_id": "mnt",
            "museum_slug": "mnt",
            "museum_name": "Museu Nacional do Traje",
            "aliases": ["museu nacional do traje", "museu do traje", "nacional do traje", "mnt"],
        },
        {
            "museum_id": "mnaz",
            "museum_slug": "mnaz",
            "museum_name": "Museu Nacional do Azulejo",
            "aliases": ["museu nacional do azulejo", "museu do azulejo", "nacional do azulejo", "mnaz"],
        },
        {
            "museum_id": "mj",
            "museum_slug": "mj",
            "museum_name": "Mosteiro dos Jeronimos",
            "aliases": ["mosteiro dos jeronimos", "jeronimos", "jeronimo", "mj"],
        },
        {
            "museum_id": "mnsr",
            "museum_slug": "mnsr",
            "museum_name": "Museu Nacional Soares dos Reis",
            "aliases": ["museu nacional soares dos reis", "museu soares dos reis", "soares dos reis", "soares dos resi", "mnsr"],
        },
    ]
    try:
        raw = json.loads(_SEARCH_MUSEUMS_CONFIG_PATH.read_text(encoding="utf-8"))
        entries = raw if isinstance(raw, list) else fallback
    except Exception:
        entries = fallback

    museums: list[SearchMuseum] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        museum_id = str(entry.get("museum_id") or "").strip()
        museum_slug = str(entry.get("museum_slug") or museum_id).strip()
        if not museum_id or not museum_slug:
            continue
        raw_aliases = entry.get("aliases")
        aliases = tuple(
            str(alias or "").strip()
            for alias in (raw_aliases if isinstance(raw_aliases, list) else [])
            if str(alias or "").strip()
        )
        museums.append(
            SearchMuseum(
                museum_id=museum_id,
                museum_slug=museum_slug,
                museum_name=str(entry.get("museum_name") or "").strip() or None,
                aliases=aliases or (museum_slug, museum_id),
            )
        )
    return tuple(museums)


_TOUR_SCOPE_PATTERNS: tuple[str, ...] = (
    r"\b(?:nesta|neste|nessa|nesse|na|no|desta|deste|da|do)\s+(?:visita|tour|percurso)(?:\s+(?:virtual|360|digital|guiada|guiado))?\b",
    r"\b(?:visita|tour|percurso)\s+(?:virtual|360|digital)\b",
    r"\b(?:in\s+)?this\s+(?:virtual\s+|360\s+|digital\s+)?(?:tour|visit)\b",
    r"\b(?:in\s+the\s+)?(?:virtual|360|digital)\s+(?:tour|visit)\b",
)

_INVENTORY_PREFIXES: tuple[str, ...] = (
    "mj",
    "mnaa",
    "mnac",
    "mnaj",
    "mnar",
    "mnap",
    "mnaz",
    "mnmc",
    "mnsr",
    "mnt",
)

_EXPLICIT_INVENTORY_MARKERS: tuple[tuple[str, ...], ...] = (
    ("numero", "de", "inventario"),
    ("n", "de", "inventario"),
    ("inventario",),
    ("referencia",),
    ("ref",),
)

_ARTIFACT_REFERENCE_MARKERS: set[str] = {
    "artefacto",
    "artefactos",
    "artefato",
    "artefatos",
    "item",
    "itens",
    "objeto",
    "objetos",
    "peca",
    "pecas",
}

_ARTIFACT_REFERENCE_NUMBER_CONNECTORS: set[str] = {
    "n",
    "no",
    "nr",
    "num",
    "numero",
}


_HARDCODED_HISTORICAL_PERIODS: tuple[tuple[tuple[str, ...], TemporalQuery], ...] = (
    (
        (
            "periodo pombalino",
            "epoca pombalina",
            "era pombalina",
        ),
        TemporalQuery(1750, 1777, "periodo pombalino", 1.0),
    ),
    (
        (
            "periodo joanino",
            "periodo joanino de d joao v",
        ),
        TemporalQuery(1706, 1750, "periodo joanino", 1.0),
    ),
    (
        (
            "periodo manuelino",
            "epoca manuelina",
        ),
        TemporalQuery(1495, 1521, "periodo manuelino", 1.0),
    ),
    (
        (
            "periodo sebastianista",
            "periodo sebastianistas",
            "sebastianista",
            "sebastianistas",
            "crise de sucessao",
        ),
        TemporalQuery(1578, 1580, "periodo sebastianista", 1.0),
    ),
    (
        ("periodo miguelista", "miguelista", "miguelistas"),
        TemporalQuery(1828, 1834, "periodo miguelista", 1.0),
    ),
)

_ROMAN_CENTURY_NUMERALS: dict[str, int] = {
    "xv": 15,
    "xvi": 16,
    "xvii": 17,
    "xviii": 18,
    "xix": 19,
    "xx": 20,
    "xxi": 21,
    "xxii": 22,
    "xxiii": 23,
    "xxiv": 24,
    "xxv": 25,
    "xxvi": 26,
    "xxvii": 27,
    "xxviii": 28,
    "xxix": 29,
    "xxx": 30,
}



_PT_QUERY_LANGUAGE_HINTS: set[str] = {
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "ao",
    "aos",
    "para",
    "por",
    "com",
    "sem",
    "sobre",
    "que",
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "encontra",
    "encontrar",
    "procura",
    "procurar",
    "mostra",
    "museu",
    "peca",
    "pecas",
    "quero",
    "ver",
    "crianca",
    "criancas",
    "infantil",
    "infantis",
    "traje",
    "trajes",
    "vestuario",
    "vestuarios",
    "roupa",
    "roupas",
    "vestido",
    "vestidos",
    "chapeu",
    "chapeus",
    "cabeca",
    "arte",
    "retrato",
    "retratos",
    "azulejo",
    "azulejos",
}

_EN_QUERY_LANGUAGE_HINTS: set[str] = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "with",
    "without",
    "about",
    "find",
    "search",
    "show",
    "museum",
    "piece",
    "pieces",
    "child",
    "children",
    "dress",
    "dresses",
    "hat",
    "hats",
    "headgear",
    "headwear",
    "tile",
    "tiles",
}

class ChatService:
    """Chat service.

    Route-facing service layer.
    Uses a hybrid chat pipeline with optional RAG retrieval.
    """

    def __init__(
        self,
        settings: Settings,
        opensearch_gateway: OpenSearchGateway,
        embedding_provider: EmbeddingProvider,
        model_retrieval_service: ModelRetrievalService,
        tour_navigation_service: TourNavigationService,
        llm_service: LLMService,
        session_store: ChatSessionStore,
        query_logger: QueryLogger | None = None,
    ) -> None:
        self.settings = settings
        self.opensearch_gateway = opensearch_gateway
        self.embedding_provider = embedding_provider
        self.model_retrieval_service = model_retrieval_service
        self.tour_navigation_service = tour_navigation_service
        self.llm_service = llm_service
        self.session_store = session_store
        self.query_logger = query_logger or get_query_logger()


    def health(self) -> ChatHealthResponse:
        return ChatHealthResponse(
            llm_provider=self.settings.LLM_PROVIDER,
            llm_base_url=self.settings.llm_base_url_resolved,
            llm_text_model=self.settings.llm_model_resolved,
            llm_json_model=self.settings.llm_model_resolved,
            text_embedding_model=self.settings.text_embedding_model_resolved,
            multimodal_embedding_model=self.settings.multimodal_embedding_model_resolved,
        )

    async def _emit_status(
        self,
        status_cb: StatusCallback | None,
        key: str,
        *,
        language: str | None = None,
        **fields: object,
    ) -> None:
        if status_cb is None:
            return
        payload: dict[str, Any] = {
            "type": "status",
            "key": key,
            "message": translate(key, language, **fields),
            **fields,
        }
        try:
            await status_cb(payload)
        except Exception:
            pass

    def _sync_state_language(
        self,
        state: ChatSessionState,
        requested_language: str | None,
    ) -> str:
        language = normalize_language(requested_language or state.language)
        state.language = language
        return language

    def _final_system_prompt(self, system_prompt: str | None, language: str | None) -> str:
        language_guard = translate("llm.final_language_guard", language)
        base_prompt = (system_prompt or "").strip()
        if not base_prompt:
            return language_guard
        return f"{base_prompt}\n\n{language_guard}"

    def _resolve_results_page_size(self, requested_page_size: int | None, *, default_size: int) -> int:
        base = max(int(default_size), 1)
        if requested_page_size is None:
            return base
        return max(1, min(int(requested_page_size), 50))

    def _text_results_default_page_size(self) -> int:
        return max(int(self.settings.CHAT_RETRIEVAL_RESULTS_PAGE_SIZE), 1)

    def _text_retrieval_window_size(self, *, minimum: int = 1) -> int:
        configured = int(getattr(self.settings, "CHAT_RETRIEVAL_PAGINATION_WINDOW", 0) or 0)
        return max(configured, int(minimum), 1)

    def _image_retrieval_window_size(self, *, minimum: int = 1) -> int:
        configured = int(
            getattr(self.settings, "CHAT_IMAGE_RETRIEVAL_PAGINATION_WINDOW", 0) or 0
        )
        return max(configured, int(minimum), 1)

    def _retrieval_window_from_request(self, request: dict[str, Any]) -> int | None:
        try:
            value = int(request.get("retrieval_window_size") or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _bounded_retrieval_total(self, total: int, retrieval_window_size: int | None) -> int:
        resolved_total = max(int(total), 0)
        if retrieval_window_size is None:
            return resolved_total
        return min(resolved_total, max(int(retrieval_window_size), 1))

    def _paginate_retrieval_results(
        self,
        *,
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        navigation_targets: list[TourNavigationTarget],
        page: int,
        page_size: int | None,
        default_page_size: int,
        total_override: int | None = None,
    ) -> tuple[
        list[ArtifactResult],
        list[ImageMatchResult],
        list[TourNavigationTarget],
        int,
        int,
        int,
        bool,
    ]:
        resolved_page = max(int(page), 1)
        resolved_page_size = self._resolve_results_page_size(
            page_size,
            default_size=default_page_size,
        )
        available_total = len(artifact_results) if artifact_results else len(image_matches)
        reported_total = (
            max(int(total_override), available_total, 0)
            if total_override is not None
            else available_total
        )
        if available_total <= 0:
            start = (resolved_page - 1) * resolved_page_size
            return (
                [],
                [],
                [],
                resolved_page,
                resolved_page_size,
                reported_total,
                start < reported_total,
            )

        start = (resolved_page - 1) * resolved_page_size
        if start >= available_total:
            return [], [], [], resolved_page, resolved_page_size, reported_total, start < reported_total

        end = min(start + resolved_page_size, available_total)
        has_more = end < reported_total

        artifact_slice: list[ArtifactResult] = []
        if artifact_results:
            artifact_slice = artifact_results[start:end]

        def _norm(value: str | None) -> str:
            return str(value or "").strip().casefold()

        if artifact_slice:
            artifact_ids = {
                str(artifact.artifact_id or "").strip()
                for artifact in artifact_slice
                if str(artifact.artifact_id or "").strip()
            }
            inventories = {
                _norm(artifact.inventory_number)
                for artifact in artifact_slice
                if _norm(artifact.inventory_number)
            }
            image_slice = [
                match
                for match in image_matches
                if (
                    str(match.artifact_id or "").strip() in artifact_ids
                    or _norm(match.inventory) in inventories
                )
            ]
            if not image_slice and image_matches:
                image_slice = image_matches[start:end]
            navigation_slice = [
                target
                for target in navigation_targets
                if _norm(target.inventory_id) in inventories
            ]
        else:
            image_slice = image_matches[start:end]
            image_inventories = {
                _norm(match.inventory)
                for match in image_slice
                if _norm(match.inventory)
            }
            if image_inventories:
                navigation_slice = [
                    target
                    for target in navigation_targets
                    if _norm(target.inventory_id) in image_inventories
                ]
            else:
                navigation_slice = navigation_targets[start:end]

        return (
            artifact_slice,
            image_slice,
            navigation_slice,
            resolved_page,
            resolved_page_size,
            reported_total,
            has_more,
        )

    def _reported_results_total(self, request: dict[str, Any], page_total: int) -> int:
        try:
            stored_total = int(request.get("results_total") or 0)
        except (TypeError, ValueError):
            stored_total = 0
        reported_total = max(stored_total, int(page_total), 0)
        retrieval_window_size = self._retrieval_window_from_request(request)
        reported_total = self._bounded_retrieval_total(
            reported_total,
            retrieval_window_size,
        )
        request["results_total"] = reported_total
        return reported_total

    def _cache_last_retrieval_results(
        self,
        *,
        state: ChatSessionState,
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        navigation_targets: list[TourNavigationTarget],
        default_page_size: int,
        retrieval_request: dict[str, Any] | None = None,
    ) -> str:
        results_request = dict(retrieval_request or {})
        results_request_id = str(results_request.get("results_request_id") or "").strip()
        if not results_request_id:
            results_request_id = str(uuid4())
        results_request["results_request_id"] = results_request_id

        artifact_payload = [
            artifact.model_dump(mode="json")
            for artifact in artifact_results
        ]
        image_payload = [
            match.model_dump(mode="json")
            for match in image_matches
        ]
        navigation_payload = [
            target.model_dump(mode="json")
            for target in navigation_targets
        ]
        resolved_default_page_size = max(int(default_page_size), 1)

        state.last_paged_artifact_results = artifact_payload
        state.last_paged_image_matches = image_payload
        state.last_paged_navigation_targets = navigation_payload
        state.last_paged_results_default_page_size = resolved_default_page_size
        state.last_paged_retrieval_request = results_request

        paged_results_by_request_id = getattr(state, "paged_results_by_request_id", None)
        if not isinstance(paged_results_by_request_id, dict):
            paged_results_by_request_id = {}
            state.paged_results_by_request_id = paged_results_by_request_id

        if results_request_id in paged_results_by_request_id:
            paged_results_by_request_id.pop(results_request_id, None)
        paged_results_by_request_id[results_request_id] = {
            "artifact_results": artifact_payload,
            "image_matches": image_payload,
            "navigation_targets": navigation_payload,
            "default_page_size": resolved_default_page_size,
            "retrieval_request": results_request,
        }
        while len(paged_results_by_request_id) > 20:
            oldest_key = next(iter(paged_results_by_request_id))
            paged_results_by_request_id.pop(oldest_key, None)

        return results_request_id

    def _build_paged_results(
        self,
        *,
        state: ChatSessionState,
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        navigation_targets: list[TourNavigationTarget],
        page: int,
        page_size: int | None,
        default_page_size: int,
        total_override: int | None = None,
        retrieval_request: dict[str, Any] | None = None,
    ) -> tuple[
        list[ArtifactResult],
        list[ImageMatchResult],
        list[TourNavigationTarget],
        int,
        int,
        int,
        bool,
    ]:
        self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            default_page_size=default_page_size,
            retrieval_request=retrieval_request,
        )
        return self._paginate_retrieval_results(
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=page,
            page_size=page_size,
            default_page_size=default_page_size,
            total_override=total_override,
        )

    async def _fetch_artifact_docs_for_image_hits(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_museum_id: str | None,
        image_hits: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        if not image_hits:
            return []

        inventory_numbers = [
            self._image_hit_inventory(hit)
            for hit in image_hits
            if self._image_hit_inventory(hit)
        ]
        artifact_docs: list[dict[str, object]] = []
        if inventory_numbers:
            artifact_docs = await self.opensearch_gateway.fetch_artifacts_by_inventory_numbers(
                museum_slug=museum_slug,
                museum_id=artifact_museum_id,
                inventory_numbers=inventory_numbers,
                top_k=max(top_k, 1),
            )
        if artifact_docs:
            return artifact_docs

        artifact_ids = [
            str(hit.get("artifact_id") or "").strip()
            for hit in image_hits
            if str(hit.get("artifact_id") or "").strip()
        ]
        if not artifact_ids:
            return []
        return await self.opensearch_gateway.fetch_artifacts_by_ids(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_ids=artifact_ids,
            top_k=max(top_k, 1),
        )

    def _artifact_results_as_prompt_docs(
        self,
        artifact_results: list[ArtifactResult],
    ) -> list[dict[str, object]]:
        docs: list[dict[str, object]] = []
        excluded_fields = {"images"}
        if not self.settings.CHAT_INCLUDE_ORIGIN_HISTORY_IN_LLM_CONTEXT:
            excluded_fields.add("origin_history")
        for artifact in artifact_results:
            payload = artifact.model_dump(
                mode="json",
                exclude_none=True,
                exclude=excluded_fields,
            )
            if payload:
                docs.append(payload)
        return docs

    def _image_matches_as_visible_prompt_docs(
        self,
        image_matches: list[ImageMatchResult],
    ) -> list[dict[str, object]]:
        docs: list[dict[str, object]] = []
        artifact_fields = (
            "inventory_number",
            "title",
            "museum",
            "category",
            "super_category",
            "creator",
            "creators",
            "date_or_period",
            "support_or_material",
            "technique",
            "origin_history",
            "production_center",
            "description",
            "detail_type",
            "sets",
            "exhibitions",
            "image_count",
        )
        for match in image_matches:
            artifact = match.artifact if isinstance(match.artifact, dict) else {}
            inventory = (
                str(match.inventory or "").strip()
                or str(artifact.get("inventory_number") or artifact.get("inventory") or "").strip()
            )
            title = str(match.title or "").strip() or str(artifact.get("title") or "").strip()
            payload: dict[str, object] = {"result_card_type": "image_match"}
            if title:
                payload["title"] = title
            if inventory:
                payload["inventory"] = inventory
            for field in artifact_fields:
                value = artifact.get(field)
                if value in (None, "", [], {}):
                    continue
                if field == "origin_history" and not self.settings.CHAT_INCLUDE_ORIGIN_HISTORY_IN_LLM_CONTEXT:
                    continue
                if field == "inventory_number" and inventory:
                    continue
                if field == "title" and title:
                    continue
                payload[field] = value
            docs.append(payload)
        return docs

    def _build_visible_results_retrieval_context(
        self,
        *,
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        page: int,
        page_size: int,
        total: int,
        match_section_label: str = "visible_image_matches",
        prefer_artifact_results: bool = False,
        include_image_matches_section: bool = True,
    ) -> str:
        artifact_docs = self._artifact_results_as_prompt_docs(artifact_results)
        image_docs = (
            self._image_matches_as_visible_prompt_docs(image_matches)
            if image_matches
            else []
        )
        if prefer_artifact_results and artifact_docs:
            docs_for_prompt = artifact_docs
        else:
            docs_for_prompt = image_docs or artifact_docs
        visible_count = len(docs_for_prompt)
        if visible_count <= 0:
            return ""

        sections = [
            f"visible_results_page: {max(int(page), 1)}",
            f"visible_results_page_size: {max(int(page_size), 1)}",
            f"visible_results_count: {visible_count}",
            f"visible_results_total: {max(int(total), 0)}",
            "The frontend displays exactly visible_results_count result cards from current_visible_results for this answer.",
        ]

        docs_context = self._format_docs_for_prompt(
            docs=docs_for_prompt,
            top_k=max(len(docs_for_prompt), 1),
        )
        if docs_context:
            sections.append("current_visible_results:")
            sections.append(docs_context)

        if image_matches and include_image_matches_section:
            image_match_payload = [
                match.model_dump(
                    exclude_none=True,
                    exclude={"artifact", "navigation_target"},
                )
                for match in image_matches
            ]
            if not docs_context:
                sections.append("current_visible_results:")
                sections.append(json.dumps(image_match_payload, ensure_ascii=True))
            sections.append(f"{match_section_label}:")
            sections.append(json.dumps(image_match_payload, ensure_ascii=True))

        return "\n".join(sections).strip()

    def _is_llm_token_limit_error(self, exc: Exception) -> bool:
        message = f"{exc} {getattr(exc, '__cause__', '')}".casefold()
        return any(
            marker in message
            for marker in (
                "length limit",
                "context length",
                "maximum context",
                "max context",
                "token limit",
                "too many tokens",
                "prompt is too long",
                "input tokens",
                "prompt_tokens",
                "completionusage",
                "maximum number of tokens",
                "exceeded the model",
                "request too large",
            )
        )

    async def _generate_final_answer_with_context_retries(
        self,
        *,
        initial_message: str,
        response_format: ResponseFormatObject,
        system_prompt: str | None,
        model_override: str | None,
        visible_result_count: int,
        build_message_for_result_count: Callable[[int], str],
    ):
        try:
            return await self.llm_service.generate(
                message=initial_message,
                response_format=response_format,
                system_prompt=system_prompt,
                model_override=model_override,
            )
        except LLMServiceError as exc:
            if not self._is_llm_token_limit_error(exc) or visible_result_count <= 0:
                raise
            last_error: LLMServiceError = exc

        for result_count in range(visible_result_count - 1, -1, -1):
            retry_message = build_message_for_result_count(result_count)
            if not retry_message.strip():
                continue
            try:
                return await self.llm_service.generate(
                    message=retry_message,
                    response_format=response_format,
                    system_prompt=system_prompt,
                    model_override=model_override,
                )
            except LLMServiceError as exc:
                if not self._is_llm_token_limit_error(exc):
                    raise
                last_error = exc

        raise last_error

    def _results_page_query_text(self, request: dict[str, Any]) -> str:
        for key in ("query_text", "lexical_query", "original_query", "user_message"):
            value = str(request.get(key) or "").strip()
            if value:
                return value
        plan = request.get("plan")
        if isinstance(plan, dict):
            for key in ("query_text", "semantic_query"):
                value = str(plan.get(key) or "").strip()
                if value:
                    return value
        kind = str(request.get("kind") or "").strip()
        return kind

    def _build_results_page_reply_prompt(
        self,
        *,
        museum_slug: str,
        language: str | None,
        query_text: str,
        page: int,
        total: int,
        docs_context: str,
    ) -> str:
        resolved_language = normalize_language(language)
        if resolved_language == "en":
            return "\n".join(
                [
                    "You are a virtual assistant for a 360 museum tour.",
                    "Final answer language: English.",
                    "The user clicked a UI control to view the next page of search results.",
                    "Write a concise answer about ONLY the visible results in current_page_results.",
                    "Do not summarize, reuse, or mention previous result pages.",
                    "Do not use conversation history as a source of facts.",
                    "Never expose artifact_id or internal doc markers such as [doc_1].",
                    "When referring to an object, prefer its title and optionally inventory reference.",
                    "If using an ordered list, use consecutive markers (1., 2., 3.) without blank lines between items.",
                    f"museum_slug: {museum_slug}",
                    f"search_query: {query_text or 'unknown'}",
                    f"results_page: {page}",
                    f"reported_total: {total}",
                    "current_page_results:",
                    docs_context,
                ]
            )

        return "\n".join(
            [
                "Es um assistente virtual para uma visita 360 ao museu.",
                "Idioma final da resposta: portugues.",
                "O utilizador clicou num controlo de UI para ver a proxima pagina de resultados.",
                "Escreve uma resposta curta sobre APENAS os resultados visiveis em current_page_results.",
                "Nao resumas, reutilizes ou menciones paginas de resultados anteriores.",
                "Nao uses historico da conversa como fonte de factos.",
                "Nunca exponhas artifact_id nem marcadores internos como [doc_1].",
                "Ao referires um objeto, privilegia o titulo e opcionalmente a referencia de inventario.",
                "Se usares lista numerada, usa marcadores consecutivos (1., 2., 3.) sem linhas em branco entre itens.",
                f"museum_slug: {museum_slug}",
                f"search_query: {query_text or 'desconhecida'}",
                f"results_page: {page}",
                f"reported_total: {total}",
                "current_page_results:",
                docs_context,
            ]
        )

    async def _generate_results_page_reply(
        self,
        *,
        payload: ChatResultsPageRequest,
        request: dict[str, Any],
        docs: list[dict[str, object]],
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult] | None = None,
        page: int,
        total: int,
    ) -> str:
        language = normalize_language(payload.language)
        docs_for_prompt = (
            self._image_matches_as_visible_prompt_docs(image_matches or [])
            if image_matches
            else docs or self._artifact_results_as_prompt_docs(artifact_results)
        )
        docs_context = self._format_docs_for_prompt(
            docs=docs_for_prompt,
            top_k=max(len(docs_for_prompt), 1),
        )
        if not docs_context:
            return translate("message.results_page_fallback", language)

        prompt = self._build_results_page_reply_prompt(
            museum_slug=self._request_museum_slug(payload, request),
            language=language,
            query_text=self._results_page_query_text(request),
            page=page,
            total=total,
            docs_context=docs_context,
        )
        try:
            llm_response = await self.llm_service.generate(
                message=prompt,
                response_format=ResponseFormatObject(type="text"),
                system_prompt=self._final_system_prompt(None, language),
                model_override=None,
            )
            reply = self._sanitize_assistant_reply(
                str(getattr(llm_response, "text", "") or ""),
                docs=docs_for_prompt,
                language=language,
            )
            if reply:
                return reply
        except Exception as exc:
            pass
        return translate("message.results_page_fallback", language)

    async def _materialize_text_results_page(
        self,
        *,
        state: ChatSessionState,
        payload: ChatResultsPageRequest,
        request: dict[str, Any],
        page: int,
        page_size: int,
    ) -> ChatResultsPageResponse:
        from_offset = (page - 1) * page_size
        request_museum_slug = self._request_museum_slug(payload, request)
        request_museum_id = self._request_museum_id(payload, request)
        search_scope = self._search_scope_from_request(
            request,
            fallback_museum_slug=request_museum_slug,
            fallback_museum_id=request_museum_id,
        )
        page_result = await self.opensearch_gateway.search_relevant_context_page(
            museum_slug=request_museum_slug,
            museum_id=request_museum_id,
            query_text=str(request.get("query_text") or ""),
            lexical_query=str(request.get("lexical_query") or "") or None,
            query_embedding=list(request.get("query_embedding") or []),
            from_offset=from_offset,
            page_size=page_size,
            filters=dict(request.get("filters") or {}),
            sort=dict(request.get("sort") or {}),
            retrieval_window_size=self._retrieval_window_from_request(request),
        )
        docs = page_result.results
        image_matches: list[ImageMatchResult] = []
        image_hits: list[dict[str, Any]] = []
        artifact_ids = [
            str(doc.get("artifact_id") or "").strip()
            for doc in docs
            if str(doc.get("artifact_id") or "").strip()
        ]
        if artifact_ids:
            try:
                image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                    museum_slug=request_museum_slug,
                    museum_id=request_museum_id,
                    artifact_ids=artifact_ids,
                    per_artifact=1,
                    max_total=max(len(artifact_ids), 1),
                )
            except Exception as exc:
                image_hits = []
            image_matches = self._build_image_matches(
                image_hits=image_hits,
                artifact_docs=docs,
            )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=request_museum_slug,
            museum_id=request_museum_id,
            docs=docs,
        )
        # Build the full artifact records (all images) so the detail modal matches
        # the image/model search behaviour. The 1-image `image_hits` above only feed
        # the result-card thumbnails, not the modal gallery.
        artifact_results = await self._build_artifact_results(
            museum_slug=request_museum_slug,
            museum_id=request_museum_id,
            artifact_docs=docs,
        )
        image_matches = self._enrich_image_matches(
            context="text_page",
            conversation_id=payload.conversation_id,
            museum_slug=request_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        reported_total = self._reported_results_total(request, page_result.total)
        reply = await self._generate_results_page_reply(
            payload=payload,
            request=request,
            docs=docs,
            artifact_results=artifact_results,
            image_matches=image_matches,
            page=page,
            total=reported_total,
        )
        results_request_id = self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            default_page_size=page_size,
            retrieval_request=request,
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            reply=reply,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            results_page=page,
            results_page_size=page_size,
            results_total=reported_total,
            results_has_more=(from_offset + page_size) < reported_total,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    async def _materialize_media_results_page(
        self,
        *,
        state: ChatSessionState,
        payload: ChatResultsPageRequest,
        request: dict[str, Any],
        page: int,
        page_size: int,
    ) -> ChatResultsPageResponse:
        from_offset = (page - 1) * page_size
        kind = str(request.get("kind") or "")
        request_museum_slug = self._request_museum_slug(payload, request)
        museum_id = self._request_museum_id(payload, request)
        search_scope = self._search_scope_from_request(
            request,
            fallback_museum_slug=request_museum_slug,
            fallback_museum_id=museum_id,
        )
        if kind == "model":
            page_result = await self.opensearch_gateway.search_similar_images_multi_page(
                museum_slug=request_museum_slug,
                museum_id=museum_id,
                image_embeddings=list(request.get("image_embeddings") or []),
                from_offset=from_offset,
                page_size=page_size,
                retrieval_window_size=self._retrieval_window_from_request(request),
            )
        else:
            page_result = await self.opensearch_gateway.search_similar_images_page(
                museum_slug=request_museum_slug,
                museum_id=museum_id,
                image_embedding=list(request.get("image_embedding") or []),
                from_offset=from_offset,
                page_size=page_size,
                retrieval_window_size=self._retrieval_window_from_request(request),
            )

        image_hits = page_result.results
        artifact_docs = await self._fetch_artifact_docs_for_image_hits(
            museum_slug=request_museum_slug,
            museum_id=museum_id,
            artifact_museum_id=str(request.get("artifact_museum_id") or "").strip() or None,
            image_hits=image_hits,
            top_k=page_size,
        )
        image_matches = self._build_image_matches(
            image_hits=image_hits,
            artifact_docs=artifact_docs,
        )
        artifact_results = await self._build_artifact_results(
            museum_slug=request_museum_slug,
            museum_id=museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=request_museum_slug,
            museum_id=museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context=f"{kind}_page",
            conversation_id=payload.conversation_id,
            museum_slug=request_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        reported_total = self._reported_results_total(request, page_result.total)
        reply = await self._generate_results_page_reply(
            payload=payload,
            request=request,
            docs=artifact_docs,
            artifact_results=artifact_results,
            image_matches=image_matches,
            page=page,
            total=reported_total,
        )
        results_request_id = self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            default_page_size=page_size,
            retrieval_request=request,
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            reply=reply,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            results_page=page,
            results_page_size=page_size,
            results_total=reported_total,
            results_has_more=(from_offset + page_size) < reported_total,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    async def get_results_page(self, payload: ChatResultsPageRequest) -> ChatResultsPageResponse:
        state = self.session_store.get(payload.conversation_id)
        if state is None or state.museum_slug != payload.museum_slug:
            return ChatResultsPageResponse(
                conversation_id=payload.conversation_id,
                results_page=max(payload.results_page, 1),
                results_page_size=self._resolve_results_page_size(
                    payload.results_page_size,
                    default_size=self._text_results_default_page_size(),
                ),
                results_total=0,
                results_has_more=False,
                results_request_id=payload.results_request_id,
            )
        self._sync_state_language(state, payload.language)

        requested_results_request_id = str(payload.results_request_id or "").strip()
        paged_results_by_request_id = getattr(state, "paged_results_by_request_id", None)
        if not isinstance(paged_results_by_request_id, dict):
            paged_results_by_request_id = {}
            state.paged_results_by_request_id = paged_results_by_request_id
        snapshot = (
            paged_results_by_request_id.get(requested_results_request_id)
            if requested_results_request_id
            else None
        )
        latest_results_request_id = str(
            (state.last_paged_retrieval_request or {}).get("results_request_id") or ""
        ).strip()
        if (
            requested_results_request_id
            and snapshot is None
            and requested_results_request_id != latest_results_request_id
        ):
            return ChatResultsPageResponse(
                conversation_id=payload.conversation_id,
                results_page=max(payload.results_page, 1),
                results_page_size=self._resolve_results_page_size(
                    payload.results_page_size,
                    default_size=self._text_results_default_page_size(),
                ),
                results_total=0,
                results_has_more=False,
                results_request_id=requested_results_request_id,
            )
        retrieval_request = dict(
            (snapshot or {}).get("retrieval_request")
            or state.last_paged_retrieval_request
            or {}
        )
        retrieval_kind = str(retrieval_request.get("kind") or "")
        default_page_size_floor = (
            self._text_results_default_page_size()
            if retrieval_kind == "text"
            else max(self.settings.CHAT_RETRIEVAL_TOP_K, 1)
        )
        default_page_size = max(
            int((snapshot or {}).get("default_page_size") or state.last_paged_results_default_page_size or 0),
            default_page_size_floor,
        )
        results_page = max(payload.results_page, 1)
        results_page_size = self._resolve_results_page_size(
            payload.results_page_size,
            default_size=default_page_size,
        )
        if retrieval_kind == "text":
            response = await self._materialize_text_results_page(
                state=state,
                payload=payload,
                request=retrieval_request,
                page=results_page,
                page_size=results_page_size,
            )
            self.session_store.save(state)
            return response
        if retrieval_kind in {"image", "model"}:
            response = await self._materialize_media_results_page(
                state=state,
                payload=payload,
                request=retrieval_request,
                page=results_page,
                page_size=results_page_size,
            )
            self.session_store.save(state)
            return response
        artifact_results = [
            ArtifactResult(**item)
            for item in (
                (snapshot or {}).get("artifact_results")
                or state.last_paged_artifact_results
            )
        ]
        image_matches = [
            ImageMatchResult(**item)
            for item in (
                (snapshot or {}).get("image_matches")
                or state.last_paged_image_matches
            )
        ]
        navigation_targets = [
            TourNavigationTarget(**item)
            for item in (
                (snapshot or {}).get("navigation_targets")
                or state.last_paged_navigation_targets
            )
        ]
        (
            paged_artifacts,
            paged_image_matches,
            paged_navigation_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._paginate_retrieval_results(
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=default_page_size,
        )
        reply = await self._generate_results_page_reply(
            payload=payload,
            request=retrieval_request,
            docs=[],
            artifact_results=paged_artifacts,
            image_matches=paged_image_matches,
            page=results_page,
            total=results_total,
        )
        search_scope = self._search_scope_from_request(
            retrieval_request,
            fallback_museum_slug=self._request_museum_slug(payload, retrieval_request),
            fallback_museum_id=self._request_museum_id(payload, retrieval_request),
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            reply=reply,
            artifact_results=paged_artifacts,
            image_matches=paged_image_matches,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=(
                str(retrieval_request.get("results_request_id") or requested_results_request_id).strip()
                or None
            ),
            search_scope=search_scope,
        )

    def _log_correlation_ids(
        self,
        *,
        metadata: dict[str, Any] | None,
        museum_slug: str | None,
    ) -> dict[str, str | None]:
        """Resolve the IDs required by the backend_query log event.

        `session_id`, `participant_id` and `task_id` have no first-class field
        on the chat request schemas today; they are read from the optional
        `metadata` payload (`ChatMessageRequest.metadata` and friends) so a
        future frontend/evaluation harness can populate them without a schema
        change. `tour_id` falls back to `museum_slug`, which is the identifier
        actually used to resolve a virtual tour in this codebase (see
        `poi_tours/panorama-overlays-inventory-<museum_slug>.json`).
        """
        meta = metadata or {}

        def _meta_str(key: str) -> str | None:
            value = meta.get(key)
            text = str(value).strip() if value is not None else ""
            return text or None

        resolved_tour_id = _meta_str("tour_id") or (str(museum_slug).strip() or None if museum_slug else None)
        return {
            "session_id": _meta_str("session_id"),
            "participant_id": _meta_str("participant_id"),
            "task_id": _meta_str("task_id"),
            "tour_id": resolved_tour_id,
        }

    def _selected_artifact_from_metadata(
        self,
        metadata: dict[str, Any] | None,
    ) -> dict[str, str | None] | None:
        if not isinstance(metadata, dict):
            return None

        selected = metadata.get("selected_artifact")
        selected_payload = selected if isinstance(selected, dict) else {}

        def _meta_str(*keys: str) -> str | None:
            for key in keys:
                value = selected_payload.get(key)
                if value is None:
                    value = metadata.get(key)
                text = str(value).strip() if value is not None else ""
                if text:
                    return text
            return None

        artifact_id = _meta_str("artifact_id", "selected_artifact_id")
        if not artifact_id:
            return None

        return {
            "artifact_id": artifact_id,
            "inventory_number": _meta_str("inventory_number", "selected_inventory_number"),
            "title": _meta_str("title", "selected_artifact_title"),
            "source_query_id": _meta_str("source_query_id", "selected_artifact_query_id"),
            "source": _meta_str("source", "selected_artifact_source"),
            "museum_id": _meta_str("museum_id", "selected_museum_id"),
            "museum_slug": _meta_str("museum_slug", "selected_museum_slug"),
            "museum_name": _meta_str("museum_name", "selected_museum_name"),
        }

    def _selected_artifact_context_mode(self, message: str) -> str:
        folded = f" {self._fold_query_text(message)} "
        anchored_patterns = (
            r"\b(outro|outros|outra|outras)\b",
            r"\bparecid[oa]s?\b",
            r"\bsemelhantes?\b",
            r"\bsimilares?\b",
            r"\brelacionad[oa]s?\b",
            r"\bmais como (este|esta|isto|isso)\b",
            r"\bmore like (this|it)\b",
            r"\bother\b",
            r"\bsimilar\b",
            r"\blike this\b",
            r"\brelated\b",
        )
        if any(re.search(pattern, folded) for pattern in anchored_patterns):
            return "anchored_similarity"
        return "selected_artifact"

    def _is_targeted_artifact_search_request(self, message: str) -> bool:
        folded = f" {self._fold_query_text(message)} "
        targeted_patterns = (
            r"\b(encontra|encontrar|procura|procurar|pesquisa|pesquisar)\b",
            r"\b(existe|existem|ha|haver|tem|tens)\b",
            r"\b(find|search|look for|are there|is there|exists?)\b",
        )
        return any(re.search(pattern, folded) for pattern in targeted_patterns)

    def _resolve_requested_search_museum(self, query: str) -> SearchMuseum | None:
        folded_query = f" {self._fold_query_text(query)} "
        if not folded_query.strip():
            return None

        candidates: list[tuple[int, SearchMuseum]] = []
        for museum in _load_search_museums():
            alias_matches = False
            for alias in museum.aliases:
                folded_alias = self._fold_query_text(alias)
                if not folded_alias:
                    continue
                phrase = r"\s+".join(re.escape(token) for token in folded_alias.split())
                if re.search(rf"\b{phrase}\b", folded_query):
                    alias_matches = True
                    candidates.append((len(folded_alias), museum))
                    break
            if alias_matches:
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _strip_search_museum_expression_from_query(
        self,
        text: str,
        target_museum: SearchMuseum | None,
    ) -> str:
        raw_text = (text or "").strip()
        if not raw_text or target_museum is None:
            return raw_text

        folded_text, source_indices = self._fold_query_text_with_index(raw_text)
        if not folded_text or not source_indices:
            return raw_text

        remove_mask = [False] * len(raw_text)
        patterns: list[str] = []
        aliases = sorted(
            {self._fold_query_text(alias) for alias in target_museum.aliases},
            key=len,
            reverse=True,
        )
        for folded_alias in aliases:
            tokens = folded_alias.split()
            if not tokens:
                continue
            phrase = r"\s+".join(re.escape(token) for token in tokens)
            patterns.append(
                rf"\b(?:(?:no|na|nos|nas|em|do|da|dos|das|de|in|at|the)\s+)?"
                rf"(?:(?:museu|museu\s+nacional|mosteiro)\s+)?{phrase}\b"
            )

        for pattern in patterns:
            for match in re.finditer(pattern, folded_text):
                start, end = match.span()
                if start >= len(source_indices) or end <= 0:
                    continue
                source_start = source_indices[max(start, 0)]
                source_end = source_indices[min(end - 1, len(source_indices) - 1)] + 1
                for index in range(max(source_start, 0), min(source_end, len(remove_mask))):
                    remove_mask[index] = True

        cleaned = "".join(
            char for index, char in enumerate(raw_text) if not remove_mask[index]
        )
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,.;:!?") or raw_text

    def _build_search_scope(
        self,
        *,
        current_museum_slug: str,
        current_museum_id: str | None,
        current_museum_name: str | None,
        search_museum_slug: str,
        search_museum_id: str | None,
        search_museum_name: str | None,
    ) -> ChatSearchScope:
        current_key = (current_museum_id or current_museum_slug or "").strip()
        search_key = (search_museum_id or search_museum_slug or "").strip()
        return ChatSearchScope(
            museum_id=search_museum_id,
            museum_slug=search_museum_slug,
            museum_name=search_museum_name or current_museum_name,
            is_cross_museum=bool(current_key and search_key and current_key != search_key),
        )

    def _search_scope_from_request(
        self,
        request: dict[str, Any],
        *,
        fallback_museum_slug: str,
        fallback_museum_id: str | None,
    ) -> ChatSearchScope | None:
        raw_scope = request.get("search_scope")
        if isinstance(raw_scope, ChatSearchScope):
            return raw_scope
        if isinstance(raw_scope, dict):
            museum_slug = str(raw_scope.get("museum_slug") or "").strip()
            if museum_slug:
                return ChatSearchScope(
                    museum_id=str(raw_scope.get("museum_id") or "").strip() or None,
                    museum_slug=museum_slug,
                    museum_name=str(raw_scope.get("museum_name") or "").strip() or None,
                    is_cross_museum=bool(raw_scope.get("is_cross_museum")),
                )
        if not fallback_museum_slug:
            return None
        return ChatSearchScope(
            museum_id=fallback_museum_id,
            museum_slug=fallback_museum_slug,
            museum_name=None,
            is_cross_museum=False,
        )

    def _request_museum_slug(self, payload: Any, request: dict[str, Any]) -> str:
        return str(request.get("museum_slug") or getattr(payload, "museum_slug", "") or "").strip()

    def _request_museum_id(self, payload: Any, request: dict[str, Any]) -> str | None:
        return str(request.get("museum_id") or getattr(payload, "museum_id", "") or "").strip() or None

    def _selected_artifact_reference_terms(
        self,
        doc: dict[str, object],
        selected_artifact: dict[str, str | None],
    ) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        def _append(value: object) -> None:
            if isinstance(value, list):
                for item in value:
                    _append(item)
                return
            text = str(value or "").strip()
            if not text:
                return
            folded = self._fold_query_text(text)
            if not folded or folded in seen:
                return
            seen.add(folded)
            terms.append(text)

        for key in (
            "title",
            "category",
            "super_category",
            "support_or_material",
            "technique",
            "date_or_period",
            "creator",
            "creators",
        ):
            _append(doc.get(key))
        _append(selected_artifact.get("title"))
        return terms[:12]

    def _build_anchored_similarity_query(
        self,
        *,
        query: str,
        reference_doc: dict[str, object],
        selected_artifact: dict[str, str | None],
    ) -> str:
        reference_terms = self._selected_artifact_reference_terms(
            reference_doc,
            selected_artifact,
        )
        parts = [str(query or "").strip(), *reference_terms]
        return " ".join(part for part in parts if part).strip() or "objetos semelhantes"

    async def _retrieve_selected_artifact_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        selected_artifact: dict[str, str | None],
        query: str,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]]:
        artifact_id = str(selected_artifact.get("artifact_id") or "").strip()
        retrieval_request: dict[str, Any] = {
            "kind": "selected_artifact",
            "museum_id": museum_id,
            "museum_slug": museum_slug,
            "original_query": query,
            "search_query": query,
            "query_text": query,
            "lexical_query": query,
            "selected_artifact": dict(selected_artifact),
            "selected_context_mode": "selected_artifact",
            "filters": {"artifact_id": artifact_id} if artifact_id else {},
            "sort": {},
            "results_total": 0,
        }
        if not artifact_id:
            return "", 0, [], retrieval_request

        try:
            docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
                museum_slug=museum_slug,
                museum_id=museum_id,
                artifact_ids=[artifact_id],
                top_k=1,
            )
        except Exception:
            docs = []

        retrieval_request["results_total"] = len(docs)
        if not docs:
            context = "\n".join(
                [
                    "selected_result_mode: true",
                    "selected_artifact_found: false",
                    f"selected_artifact_id: {artifact_id}",
                    "Instruction: The user selected one result card, but the artifact was not found in retrieval. Do not answer from unrelated collection results.",
                ]
            )
            return context, 0, [], retrieval_request

        docs_context = self._format_docs_for_prompt(docs=docs, top_k=1)
        context = "\n".join(
            [
                "selected_result_mode: true",
                "selected_artifact_found: true",
                "Instruction: The user selected exactly one result card in the UI. Answer ONLY about selected_artifact. If the requested fact is not present in selected_artifact, say that the available record does not contain that information. Do not use other search results as factual context.",
                "selected_artifact:",
                docs_context,
            ]
        ).strip()
        return context, len(docs), docs, retrieval_request

    async def _retrieve_anchored_similarity_context(
        self,
        *,
        selected_artifact: dict[str, str | None],
        query: str,
        filters: dict[str, object],
        sort: dict[str, object],
        result_window_size: int | None,
        reference_museum_slug: str | None = None,
        reference_museum_id: str | None = None,
        search_museum_slug: str | None = None,
        search_museum_id: str | None = None,
        museum_slug: str | None = None,
        museum_id: str | None = None,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]]:
        reference_museum_slug = (reference_museum_slug or museum_slug or "").strip()
        reference_museum_id = (reference_museum_id or museum_id or "").strip() or None
        search_museum_slug = (search_museum_slug or museum_slug or reference_museum_slug).strip()
        search_museum_id = (search_museum_id or museum_id or reference_museum_id or "").strip() or None
        artifact_id = str(selected_artifact.get("artifact_id") or "").strip()
        retrieval_request: dict[str, Any] = {
            "kind": "anchored_similarity",
            "museum_id": search_museum_id,
            "museum_slug": search_museum_slug,
            "reference_museum_id": reference_museum_id,
            "reference_museum_slug": reference_museum_slug,
            "original_query": query,
            "search_query": query,
            "query_text": query,
            "lexical_query": query,
            "selected_artifact": dict(selected_artifact),
            "selected_context_mode": "anchored_similarity",
            "filters": dict(filters),
            "sort": dict(sort),
            "results_total": 0,
        }
        if not artifact_id:
            return "", 0, [], retrieval_request

        try:
            reference_docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
                museum_slug=reference_museum_slug,
                museum_id=reference_museum_id,
                artifact_ids=[artifact_id],
                top_k=1,
            )
        except Exception:
            reference_docs = []

        if not reference_docs:
            context = "\n".join(
                [
                    "selected_result_mode: true",
                    "selected_context_mode: anchored_similarity",
                    "reference_artifact_found: false",
                    f"selected_artifact_id: {artifact_id}",
                    "Instruction: The user selected one result card as a reference for finding similar objects, but the reference artifact was not found. Do not answer from unrelated collection results.",
                ]
            )
            return context, 0, [], retrieval_request

        reference_doc = reference_docs[0]
        anchored_query = self._build_anchored_similarity_query(
            query=query,
            reference_doc=reference_doc,
            selected_artifact=selected_artifact,
        )
        retrieval_window = (
            max(int(result_window_size), 1) + 1
            if result_window_size is not None
            else None
        )
        (
            _candidate_context,
            _candidate_total,
            candidate_docs,
            candidate_request,
        ) = await self._retrieve_context(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            query=anchored_query,
            filters=filters,
            sort=sort,
            result_window_size=retrieval_window,
        )

        filtered_docs = [
            doc
            for doc in candidate_docs
            if str(doc.get("artifact_id") or "").strip() != artifact_id
        ]
        if result_window_size is not None:
            filtered_docs = filtered_docs[: max(int(result_window_size), 1)]
        retrieval_request = {
            **candidate_request,
            "kind": "anchored_similarity",
            "original_query": query,
            "anchor_query": anchored_query,
            "selected_artifact": dict(selected_artifact),
            "selected_context_mode": "anchored_similarity",
            "reference_artifact_found": True,
            "reference_artifact_id": artifact_id,
            "results_total": len(filtered_docs),
        }

        reference_context = self._format_docs_for_prompt(
            docs=reference_docs,
            top_k=1,
        )
        candidate_context = self._format_docs_for_prompt(
            docs=filtered_docs,
            top_k=max(len(filtered_docs), 1),
        )
        context_parts = [
            "selected_result_mode: true",
            "selected_context_mode: anchored_similarity",
            "Instruction: The user selected reference_artifact in the UI and is asking for other/similar objects. Use reference_artifact only as the comparison anchor. Answer about similar_candidate_results only, and do not include the reference artifact as a candidate.",
            "reference_artifact:",
            reference_context,
            "similar_candidate_results:",
            candidate_context or "No similar candidate results were found.",
        ]
        return "\n".join(context_parts).strip(), len(filtered_docs), filtered_docs, retrieval_request

    def _log_retrieved_artifact_entries(
        self,
        docs: list[dict[str, Any]],
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for doc in docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            entries.append(
                {
                    "artifact_id": artifact_id,
                    "inventory_number": doc.get("inventory_number") or None,
                    "title": doc.get("title") or None,
                    "score": doc.get("score"),
                    "source": source,
                }
            )
        return entries

    def _log_shown_artifact_entries(
        self,
        artifact_results: list[ArtifactResult],
    ) -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": artifact.artifact_id,
                "inventory_number": artifact.inventory_number,
                "title": artifact.title,
            }
            for artifact in artifact_results
        ]

    def _log_navigation_target_entries(
        self,
        *,
        navigation_targets: list[TourNavigationTarget],
        artifact_results: list[ArtifactResult],
    ) -> list[dict[str, Any]]:
        artifact_id_by_inventory = {
            str(artifact.inventory_number or "").strip().casefold(): artifact.artifact_id
            for artifact in artifact_results
            if str(artifact.inventory_number or "").strip()
        }
        entries: list[dict[str, Any]] = []
        for target in navigation_targets:
            inventory_key = str(target.inventory_id or "").strip().casefold()
            entries.append(
                {
                    "artifact_id": artifact_id_by_inventory.get(inventory_key),
                    "inventory_number": target.inventory_id,
                    "has_navigation_target": True,
                    "panorama_id": target.panorama_key,
                    "overlay_id": target.overlay_id,
                }
            )
        return entries

    async def _emit_query_log(
        self,
        *,
        payload: Any,
        conversation_id: str,
        query_id: str,
        language: str | None,
        raw_query: str | None,
        resolved_query: str | None,
        lexical_query: str | None,
        embedding_query: str | None,
        route: str | None,
        retrieval_mode: str,
        filters_applied: dict[str, Any] | None,
        boosts_applied: dict[str, Any] | None,
        retrieved_artifacts: list[dict[str, Any]],
        shown_artifacts: list[dict[str, Any]],
        navigation_targets: list[dict[str, Any]],
        latency_retrieval_ms: float | None,
        latency_llm_ms: float | None,
        latency_total_ms: float | None,
        status: str,
        error: str | None,
        opensearch_query: dict[str, Any] | None = None,
    ) -> None:
        metadata = getattr(payload, "metadata", None)
        museum_slug = getattr(payload, "museum_slug", None)
        correlation = self._log_correlation_ids(metadata=metadata, museum_slug=museum_slug)
        event: dict[str, Any] = {
            "event_type": "backend_query",
            "timestamp": utc_timestamp(),
            "session_id": correlation["session_id"],
            "conversation_id": conversation_id,
            "query_id": query_id,
            "participant_id": correlation["participant_id"],
            "task_id": correlation["task_id"],
            "tour_id": correlation["tour_id"],
            "museum_slug": museum_slug,
            "language": language,
            "raw_query": raw_query,
            "resolved_query": resolved_query,
            "lexical_query": lexical_query,
            "embedding_query": embedding_query,
            "route": route,
            "retrieval_mode": retrieval_mode,
            "filters_applied": filters_applied or {},
            "boosts_applied": boosts_applied or {},
            "retrieved_artifacts": retrieved_artifacts,
            "shown_artifacts": shown_artifacts,
            "navigation_targets": navigation_targets,
            "latency_retrieval_ms": latency_retrieval_ms,
            "latency_llm_ms": latency_llm_ms,
            "latency_total_ms": latency_total_ms,
            "status": status,
            "error": error,
            "opensearch_query": opensearch_query,
        }
        await self.query_logger.log_event(event)

    def _matched_retrieval_boosts_for_log(
        self,
        *,
        query_text: str | None,
        lexical_query: str | None,
    ) -> list[dict[str, Any]]:
        if self.settings.CHAT_RETRIEVAL_EMBEDDING_ONLY:
            return []
        boost_resolver = getattr(self.opensearch_gateway, "matched_retrieval_boosts", None)
        if not callable(boost_resolver):
            return []
        try:
            boosts = boost_resolver(
                query_text=str(query_text or ""),
                lexical_query=lexical_query,
            )
        except Exception:
            return []
        return boosts if isinstance(boosts, list) else []

    def _text_boosts_applied_for_log(
        self,
        *,
        router_decision: dict[str, object],
        retrieval_request: dict[str, Any],
    ) -> dict[str, Any]:
        if router_decision.get("mode") != "rag":
            return {}
        if retrieval_request.get("kind") == "selected_artifact":
            return {}

        boosts: dict[str, Any] = {"in_tour_boost": self.settings.CHAT_IN_TOUR_BOOST}
        retrieval_boosts = retrieval_request.get("retrieval_boosts")
        if isinstance(retrieval_boosts, list) and retrieval_boosts:
            boosts["retrieval_boosts"] = retrieval_boosts
        return boosts

    def _rag_debug_json(self, event: str, payload: dict[str, Any]) -> None:
        if not self.settings.BACKEND_RAG_DEBUG_ENABLED:
            return
        log_json_event(
            logger,
            logging.INFO,
            event,
            payload,
            max_chars=max(int(self.settings.BACKEND_RAG_DEBUG_MAX_CHARS), 1000),
        )

    def _rag_debug_docs(self, docs: list[dict[str, object]], *, limit: int = 8) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for index, doc in enumerate(docs[: max(limit, 0)], start=1):
            entries.append(
                {
                    "rank": index,
                    "artifact_id": doc.get("artifact_id"),
                    "inventory_number": doc.get("inventory_number") or doc.get("inventory"),
                    "title": doc.get("title"),
                    "score": doc.get("score"),
                    "rerank_score": doc.get("rerank_score"),
                    "museum_id": doc.get("museum_id"),
                    "category": doc.get("category"),
                    "in_tour": doc.get("in_tour"),
                    "snippet": doc.get("snippet"),
                }
            )
        return entries

    async def handle_message(
        self,
        payload: ChatMessageRequest,
        *,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        conversation_id = payload.conversation_id or str(uuid4())
        requested_format = payload.response_format or ResponseFormatObject(type="text")
        query_id = str(uuid4())
        request_started_at = time.monotonic()
        retrieval_latency_ms: float | None = None
        llm_latency_ms: float | None = None

        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        self.session_store.append_turn(state, role="user", text=payload.message)
        log_event(
            logger,
            logging.INFO,
            "chat.message.start",
            conversation_id=conversation_id,
            query_id=query_id,
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            language=language,
            response_format=requested_format.type,
            message_chars=len((payload.message or "").strip()),
            has_metadata=bool(payload.metadata),
        )
        await self._emit_status(status_cb, "status.analyzing_request", language=language)

        context_policy = self._derive_context_policy(
            message=payload.message,
            state=state,
        )

        try:
            router_decision = await self._route_message(
                payload=payload,
                state=state,
                context_policy=context_policy,
            )
        except LLMServiceError:
            router_decision = self._fallback_router_decision(
                payload.message,
                context_policy=context_policy,
            )
        router_decision = self._apply_context_policy_guardrails(
            router_decision=router_decision,
            context_policy=context_policy,
            user_message=payload.message,
        )
        rewritten_query = str(router_decision.get("rewritten_query", payload.message)).strip()
        original_message = (payload.message or "").strip()
        current_museum_id = self._resolve_museum_id(payload)
        requested_museum = self._resolve_requested_search_museum(original_message)
        search_museum_slug = requested_museum.museum_slug if requested_museum else payload.museum_slug
        search_museum_id = requested_museum.museum_id if requested_museum else current_museum_id
        search_museum_name = (
            requested_museum.museum_name
            if requested_museum
            else payload.museum_name
        )
        search_query_message = self._strip_search_museum_expression_from_query(
            original_message,
            requested_museum,
        )
        search_scope = self._build_search_scope(
            current_museum_slug=payload.museum_slug,
            current_museum_id=current_museum_id,
            current_museum_name=payload.museum_name,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            search_museum_name=search_museum_name,
        )
        selected_artifact = self._selected_artifact_from_metadata(
            payload.metadata if isinstance(payload.metadata, dict) else None
        )
        selected_context_mode = (
            self._selected_artifact_context_mode(payload.message)
            if selected_artifact
            else None
        )
        reference_museum_slug = payload.museum_slug
        reference_museum_id = current_museum_id
        reference_museum_name = payload.museum_name
        if selected_artifact:
            reference_museum_slug = str(selected_artifact.get("museum_slug") or payload.museum_slug).strip() or payload.museum_slug
            reference_museum_id = str(selected_artifact.get("museum_id") or current_museum_id or "").strip() or None
            reference_museum_name = str(selected_artifact.get("museum_name") or payload.museum_name or "").strip() or payload.museum_name
        if (
            selected_artifact
            and requested_museum is not None
            and selected_context_mode == "selected_artifact"
            and self._is_targeted_artifact_search_request(payload.message)
        ):
            selected_context_mode = "anchored_similarity"
        if selected_artifact and selected_context_mode == "selected_artifact":
            search_museum_slug = reference_museum_slug
            search_museum_id = reference_museum_id
            search_museum_name = reference_museum_name
        elif (
            selected_artifact
            and selected_context_mode == "anchored_similarity"
            and requested_museum is None
        ):
            search_museum_slug = reference_museum_slug
            search_museum_id = reference_museum_id
            search_museum_name = reference_museum_name
        if selected_artifact:
            search_scope = self._build_search_scope(
                current_museum_slug=payload.museum_slug,
                current_museum_id=current_museum_id,
                current_museum_name=payload.museum_name,
                search_museum_slug=search_museum_slug,
                search_museum_id=search_museum_id,
                search_museum_name=search_museum_name,
            )
        if selected_artifact:
            router_decision = {
                **router_decision,
                "mode": "rag",
                "intent": (
                    "anchored_similarity_search"
                    if selected_context_mode == "anchored_similarity"
                    else "selected_artifact_question"
                ),
                "use_history_for_query": False,
                "use_history_for_answer": False,
                "carry_filters": False,
                "carry_sort": False,
            }

        log_event(
            logger,
            logging.INFO,
            "chat.message.route",
            conversation_id=conversation_id,
            query_id=query_id,
            mode=router_decision.get("mode"),
            intent=router_decision.get("intent"),
            rewritten_query_chars=len(rewritten_query),
            selected_artifact=bool(selected_artifact),
            selected_context_mode=selected_context_mode,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
        )
        self._rag_debug_json(
            "rag.router_decision",
            {
                "conversation_id": conversation_id,
                "query_id": query_id,
                "input": {
                    "museum_slug": payload.museum_slug,
                    "museum_id": payload.museum_id,
                    "language": language,
                    "message": original_message,
                    "message_chars": len(original_message),
                    "has_metadata": bool(payload.metadata),
                },
                "router_decision": router_decision,
                "context_policy": context_policy,
                "selected_artifact": selected_artifact,
                "selected_context_mode": selected_context_mode,
                "search_scope": search_scope.model_dump(mode="json"),
            },
        )

        carry_filters = bool(router_decision.get("carry_filters", False))
        carry_sort = bool(router_decision.get("carry_sort", False))
        use_history_for_answer = bool(router_decision.get("use_history_for_answer", False))

        base_filters = state.filters if carry_filters else {}
        base_sort = state.sort if carry_sort else {}
        effective_filters = self._merge_state_with_delta(
            base=base_filters,
            delta=router_decision.get("filters_delta"),
        )
        effective_sort = self._merge_state_with_delta(
            base=base_sort,
            delta=router_decision.get("sort_delta"),
        )

        retrieval_context = ""
        retrieved_docs_count = 0
        retrieved_docs: list[dict[str, object]] = []
        retrieval_request: dict[str, Any] = {}
        image_matches: list[ImageMatchResult] = []
        artifact_results: list[ArtifactResult] = []
        artifact_image_hits: list[dict[str, Any]] = []
        text_results_default_page_size = self._text_results_default_page_size()
        text_results_page = max(payload.results_page, 1)
        text_results_page_size = self._resolve_results_page_size(
            payload.results_page_size,
            default_size=text_results_default_page_size,
        )
        text_results_window_size = min(text_results_page * text_results_page_size, 50)
        if (
            selected_artifact
            and selected_context_mode == "selected_artifact"
            and self.settings.CHAT_ENABLE_RAG
        ):
            await self._emit_status(status_cb, "status.searching_collection", language=language)
            retrieval_started_at = time.monotonic()
            (
                retrieval_context,
                retrieved_docs_count,
                retrieved_docs,
                retrieval_request,
            ) = await self._retrieve_selected_artifact_context(
                museum_slug=reference_museum_slug,
                museum_id=reference_museum_id,
                selected_artifact=selected_artifact,
                query=payload.message,
            )
            retrieval_request["search_scope"] = search_scope.model_dump(mode="json")
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=retrieved_docs_count,
            )
            if retrieved_docs:
                artifact_ids = [
                    str(doc.get("artifact_id") or "").strip()
                    for doc in retrieved_docs
                    if str(doc.get("artifact_id") or "").strip()
                ]
                if artifact_ids:
                    try:
                        artifact_image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                            museum_slug=reference_museum_slug,
                            museum_id=reference_museum_id,
                            artifact_ids=artifact_ids,
                            per_artifact=1,
                            max_total=max(len(artifact_ids), 1),
                        )
                    except Exception as exc:
                        log_event(
                            logger,
                            logging.WARNING,
                            "chat.message.thumbnail_fetch.error",
                            conversation_id=conversation_id,
                            query_id=query_id,
                            artifact_count=len(artifact_ids),
                            error=exc,
                        )
                        artifact_image_hits = []
                    image_matches = self._build_image_matches(
                        image_hits=artifact_image_hits,
                        artifact_docs=retrieved_docs,
                    )
        elif (
            selected_artifact
            and selected_context_mode == "anchored_similarity"
            and self.settings.CHAT_ENABLE_RAG
        ):
            await self._emit_status(status_cb, "status.searching_collection", language=language)
            retrieval_started_at = time.monotonic()
            (
                retrieval_context,
                retrieved_docs_count,
                retrieved_docs,
                retrieval_request,
            ) = await self._retrieve_anchored_similarity_context(
                reference_museum_slug=reference_museum_slug,
                reference_museum_id=reference_museum_id,
                search_museum_slug=search_museum_slug,
                search_museum_id=search_museum_id,
                selected_artifact=selected_artifact,
                query=search_query_message,
                filters=effective_filters,
                sort=effective_sort,
                result_window_size=text_results_window_size,
            )
            retrieval_request["search_scope"] = search_scope.model_dump(mode="json")
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=retrieved_docs_count,
            )
            if retrieved_docs:
                artifact_ids = [
                    str(doc.get("artifact_id") or "").strip()
                    for doc in retrieved_docs
                    if str(doc.get("artifact_id") or "").strip()
                ]
                if artifact_ids:
                    try:
                        artifact_image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                            museum_slug=search_museum_slug,
                            museum_id=search_museum_id,
                            artifact_ids=artifact_ids,
                            per_artifact=1,
                            max_total=max(len(artifact_ids), 1),
                        )
                    except Exception as exc:
                        log_event(
                            logger,
                            logging.WARNING,
                            "chat.message.thumbnail_fetch.error",
                            conversation_id=conversation_id,
                            query_id=query_id,
                            artifact_count=len(artifact_ids),
                            error=exc,
                        )
                        artifact_image_hits = []
                    image_matches = self._build_image_matches(
                        image_hits=artifact_image_hits,
                        artifact_docs=retrieved_docs,
                    )
        elif router_decision["mode"] == "rag" and self.settings.CHAT_ENABLE_RAG:
            # Retrieval must stay strict to user wording (no expansion/rewrite additions).
            retrieval_query = search_query_message
            await self._emit_status(status_cb, "status.searching_collection", language=language)
            retrieval_started_at = time.monotonic()
            (
                retrieval_context,
                retrieved_docs_count,
                retrieved_docs,
                retrieval_request,
            ) = await self._retrieve_context(
                museum_slug=search_museum_slug,
                museum_id=search_museum_id,
                query=retrieval_query,
                filters=effective_filters,
                sort=effective_sort,
                result_window_size=text_results_window_size,
            )
            retrieval_request["museum_slug"] = search_museum_slug
            retrieval_request["search_scope"] = search_scope.model_dump(mode="json")
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=retrieved_docs_count,
            )
            if retrieved_docs:
                artifact_ids = [
                    str(doc.get("artifact_id") or "").strip()
                    for doc in retrieved_docs
                    if str(doc.get("artifact_id") or "").strip()
                ]
                if artifact_ids:
                    try:
                        artifact_image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                            museum_slug=search_museum_slug,
                            museum_id=search_museum_id,
                            artifact_ids=artifact_ids,
                            per_artifact=1,
                            max_total=max(len(artifact_ids), 1),
                        )
                    except Exception as exc:
                        log_event(
                            logger,
                            logging.WARNING,
                            "chat.message.thumbnail_fetch.error",
                            conversation_id=conversation_id,
                            query_id=query_id,
                            artifact_count=len(artifact_ids),
                            error=exc,
                        )
                        artifact_image_hits = []
                    image_matches = self._build_image_matches(
                        image_hits=artifact_image_hits,
                        artifact_docs=retrieved_docs,
                    )
        else:
            pass
        log_event(
            logger,
            logging.INFO,
            "chat.message.retrieval.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            mode=router_decision.get("mode"),
            retrieval_kind=retrieval_request.get("kind"),
            retrieved_docs_count=retrieved_docs_count,
            image_matches=len(image_matches),
            duration_ms=round(retrieval_latency_ms, 1) if retrieval_latency_ms is not None else None,
        )
        self._rag_debug_json(
            "rag.retrieval_result",
            {
                "conversation_id": conversation_id,
                "query_id": query_id,
                "route": router_decision.get("mode"),
                "intent": router_decision.get("intent"),
                "search_scope": search_scope.model_dump(mode="json"),
                "effective_filters": effective_filters,
                "effective_sort": effective_sort,
                "latency_ms": round(retrieval_latency_ms, 1) if retrieval_latency_ms is not None else None,
                "context_chars": len(retrieval_context),
                "retrieved_docs_count": retrieved_docs_count,
                "retrieval_request": retrieval_request,
                "retrieved_docs": self._rag_debug_docs(retrieved_docs),
                "thumbnail_image_hits": self._rag_debug_docs(artifact_image_hits),
                "image_matches_count": len(image_matches),
            },
        )
        # Build the full artifact records (all images) so the detail modal matches
        # the image/model search behaviour. `artifact_image_hits` above only carries
        # one image per artifact (for the result-card thumbnails), so passing it here
        # would cap the modal gallery at a single image.
        artifact_results = await self._build_artifact_results(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            artifact_docs=retrieved_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            docs=retrieved_docs,
        )
        image_matches = self._enrich_image_matches(
            context="text",
            conversation_id=conversation_id,
            museum_slug=search_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        (
            paged_artifact_results,
            paged_image_matches,
            paged_navigation_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._build_paged_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=text_results_default_page_size,
            total_override=retrieved_docs_count,
            retrieval_request=retrieval_request,
        )
        results_request_id = str(
            state.last_paged_retrieval_request.get("results_request_id") or ""
        ).strip() or None
        visible_retrieval_context = self._build_visible_results_retrieval_context(
            artifact_results=paged_artifact_results,
            image_matches=[],
            page=results_page,
            page_size=results_page_size,
            total=results_total,
            prefer_artifact_results=True,
            include_image_matches_section=False,
        )
        retrieval_kind_for_log = str(retrieval_request.get("kind") or "")
        retrieval_mode_for_log = (
            retrieval_kind_for_log
            if retrieval_kind_for_log in {"selected_artifact", "anchored_similarity"}
            else ("hybrid_text" if router_decision.get("mode") == "rag" else "none")
        )
        retrieved_source_for_log = (
            retrieval_kind_for_log
            if retrieval_kind_for_log in {"selected_artifact", "anchored_similarity"}
            else "hybrid"
        )
        final_retrieval_context = (
            retrieval_context
            if retrieval_kind_for_log == "anchored_similarity"
            else (visible_retrieval_context or retrieval_context)
        )
        retry_context_result_count = (
            0
            if retrieval_kind_for_log == "anchored_similarity"
            else len(paged_artifact_results)
        )
        self._rag_debug_json(
            "rag.visible_context",
            {
                "conversation_id": conversation_id,
                "query_id": query_id,
                "retrieval_kind": retrieval_kind_for_log,
                "retrieval_mode": retrieval_mode_for_log,
                "results": {
                    "page": results_page,
                    "page_size": results_page_size,
                    "total": results_total,
                    "has_more": results_has_more,
                    "shown_artifacts": len(paged_artifact_results),
                    "shown_images": len(paged_image_matches),
                    "navigation_targets": len(paged_navigation_targets),
                },
                "visible_context_chars": len(visible_retrieval_context),
                "final_retrieval_context_chars": len(final_retrieval_context),
                "artifact_results": [
                    result.model_dump(mode="json")
                    for result in paged_artifact_results[:8]
                ],
                "image_matches": [
                    result.model_dump(mode="json")
                    for result in paged_image_matches[:8]
                ],
                "navigation_targets": [
                    result.model_dump(mode="json")
                    for result in paged_navigation_targets[:8]
                ],
            },
        )

        def build_text_final_message_for_result_count(result_count: int) -> str:
            limited_visible_retrieval_context = self._build_visible_results_retrieval_context(
                artifact_results=paged_artifact_results[: max(result_count, 0)],
                image_matches=[],
                page=results_page,
                page_size=results_page_size,
                total=results_total,
                prefer_artifact_results=True,
                include_image_matches_section=False,
            )
            limited_final_retrieval_context = (
                retrieval_context
                if retrieval_kind_for_log == "anchored_similarity"
                else limited_visible_retrieval_context
            )
            return self._build_final_prompt(
                payload=payload,
                state=state,
                router_decision=router_decision,
                retrieval_context=limited_final_retrieval_context,
                effective_filters=effective_filters,
                effective_sort=effective_sort,
                use_history_for_answer=use_history_for_answer,
                museum_slug=search_museum_slug,
                museum_name=search_museum_name,
            )

        final_message = self._build_final_prompt(
            payload=payload,
            state=state,
            router_decision=router_decision,
            retrieval_context=final_retrieval_context,
            effective_filters=effective_filters,
            effective_sort=effective_sort,
            use_history_for_answer=use_history_for_answer,
            museum_slug=search_museum_slug,
            museum_name=search_museum_name,
        )
        self._rag_debug_json(
            "rag.final_prompt",
            {
                "conversation_id": conversation_id,
                "query_id": query_id,
                "model_override": payload.model_override,
                "response_format": requested_format.model_dump(mode="json"),
                "final_prompt_chars": len(final_message),
                "retrieval_context_chars": len(final_retrieval_context),
                "retry_context_result_count": retry_context_result_count,
                "use_history_for_answer": use_history_for_answer,
                "router_decision": router_decision,
            },
        )
        await self._emit_status(status_cb, "status.generating_final_answer", language=language)

        llm_started_at = time.monotonic()
        try:
            llm_response = await self._generate_final_answer_with_context_retries(
                initial_message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
                visible_result_count=retry_context_result_count,
                build_message_for_result_count=build_text_final_message_for_result_count,
            )
        except LLMServiceError as exc:
            llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.message.llm.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round(llm_latency_ms, 1),
                error=exc,
            )
            # Soft-fail in dev so frontend keeps moving while LLM infra is still unstable.
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=original_message,
                resolved_query=rewritten_query,
                lexical_query=retrieval_request.get("lexical_query"),
                embedding_query=retrieval_request.get("query_text"),
                route=str(router_decision.get("mode")),
                retrieval_mode=retrieval_mode_for_log,
                filters_applied={
                    "filters": effective_filters,
                    "sort": effective_sort,
                    "temporal_query": retrieval_request.get("temporal_query"),
                    "tour_scope": retrieval_request.get("tour_scope"),
                    "selected_artifact": retrieval_request.get("selected_artifact"),
                    "selected_context_mode": retrieval_request.get("selected_context_mode"),
                    "target_museum": retrieval_request.get("search_scope"),
                },
                boosts_applied=self._text_boosts_applied_for_log(
                    router_decision=router_decision,
                    retrieval_request=retrieval_request,
                ),
                retrieved_artifacts=self._log_retrieved_artifact_entries(retrieved_docs, source=retrieved_source_for_log),
                shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
                navigation_targets=self._log_navigation_target_entries(
                    navigation_targets=paged_navigation_targets,
                    artifact_results=paged_artifact_results,
                ),
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=llm_latency_ms,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
                opensearch_query=retrieval_request.get("opensearch_query"),
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override
                or self.settings.llm_model_resolved,
                image_matches=paged_image_matches,
                artifact_results=paged_artifact_results,
                navigation_targets=paged_navigation_targets,
                results_page=results_page,
                results_page_size=results_page_size,
                results_total=results_total,
                results_has_more=results_has_more,
                results_request_id=results_request_id,
                search_scope=search_scope,
            )
        llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.message.llm.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            model=llm_response.model,
            response_format=llm_response.response_format.type,
            duration_ms=round(llm_latency_ms, 1),
        )

        reply_docs = self._artifact_results_as_prompt_docs(paged_artifact_results) or retrieved_docs
        final_reply_text = self._sanitize_assistant_reply(
            llm_response.text,
            docs=reply_docs,
            language=language,
        )

        if not carry_filters:
            state.filters = {}
        if not carry_sort:
            state.sort = {}

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(
            state=state,
            artifact_docs=retrieved_docs,
            preserve_existing_when_empty=bool(router_decision.get("use_history_for_query", False)),
        )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=payload.message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=original_message,
            resolved_query=rewritten_query,
            lexical_query=retrieval_request.get("lexical_query"),
            embedding_query=retrieval_request.get("query_text"),
            route=str(router_decision.get("mode")),
            retrieval_mode=retrieval_mode_for_log,
            filters_applied={
                "filters": effective_filters,
                "sort": effective_sort,
                "temporal_query": retrieval_request.get("temporal_query"),
                "tour_scope": retrieval_request.get("tour_scope"),
                "selected_artifact": retrieval_request.get("selected_artifact"),
                "selected_context_mode": retrieval_request.get("selected_context_mode"),
                "target_museum": retrieval_request.get("search_scope"),
            },
            boosts_applied=self._text_boosts_applied_for_log(
                router_decision=router_decision,
                retrieval_request=retrieval_request,
            ),
            retrieved_artifacts=self._log_retrieved_artifact_entries(retrieved_docs, source=retrieved_source_for_log),
            shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
            navigation_targets=self._log_navigation_target_entries(
                navigation_targets=paged_navigation_targets,
                artifact_results=paged_artifact_results,
            ),
            latency_retrieval_ms=retrieval_latency_ms,
            latency_llm_ms=llm_latency_ms,
            latency_total_ms=(time.monotonic() - request_started_at) * 1000,
            status="ok",
            error=None,
            opensearch_query=retrieval_request.get("opensearch_query"),
        )

        log_event(
            logger,
            logging.INFO,
            "chat.message.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status="ok",
            results_total=results_total,
            shown_artifacts=len(paged_artifact_results),
            shown_images=len(paged_image_matches),
            duration_ms=round((time.monotonic() - request_started_at) * 1000, 1),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=paged_image_matches,
            artifact_results=paged_artifact_results,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    async def regenerate_last_reply(
        self,
        payload: ChatRegenerateRequest,
        *,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        state = self.session_store.load_or_create(
            conversation_id=payload.conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        if not state.history:
            raise ValueError(translate("error.no_history_regenerate", language))

        while state.history and state.history[-1].role == "assistant":
            state.history.pop()

        if not state.history or state.history[-1].role != "user":
            raise ValueError(translate("error.no_user_message_regenerate", language))

        last_user_message = state.history[-1].text
        state.history.pop()
        self.session_store.save(state)

        regenerated_payload = ChatMessageRequest(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            museum_name=payload.museum_name,
            language=language,
            message=last_user_message,
            conversation_id=payload.conversation_id,
            response_format=payload.response_format,
            system_prompt=payload.system_prompt,
            model_override=payload.model_override,
            metadata=payload.metadata,
        )
        return await self.handle_message(regenerated_payload, status_cb=status_cb)

    async def handle_image_message(
        self,
        payload: ChatImageMessageRequest,
        *,
        image_bytes: bytes,
        image_filename: str | None = None,
        image_content_type: str | None = None,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        conversation_id = payload.conversation_id or str(uuid4())
        requested_format = payload.response_format or ResponseFormatObject(type="text")
        query_id = str(uuid4())
        request_started_at = time.monotonic()
        retrieval_latency_ms: float | None = None
        llm_latency_ms: float | None = None
        explicit_museum_id = self._resolve_explicit_museum_id_values(
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        museum_id = self._resolve_museum_id_values(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        user_message = (payload.message or "").strip() or translate(
            "message.default_image_query",
            language,
        )
        requested_museum = self._resolve_requested_search_museum(user_message)
        if requested_museum is not None:
            museum_id = requested_museum.museum_id
            explicit_museum_id = requested_museum.museum_id
        search_museum_slug = requested_museum.museum_slug if requested_museum else payload.museum_slug
        search_museum_id = museum_id
        search_museum_name = (
            requested_museum.museum_name
            if requested_museum
            else payload.museum_name
        )
        search_scope = self._build_search_scope(
            current_museum_slug=payload.museum_slug,
            current_museum_id=self._resolve_museum_id_values(
                museum_slug=payload.museum_slug,
                museum_id=payload.museum_id,
                metadata=payload.metadata,
            ),
            current_museum_name=payload.museum_name,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            search_museum_name=search_museum_name,
        )

        self.session_store.append_turn(state, role="user", text=user_message)
        log_event(
            logger,
            logging.INFO,
            "chat.image.start",
            conversation_id=conversation_id,
            query_id=query_id,
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            language=language,
            response_format=requested_format.type,
            image_filename=image_filename,
            image_content_type=image_content_type,
            image_bytes=len(image_bytes or b""),
            message_chars=len(user_message),
        )
        await self._emit_status(status_cb, "status.analyzing_image", language=language)

        image_matches: list[ImageMatchResult] = []
        artifact_docs: list[dict[str, object]] = []
        artifact_results: list[ArtifactResult] = []
        navigation_targets: list[TourNavigationTarget] = []
        image_results_total = 0
        image_opensearch_query: dict[str, Any] | None = None
        retrieval_started_at = time.monotonic()
        try:
            image_embedding = await self.embedding_provider.embed_multimodal_image_bytes(
                image_bytes=image_bytes,
                text=None,
            )
        except Exception as exc:
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.image.embedding.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round(retrieval_latency_ms, 1),
                error=exc,
            )
            fallback = translate("error.image_processing_failed", language)
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=user_message,
                resolved_query=user_message,
                lexical_query=None,
                embedding_query=None,
                route="image_search",
                retrieval_mode="image_similarity",
                filters_applied={
                    "filters": dict(state.filters),
                    "sort": dict(state.sort),
                    "target_museum": search_scope.model_dump(mode="json"),
                },
                boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
                retrieved_artifacts=[],
                shown_artifacts=[],
                navigation_targets=[],
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=None,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                search_scope=search_scope,
            )

        await self._emit_status(status_cb, "status.searching_collection", language=language)
        image_candidates_k = max(
            self.settings.CHAT_IMAGE_RETRIEVAL_TOP_K,
            self.settings.CHAT_IMAGE_ARTIFACT_TOP_K,
            self.settings.CHAT_RETRIEVAL_CANDIDATES,
            1,
        )
        image_retrieval_window_size = self._image_retrieval_window_size(
            minimum=image_candidates_k,
        )
        try:
            image_page = await self.opensearch_gateway.search_similar_images_page(
                museum_slug=search_museum_slug,
                museum_id=search_museum_id,
                image_embedding=image_embedding,
                from_offset=0,
                page_size=image_candidates_k,
                retrieval_window_size=image_retrieval_window_size,
            )
            image_opensearch_query = image_page.query_body
            image_hits = image_page.results
            image_results_total = self._bounded_retrieval_total(
                image_page.total,
                image_retrieval_window_size,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.image.search.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round((time.monotonic() - retrieval_started_at) * 1000, 1),
                error=exc,
            )
            image_hits = []

        if image_hits:
            try:
                artifact_docs = await self._fetch_artifact_docs_for_image_hits(
                    museum_slug=search_museum_slug,
                    museum_id=search_museum_id,
                    artifact_museum_id=explicit_museum_id,
                    image_hits=image_hits,
                    top_k=image_candidates_k,
                )
            except Exception as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "chat.image.artifact_fetch.error",
                    conversation_id=conversation_id,
                    query_id=query_id,
                    image_hits=len(image_hits),
                    error=exc,
                )
                artifact_docs = []

            image_matches = self._build_image_matches(image_hits=image_hits, artifact_docs=artifact_docs)
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=len(artifact_docs),
            )
        else:
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=0,
            )
        retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.image.retrieval.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            image_hits=len(image_hits),
            artifact_docs=len(artifact_docs),
            results_total=image_results_total,
            duration_ms=round(retrieval_latency_ms, 1),
        )

        artifact_results = await self._build_artifact_results(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context="image",
            conversation_id=conversation_id,
            museum_slug=search_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        (
            paged_artifact_results,
            paged_image_matches,
            paged_navigation_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._build_paged_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=max(self.settings.CHAT_IMAGE_ARTIFACT_TOP_K, 1),
            total_override=image_results_total,
            retrieval_request={
                "kind": "image",
                "museum_id": search_museum_id,
                "museum_slug": search_museum_slug,
                "artifact_museum_id": explicit_museum_id,
                "search_scope": search_scope.model_dump(mode="json"),
                "image_embedding": image_embedding,
                "retrieval_window_size": image_retrieval_window_size,
                "results_total": image_results_total,
                "opensearch_query": image_opensearch_query,
            },
        )
        results_request_id = str(
            state.last_paged_retrieval_request.get("results_request_id") or ""
        ).strip() or None

        retrieval_context = self._build_visible_results_retrieval_context(
            artifact_results=paged_artifact_results,
            image_matches=paged_image_matches,
            page=results_page,
            page_size=results_page_size,
            total=results_total,
            match_section_label="visible_image_retrieval_matches",
        )
        retry_context_result_count = (
            len(paged_image_matches) if paged_image_matches else len(paged_artifact_results)
        )

        def build_image_final_message_for_result_count(result_count: int) -> str:
            limited_retrieval_context = self._build_visible_results_retrieval_context(
                artifact_results=paged_artifact_results[: max(result_count, 0)],
                image_matches=paged_image_matches[: max(result_count, 0)],
                page=results_page,
                page_size=results_page_size,
                total=results_total,
                match_section_label="visible_image_retrieval_matches",
            )
            return build_final_answer_prompt(
                museum_slug=search_museum_slug,
                museum_name=search_museum_name,
                input_modality="image",
                mode="rag",
                intent="image_search",
                rolling_summary="",
                filters_state=effective_filters,
                sort_state=effective_sort,
                user_message=user_message,
                rewritten_query=user_message,
                retrieval_context=limited_retrieval_context,
                use_history_for_answer=False,
                language=language,
            )

        router_decision: dict[str, object] = {
            "mode": "rag",
            "intent": "image_search",
            "rewritten_query": user_message,
            "needs_retrieval": True,
            "reason": "image_upload",
            "filters_delta": {},
            "sort_delta": {},
        }
        effective_filters = dict(state.filters)
        effective_sort = dict(state.sort)

        final_message = build_final_answer_prompt(
            museum_slug=search_museum_slug,
            museum_name=search_museum_name,
            input_modality="image",
            mode="rag",
            intent="image_search",
            rolling_summary="",
            filters_state=effective_filters,
            sort_state=effective_sort,
            user_message=user_message,
            rewritten_query=user_message,
            retrieval_context=retrieval_context,
            use_history_for_answer=False,
            language=language,
        )
        await self._emit_status(status_cb, "status.generating_final_answer", language=language)

        llm_started_at = time.monotonic()
        try:
            llm_response = await self._generate_final_answer_with_context_retries(
                initial_message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
                visible_result_count=retry_context_result_count,
                build_message_for_result_count=build_image_final_message_for_result_count,
            )
        except LLMServiceError as exc:
            llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.image.llm.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round(llm_latency_ms, 1),
                error=exc,
            )
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=user_message,
                resolved_query=user_message,
                lexical_query=None,
                embedding_query=None,
                route="image_search",
                retrieval_mode="image_similarity",
                filters_applied={
                    "filters": effective_filters,
                    "sort": effective_sort,
                    "target_museum": search_scope.model_dump(mode="json"),
                },
                boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
                retrieved_artifacts=self._log_retrieved_artifact_entries(artifact_docs, source="image"),
                shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
                navigation_targets=self._log_navigation_target_entries(
                    navigation_targets=paged_navigation_targets,
                    artifact_results=paged_artifact_results,
                ),
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=llm_latency_ms,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
                opensearch_query=image_opensearch_query,
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                image_matches=paged_image_matches,
                artifact_results=paged_artifact_results,
                navigation_targets=paged_navigation_targets,
                results_page=results_page,
                results_page_size=results_page_size,
                results_total=results_total,
                results_has_more=results_has_more,
                results_request_id=results_request_id,
                search_scope=search_scope,
            )
        llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.image.llm.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            model=llm_response.model,
            response_format=llm_response.response_format.type,
            duration_ms=round(llm_latency_ms, 1),
        )

        reply_docs = self._artifact_results_as_prompt_docs(paged_artifact_results) or artifact_docs
        final_reply_text = self._sanitize_assistant_reply(
            llm_response.text,
            docs=reply_docs,
            language=language,
        )

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(
            state=state,
            artifact_docs=artifact_docs,
        )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=user_message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=user_message,
            resolved_query=user_message,
            lexical_query=None,
            embedding_query=None,
            route="image_search",
            retrieval_mode="image_similarity",
            filters_applied={
                "filters": effective_filters,
                "sort": effective_sort,
                "target_museum": search_scope.model_dump(mode="json"),
            },
            boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
            retrieved_artifacts=self._log_retrieved_artifact_entries(artifact_docs, source="image"),
            shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
            navigation_targets=self._log_navigation_target_entries(
                navigation_targets=paged_navigation_targets,
                artifact_results=paged_artifact_results,
            ),
            latency_retrieval_ms=retrieval_latency_ms,
            latency_llm_ms=llm_latency_ms,
            latency_total_ms=(time.monotonic() - request_started_at) * 1000,
            status="ok",
            error=None,
            opensearch_query=image_opensearch_query,
        )

        log_event(
            logger,
            logging.INFO,
            "chat.image.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status="ok",
            results_total=results_total,
            shown_artifacts=len(paged_artifact_results),
            shown_images=len(paged_image_matches),
            duration_ms=round((time.monotonic() - request_started_at) * 1000, 1),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=paged_image_matches,
            artifact_results=paged_artifact_results,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    async def handle_model_message(
        self,
        payload: ChatModelMessageRequest,
        *,
        model_bytes: bytes,
        model_filename: str | None = None,
        model_content_type: str | None = None,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        conversation_id = payload.conversation_id or str(uuid4())
        requested_format = payload.response_format or ResponseFormatObject(type="text")
        query_id = str(uuid4())
        request_started_at = time.monotonic()
        retrieval_latency_ms: float | None = None
        llm_latency_ms: float | None = None
        explicit_museum_id = self._resolve_explicit_museum_id_values(
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        museum_id = self._resolve_museum_id_values(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        user_message = (payload.message or "").strip() or translate(
            "message.default_model_query",
            language,
        )
        requested_museum = self._resolve_requested_search_museum(user_message)
        if requested_museum is not None:
            museum_id = requested_museum.museum_id
            explicit_museum_id = requested_museum.museum_id
        search_museum_slug = requested_museum.museum_slug if requested_museum else payload.museum_slug
        search_museum_id = museum_id
        search_museum_name = (
            requested_museum.museum_name
            if requested_museum
            else payload.museum_name
        )
        search_scope = self._build_search_scope(
            current_museum_slug=payload.museum_slug,
            current_museum_id=self._resolve_museum_id_values(
                museum_slug=payload.museum_slug,
                museum_id=payload.museum_id,
                metadata=payload.metadata,
            ),
            current_museum_name=payload.museum_name,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            search_museum_name=search_museum_name,
        )

        self.session_store.append_turn(state, role="user", text=user_message)
        log_event(
            logger,
            logging.INFO,
            "chat.model.start",
            conversation_id=conversation_id,
            query_id=query_id,
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            language=language,
            response_format=requested_format.type,
            model_filename=model_filename,
            model_content_type=model_content_type,
            model_bytes=len(model_bytes or b""),
            message_chars=len(user_message),
        )
        await self._emit_status(status_cb, "status.preparing_model", language=language)

        image_matches: list[ImageMatchResult] = []
        artifact_docs: list[dict[str, object]] = []
        artifact_results: list[ArtifactResult] = []
        navigation_targets: list[TourNavigationTarget] = []
        file_name = (model_filename or "model.glb").strip() or "model.glb"
        model_results_total = 0
        model_image_embeddings: list[list[float]] = []
        model_opensearch_query: dict[str, Any] | None = None

        retrieval_started_at = time.monotonic()
        try:
            retrieval_result = await self.model_retrieval_service.retrieve(
                museum_slug=search_museum_slug,
                museum_id=search_museum_id,
                model_bytes=model_bytes,
                file_name=file_name,
                artifact_museum_id=explicit_museum_id,
                progress_cb=lambda message, fields: self._emit_status(
                    status_cb,
                    message,
                    language=language,
                    **fields,
                ),
            )
            artifact_docs = retrieval_result.artifact_docs
            model_results_total = retrieval_result.image_hits_total
            model_image_embeddings = retrieval_result.image_embeddings
            model_opensearch_query = retrieval_result.opensearch_query
            image_matches = self._build_image_matches(
                image_hits=retrieval_result.image_hits,
                artifact_docs=artifact_docs,
            )
        except Exception as exc:
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.model.retrieval.error",
                conversation_id=conversation_id,
                query_id=query_id,
                file_name=file_name,
                duration_ms=round(retrieval_latency_ms, 1),
                error=exc,
            )
            fallback = translate("error.model_processing_failed", language)
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=user_message,
                resolved_query=user_message,
                lexical_query=None,
                embedding_query=None,
                route="model_search",
                retrieval_mode="model_multiview_similarity",
                filters_applied={
                    "filters": dict(state.filters),
                    "sort": dict(state.sort),
                    "target_museum": search_scope.model_dump(mode="json"),
                },
                boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
                retrieved_artifacts=[],
                shown_artifacts=[],
                navigation_targets=[],
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=None,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                search_scope=search_scope,
            )
        retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.model.retrieval.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            file_name=file_name,
            artifact_docs=len(artifact_docs),
            image_matches=len(image_matches),
            results_total=model_results_total,
            duration_ms=round(retrieval_latency_ms, 1),
        )

        await self._emit_status(
            status_cb,
            "status.artifacts_found",
            language=language,
            artifact_count=len(artifact_docs),
        )

        artifact_results = await self._build_artifact_results(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context="model",
            conversation_id=conversation_id,
            museum_slug=search_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        (
            paged_artifact_results,
            paged_image_matches,
            paged_navigation_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._build_paged_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=max(self.settings.CHAT_IMAGE_ARTIFACT_TOP_K, 1),
            total_override=model_results_total,
            retrieval_request={
                "kind": "model",
                "museum_id": search_museum_id,
                "museum_slug": search_museum_slug,
                "artifact_museum_id": explicit_museum_id,
                "search_scope": search_scope.model_dump(mode="json"),
                "image_embeddings": model_image_embeddings,
                "retrieval_window_size": retrieval_result.retrieval_window_size,
                "results_total": model_results_total,
                "opensearch_query": model_opensearch_query,
            },
        )
        results_request_id = str(
            state.last_paged_retrieval_request.get("results_request_id") or ""
        ).strip() or None

        retrieval_context = self._build_visible_results_retrieval_context(
            artifact_results=paged_artifact_results,
            image_matches=paged_image_matches,
            page=results_page,
            page_size=results_page_size,
            total=results_total,
            match_section_label="visible_model_view_matches",
        )
        retry_context_result_count = (
            len(paged_image_matches) if paged_image_matches else len(paged_artifact_results)
        )

        def build_model_final_message_for_result_count(result_count: int) -> str:
            limited_retrieval_context = self._build_visible_results_retrieval_context(
                artifact_results=paged_artifact_results[: max(result_count, 0)],
                image_matches=paged_image_matches[: max(result_count, 0)],
                page=results_page,
                page_size=results_page_size,
                total=results_total,
                match_section_label="visible_model_view_matches",
            )
            return build_final_answer_prompt(
                museum_slug=search_museum_slug,
                museum_name=search_museum_name,
                input_modality="model",
                mode="rag",
                intent="model_search",
                rolling_summary="",
                filters_state=effective_filters,
                sort_state=effective_sort,
                user_message=user_message,
                rewritten_query=user_message,
                retrieval_context=limited_retrieval_context,
                use_history_for_answer=False,
                language=language,
            )

        router_decision: dict[str, object] = {
            "mode": "rag",
            "intent": "model_search",
            "rewritten_query": user_message,
            "needs_retrieval": True,
            "reason": "model_upload",
            "filters_delta": {},
            "sort_delta": {},
        }
        effective_filters = dict(state.filters)
        effective_sort = dict(state.sort)

        final_message = build_final_answer_prompt(
            museum_slug=search_museum_slug,
            museum_name=search_museum_name,
            input_modality="model",
            mode="rag",
            intent="model_search",
            rolling_summary="",
            filters_state=effective_filters,
            sort_state=effective_sort,
            user_message=user_message,
            rewritten_query=user_message,
            retrieval_context=retrieval_context,
            use_history_for_answer=False,
            language=language,
        )
        await self._emit_status(status_cb, "status.generating_final_answer", language=language)

        llm_started_at = time.monotonic()
        try:
            llm_response = await self._generate_final_answer_with_context_retries(
                initial_message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
                visible_result_count=retry_context_result_count,
                build_message_for_result_count=build_model_final_message_for_result_count,
            )
        except LLMServiceError as exc:
            llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.model.llm.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round(llm_latency_ms, 1),
                error=exc,
            )
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=user_message,
                resolved_query=user_message,
                lexical_query=None,
                embedding_query=None,
                route="model_search",
                retrieval_mode="model_multiview_similarity",
                filters_applied={
                    "filters": effective_filters,
                    "sort": effective_sort,
                    "target_museum": search_scope.model_dump(mode="json"),
                },
                boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
                retrieved_artifacts=self._log_retrieved_artifact_entries(artifact_docs, source="model"),
                shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
                navigation_targets=self._log_navigation_target_entries(
                    navigation_targets=paged_navigation_targets,
                    artifact_results=paged_artifact_results,
                ),
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=llm_latency_ms,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
                opensearch_query=model_opensearch_query,
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                image_matches=paged_image_matches,
                artifact_results=paged_artifact_results,
                navigation_targets=paged_navigation_targets,
                results_page=results_page,
                results_page_size=results_page_size,
                results_total=results_total,
                results_has_more=results_has_more,
                results_request_id=results_request_id,
                search_scope=search_scope,
            )
        llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.model.llm.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            model=llm_response.model,
            response_format=llm_response.response_format.type,
            duration_ms=round(llm_latency_ms, 1),
        )

        reply_docs = self._artifact_results_as_prompt_docs(paged_artifact_results) or artifact_docs
        final_reply_text = self._sanitize_assistant_reply(
            llm_response.text,
            docs=reply_docs,
            language=language,
        )

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(
            state=state,
            artifact_docs=artifact_docs,
        )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=user_message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=user_message,
            resolved_query=user_message,
            lexical_query=None,
            embedding_query=None,
            route="model_search",
            retrieval_mode="model_multiview_similarity",
            filters_applied={
                "filters": effective_filters,
                "sort": effective_sort,
                "target_museum": search_scope.model_dump(mode="json"),
            },
            boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
            retrieved_artifacts=self._log_retrieved_artifact_entries(artifact_docs, source="model"),
            shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
            navigation_targets=self._log_navigation_target_entries(
                navigation_targets=paged_navigation_targets,
                artifact_results=paged_artifact_results,
            ),
            latency_retrieval_ms=retrieval_latency_ms,
            latency_llm_ms=llm_latency_ms,
            latency_total_ms=(time.monotonic() - request_started_at) * 1000,
            status="ok",
            error=None,
            opensearch_query=model_opensearch_query,
        )

        log_event(
            logger,
            logging.INFO,
            "chat.model.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status="ok",
            results_total=results_total,
            shown_artifacts=len(paged_artifact_results),
            shown_images=len(paged_image_matches),
            duration_ms=round((time.monotonic() - request_started_at) * 1000, 1),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=paged_image_matches,
            artifact_results=paged_artifact_results,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    def _is_obvious_greeting_or_smalltalk(self, message: str) -> bool:
        greetings = {
            "ola",
            "olá",
            "oi",
            "bom dia",
            "boa tarde",
            "boa noite",
            "tudo bem",
            "como estas",
            "como estás",
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
        }
        folded_message = self._fold_query_text(message)
        folded_greetings = {
            self._fold_query_text(greeting)
            for greeting in greetings
        }
        return bool(folded_message and folded_message in folded_greetings)

    def _fallback_router_decision(
        self,
        message: str,
        *,
        context_policy: dict[str, object],
    ) -> dict[str, object]:
        if self._is_obvious_greeting_or_smalltalk(message):
            return {
                "mode": "llm_only",
                "intent": "fallback",
                "rewritten_query": message,
                "needs_retrieval": False,
                "reason": "router_unavailable_deterministic_fallback_llm_only",
                "is_follow_up": bool(context_policy.get("is_follow_up", False)),
                "use_history_for_query": False,
                "use_history_for_answer": bool(
                    context_policy.get("use_history_for_answer", False)
                ),
                "carry_filters": False,
                "carry_sort": False,
                "filters_delta": {},
                "sort_delta": {},
            }

        return {
            "mode": "rag",
            "intent": "search",
            "rewritten_query": message,
            "needs_retrieval": True,
            "reason": "router_unavailable_deterministic_fallback_rag",
            "is_follow_up": bool(context_policy.get("is_follow_up", False)),
            "use_history_for_query": False,
            "use_history_for_answer": bool(
                context_policy.get("use_history_for_answer", False)
            ),
            "carry_filters": False,
            "carry_sort": False,
            "filters_delta": {},
            "sort_delta": {},
        }

    async def _route_message(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        context_policy: dict[str, object],
    ) -> dict[str, object]:
        router_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string", "enum": ["rag", "llm_only"]},
                "intent": {"type": "string", "enum": ["overview", "search", "refine", "fallback"]},
                "is_follow_up": {"type": "boolean"},
                "use_history_for_query": {"type": "boolean"},
                "use_history_for_answer": {"type": "boolean"},
                "carry_filters": {"type": "boolean"},
                "carry_sort": {"type": "boolean"},
                "rewritten_query": {"type": "string"},
                "needs_retrieval": {"type": "boolean"},
                "reason": {"type": "string"},
                "filters_delta": {
                    "type": "object",
                    "additionalProperties": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {"type": "null"},
                        ]
                    },
                },
                "sort_delta": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": [
                "mode",
                "intent",
                "is_follow_up",
                "use_history_for_query",
                "use_history_for_answer",
                "carry_filters",
                "carry_sort",
                "rewritten_query",
                "needs_retrieval",
                "reason",
                "filters_delta",
                "sort_delta",
            ],
        }

        use_history_for_query_hint = bool(context_policy.get("use_history_for_query", False))
        use_history_for_answer_hint = bool(context_policy.get("use_history_for_answer", False))
        include_aux_context = use_history_for_query_hint or use_history_for_answer_hint
        history_lines = (
            [
                f"- {turn.role}: {turn.text}"
                for turn in state.history[-min(self.settings.CHAT_HISTORY_WINDOW, 4) :]
            ]
            if include_aux_context
            else []
        )
        conversation_state_aux: dict[str, object] = {
            "intent_prev": state.intent,
            "history_size": len(state.history),
        }
        router_prompt = build_router_user_prompt(
            museum_slug=payload.museum_slug,
            museum_name=payload.museum_name,
            rolling_summary=state.rolling_summary if include_aux_context else "",
            filters_state={
                key: value
                for key, value in state.filters.items()
                if key != "artifact_id"
            },
            sort_state=state.sort,
            history_lines=history_lines,
            current_user_message=payload.message,
            router_schema=router_schema,
            context_policy_hint=context_policy,
            conversation_state_aux=conversation_state_aux,
            language=state.language,
        )

        router_response = await self.llm_service.generate(
            message=router_prompt,
            response_format=ResponseFormatObject(type="json_object"),
            system_prompt=get_router_system_prompt(language=state.language),
            model_override=payload.model_override,
        )

        parsed = router_response.parsed_json
        if not isinstance(parsed, dict):
            raise LLMServiceError("Router did not return an object.")

        mode = str(parsed.get("mode", "llm_only"))
        intent = str(parsed.get("intent", "fallback"))
        rewritten_query = str(parsed.get("rewritten_query", payload.message))
        needs_retrieval = bool(parsed.get("needs_retrieval", mode == "rag"))
        reason = str(parsed.get("reason", ""))
        is_follow_up = bool(parsed.get("is_follow_up", context_policy.get("is_follow_up", False)))
        use_history_for_query = bool(
            parsed.get(
                "use_history_for_query",
                context_policy.get("use_history_for_query", False),
            )
        )
        use_history_for_answer = bool(
            parsed.get(
                "use_history_for_answer",
                context_policy.get("use_history_for_answer", False),
            )
        )
        carry_filters = bool(parsed.get("carry_filters", context_policy.get("carry_filters", False)))
        carry_sort = bool(parsed.get("carry_sort", context_policy.get("carry_sort", False)))
        filters_delta = parsed.get("filters_delta") or {}
        sort_delta = parsed.get("sort_delta") or {}

        if not isinstance(filters_delta, dict):
            filters_delta = {}
        if not isinstance(sort_delta, dict):
            sort_delta = {}

        if self._extract_tour_scope_expression(payload.message):
            is_follow_up = False
            use_history_for_query = False
            carry_filters = False
            carry_sort = False
            filters_delta.pop("artifact_id", None)
            reason = self._append_reason_tag(reason, "guardrail_tour_scope_query")

        if mode not in {"rag", "llm_only"}:
            mode = "llm_only"
        if intent not in {"overview", "search", "refine", "fallback"}:
            intent = "fallback"
        if not is_follow_up:
            use_history_for_query = False
            carry_filters = False
            carry_sort = False
        if not use_history_for_query:
            carry_filters = False
            carry_sort = False
        if intent in {"search", "refine"}:
            if mode != "rag" or not needs_retrieval:
                reason = self._append_reason_tag(
                    reason,
                    "guardrail_search_requires_retrieval",
                )
            mode = "rag"
            needs_retrieval = True
        elif needs_retrieval:
            if mode != "rag":
                reason = self._append_reason_tag(
                    reason,
                    "guardrail_needs_retrieval_requires_rag",
                )
            mode = "rag"
            needs_retrieval = True
        elif mode == "rag":
            needs_retrieval = True
        else:
            needs_retrieval = False

        return {
            "mode": mode,
            "intent": intent,
            "is_follow_up": is_follow_up,
            "use_history_for_query": use_history_for_query,
            "use_history_for_answer": use_history_for_answer,
            "carry_filters": carry_filters,
            "carry_sort": carry_sort,
            "rewritten_query": rewritten_query,
            "needs_retrieval": needs_retrieval,
            "reason": reason,
            "filters_delta": filters_delta,
            "sort_delta": sort_delta,
        }

    @staticmethod
    def _append_reason_tag(reason: str, tag: str) -> str:
        cleaned_reason = str(reason or "").strip()
        cleaned_tag = str(tag or "").strip()
        if not cleaned_tag:
            return cleaned_reason
        if not cleaned_reason:
            return cleaned_tag
        if cleaned_tag in cleaned_reason:
            return cleaned_reason
        return f"{cleaned_reason} | {cleaned_tag}"

    def _apply_context_policy_guardrails(
        self,
        *,
        router_decision: dict[str, object],
        context_policy: dict[str, object],
        user_message: str,
    ) -> dict[str, object]:
        guarded = dict(router_decision)
        if not bool(context_policy.get("use_history_for_query", False)):
            return guarded

        guarded["is_follow_up"] = True
        guarded["use_history_for_query"] = True
        guarded["carry_filters"] = True
        guarded["carry_sort"] = True

        # Follow-up queries can still use non-artifact state filters.
        guarded["mode"] = "rag"
        guarded["needs_retrieval"] = True

        rewritten_query = str(guarded.get("rewritten_query", "")).strip()
        if not rewritten_query:
            guarded["rewritten_query"] = user_message

        reason = str(guarded.get("reason", "")).strip()
        guarded["reason"] = self._append_reason_tag(
            reason,
            "guardrail_context_policy_follow_up",
        )
        return guarded

    def _inventory_candidate_variants(self, value: str) -> list[str]:
        folded = self._fold_query_text(value)
        compact = re.sub(r"[^a-z0-9]+", "", folded, flags=re.UNICODE)

        variants: list[str] = []

        def add(candidate: str) -> None:
            cleaned = candidate.strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        add(folded)
        for prefix in sorted(_INVENTORY_PREFIXES, key=len, reverse=True):
            if not compact.startswith(prefix):
                continue
            suffix = compact[len(prefix):]
            if suffix and re.search(r"\d", suffix):
                add(f"{prefix} {suffix}")
                break
        add(compact)
        return variants

    def _looks_like_inventory_suffix(self, token: str) -> bool:
        cleaned = str(token or "").strip()
        return bool(cleaned and len(cleaned) <= 32 and re.search(r"\d", cleaned))

    def _looks_like_common_year(self, token: str) -> bool:
        cleaned = str(token or "").strip()
        if not re.fullmatch(r"\d{4}", cleaned):
            return False
        year = int(cleaned)
        return 1500 <= year <= 2099

    def _add_inventory_candidate(
        self,
        candidates: list[str],
        seen: set[str],
        value: str,
    ) -> None:
        for candidate in self._inventory_candidate_variants(value):
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    def _extract_inventory_candidates(self, query: str) -> list[str]:
        folded = self._fold_query_text(query)
        if not folded:
            return []

        tokens = folded.split()
        prefix_set = set(_INVENTORY_PREFIXES)
        candidates: list[str] = []
        seen: set[str] = set()

        for index, token in enumerate(tokens):
            if token in prefix_set and index + 1 < len(tokens):
                suffix = tokens[index + 1]
                if self._looks_like_inventory_suffix(suffix):
                    self._add_inventory_candidate(candidates, seen, f"{token} {suffix}")
                continue

            for prefix in sorted(_INVENTORY_PREFIXES, key=len, reverse=True):
                if not token.startswith(prefix) or token == prefix:
                    continue
                suffix = token[len(prefix):]
                if self._looks_like_inventory_suffix(suffix):
                    self._add_inventory_candidate(candidates, seen, token)
                break

        for index, token in enumerate(tokens):
            if token not in _ARTIFACT_REFERENCE_MARKERS:
                continue
            value_index = index + 1
            if (
                value_index < len(tokens)
                and tokens[value_index] in _ARTIFACT_REFERENCE_NUMBER_CONNECTORS
            ):
                value_index += 1
            if value_index >= len(tokens):
                continue
            value = tokens[value_index]
            if not self._looks_like_inventory_suffix(value):
                continue
            if self._looks_like_common_year(value):
                continue

            parts = [value]
            next_index = value_index + 1
            if (
                next_index < len(tokens)
                and re.fullmatch(r"[a-z0-9]{1,6}", tokens[next_index])
                and re.search(r"\d", tokens[next_index])
            ):
                parts.append(tokens[next_index])
            self._add_inventory_candidate(candidates, seen, " ".join(parts))

        for index in range(len(tokens)):
            for marker in _EXPLICIT_INVENTORY_MARKERS:
                marker_size = len(marker)
                if tuple(tokens[index : index + marker_size]) != marker:
                    continue
                window_start = index + marker_size
                window = tokens[window_start : window_start + 4]
                for offset, token in enumerate(window):
                    if token in prefix_set and offset + 1 < len(window):
                        suffix = window[offset + 1]
                        if self._looks_like_inventory_suffix(suffix):
                            self._add_inventory_candidate(
                                candidates,
                                seen,
                                f"{token} {suffix}",
                            )
                    elif self._looks_like_inventory_suffix(token):
                        self._add_inventory_candidate(candidates, seen, token)
                break

        return candidates

    async def _retrieve_inventory_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        raw_query: str,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]] | None:
        inventory_candidates = self._extract_inventory_candidates(raw_query)
        if not inventory_candidates:
            return None

        top_k = min(max(self.settings.CHAT_RETRIEVAL_TOP_K, 1), 5)
        try:
            docs = await self.opensearch_gateway.search_artifacts_by_inventory_candidates(
                museum_slug=museum_slug,
                museum_id=museum_id,
                inventory_numbers=inventory_candidates,
                top_k=top_k,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.retrieval.inventory.error",
                museum_slug=museum_slug,
                museum_id=museum_id,
                inventory_count=len(inventory_candidates),
                error=exc,
            )
            return None

        if not docs:
            return None

        docs_for_context = docs[:top_k]
        context = self._format_docs_for_prompt(docs=docs_for_context, top_k=top_k)
        retrieval_request = {
            "kind": "inventory",
            "museum_id": museum_id,
            "museum_slug": museum_slug,
            "original_query": raw_query,
            "search_query": raw_query,
            "query_text": raw_query,
            "inventory_candidates": inventory_candidates,
            "filters": {},
            "sort": {},
            "results_total": len(docs),
        }
        return context, len(docs), docs, retrieval_request

    async def _retrieve_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        query: str,
        filters: dict[str, object],
        sort: dict[str, object],
        result_window_size: int | None = None,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]]:
        raw_query = (query or "").strip()
        inventory_result = await self._retrieve_inventory_context(
            museum_slug=museum_slug,
            museum_id=museum_id,
            raw_query=raw_query,
        )
        if inventory_result is not None:
            return inventory_result

        temporal_query = await self._interpret_temporal_query(raw_query)
        tour_scope_expression = self._extract_tour_scope_expression(raw_query)
        retrieval_filters = self._merge_temporal_filter(filters, temporal_query)
        retrieval_filters = self._merge_tour_scope_filter(
            retrieval_filters,
            tour_scope_expression,
        )
        search_query = self._strip_temporal_expression_from_query(raw_query, temporal_query)
        search_query = self._strip_tour_scope_expression_from_query(
            search_query,
            tour_scope_expression,
        )
        if not search_query:
            search_query = "objetos"
        original_search_query = search_query
        lexical_query_fallback = self._build_lexical_query(
            query=search_query,
            museum_slug=museum_slug,
            museum_id=museum_id,
        )
        # lexical_query is intentionally short and optimized for BM25.
        lexical_query = lexical_query_fallback
        # embedding_query keeps the original/resolved natural-language request for semantic retrieval.
        embedding_query = original_search_query
        query_rewrite_source = "heuristic"

        llm_lexical_query, _ = await self._rewrite_retrieval_query_with_llm(
            query=search_query,
            museum_slug=museum_slug,
            museum_id=museum_id,
            filters=filters,
            sort=sort,
        )
        if llm_lexical_query:
            llm_lexical_query = self._strip_temporal_expression_from_query(
                llm_lexical_query,
                temporal_query,
            )
            llm_lexical_query = self._strip_tour_scope_expression_from_query(
                llm_lexical_query,
                tour_scope_expression,
            )
        if llm_lexical_query:
            lexical_query = llm_lexical_query
            query_rewrite_source = "llm"

        temporal_filter_payload = self._temporal_query_filter_payload(temporal_query)

        tour_scope_filter_payload = self._tour_scope_filter_payload(tour_scope_expression)

        if self.settings.CHAT_USE_QUERY_EMBEDDINGS:
            try:
                query_embedding = await self.embedding_provider.embed_text(embedding_query)
            except NotImplementedError:
                log_event(
                    logger,
                    logging.WARNING,
                    "chat.retrieval.embedding.unsupported",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    query_chars=len(embedding_query),
                )
                return "", 0, [], {}
            except Exception as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "chat.retrieval.embedding.error",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    query_chars=len(embedding_query),
                    error=exc,
                )
                return "", 0, [], {}
        else:
            log_event(
                logger,
                logging.INFO,
                "chat.retrieval.embedding.disabled",
                museum_slug=museum_slug,
                museum_id=museum_id,
            )
            return "", 0, [], {}

        final_top_k = max(self.settings.CHAT_RETRIEVAL_TOP_K, 1)
        if result_window_size is None:
            retrieval_page_size = max(
                self.settings.CHAT_RETRIEVAL_CANDIDATES,
                final_top_k,
                1,
            )
        else:
            retrieval_page_size = max(int(result_window_size), final_top_k, 1)
        retrieval_window_size = self._text_retrieval_window_size(
            minimum=retrieval_page_size,
        )
        retrieval_boosts = self._matched_retrieval_boosts_for_log(
            query_text=embedding_query,
            lexical_query=lexical_query,
        )
        self._rag_debug_json(
            "rag.text_query_plan",
            {
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "raw_query": raw_query,
                "search_query": search_query,
                "embedding_query": embedding_query,
                "lexical_query": lexical_query,
                "query_rewrite_source": query_rewrite_source,
                "filters": dict(retrieval_filters),
                "sort": dict(sort),
                "temporal_query": self._temporal_query_filter_payload(temporal_query),
                "tour_scope": self._tour_scope_filter_payload(tour_scope_expression),
                "retrieval_page_size": retrieval_page_size,
                "retrieval_window_size": retrieval_window_size,
                "retrieval_boosts": retrieval_boosts,
                "query_embedding": query_embedding,
            },
        )

        try:
            page_result = await self.opensearch_gateway.search_relevant_context_page(
                museum_slug=museum_slug,
                museum_id=museum_id,
                query_text=embedding_query,
                lexical_query=lexical_query,
                query_embedding=query_embedding,
                from_offset=0,
                page_size=retrieval_page_size,
                filters=retrieval_filters,
                sort=sort,
                retrieval_window_size=retrieval_window_size,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.retrieval.opensearch.error",
                museum_slug=museum_slug,
                museum_id=museum_id,
                page_size=retrieval_page_size,
                retrieval_window_size=retrieval_window_size,
                error=exc,
            )
            return "", 0, [], {}

        docs = page_result.results
        results_total = self._bounded_retrieval_total(page_result.total, retrieval_window_size)
        retrieval_request = {
            "kind": "text",
            "museum_id": museum_id,
            "museum_slug": museum_slug,
            "original_query": raw_query,
            "search_query": search_query,
            "query_text": embedding_query,
            "lexical_query": lexical_query,
            "query_rewrite_source": query_rewrite_source,
            "query_embedding": query_embedding,
            "filters": dict(retrieval_filters),
            "sort": dict(sort),
            "retrieval_boosts": retrieval_boosts,
            "retrieval_window_size": retrieval_window_size,
            "results_total": results_total,
            "temporal_query": temporal_filter_payload,
            "tour_scope": tour_scope_filter_payload,
            "opensearch_query": getattr(page_result, "query_body", None),
        }

        if not docs:
            return "", results_total, [], retrieval_request

        context_top_k = final_top_k
        if result_window_size is not None:
            try:
                context_top_k = max(
                    context_top_k,
                    min(max(int(result_window_size), 1), len(docs)),
                )
            except (TypeError, ValueError):
                pass
        docs_for_context = docs[:context_top_k]
        context = self._format_docs_for_prompt(docs=docs_for_context, top_k=context_top_k)
        return context, results_total, docs, retrieval_request

    def _tokenize_for_language_guardrail(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKD", (text or "").casefold())
        folded = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        return re.findall(r"[a-z0-9]+", folded, flags=re.UNICODE)

    def _is_probably_portuguese_query(self, text: str) -> bool:
        tokens = self._tokenize_for_language_guardrail(text)
        if not tokens:
            return False
        pt_hits = sum(token in _PT_QUERY_LANGUAGE_HINTS for token in tokens)
        en_hits = sum(token in _EN_QUERY_LANGUAGE_HINTS for token in tokens)
        if re.search(r"[ãõáàâéêíóôúç]", (text or "").casefold()):
            pt_hits += 1
        return pt_hits > en_hits and pt_hits > 0

    def _is_probably_english_query(self, text: str) -> bool:
        tokens = self._tokenize_for_language_guardrail(text)
        if not tokens:
            return False
        en_hits = sum(token in _EN_QUERY_LANGUAGE_HINTS for token in tokens)
        pt_hits = sum(token in _PT_QUERY_LANGUAGE_HINTS for token in tokens)
        return en_hits > pt_hits and en_hits > 0

    def _query_terms_overlap_source(self, source: str, candidate: str) -> bool:
        source_tokens = {
            token for token in self._tokenize_for_language_guardrail(source) if len(token) > 1
        }
        candidate_tokens = [
            token for token in self._tokenize_for_language_guardrail(candidate) if len(token) > 1
        ]
        if not source_tokens or not candidate_tokens:
            return False
        return any(token in source_tokens for token in candidate_tokens)

    def _fold_query_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", (text or "").casefold())
        folded = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        return re.sub(r"[^a-z0-9]+", " ", folded, flags=re.UNICODE).strip()

    def _fold_query_text_with_index(self, text: str) -> tuple[str, list[int]]:
        folded_chars: list[str] = []
        source_indices: list[int] = []
        for index, char in enumerate(text or ""):
            normalized = unicodedata.normalize("NFKD", char.casefold())
            for normalized_char in normalized:
                if unicodedata.combining(normalized_char):
                    continue
                folded_chars.append(normalized_char if normalized_char.isalnum() else " ")
                source_indices.append(index)
        return "".join(folded_chars), source_indices

    def _temporal_alias_expressions(self, temporal_query: TemporalQuery) -> list[str]:
        expressions: list[str] = []
        if temporal_query.expression:
            expressions.append(temporal_query.expression)
        for aliases, candidate in _HARDCODED_HISTORICAL_PERIODS:
            if (
                candidate.start_year == temporal_query.start_year
                and candidate.end_year == temporal_query.end_year
                and candidate.expression == temporal_query.expression
            ):
                expressions.extend(aliases)
        seen: set[str] = set()
        deduped: list[str] = []
        for expression in expressions:
            folded = self._fold_query_text(expression)
            if not folded or folded in seen:
                continue
            seen.add(folded)
            deduped.append(expression)
        return deduped

    def _strip_temporal_expression_from_query(
        self,
        text: str,
        temporal_query: TemporalQuery,
    ) -> str:
        raw_text = (text or "").strip()
        if (
            not raw_text
            or temporal_query.start_year is None
            or temporal_query.end_year is None
        ):
            return raw_text

        folded_text, source_indices = self._fold_query_text_with_index(raw_text)
        if not folded_text or not source_indices:
            return raw_text

        year = r"(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})"
        patterns = [
            rf"\b(?:entre|de)\s+{year}\s+(?:e|a|ate)\s+{year}\b",
            rf"\b(?:no\s+periodo|do\s+periodo|periodo)\s+{year}\s+(?:(?:e|a|ate)\s+)?{year}\b",
            rf"\b{year}\s+(?:e|a|ate)\s+{year}\b",
        ]

        for expression in self._temporal_alias_expressions(temporal_query):
            tokens = self._fold_query_text(expression).split()
            if not tokens:
                continue
            phrase = r"\s+".join(re.escape(token) for token in tokens)
            patterns.append(
                rf"\b(?:(?:de|do|da|dos|das|no|na|nos|nas|em)\s+)?"
                rf"{phrase}(?:\s+de\s+d\s+joao\s+(?:v|vi))?\b"
            )

        remove_mask = [False] * len(raw_text)
        for pattern in patterns:
            for match in re.finditer(pattern, folded_text):
                if match.end() <= match.start():
                    continue
                start = source_indices[match.start()]
                end = source_indices[match.end() - 1] + 1
                for index in range(start, min(end, len(remove_mask))):
                    remove_mask[index] = True

        cleaned = "".join(
            char for index, char in enumerate(raw_text) if not remove_mask[index]
        )
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,.;:!?")

    def _extract_tour_scope_expression(self, text: str) -> str | None:
        raw_text = (text or "").strip()
        if not raw_text:
            return None

        folded_text, source_indices = self._fold_query_text_with_index(raw_text)
        if not folded_text or not source_indices:
            return None

        for pattern in _TOUR_SCOPE_PATTERNS:
            match = re.search(pattern, folded_text)
            if not match:
                continue
            start = source_indices[match.start()]
            end = source_indices[match.end() - 1] + 1
            expression = raw_text[start:end].strip(" ,.;:!?")
            return expression or None

        return None

    def _strip_tour_scope_expression_from_query(
        self,
        text: str,
        tour_scope_expression: str | None,
    ) -> str:
        raw_text = (text or "").strip()
        if not raw_text or not tour_scope_expression:
            return raw_text

        folded_text, source_indices = self._fold_query_text_with_index(raw_text)
        if not folded_text or not source_indices:
            return raw_text

        remove_mask = [False] * len(raw_text)
        for pattern in _TOUR_SCOPE_PATTERNS:
            for match in re.finditer(pattern, folded_text):
                if match.end() <= match.start():
                    continue
                start = source_indices[match.start()]
                end = source_indices[match.end() - 1] + 1
                for index in range(start, min(end, len(remove_mask))):
                    remove_mask[index] = True

        cleaned = "".join(
            char for index, char in enumerate(raw_text) if not remove_mask[index]
        )
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,.;:!?")

    def _extract_explicit_temporal_query(self, text: str) -> TemporalQuery:
        folded = self._fold_query_text(text)
        if not folded:
            return TemporalQuery(None, None, None, None)

        year = r"(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})"
        explicit_patterns = (
            rf"\b(?:entre|de)\s+({year})\s+(?:e|a|ate)\s+({year})\b",
            rf"\b(?:no\s+periodo|do\s+periodo|periodo)\s+({year})\s+(?:(?:e|a|ate)\s+)?({year})\b",
            rf"\b({year})\s+(?:e|a|ate)\s+({year})\b",
        )
        for pattern in explicit_patterns:
            match = re.search(pattern, folded)
            if not match:
                continue
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            if start_year > end_year:
                start_year, end_year = end_year, start_year
            return TemporalQuery(
                start_year=start_year,
                end_year=end_year,
                expression=match.group(0),
                confidence=1.0,
            )

        return TemporalQuery(None, None, None, None)

    def _resolve_hardcoded_historical_period(self, text: str) -> TemporalQuery:
        folded = self._fold_query_text(text)
        if not folded:
            return TemporalQuery(None, None, None, None)

        for aliases, temporal_query in _HARDCODED_HISTORICAL_PERIODS:
            for alias in aliases:
                folded_alias = self._fold_query_text(alias)
                if not folded_alias:
                    continue
                if (
                    temporal_query.expression == "periodo joanino"
                    and re.search(r"\b(?:d\s*)?joao\s+vi\b", folded)
                ):
                    continue
                if re.search(rf"\b{re.escape(folded_alias)}\b", folded):
                    return temporal_query

        return TemporalQuery(None, None, None, None)

    def _parse_supported_century(self, token: str) -> int | None:
        normalized = (token or "").casefold().strip()
        if not normalized:
            return None
        if normalized in _ROMAN_CENTURY_NUMERALS:
            return _ROMAN_CENTURY_NUMERALS[normalized]

        match = re.match(r"^([0-9]{1,2})", normalized)
        if not match:
            return None
        century = int(match.group(1))
        if not 15 <= century <= 30:
            return None
        return century

    def _extract_century_temporal_query(self, text: str) -> TemporalQuery:
        folded = self._fold_query_text(text)
        if not folded:
            return TemporalQuery(None, None, None, None)

        century_token = r"([ivxlcdm]+|[0-9]{1,2}(?:o|a|st|nd|rd|th)?)"
        century_patterns = (
            rf"\b(?:seculo|seculos|sec|secs|century|centuries)\s+{century_token}\b",
            rf"\b{century_token}\s+(?:seculo|seculos|sec|secs|century|centuries)\b",
        )
        for pattern in century_patterns:
            match = re.search(pattern, folded)
            if not match:
                continue
            century = self._parse_supported_century(match.group(1))
            if century is None:
                continue
            start_year = (century - 1) * 100
            return TemporalQuery(
                start_year=start_year,
                end_year=start_year + 99,
                expression=match.group(0),
                confidence=1.0,
            )

        return TemporalQuery(None, None, None, None)

    def _build_temporal_interpretation_prompt(self, text: str) -> str:
        return "\n".join(
            [
                "Interpreta referencias temporais numa pergunta sobre patrimonio/museus.",
                "A tarefa NAO e responder ao utilizador.",
                "Extrai um intervalo de anos apenas se a pergunta mencionar anos, datas, seculos, decadas ou um periodo historico reconhecivel.",
                "Para periodos historicos implicitos, resolve para o intervalo historico mais usado no contexto portugues/europeu.",
                "Nao inventes datas se nao houver referencia temporal.",
                "Responde apenas JSON neste formato:",
                '{"start_year":1900,"end_year":1910,"expression":"expressao temporal mencionada","confidence":0.9}',
                "Se nao houver referencia temporal, responde:",
                '{"start_year":null,"end_year":null,"expression":null,"confidence":0.0}',
                "Regras:",
                "- start_year e end_year devem ser inteiros ou null.",
                "- expression deve ser a expressao temporal presente na pergunta, sem traduzir.",
                "- confidence deve estar entre 0 e 1.",
                "- Se tiveres duvida, usa null/null com confidence baixa.",
                f"pergunta: {text}",
            ]
        )

    async def _interpret_temporal_query(self, text: str) -> TemporalQuery:
        explicit = self._extract_explicit_temporal_query(text)
        if explicit.start_year is not None and explicit.end_year is not None:
            return explicit

        hardcoded_period = self._resolve_hardcoded_historical_period(text)
        if hardcoded_period.start_year is not None and hardcoded_period.end_year is not None:
            return hardcoded_period

        century = self._extract_century_temporal_query(text)
        if century.start_year is not None and century.end_year is not None:
            return century

        folded = self._fold_query_text(text)
        temporal_hints = (
            "periodo",
            "epoca",
            "era",
            "seculo",
            "seculos",
            "sec",
            "secs",
            "century",
            "centuries",
            "decada",
            "anos",
        )
        if not folded or not any(re.search(rf"\b{hint}\b", folded) for hint in temporal_hints):
            return TemporalQuery(None, None, None, None)

        try:
            response = await self.llm_service.generate(
                message=self._build_temporal_interpretation_prompt(text),
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=(
                    "Es um interpretador temporal para retrieval museologico. "
                    "Responde sempre JSON valido e nunca texto livre."
                ),
                model_override=None,
            )
        except Exception:
            return TemporalQuery(None, None, None, None)

        payload = response.parsed_json
        if not isinstance(payload, dict):
            return TemporalQuery(None, None, None, None)

        start_year = payload.get("start_year")
        end_year = payload.get("end_year")
        confidence = payload.get("confidence")
        if not isinstance(start_year, int) or not isinstance(end_year, int):
            return TemporalQuery(None, None, None, None)
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        resolved_confidence = (
            float(confidence)
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else 0.0
        )
        if resolved_confidence < 0.55:
            return TemporalQuery(None, None, None, None)

        expression = str(payload.get("expression") or "").strip() or None
        return TemporalQuery(
            start_year=start_year,
            end_year=end_year,
            expression=expression,
            confidence=max(0.0, min(resolved_confidence, 1.0)),
        )

    def _temporal_query_filter_payload(
        self,
        temporal_query: TemporalQuery,
    ) -> dict[str, object] | None:
        if temporal_query.start_year is None or temporal_query.end_year is None:
            return None
        return {
            "start_year": temporal_query.start_year,
            "end_year": temporal_query.end_year,
            "expression": temporal_query.expression,
            "confidence": temporal_query.confidence,
            "include_unknown": False,
        }

    def _merge_temporal_filter(
        self,
        filters: dict[str, object],
        temporal_query: TemporalQuery,
    ) -> dict[str, object]:
        temporal_filter = self._temporal_query_filter_payload(temporal_query)
        if not temporal_filter:
            return dict(filters)
        merged = dict(filters)
        merged["_temporal_interval"] = temporal_filter
        return merged

    def _tour_scope_filter_payload(
        self,
        tour_scope_expression: str | None,
    ) -> dict[str, object] | None:
        expression = str(tour_scope_expression or "").strip()
        if not expression:
            return None
        return {
            "in_tour": True,
            "expression": expression,
        }

    def _merge_tour_scope_filter(
        self,
        filters: dict[str, object],
        tour_scope_expression: str | None,
    ) -> dict[str, object]:
        if not tour_scope_expression:
            return dict(filters)
        merged = dict(filters)
        merged["in_tour"] = True
        return merged

    def _has_query_language_mismatch(self, source: str, candidate: str) -> bool:
        source_pt = self._is_probably_portuguese_query(source)
        source_en = self._is_probably_english_query(source)
        candidate_pt = self._is_probably_portuguese_query(candidate)
        candidate_en = self._is_probably_english_query(candidate)
        if source_pt:
            if candidate_en and not candidate_pt:
                return True
            if candidate_pt or self._query_terms_overlap_source(source, candidate):
                return False
            return True
        if source_en and candidate_pt and not candidate_en:
            return True
        return False

    async def _rewrite_retrieval_query_with_llm(
        self,
        *,
        query: str,
        museum_slug: str,
        museum_id: str | None,
        filters: dict[str, object],
        sort: dict[str, object],
    ) -> tuple[str, str]:
        raw_query = (query or "").strip()
        if not raw_query:
            return "", ""

        prompt = build_retrieval_query_rewrite_prompt(
            user_query=raw_query,
            filters=dict(filters),
            sort=dict(sort),
        )
        try:
            response = await self.llm_service.generate(
                message=prompt,
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT,
                model_override=None,
            )
        except Exception:
            return "", ""

        payload = response.parsed_json
        if not isinstance(payload, dict):
            return "", ""

        lexical_query = str(payload.get("lexical_query") or "").strip()
        if lexical_query and self._has_query_language_mismatch(raw_query, lexical_query):
            lexical_query = ""
        return lexical_query, ""

    def _build_lexical_query(
        self,
        *,
        query: str,
        museum_slug: str,
        museum_id: str | None,
    ) -> str:
        def _fold_token(value: str) -> str:
            normalized_value = unicodedata.normalize("NFKD", value.casefold())
            return "".join(
                char for char in normalized_value if not unicodedata.combining(char)
            )

        text = (query or "").strip().lower()
        if not text:
            return ""

        for pattern in (
            r"\b(no|na|do|da|em)\s+museu\s+[a-z0-9_-]+\b",
            r"\bmuseu\s+[a-z0-9_-]+\b",
        ):
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        normalized = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        tokens = [token for token in normalized.split() if token]

        drop_tokens = {
            "no",
            "na",
            "nos",
            "nas",
            "do",
            "da",
            "dos",
            "das",
            "de",
            "a",
            "o",
            "as",
            "os",
            "um",
            "uma",
            "uns",
            "umas",
            "e",
            "ou",
            "que",
            "para",
            "por",
            "com",
            "sem",
            "sobre",
            "em",
            "ao",
            "aos",
            "encontra",
            "encontrar",
            "procura",
            "procurar",
            "procuro",
            "mostrar",
            "mostra",
            "dizer",
            "diz",
            "fala",
            "falar",
            "ver",
            "podes",
            "pode",
            "podem",
            "conseguir",
            "consegues",
            "eu",
            "me",
            "mim",
            "tu",
            "te",
            "ti",
            "voce",
            "voces",
            "isto",
            "isso",
            "este",
            "esta",
            "estes",
            "estas",
            "esse",
            "essa",
            "esses",
            "essas",
            "desse",
            "dessa",
            "desses",
            "dessas",
            "neste",
            "nesta",
            "nestes",
            "nestas",
            "nesse",
            "nessa",
            "nesses",
            "nessas",
            "quero",
            "queria",
            "queremos",
            "preciso",
            "precisava",
            "favor",
            "porfavor",
            "ok",
            "okay",
            "nope",
            "nao",
            "sim",
            "entao",
            "bom",
            "ora",
            "tipo",
            "agora",
            "antes",
            "obra",
            "peca",
            "pecas",
            "museu",
            "colecao",
            "colecoes",
            "acervo",
            "artefacto",
            "artefactos",
        }
        drop_tokens = {_fold_token(token) for token in drop_tokens}
        drop_tokens.add(_fold_token((museum_slug or "").strip().lower()))
        if museum_id:
            drop_tokens.add(_fold_token(museum_id.strip().lower()))

        filtered: list[str] = []
        fallback: list[str] = []
        seen_filtered: set[str] = set()
        seen_fallback: set[str] = set()
        for token in tokens:
            folded = _fold_token(token)
            if len(folded) <= 1:
                continue
            if folded not in seen_fallback:
                seen_fallback.add(folded)
                fallback.append(token)
            if folded in drop_tokens:
                continue
            if folded in seen_filtered:
                continue
            seen_filtered.add(folded)
            filtered.append(token)

        if filtered:
            return " ".join(filtered[:12])

        if fallback:
            return " ".join(fallback[:12])

        return ""

    def _derive_context_policy(
        self,
        *,
        message: str,
        state: ChatSessionState,
    ) -> dict[str, object]:
        def _policy(
            *,
            is_follow_up: bool,
            use_history_for_query: bool,
            use_history_for_answer: bool,
            carry_filters: bool,
            carry_sort: bool,
            reason: str,
        ) -> dict[str, object]:
            return {
                "is_follow_up": is_follow_up,
                "use_history_for_query": use_history_for_query,
                "use_history_for_answer": use_history_for_answer,
                "carry_filters": carry_filters,
                "carry_sort": carry_sort,
                "reason": reason,
            }

        text = (message or "").strip().lower()
        if not text:
            return _policy(
                is_follow_up=False,
                use_history_for_query=False,
                use_history_for_answer=False,
                carry_filters=False,
                carry_sort=False,
                reason="empty_message",
            )

        has_previous_context = bool(state.history[:-1] or state.rolling_summary.strip())
        if not has_previous_context:
            return _policy(
                is_follow_up=False,
                use_history_for_query=False,
                use_history_for_answer=False,
                carry_filters=False,
                carry_sort=False,
                reason="no_previous_context",
            )

        standalone_patterns = (
            r"\bnova\s+(pesquisa|busca|consulta|quest[ãa]o|pergunta)\b",
            r"\b(do zero|novo tema|mudar de tema|outro tema|novo pedido)\b",
            r"\bsem rela[cç][ãa]o com (a )?(anterior|ultima)\b",
            r"\bmudando de assunto\b",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in standalone_patterns):
            return _policy(
                is_follow_up=False,
                use_history_for_query=False,
                use_history_for_answer=False,
                carry_filters=False,
                carry_sort=False,
                reason="explicit_new_query",
            )

        if self._extract_tour_scope_expression(text):
            return _policy(
                is_follow_up=False,
                use_history_for_query=False,
                use_history_for_answer=False,
                carry_filters=False,
                carry_sort=False,
                reason="tour_scope_query",
            )

        referential_query_patterns = (
            r"^\s*(e|ent[aã]o|agora)\b",
            r"\b(deste|desta|desses|dessas|disso|nisso|daquilo)\b",
            r"\b(neste|nesta|nestes|nestas|nesse|nessa|nesses|nessas)\b",
            r"\b(esse|essa|esses|essas)\b",
            r"\b(nele|nela|dele|dela|disto|nisto)\b",
            r"\b(anterior|ultima|acima|mesmo|mesma)\b",
            r"\b(mais recente|mais antigo|mais antiga)\b",
            r"\b(do s[eé]culo|desse s[eé]culo)\b",
            r"\b(s[oó] os|s[oó] as|apenas os|apenas as)\b",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in referential_query_patterns):
            return _policy(
                is_follow_up=True,
                use_history_for_query=True,
                use_history_for_answer=True,
                carry_filters=True,
                carry_sort=True,
                reason="referential_follow_up_query",
            )

        referential_answer_patterns = (
            r"\b(explica melhor|detalha|resume|reformula|desenvolve|continua)\b",
            r"\b(com mais detalhe|mais detalhe)\b",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in referential_answer_patterns):
            return _policy(
                is_follow_up=True,
                use_history_for_query=False,
                use_history_for_answer=True,
                carry_filters=False,
                carry_sort=False,
                reason="referential_follow_up_answer_only",
            )

        return _policy(
            is_follow_up=False,
            use_history_for_query=False,
            use_history_for_answer=False,
            carry_filters=False,
            carry_sort=False,
            reason="standalone_query_default",
        )

    def _doc_inventory(self, doc: dict[str, object]) -> str:
        return str(doc.get("inventory_number") or doc.get("inventory") or "").strip()

    def _doc_search_text(self, doc: dict[str, object]) -> str:
        return str(doc.get("search_text") or doc.get("full_text") or "").strip()

    def _image_hit_inventory(self, hit: dict[str, object]) -> str:
        return str(hit.get("inventory_number") or hit.get("inventory") or "").strip()

    def _image_match_name(self, hit: dict[str, object]) -> str:
        local_path = str(hit.get("local_path") or "").strip()
        if local_path:
            return local_path.replace("\\", "/").strip("/")

        original_image_name = str(hit.get("original_image_name") or "").strip()
        if original_image_name:
            return original_image_name

        image_id = str(hit.get("image_id") or hit.get("id") or "").strip()
        if image_id:
            return f"{image_id.replace(':', '_')}.jpg"
        return ""

    def _format_docs_for_prompt(self, *, docs: list[dict[str, object]], top_k: int) -> str:
        snippets: list[str] = []
        excluded_keys = {
            "artifact_id",
            "search_text",
            "full_text",
            "creator_ids",
            "set_ids",
            "set_numbers",
            "exhibition_ids",
            "exhibition_types",
            "bibliography",
        }
        if not self.settings.CHAT_INCLUDE_ORIGIN_HISTORY_IN_LLM_CONTEXT:
            excluded_keys.update({"origin_history", "historical_origin"})
        for index, doc in enumerate(docs[: max(top_k, 1)], start=1):
            prompt_doc = {
                key: value
                for key, value in doc.items()
                if key not in excluded_keys and value not in (None, "")
            }
            snippet = json.dumps(prompt_doc, ensure_ascii=True)
            snippets.append(f"[doc_{index}] {snippet}")
        return "\n".join(snippets)

    def _as_string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                items.append(text)
        return items

    async def _build_artifact_results(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_docs: list[dict[str, object]],
        max_images_per_artifact: int | None = None,
        artifact_image_hits: list[dict[str, Any]] | None = None,
    ) -> list[ArtifactResult]:
        if not artifact_docs:
            return []

        artifact_ids = self._extract_artifact_ids_from_docs(artifact_docs)
        image_hits: list[dict[str, Any]] = list(artifact_image_hits or [])
        image_limit = (
            max(int(max_images_per_artifact), 0)
            if max_images_per_artifact is not None
            else None
        )
        if artifact_ids and image_limit != 0 and artifact_image_hits is None:
            if image_limit is None:
                per_artifact = 1
                total_expected = 0
                for doc in artifact_docs:
                    raw_count = doc.get("image_count")
                    if isinstance(raw_count, str) and raw_count.strip().isdigit():
                        count = int(raw_count.strip())
                    elif isinstance(raw_count, (int, float)):
                        count = int(raw_count)
                    else:
                        count = 0
                    count = max(count, 1)
                    per_artifact = max(per_artifact, count)
                    total_expected += count

                per_artifact = min(per_artifact, 120)
                max_total = min(max(total_expected, len(artifact_ids)), 1000)
            else:
                per_artifact = max(image_limit, 1)
                max_total = max(len(artifact_ids) * per_artifact, 1)

            try:
                image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    artifact_ids=artifact_ids,
                    per_artifact=per_artifact,
                    max_total=max_total,
                )
            except Exception as exc:
                image_hits = []

        images_by_artifact: dict[str, list[dict[str, Any]]] = {}
        for hit in image_hits:
            artifact_id = str(hit.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            images_by_artifact.setdefault(artifact_id, []).append(hit)

        results: list[ArtifactResult] = []
        for doc in artifact_docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id:
                continue

            images: list[ArtifactImageResult] = []
            seen_keys: set[str] = set()

            def _append_image(
                *,
                image_id: str | None = None,
                image_order: int | None = None,
                local_path: str | None = None,
                source_url: str | None = None,
                caption: str | None = None,
                alt_text: str | None = None,
                original_image_name: str | None = None,
            ) -> None:
                normalized_image_id = str(image_id or "").strip() or None
                normalized_local_path = str(local_path or "").strip().replace("\\", "/") or None
                normalized_source_url = str(source_url or "").strip() or None
                normalized_original_name = str(original_image_name or "").strip() or None
                if normalized_original_name is None and normalized_local_path:
                    normalized_original_name = normalized_local_path.strip("/")
                if normalized_original_name is None and normalized_source_url:
                    normalized_original_name = Path(
                        urlparse(normalized_source_url).path
                    ).name.strip() or None
                if normalized_original_name is None and normalized_image_id:
                    normalized_original_name = f"{normalized_image_id.replace(':', '_')}.jpg"

                dedupe_key = (
                    normalized_image_id
                    or normalized_local_path
                    or normalized_source_url
                    or normalized_original_name
                )
                if image_limit is not None and len(images) >= image_limit:
                    return
                if not dedupe_key:
                    return
                if dedupe_key in seen_keys:
                    return
                seen_keys.add(dedupe_key)

                images.append(
                    ArtifactImageResult(
                        original_image_name=normalized_original_name,
                        image_id=normalized_image_id,
                        image_order=image_order,
                        local_path=normalized_local_path,
                        source_url=normalized_source_url,
                        caption=str(caption or "").strip() or None,
                        alt_text=str(alt_text or "").strip() or None,
                    )
                )

            for hit in images_by_artifact.get(artifact_id, []):
                _append_image(
                    image_id=str(hit.get("image_id") or hit.get("id") or "").strip() or None,
                    image_order=(
                        hit.get("image_order")
                        if isinstance(hit.get("image_order"), int)
                        else None
                    ),
                    local_path=str(hit.get("local_path") or "").strip() or None,
                    source_url=str(hit.get("source_url") or "").strip() or None,
                    caption=str(hit.get("caption") or "").strip() or None,
                    alt_text=str(hit.get("alt_text") or "").strip() or None,
                    original_image_name=str(hit.get("original_image_name") or "").strip() or None,
                )

            image_paths = self._as_string_list(doc.get("image_paths"))
            image_urls = self._as_string_list(doc.get("image_urls"))
            image_ids = self._as_string_list(doc.get("image_ids"))
            max_len = max(len(image_paths), len(image_urls), len(image_ids))
            for index in range(max_len):
                _append_image(
                    image_id=image_ids[index] if index < len(image_ids) else None,
                    local_path=image_paths[index] if index < len(image_paths) else None,
                    source_url=image_urls[index] if index < len(image_urls) else None,
                )

            raw_image_count = doc.get("image_count")
            image_count_value: int | None = None
            if isinstance(raw_image_count, str) and raw_image_count.strip().isdigit():
                image_count_value = int(raw_image_count.strip())
            elif isinstance(raw_image_count, (int, float)):
                image_count_value = int(raw_image_count)
            if image_count_value is not None and image_count_value < 0:
                image_count_value = None

            # --- Campos relacionais (export relacional do RAIZ) ---
            def _as_str_list(value: Any) -> list[str]:
                if not isinstance(value, list):
                    return []
                out: list[str] = []
                for item in value:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if text:
                        out.append(text)
                return out

            def _as_int(value: Any) -> int | None:
                if isinstance(value, bool):
                    return None
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
                if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                    return int(value.strip())
                return None

            creators_list = _as_str_list(doc.get("creators"))
            legacy_creator = str(doc.get("creator") or "").strip()
            if not creators_list and legacy_creator:
                creators_list = [legacy_creator]
            creator_value = legacy_creator or (creators_list[0] if creators_list else None)

            raw_in_tour = doc.get("in_tour")
            in_tour_bool = bool(raw_in_tour) if not isinstance(raw_in_tour, str) else raw_in_tour.strip().lower() in {"true", "1", "yes"}

            results.append(
                ArtifactResult(
                    artifact_id=artifact_id,
                    tipo_inventario=str(doc.get("tipo_inventario") or "").strip() or None,
                    inventory_number=self._doc_inventory(doc) or None,
                    title=str(doc.get("title") or "").strip() or None,
                    museum_id=str(doc.get("museum_id") or "").strip() or None,
                    museum=str(doc.get("museum") or doc.get("museum_name") or "").strip() or None,
                    category=str(doc.get("category") or "").strip() or None,
                    super_category=str(doc.get("super_category") or "").strip() or None,
                    creator=creator_value,
                    creators=creators_list,
                    creator_ids=_as_str_list(doc.get("creator_ids")),
                    date_or_period=str(doc.get("date_or_period") or "").strip() or None,
                    start_year=_as_int(doc.get("start_year")),
                    end_year=_as_int(doc.get("end_year")),
                    support_or_material=str(
                        doc.get("support_or_material") or doc.get("support") or ""
                    ).strip()
                    or None,
                    technique=str(doc.get("technique") or "").strip() or None,
                    origin_history=str(
                        doc.get("origin_history") or doc.get("historical_origin") or ""
                    ).strip()
                    or None,
                    incorporation=str(doc.get("incorporation") or "").strip() or None,
                    production_center=str(
                        doc.get("production_center")
                        or doc.get("manufacturer_location")
                        or doc.get("location")
                        or ""
                    ).strip()
                    or None,
                    description=str(doc.get("description") or "").strip() or None,
                    search_text=self._doc_search_text(doc) or None,
                    detail_type=str(doc.get("detail_type") or "").strip() or None,
                    detail_url=str(doc.get("detail_url") or "").strip() or None,
                    in_tour=in_tour_bool,
                    sets=_as_str_list(doc.get("sets")),
                    set_ids=_as_str_list(doc.get("set_ids")),
                    set_numbers=_as_str_list(doc.get("set_numbers")),
                    exhibitions=_as_str_list(doc.get("exhibitions")),
                    exhibition_ids=_as_str_list(doc.get("exhibition_ids")),
                    exhibition_types=_as_str_list(doc.get("exhibition_types")),
                    exhibition_count=_as_int(doc.get("exhibition_count")),
                    bibliography=str(doc.get("bibliography") or "").strip() or None,
                    bibliography_count=_as_int(doc.get("bibliography_count")),
                    image_count=image_count_value,
                    images=images,
                )
            )

        return results

    async def get_artifact_full(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_id: str,
    ) -> ArtifactResult | None:
        """Devolve um ArtifactResult completo (com imagens) por id, ou None."""
        artifact_id = (artifact_id or "").strip()
        if not artifact_id:
            return None
        docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_ids=[artifact_id],
            top_k=1,
        )
        if not docs:
            return None
        results = await self._build_artifact_results(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_docs=docs,
        )
        return results[0] if results else None

    async def get_artifact_full_by_inventory(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        inventory_number: str,
    ) -> ArtifactResult | None:
        """Devolve um ArtifactResult completo por numero de inventario, ou None."""
        inventory_number = (inventory_number or "").strip()
        if not inventory_number:
            return None
        docs = await self.opensearch_gateway.fetch_artifacts_by_inventory_numbers(
            museum_slug=museum_slug,
            museum_id=museum_id,
            inventory_numbers=[inventory_number],
            top_k=1,
        )
        if not docs:
            docs = await self.opensearch_gateway.search_artifacts_by_inventory_candidates(
                museum_slug=museum_slug,
                museum_id=museum_id,
                inventory_numbers=[inventory_number],
                top_k=1,
            )
        if not docs:
            return None
        results = await self._build_artifact_results(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_docs=docs,
        )
        return results[0] if results else None

    def _build_related_artifact_images(self, art: dict[str, Any]) -> list[ArtifactImageResult]:
        related_images: list[ArtifactImageResult] = []
        image_paths = self._as_string_list(art.get("image_paths"))
        image_urls = self._as_string_list(art.get("image_urls"))
        image_ids = self._as_string_list(art.get("image_ids") or art.get("image_file_ids"))
        max_len = min(max(len(image_paths), len(image_urls), len(image_ids)), 1)
        for index in range(max_len):
            image_id = image_ids[index] if index < len(image_ids) else None
            local_path = image_paths[index] if index < len(image_paths) else None
            source_url = image_urls[index] if index < len(image_urls) else None
            original_name = local_path
            if not original_name and image_id:
                original_name = f"{image_id.replace(':', '_')}.jpg"
            if original_name or image_id or local_path or source_url:
                related_images.append(
                    ArtifactImageResult(
                        original_image_name=original_name,
                        image_id=image_id,
                        local_path=local_path,
                        source_url=source_url,
                    )
                )
        return related_images

    async def _build_related_images_lookup(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        related_docs: list[dict[str, Any]],
    ) -> dict[str, list[ArtifactImageResult]]:
        if not related_docs:
            return {}

        artifact_ids = self._extract_artifact_ids_from_docs(related_docs)
        image_hits: list[dict[str, Any]] = []
        if artifact_ids:
            try:
                image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    artifact_ids=artifact_ids,
                    per_artifact=1,
                    max_total=max(len(artifact_ids), 1),
                )
            except Exception as exc:
                image_hits = []

        artifact_results = await self._build_artifact_results(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_docs=related_docs,
            max_images_per_artifact=1,
            artifact_image_hits=image_hits,
        )
        return {
            result.artifact_id: result.images
            for result in artifact_results
            if result.artifact_id
        }

    def _build_navigation_lookup_for_related(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        related_docs: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        inventories: list[str] = []
        for art in related_docs:
            inv = str(art.get("inventory_number") or "").strip()
            if inv:
                inventories.append(inv)

        navigation_lookup: dict[str, dict[str, Any]] = {}
        if not inventories:
            return navigation_lookup

        try:
            navigation_targets_raw = self.tour_navigation_service.resolve_targets(
                museum_slug=museum_slug,
                museum_id=museum_id,
                inventories=inventories,
                limit=max(len(inventories), 1),
            )
        except Exception:
            return navigation_lookup

        for tgt in navigation_targets_raw:
            inv_id = str(tgt.get("inventory_id") or "").strip()
            if inv_id and inv_id not in navigation_lookup:
                navigation_lookup[inv_id] = tgt
        return navigation_lookup

    def _build_related_item(
        self,
        art: dict[str, Any],
        *,
        navigation_lookup: dict[str, dict[str, Any]],
        images_by_artifact: dict[str, list[ArtifactImageResult]] | None = None,
    ) -> RelatedArtifactItem:
        artifact_id = str(art.get("artifact_id") or "")
        inv = str(art.get("inventory_number") or "").strip() or None
        nav = navigation_lookup.get(inv or "") if inv else None
        nav_target = None
        if nav:
            nav_target = TourNavigationTarget(
                overlay_id=str(nav.get("overlay_id") or ""),
                panorama_key=str(nav.get("panorama_key") or ""),
                inventory_id=str(nav.get("inventory_id") or ""),
                location=nav.get("location"),
                title=nav.get("title"),
            )
        related_images = (
            (images_by_artifact or {}).get(artifact_id)
            or self._build_related_artifact_images(art)
        )
        return RelatedArtifactItem(
            artifact_id=artifact_id,
            inventory_number=inv,
            title=str(art.get("title") or "").strip() or None,
            museum_id=str(art.get("museum_id") or "").strip() or None,
            museum=str(art.get("museum") or "").strip() or None,
            category=str(art.get("category") or "").strip() or None,
            creators=list(art.get("creators") or []),
            date_or_period=str(art.get("date_or_period") or "").strip() or None,
            detail_type=str(art.get("detail_type") or "").strip() or None,
            detail_url=str(art.get("detail_url") or "").strip() or None,
            in_tour=bool(art.get("in_tour")),
            image_count=art.get("image_count") if isinstance(art.get("image_count"), int) else None,
            images=related_images,
            navigation_target=nav_target,
        )

    async def get_related_artifacts_page(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_id: str,
        tipo: str,
        entity_id: str,
        offset: int,
        limit: int,
    ) -> RelatedArtifactsPageResponse:
        cleaned_tipo = (tipo or "").strip()
        if cleaned_tipo not in {"conjunto", "exposicao"}:
            raise ValueError("tipo invalido.")
        cleaned_entity_id = (entity_id or "").strip()
        cleaned_artifact_id = (artifact_id or "").strip()
        resolved_offset = max(int(offset), 0)
        resolved_limit = max(1, min(int(limit), 50))

        related, total = await self.opensearch_gateway.fetch_artifacts_by_entity(
            tipo=cleaned_tipo,
            entity_id=cleaned_entity_id,
            museum_slug=museum_slug,
            museum_id=museum_id,
            top_k=resolved_limit,
            from_offset=resolved_offset,
            exclude_artifact_id=cleaned_artifact_id or None,
        )
        navigation_lookup = await asyncio.to_thread(
            self._build_navigation_lookup_for_related,
            museum_slug=museum_slug,
            museum_id=museum_id,
            related_docs=related,
        )
        images_by_artifact = await self._build_related_images_lookup(
            museum_slug=museum_slug,
            museum_id=museum_id,
            related_docs=related,
        )
        artifacts = [
            self._build_related_item(
                art,
                navigation_lookup=navigation_lookup,
                images_by_artifact=images_by_artifact,
            )
            for art in related
        ]
        return RelatedArtifactsPageResponse(
            artifact_id=cleaned_artifact_id,
            tipo=cleaned_tipo,  # type: ignore[arg-type]
            entity_id=cleaned_entity_id,
            artifacts=artifacts,
            artifacts_offset=resolved_offset,
            artifacts_limit=resolved_limit,
            artifacts_total=max(int(total), 0),
            artifacts_has_more=resolved_offset + len(artifacts) < max(int(total), 0),
        )

    # ------------------------------------------------------------------ #
    # Modal de detalhe: autores + conjuntos + exposicoes do artefacto.
    # ------------------------------------------------------------------ #
    async def get_artifact_detail_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_id: str,
    ) -> ArtifactDetailContextResponse:
        """Devolve, para um artefacto, info expandida de autores/conjuntos/exposicoes
        + listas de outros artefactos ligados a cada conjunto/exposicao."""
        artifact_id = (artifact_id or "").strip()
        if not artifact_id:
            return ArtifactDetailContextResponse(artifact_id="")

        # 1) Obter o artifact_doc para saber os ids das entidades a buscar.
        artifact_docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_ids=[artifact_id],
            top_k=1,
        )
        if not artifact_docs:
            return ArtifactDetailContextResponse(artifact_id=artifact_id)
        doc = artifact_docs[0]

        def _as_list(value: Any) -> list[str]:
            if isinstance(value, str):
                return [
                    item.strip()
                    for item in re.split(r"[;,\n|]+", value)
                    if item.strip()
                ]
            if not isinstance(value, list):
                return []
            return [str(v).strip() for v in value if str(v or "").strip()]

        def _entity_lookup_ids(tipo: str, values: list[str]) -> list[str]:
            prefix = f"{tipo}:"
            candidates: list[str] = []
            seen: set[str] = set()
            for value in values:
                cleaned = str(value or "").strip()
                if not cleaned:
                    continue
                variants = (
                    [cleaned, cleaned[len(prefix) :]]
                    if cleaned.startswith(prefix)
                    else [f"{prefix}{cleaned}", cleaned]
                )
                for variant in variants:
                    normalized = variant.strip()
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    candidates.append(normalized)
            return candidates

        creator_ids = _as_list(doc.get("creator_ids"))
        set_ids = _as_list(doc.get("set_ids"))
        exhibition_ids = _as_list(doc.get("exhibition_ids"))

        # Conjuntos/exposicoes usam entity_id prefixado no indice relacional. Autores
        # sao mais simples: creator_ids ja contem o _id do indice cultural_heritage_authors.
        prefixed_sets = _entity_lookup_ids("conjunto", set_ids)
        prefixed_exhibitions = _entity_lookup_ids("exposicao", exhibition_ids)

        top_k_related = int(getattr(self.settings, "CHAT_RELATED_ARTIFACTS_PAGE_SIZE", 10) or 10)

        # 2) Buscar metainfo de cada entidade em paralelo.
        authors_task = self.opensearch_gateway.fetch_authors_by_ids(
            author_ids=creator_ids,
        ) if creator_ids else asyncio.sleep(0, result=[])
        sets_task = self.opensearch_gateway.fetch_entities_by_ids(
            tipo="conjunto", entity_ids=prefixed_sets,
        ) if prefixed_sets else asyncio.sleep(0, result=[])
        exhibitions_task = self.opensearch_gateway.fetch_entities_by_ids(
            tipo="exposicao", entity_ids=prefixed_exhibitions,
        ) if prefixed_exhibitions else asyncio.sleep(0, result=[])

        authors_docs, sets_docs, exhibitions_docs = await asyncio.gather(
            authors_task, sets_task, exhibitions_task,
            return_exceptions=False,
        )

        # 3) Para cada conjunto/exposicao, buscar artefactos relacionados em paralelo.
        set_related_tasks = [
            self.opensearch_gateway.fetch_artifacts_by_entity(
                tipo="conjunto",
                entity_id=str(s.get("entity_id") or ""),
                museum_slug=museum_slug,
                museum_id=museum_id,
                top_k=top_k_related,
                exclude_artifact_id=artifact_id,
            )
            for s in sets_docs
        ]
        exhibition_related_tasks = [
            self.opensearch_gateway.fetch_artifacts_by_entity(
                tipo="exposicao",
                entity_id=str(e.get("entity_id") or ""),
                museum_slug=museum_slug,
                museum_id=museum_id,
                top_k=top_k_related,
                exclude_artifact_id=artifact_id,
            )
            for e in exhibitions_docs
        ]
        all_related = await asyncio.gather(
            *(set_related_tasks + exhibition_related_tasks), return_exceptions=False
        )
        set_related = all_related[: len(set_related_tasks)]
        exhibition_related = all_related[len(set_related_tasks) :]

        # 4) Resolver imagens e navigation_targets dos artefactos relacionados (em massa).
        all_related_docs: list[dict[str, Any]] = []
        all_inventories: list[str] = []
        for related, _total in list(set_related) + list(exhibition_related):
            all_related_docs.extend(related)
            for art in related:
                inv = str(art.get("inventory_number") or "").strip()
                if inv:
                    all_inventories.append(inv)
        images_by_artifact = await self._build_related_images_lookup(
            museum_slug=museum_slug,
            museum_id=museum_id,
            related_docs=all_related_docs,
        )
        navigation_lookup: dict[str, dict[str, Any]] = {}
        if all_inventories:
            try:
                navigation_targets_raw = await asyncio.to_thread(
                    self.tour_navigation_service.resolve_targets,
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    inventories=all_inventories,
                    limit=max(len(all_inventories), 1),
                )
            except Exception:
                navigation_targets_raw = []
            for tgt in navigation_targets_raw:
                inv_id = str(tgt.get("inventory_id") or "").strip()
                if inv_id and inv_id not in navigation_lookup:
                    navigation_lookup[inv_id] = tgt

        # 5) Construir resposta.
        authors_response: list[AuthorEntity] = []
        for a in authors_docs:
            biography = str(a.get("biografia") or a.get("biography") or "").strip() or None
            authors_response.append(
                AuthorEntity(
                    entity_id=str(a.get("entity_id") or ""),
                    name=str(a.get("name") or "").strip() or None,
                    atividade=str(a.get("atividade") or "").strip() or None,
                    data_nascimento=str(a.get("data_nascimento") or "").strip() or None,
                    data_obito=str(a.get("data_obito") or "").strip() or None,
                    local_nascimento=str(a.get("local_nascimento") or "").strip() or None,
                    local_obito=str(a.get("local_obito") or "").strip() or None,
                    biografia=biography,
                    biography=biography,
                    url=str(a.get("url") or "").strip() or None,
                    n_objetos=a.get("n_objetos") if isinstance(a.get("n_objetos"), int) else None,
                )
            )

        sets_response: list[SetEntityWithArtifacts] = []
        for s, (related, total) in zip(sets_docs, set_related):
            sets_response.append(
                SetEntityWithArtifacts(
                    entity_id=str(s.get("entity_id") or ""),
                    name=str(s.get("name") or "").strip() or None,
                    num_conjunto=str(s.get("num_conjunto") or "").strip() or None,
                    historial=str(s.get("historial") or "").strip() or None,
                    descricao=str(s.get("descricao") or "").strip() or None,
                    url=str(s.get("url") or "").strip() or None,
                    n_objetos=total,
                    artifacts=[
                        self._build_related_item(
                            a,
                            navigation_lookup=navigation_lookup,
                            images_by_artifact=images_by_artifact,
                        )
                        for a in related
                    ],
                    artifacts_returned=len(related),
                )
            )

        exhibitions_response: list[ExhibitionEntityWithArtifacts] = []
        for e, (related, total) in zip(exhibitions_docs, exhibition_related):
            exhibitions_response.append(
                ExhibitionEntityWithArtifacts(
                    entity_id=str(e.get("entity_id") or ""),
                    name=str(e.get("name") or "").strip() or None,
                    tipo_exposicao=str(e.get("tipo_exposicao") or "").strip() or None,
                    local=str(e.get("local") or "").strip() or None,
                    ano_inicial=e.get("ano_inicial") if isinstance(e.get("ano_inicial"), int) else None,
                    ano_final=e.get("ano_final") if isinstance(e.get("ano_final"), int) else None,
                    texto=str(e.get("texto") or "").strip() or None,
                    ficha_tecnica=str(e.get("ficha_tecnica") or "").strip() or None,
                    url=str(e.get("url") or "").strip() or None,
                    n_objetos=total,
                    artifacts=[
                        self._build_related_item(
                            a,
                            navigation_lookup=navigation_lookup,
                            images_by_artifact=images_by_artifact,
                        )
                        for a in related
                    ],
                    artifacts_returned=len(related),
                )
            )

        return ArtifactDetailContextResponse(
            artifact_id=artifact_id,
            authors=authors_response,
            sets=sets_response,
            exhibitions=exhibitions_response,
        )

    def _build_image_matches(
        self,
        *,
        image_hits: list[dict[str, object]],
        artifact_docs: list[dict[str, object]],
    ) -> list[ImageMatchResult]:
        doc_by_inventory: dict[str, dict[str, object]] = {}
        ordered_inventories: list[str] = []
        doc_by_id: dict[str, dict[str, object]] = {}
        ordered_artifact_ids: list[str] = []
        for doc in artifact_docs:
            inventory = self._doc_inventory(doc)
            if inventory:
                inventory_key = inventory.casefold()
                doc_by_inventory[inventory_key] = doc
                if inventory_key not in ordered_inventories:
                    ordered_inventories.append(inventory_key)

            artifact_id = str(doc.get("artifact_id") or "").strip()
            if artifact_id:
                doc_by_id[artifact_id] = doc
                if artifact_id not in ordered_artifact_ids:
                    ordered_artifact_ids.append(artifact_id)

        best_hit_by_inventory: dict[str, dict[str, object]] = {}
        best_hit_by_artifact: dict[str, dict[str, object]] = {}

        def register_best(
            bucket: dict[str, dict[str, object]],
            key: str,
            hit: dict[str, object],
            score_value: float,
            original_image_name: str,
        ) -> None:
            current_best = bucket.get(key)
            if current_best is None:
                bucket[key] = {
                    "hit": hit,
                    "score_value": score_value,
                    "original_image_name": original_image_name,
                }
                return

            current_score = current_best.get("score_value")
            current_score_value = (
                float(current_score)
                if isinstance(current_score, (int, float))
                else float("-inf")
            )
            if score_value > current_score_value:
                bucket[key] = {
                    "hit": hit,
                    "score_value": score_value,
                    "original_image_name": original_image_name,
                }

        for hit in image_hits:
            original_image_name = self._image_match_name(hit)
            if not original_image_name:
                continue
            inventory = self._image_hit_inventory(hit)
            artifact_id = str(hit.get("artifact_id") or "").strip()
            score_raw = hit.get("score")
            score_value = float(score_raw) if isinstance(score_raw, (int, float)) else float("-inf")
            if inventory:
                register_best(
                    best_hit_by_inventory,
                    inventory.casefold(),
                    hit,
                    score_value,
                    original_image_name,
                )
            if artifact_id:
                register_best(
                    best_hit_by_artifact,
                    artifact_id,
                    hit,
                    score_value,
                    original_image_name,
                )

        def build_match(
            *,
            best_entry: dict[str, object],
            doc: dict[str, object],
        ) -> ImageMatchResult | None:
            hit_obj = best_entry.get("hit")
            if not isinstance(hit_obj, dict):
                return None
            hit: dict[str, object] = hit_obj
            original_image_name = str(best_entry.get("original_image_name") or "").strip()
            if not original_image_name:
                original_image_name = self._image_match_name(hit)
            if not original_image_name:
                return None

            score_raw = hit.get("score")
            score: float | None = None
            if isinstance(score_raw, (int, float)):
                score = float(score_raw)

            return ImageMatchResult(
                original_image_name=original_image_name,
                artifact_id=str(doc.get("artifact_id") or hit.get("artifact_id") or "").strip()
                or None,
                score=score,
                title=str(doc.get("title") or hit.get("artifact_title") or "").strip() or None,
                inventory=self._doc_inventory(doc) or self._image_hit_inventory(hit) or None,
                image_id=str(hit.get("image_id") or hit.get("id") or "").strip() or None,
                local_path=str(hit.get("local_path") or "").strip() or None,
                source_url=str(hit.get("source_url") or "").strip() or None,
            )

        matches: list[ImageMatchResult] = []
        for inventory_key in ordered_inventories:
            best_entry = best_hit_by_inventory.get(inventory_key)
            if not isinstance(best_entry, dict):
                continue
            doc = doc_by_inventory.get(inventory_key, {})
            match = build_match(best_entry=best_entry, doc=doc)
            if match is not None:
                matches.append(match)

        if matches:
            return matches

        for artifact_id in ordered_artifact_ids:
            best_entry = best_hit_by_artifact.get(artifact_id)
            if not isinstance(best_entry, dict):
                continue
            doc = doc_by_id.get(artifact_id, {})
            match = build_match(best_entry=best_entry, doc=doc)
            if match is not None:
                matches.append(match)
        return matches

    def _enrich_image_matches(
        self,
        *,
        context: str,
        conversation_id: str,
        museum_slug: str,
        image_matches: list[ImageMatchResult],
        artifact_results: list[ArtifactResult],
        navigation_targets: list[TourNavigationTarget],
    ) -> list[ImageMatchResult]:
        if not image_matches:
            return []

        artifact_by_id = {
            artifact.artifact_id: artifact
            for artifact in artifact_results
            if artifact.artifact_id
        }
        artifact_by_inventory = {
            str(artifact.inventory_number or "").strip().casefold(): artifact
            for artifact in artifact_results
            if str(artifact.inventory_number or "").strip()
        }
        navigation_by_inventory = {
            str(target.inventory_id or "").strip().casefold(): target
            for target in navigation_targets
            if str(target.inventory_id or "").strip()
        }

        enriched: list[ImageMatchResult] = []
        for match in image_matches:
            artifact_id = str(match.artifact_id or "").strip()
            inventory = str(match.inventory or "").strip()
            inventory_key = inventory.casefold()
            artifact = artifact_by_inventory.get(inventory_key) if inventory_key else None
            if artifact is None and artifact_id:
                artifact = artifact_by_id.get(artifact_id)

            target: TourNavigationTarget | None = None
            if inventory_key:
                target = navigation_by_inventory.get(inventory_key)
            if target is None and artifact is not None:
                artifact_inventory = str(artifact.inventory_number or "").strip()
                if artifact_inventory:
                    target = navigation_by_inventory.get(artifact_inventory.casefold())

            enriched.append(
                match.model_copy(
                    update={
                        "artifact": artifact.model_dump(exclude_none=True) if artifact else None,
                        "navigation_target": target.model_dump(exclude_none=True) if target else None,
                    }
                )
            )

        return enriched

    def _resolve_navigation_targets(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        docs: list[dict[str, object]],
    ) -> list[TourNavigationTarget]:
        inventories: list[str] = []
        for doc in docs:
            inventory = self._doc_inventory(doc)
            if inventory:
                inventories.append(inventory)

        if not inventories:
            return []

        resolved = self.tour_navigation_service.resolve_targets(
            museum_slug=museum_slug,
            museum_id=museum_id,
            inventories=inventories,
            limit=max(len(inventories), 1),
        )

        targets: list[TourNavigationTarget] = []
        for item in resolved:
            overlay_id = str(item.get("overlay_id") or "").strip()
            panorama_key = str(item.get("panorama_key") or "").strip()
            inventory_id = str(item.get("inventory_id") or "").strip()
            if not overlay_id or not panorama_key or not inventory_id:
                continue
            targets.append(
                TourNavigationTarget(
                    overlay_id=overlay_id,
                    panorama_key=panorama_key,
                    inventory_id=inventory_id,
                    location=str(item.get("location") or "").strip() or None,
                    title=str(item.get("title") or "").strip() or None,
                )
            )
        return targets

    def _sanitize_assistant_reply(
        self,
        text: str,
        *,
        docs: list[dict[str, object]] | None = None,
        language: str | None = None,
    ) -> str:
        cleaned = text or ""
        doc_label_map = self._build_doc_label_map(docs or [], language=language)
        if doc_label_map:
            cleaned = self._replace_doc_refs_with_labels(
                text=cleaned,
                doc_label_map=doc_label_map,
                language=language,
            )

        artifact_label = translate("sanitizer.artifact_label", language)
        context_label = translate("sanitizer.context_label", language)
        cleaned = re.sub(
            r"`?\bartifact[_-]?\d+\b`?",
            artifact_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(?:retrieval_context|current_message|explicit_state|recent_history_aux|rolling_summary_aux|current_page_results|current_visible_results|visible_results_count|visible_results_page|visible_results_page_size|visible_results_total|search_query|reported_total|results_page)\b",
            context_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(?i)\bcom base na informac[aã]o fornecida pelo contexto\b[:,]?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(?i)\bbased on the information provided by the context\b[:,]?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            rf"\b(?:documento|document)\s+{re.escape(artifact_label)}\b",
            artifact_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\bdoc(?:umento|ument)?[_\s-]*\d+\b",
            artifact_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([,.;:!?])", r"\1", cleaned)
        cleaned = self._renumber_ordered_list_markers(cleaned)
        return cleaned.strip()

    def _renumber_ordered_list_markers(self, text: str) -> str:
        item_pattern = re.compile(r"^(\s*)(\d{1,2})([.)])(\s+)(.+)$")
        lines = text.splitlines()
        item_indexes: list[int] = []

        def flush_item_indexes() -> None:
            if len(item_indexes) < 2:
                item_indexes.clear()
                return
            for list_number, line_index in enumerate(item_indexes, start=1):
                match = item_pattern.match(lines[line_index])
                if not match:
                    continue
                lines[line_index] = (
                    f"{match.group(1)}{list_number}{match.group(3)}"
                    f"{match.group(4)}{match.group(5)}"
                )
            item_indexes.clear()

        for index, line in enumerate(lines):
            if item_pattern.match(line):
                item_indexes.append(index)
                continue
            if not line.strip():
                continue
            flush_item_indexes()

        flush_item_indexes()
        return "\n".join(lines)

    def _build_doc_label_map(
        self,
        docs: list[dict[str, object]],
        *,
        language: str | None = None,
    ) -> dict[int, str]:
        labels: dict[int, str] = {}
        for index, doc in enumerate(docs, start=1):
            title = str(doc.get("title") or "").strip()
            inventory = self._doc_inventory(doc)
            if title and inventory:
                labels[index] = translate(
                    "sanitizer.titled_inventory_artifact",
                    language,
                    title=title,
                    inventory=inventory,
                )
            elif title:
                labels[index] = translate(
                    "sanitizer.titled_artifact",
                    language,
                    title=title,
                )
            elif inventory:
                labels[index] = translate(
                    "sanitizer.inventory_artifact",
                    language,
                    inventory=inventory,
                )
            else:
                labels[index] = translate("sanitizer.collection_artifact", language)
        return labels

    def _compact_for_match(self, value: str) -> str:
        lowered = (value or "").casefold()
        compact = re.sub(r"[\W_]+", "", lowered, flags=re.UNICODE)
        return compact

    def _replace_doc_refs_with_labels(
        self,
        *,
        text: str,
        doc_label_map: dict[int, str],
        language: str | None = None,
    ) -> str:
        fallback_label = translate("sanitizer.collection_artifact", language)

        def _replace(match: re.Match[str]) -> str:
            idx_text = match.group(1)
            try:
                idx = int(idx_text)
            except Exception:
                return fallback_label
            return doc_label_map.get(idx, fallback_label)

        replaced = re.sub(r"\[doc_(\d+)\]", _replace, text, flags=re.IGNORECASE)
        replaced = re.sub(r"\bdoc[_-](\d+)\b", _replace, replaced, flags=re.IGNORECASE)
        replaced = re.sub(
            r"\b(?:documento|document)\s+\[?doc[_-]?(\d+)\]?\b",
            _replace,
            replaced,
            flags=re.IGNORECASE,
        )
        return replaced

    def _build_final_prompt(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        router_decision: dict[str, object],
        retrieval_context: str,
        effective_filters: dict[str, object],
        effective_sort: dict[str, object],
        use_history_for_answer: bool,
        museum_slug: str | None = None,
        museum_name: str | None = None,
    ) -> str:
        mode = str(router_decision.get("mode", "llm_only"))
        rewritten_query = str(router_decision.get("rewritten_query", payload.message))
        return build_final_answer_prompt(
            museum_slug=museum_slug or payload.museum_slug,
            museum_name=museum_name if museum_name is not None else payload.museum_name,
            input_modality="text",
            mode=mode,
            intent=str(router_decision.get("intent", "fallback")),
            rolling_summary=state.rolling_summary if use_history_for_answer else "",
            filters_state=effective_filters,
            sort_state=effective_sort,
            user_message=payload.message,
            rewritten_query=rewritten_query,
            retrieval_context=retrieval_context,
            use_history_for_answer=use_history_for_answer,
            language=state.language,
        )

    def _resolve_museum_id_values(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        metadata: dict[str, object] | None = None,
    ) -> str | None:
        if museum_id:
            return museum_id.strip() or None

        if metadata and isinstance(metadata, dict):
            metadata_museum_id = metadata.get("museum_id")
            if isinstance(metadata_museum_id, str) and metadata_museum_id.strip():
                return metadata_museum_id.strip()

        return museum_slug.strip() or None

    def _resolve_explicit_museum_id_values(
        self,
        *,
        museum_id: str | None,
        metadata: dict[str, object] | None = None,
    ) -> str | None:
        if museum_id:
            return museum_id.strip() or None

        if metadata and isinstance(metadata, dict):
            metadata_museum_id = metadata.get("museum_id")
            if isinstance(metadata_museum_id, str) and metadata_museum_id.strip():
                return metadata_museum_id.strip()

        return None

    def _resolve_museum_id(self, payload: ChatMessageRequest) -> str | None:
        return self._resolve_museum_id_values(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            metadata=payload.metadata if isinstance(payload.metadata, dict) else None,
        )

    def _extract_artifact_ids_from_docs(self, docs: list[dict[str, object]]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id or artifact_id in seen:
                continue
            seen.add(artifact_id)
            ids.append(artifact_id)
        return ids

    def _update_last_result_ids(
        self,
        *,
        state: ChatSessionState,
        artifact_docs: list[dict[str, object]],
        preserve_existing_when_empty: bool = False,
    ) -> None:
        artifact_ids = self._extract_artifact_ids_from_docs(artifact_docs)
        if artifact_ids:
            state.last_result_ids = artifact_ids
        elif not preserve_existing_when_empty:
            state.last_result_ids = []

    def _merge_state_with_delta(
        self,
        *,
        base: dict[str, object],
        delta: object,
    ) -> dict[str, object]:
        merged: dict[str, object] = dict(base)
        if not isinstance(delta, dict):
            return merged

        for key, value in delta.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value

        return merged

    def _apply_router_decision_to_state(
        self,
        *,
        state: ChatSessionState,
        router_decision: dict[str, object],
    ) -> None:
        intent = router_decision.get("intent")
        if isinstance(intent, str):
            state.intent = intent

        filters_delta = router_decision.get("filters_delta")
        if isinstance(filters_delta, dict):
            for key, value in filters_delta.items():
                if key == "artifact_id":
                    continue
                if value is None:
                    state.filters.pop(key, None)
                else:
                    state.filters[key] = value

        sort_delta = router_decision.get("sort_delta")
        if isinstance(sort_delta, dict):
            for key, value in sort_delta.items():
                if value is None:
                    state.sort.pop(key, None)
                else:
                    state.sort[key] = value

    def _update_rolling_summary(
        self,
        *,
        state: ChatSessionState,
        latest_user_message: str,
        latest_assistant_message: str,
        router_decision: dict[str, object],
    ) -> None:
        line = (
            f"intent={router_decision.get('intent', 'fallback')} | "
            f"user={latest_user_message[:120]} | "
            f"assistant={latest_assistant_message[:140]}"
        )

        previous = state.rolling_summary.strip()
        candidate = f"{previous}\n{line}".strip() if previous else line
        max_chars = max(self.settings.CHAT_ROLLING_SUMMARY_MAX_CHARS, 120)
        if len(candidate) > max_chars:
            candidate = candidate[-max_chars:]
        state.rolling_summary = candidate



@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        settings=settings,
        opensearch_gateway=get_opensearch_gateway(),
        embedding_provider=get_embedding_provider(),
        model_retrieval_service=get_model_retrieval_service(),
        tour_navigation_service=get_tour_navigation_service(),
        llm_service=get_llm_service(),
        session_store=get_chat_session_store(),
        query_logger=get_query_logger(),
    )
