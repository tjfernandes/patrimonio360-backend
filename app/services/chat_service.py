from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import unicodedata
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.prompts import (
    build_final_answer_prompt,
    build_router_user_prompt,
    get_router_system_prompt,
)
from app.prompts.query_planner_prompts import (
    RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT,
    build_retrieval_query_rewrite_prompt,
)
from app.query_planning import (
    CompiledOpenSearchDSL,
    QueryCompileError,
    QueryExecutionResult,
    QueryPlanningError,
    QueryPlan,
    QuerySchema,
    QuerySchemaField,
    classify_query,
    compile_query,
    plan_query,
)
from app.query_planning.models import ListSpec, TermFilter
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
    ImageMatchResult,
    ResponseFormatObject,
    TourNavigationTarget,
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
from app.services.tour_navigation import TourNavigationService, get_tour_navigation_service

logger = logging.getLogger(__name__)
StatusCallback = Callable[[dict[str, Any]], Awaitable[None]]
EVENT_LABELS: dict[str, str] = {
    "chat.receive": "Recebida mensagem no endpoint de chat.",
    "chat.receive_image": "Recebida mensagem com imagem no endpoint de chat.",
    "chat.receive_model": "Recebida mensagem com modelo 3D no endpoint de chat.",
    "chat.receive_message": "Conteudo de mensagem recebido.",
    "chat.rewrite": "Query reescrita para este input.",
    "chat.router_unavailable": "Router indisponivel; fallback para llm_only.",
    "chat.route": "Decisao de routing (rag vs llm_only).",
    "chat.route_delta": "Deltas de estado calculados pelo router.",
    "chat.retrieve_prepare": "Retrieval preparado antes de consultar OpenSearch.",
    "chat.retrieve_query_rewrite": "Query de retrieval reescrita para OpenSearch.",
    "chat.retrieve_embedding_ready": "Embedding de query gerado para retrieval.",
    "chat.retrieve": "Tentativa de retrieval no RAG.",
    "chat.retrieve_embedding_disabled": "Embeddings de query desativados; retrieval ignorado.",
    "chat.retrieve_embedding_error": "Embedding indisponivel; retrieval cancelado.",
    "chat.retrieve_skipped": "Retrieval ignorado nesta mensagem.",
    "chat.image_embedding_ready": "Embedding multimodal da imagem gerado.",
    "chat.image_retrieve": "Image retrieval executado com sucesso.",
    "chat.image_matches_llm_filter_skipped": "Filtro LLM de image matches ignorado para preservar recall em retrieval.",
    "chat.image_matches_llm_filter": "Image matches filtrados por decisao LLM antes da resposta final.",
    "chat.image_matches_llm_filter_error": "Erro ao filtrar image matches com LLM; fallback para lista original.",
    "chat.image_match_enrichment": "Debug de associacao entre image matches, artefactos e targets de tour.",
    "chat.docs_llm_filter": "Docs de retrieval filtrados por decisao LLM antes da resposta final.",
    "chat.docs_llm_filter_skipped": "Filtro LLM de docs ignorado para preservar recall em query de pesquisa.",
    "chat.docs_llm_filter_error": "Erro ao filtrar docs com LLM; fallback para lista original.",
    "chat.image_retrieve_empty": "Image retrieval sem resultados.",
    "chat.image_retrieve_error": "Erro inesperado durante image retrieval.",
    "chat.model_retrieve": "Model retrieval executado com sucesso.",
    "chat.model_retrieve_empty": "Model retrieval sem resultados.",
    "chat.model_retrieve_error": "Erro inesperado durante model retrieval.",
    "chat.reply_sanitized": "Resposta final sanitizada para remover IDs internos.",
    "chat.llm_error": "Erro ao chamar o LLM.",
    "chat.reply": "Resposta final gerada pelo LLM.",
    "chat.state_after_reply": "Estado da conversa apos resposta.",
    "chat.retrieve_not_implemented": "Retrieval ainda nao implementado neste ambiente.",
    "chat.retrieve_error": "Erro inesperado durante retrieval.",
    "chat.retrieve_results_page": "Pagina de resultados de retrieval devolvida.",
    "chat.navigation_targets": "Alvos de navegacao em tour resolvidos para esta resposta.",
    "chat.context_policy": "Politica de reutilizacao de contexto para o turno atual.",
    "chat.context_artifact_scope": "Filtro contextual por artifact_id aplicado para follow-up referencial.",
    "chat.structured.classify": "Classificacao da query para modo estruturado.",
    "chat.structured.plan": "Plano estruturado gerado para query analitica.",
    "chat.structured.fallback": "Fallback do modo estruturado para fluxo RAG.",
    "chat.structured.execute": "Query estruturada executada no OpenSearch.",
    "chat.structured.reply": "Resposta final gerada por executor estruturado.",
    "chat.retrieve_query_rewrite_guardrail": "Guardrail aplicado a query reescrita pelo LLM.",
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
    "encontra",
    "encontrar",
    "procura",
    "procurar",
    "mostra",
    "museu",
    "peca",
    "pecas",
    "crianca",
    "criancas",
    "vestido",
    "vestidos",
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
    "tile",
    "tiles",
}

_INTENTS_WITH_RECALL_GUARDRAIL: set[str] = {
    "search",
    "refine",
    "image_search",
    "model_search",
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
    ) -> None:
        self.settings = settings
        self.opensearch_gateway = opensearch_gateway
        self.embedding_provider = embedding_provider
        self.model_retrieval_service = model_retrieval_service
        self.tour_navigation_service = tour_navigation_service
        self.llm_service = llm_service
        self.session_store = session_store

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
                logger.log(
                    level,
                    f"{prefix} " + json.dumps(payload, ensure_ascii=False, default=str),
                )
            return

        details = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.log(level, f"{event} {details}".strip())

    def _intent_requires_recall_guardrail(self, intent: str | None) -> bool:
        return str(intent or "").strip().casefold() in _INTENTS_WITH_RECALL_GUARDRAIL

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
            logger.debug("status callback failed", exc_info=True)

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
    ) -> None:
        state.last_paged_artifact_results = [
            artifact.model_dump(mode="json")
            for artifact in artifact_results
        ]
        state.last_paged_image_matches = [
            match.model_dump(mode="json")
            for match in image_matches
        ]
        state.last_paged_navigation_targets = [
            target.model_dump(mode="json")
            for target in navigation_targets
        ]
        state.last_paged_results_default_page_size = max(int(default_page_size), 1)
        state.last_paged_retrieval_request = dict(retrieval_request or {})

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
        page_result = await self.opensearch_gateway.search_relevant_context_page(
            museum_slug=payload.museum_slug,
            museum_id=str(request.get("museum_id") or payload.museum_id or "").strip() or None,
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
                    museum_slug=payload.museum_slug,
                    museum_id=str(request.get("museum_id") or payload.museum_id or "").strip() or None,
                    artifact_ids=artifact_ids,
                    per_artifact=1,
                    max_total=max(len(artifact_ids), 1),
                )
            except Exception as exc:
                self._log(
                    logging.WARNING,
                    "chat.image_retrieve_error",
                    conversation_id=payload.conversation_id,
                    museum_slug=payload.museum_slug,
                    museum_id=payload.museum_id,
                    reason=f"artifact_image_fetch_failed: {exc}",
                )
                image_hits = []
            image_matches = self._build_image_matches(
                image_hits=image_hits,
                artifact_docs=docs,
            )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=str(request.get("museum_id") or payload.museum_id or "").strip() or None,
            docs=docs,
        )
        artifact_results = await self._build_artifact_results(
            museum_slug=payload.museum_slug,
            museum_id=str(request.get("museum_id") or payload.museum_id or "").strip() or None,
            artifact_docs=docs,
            max_images_per_artifact=1,
            artifact_image_hits=image_hits,
        )
        image_matches = self._enrich_image_matches(
            context="text_page",
            conversation_id=payload.conversation_id,
            museum_slug=payload.museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        reported_total = self._reported_results_total(request, page_result.total)
        self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            default_page_size=page_size,
            retrieval_request=request,
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            results_page=page,
            results_page_size=page_size,
            results_total=reported_total,
            results_has_more=(from_offset + page_size) < reported_total,
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
        museum_id = str(request.get("museum_id") or payload.museum_id or "").strip() or None
        if kind == "model":
            page_result = await self.opensearch_gateway.search_similar_images_multi_page(
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                image_embeddings=list(request.get("image_embeddings") or []),
                from_offset=from_offset,
                page_size=page_size,
                retrieval_window_size=self._retrieval_window_from_request(request),
            )
        else:
            page_result = await self.opensearch_gateway.search_similar_images_page(
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                image_embedding=list(request.get("image_embedding") or []),
                from_offset=from_offset,
                page_size=page_size,
                retrieval_window_size=self._retrieval_window_from_request(request),
            )

        image_hits = page_result.results
        artifact_docs = await self._fetch_artifact_docs_for_image_hits(
            museum_slug=payload.museum_slug,
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
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context=f"{kind}_page",
            conversation_id=payload.conversation_id,
            museum_slug=payload.museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        reported_total = self._reported_results_total(request, page_result.total)
        self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            default_page_size=page_size,
            retrieval_request=request,
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            results_page=page,
            results_page_size=page_size,
            results_total=reported_total,
            results_has_more=(from_offset + page_size) < reported_total,
        )

    async def _materialize_structured_results_page(
        self,
        *,
        state: ChatSessionState,
        payload: ChatResultsPageRequest,
        request: dict[str, Any],
        page: int,
        page_size: int,
    ) -> ChatResultsPageResponse:
        from_offset = (page - 1) * page_size
        plan = QueryPlan.model_validate(request.get("plan") or {})
        dsl = CompiledOpenSearchDSL.model_validate(request.get("dsl") or {})
        body = dict(dsl.body)
        body["from"] = from_offset
        body["size"] = page_size
        body["track_total_hits"] = True
        paged_dsl = CompiledOpenSearchDSL(
            endpoint=dsl.endpoint,
            index=dsl.index,
            body=body,
        )
        result = await self.opensearch_gateway.execute_structured_query(
            plan=plan,
            dsl=paged_dsl,
        )
        museum_id = str(request.get("museum_id") or payload.museum_id or "").strip() or None
        artifact_results = await self._build_artifact_results(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            artifact_docs=result.items,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            docs=result.items,
        )
        reported_total = self._reported_results_total(request, int(result.total or 0))
        self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=[],
            navigation_targets=navigation_targets,
            default_page_size=page_size,
            retrieval_request=request,
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            artifact_results=artifact_results,
            image_matches=[],
            navigation_targets=navigation_targets,
            results_page=page,
            results_page_size=page_size,
            results_total=reported_total,
            results_has_more=(from_offset + page_size) < reported_total,
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
            )
        self._sync_state_language(state, payload.language)

        retrieval_request = dict(state.last_paged_retrieval_request or {})
        retrieval_kind = str(retrieval_request.get("kind") or "")
        default_page_size_floor = (
            self._text_results_default_page_size()
            if retrieval_kind == "text"
            else max(self.settings.CHAT_RETRIEVAL_TOP_K, 1)
        )
        default_page_size = max(
            int(state.last_paged_results_default_page_size or 0),
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
            self._log(
                logging.INFO,
                "chat.retrieve_results_page",
                conversation_id=payload.conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=payload.museum_id,
                source="opensearch",
                kind=retrieval_kind,
                page=response.results_page,
                page_size=response.results_page_size,
                total=response.results_total,
                has_more=response.results_has_more,
                artifact_results=len(response.artifact_results),
                image_matches=len(response.image_matches),
            )
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
            self._log(
                logging.INFO,
                "chat.retrieve_results_page",
                conversation_id=payload.conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=payload.museum_id,
                source="opensearch",
                kind=retrieval_kind,
                page=response.results_page,
                page_size=response.results_page_size,
                total=response.results_total,
                has_more=response.results_has_more,
                artifact_results=len(response.artifact_results),
                image_matches=len(response.image_matches),
            )
            return response
        if retrieval_kind == "structured_list":
            response = await self._materialize_structured_results_page(
                state=state,
                payload=payload,
                request=retrieval_request,
                page=results_page,
                page_size=results_page_size,
            )
            self.session_store.save(state)
            self._log(
                logging.INFO,
                "chat.retrieve_results_page",
                conversation_id=payload.conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=payload.museum_id,
                source="opensearch",
                kind=retrieval_kind,
                page=response.results_page,
                page_size=response.results_page_size,
                total=response.results_total,
                has_more=response.results_has_more,
                artifact_results=len(response.artifact_results),
                image_matches=len(response.image_matches),
            )
            return response

        artifact_results = [ArtifactResult(**item) for item in state.last_paged_artifact_results]
        image_matches = [ImageMatchResult(**item) for item in state.last_paged_image_matches]
        navigation_targets = [
            TourNavigationTarget(**item)
            for item in state.last_paged_navigation_targets
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
        self._log(
            logging.INFO,
            "chat.retrieve_results_page",
            conversation_id=payload.conversation_id,
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            page=results_page,
            page_size=results_page_size,
            total=results_total,
            has_more=results_has_more,
            artifact_results=len(paged_artifacts),
            image_matches=len(paged_image_matches),
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            artifact_results=paged_artifacts,
            image_matches=paged_image_matches,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
        )

    async def handle_message(
        self,
        payload: ChatMessageRequest,
        *,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        conversation_id = payload.conversation_id or str(uuid4())
        requested_format = payload.response_format or ResponseFormatObject(type="text")
        self._log(
            logging.INFO,
            "chat.receive",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            response_format=requested_format.type,
        )
        if self.settings.LOG_CHAT_MESSAGES:
            self._log(
                logging.DEBUG,
                "chat.receive_message",
                conversation_id=conversation_id,
                text=payload.message,
            )

        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        self.session_store.append_turn(state, role="user", text=payload.message)
        await self._emit_status(status_cb, "status.analyzing_request", language=language)

        context_policy = self._derive_context_policy(
            message=payload.message,
            state=state,
        )
        self._log(
            logging.INFO,
            "chat.context_policy",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            is_follow_up=context_policy.get("is_follow_up"),
            use_history_for_query=context_policy.get("use_history_for_query"),
            use_history_for_answer=context_policy.get("use_history_for_answer"),
            carry_filters=context_policy.get("carry_filters"),
            carry_sort=context_policy.get("carry_sort"),
            reason=context_policy.get("reason"),
        )

        structured_response = await self._try_handle_structured_query(
            payload=payload,
            state=state,
            conversation_id=conversation_id,
            requested_format=requested_format,
            status_cb=status_cb,
        )
        if structured_response is not None:
            return structured_response

        try:
            router_decision = await self._route_message(
                payload=payload,
                state=state,
                context_policy=context_policy,
            )
        except LLMServiceError:
            self._log(
                logging.WARNING,
                "chat.router_unavailable",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
            )
            router_decision = {
                "mode": "llm_only",
                "intent": "fallback",
                "rewritten_query": payload.message,
                "needs_retrieval": False,
                "reason": "router_unavailable: usa contexto local",
                "is_follow_up": bool(context_policy.get("is_follow_up", False)),
                "use_history_for_query": bool(
                    context_policy.get("use_history_for_query", False)
                ),
                "use_history_for_answer": bool(
                    context_policy.get("use_history_for_answer", False)
                ),
                "carry_filters": bool(context_policy.get("carry_filters", False)),
                "carry_sort": bool(context_policy.get("carry_sort", False)),
                "filters_delta": {},
                "sort_delta": {},
            }
        router_decision = self._apply_context_policy_guardrails(
            router_decision=router_decision,
            context_policy=context_policy,
            user_message=payload.message,
        )
        self._log(
            logging.INFO,
            "chat.route",
            conversation_id=conversation_id,
            mode=router_decision.get("mode"),
            intent=router_decision.get("intent"),
            needs_retrieval=router_decision.get("needs_retrieval"),
            is_follow_up=router_decision.get("is_follow_up"),
            use_history_for_query=router_decision.get("use_history_for_query"),
            use_history_for_answer=router_decision.get("use_history_for_answer"),
            carry_filters=router_decision.get("carry_filters"),
            carry_sort=router_decision.get("carry_sort"),
            reason=router_decision.get("reason"),
        )
        rewritten_query = str(router_decision.get("rewritten_query", payload.message)).strip()
        original_message = (payload.message or "").strip()
        self._log(
            logging.INFO,
            "chat.rewrite",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            original_message=original_message,
            rewritten_query=rewritten_query,
            query_changed=rewritten_query != original_message,
            mode=router_decision.get("mode"),
            intent=router_decision.get("intent"),
        )
        self._log(
            logging.DEBUG,
            "chat.route_delta",
            conversation_id=conversation_id,
            filters_delta=router_decision.get("filters_delta"),
            sort_delta=router_decision.get("sort_delta"),
            rewritten_query=router_decision.get("rewritten_query"),
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
        pre_scope_filters = dict(effective_filters)
        effective_filters = self._apply_follow_up_artifact_scope(
            message=payload.message,
            state=state,
            router_decision=router_decision,
            filters=effective_filters,
        )
        if effective_filters != pre_scope_filters:
            self._log(
                logging.INFO,
                "chat.context_artifact_scope",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                artifact_id=effective_filters.get("artifact_id"),
                reason="referential_follow_up_single_artifact",
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
        if router_decision["mode"] == "rag" and self.settings.CHAT_ENABLE_RAG:
            # Retrieval must stay strict to user wording (no expansion/rewrite additions).
            retrieval_query = payload.message
            await self._emit_status(status_cb, "status.searching_collection", language=language)
            (
                retrieval_context,
                retrieved_docs_count,
                retrieved_docs,
                retrieval_request,
            ) = await self._retrieve_context(
                museum_slug=payload.museum_slug,
                museum_id=self._resolve_museum_id(payload),
                query=retrieval_query,
                filters=effective_filters,
                sort=effective_sort,
                result_window_size=text_results_window_size,
            )
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=retrieved_docs_count,
            )
            retrieved_docs = await self._filter_docs_with_llm(
                docs=retrieved_docs,
                user_message=payload.message,
                museum_slug=payload.museum_slug,
                intent=str(router_decision.get("intent", "")),
                model_override=payload.model_override,
                system_prompt=payload.system_prompt,
            )
            retrieved_docs_count = len(retrieved_docs)
            if retrieval_request:
                retrieved_docs_count = max(
                    retrieved_docs_count,
                    int(retrieval_request.get("results_total") or 0),
                )
            self._log(
                logging.INFO,
                "chat.retrieve",
                conversation_id=conversation_id,
                mode="rag",
                query=retrieval_query,
                context_found=bool(retrieval_context),
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
                            museum_slug=payload.museum_slug,
                            museum_id=self._resolve_museum_id(payload),
                            artifact_ids=artifact_ids,
                            per_artifact=1,
                            max_total=max(len(artifact_ids), 1),
                        )
                    except Exception as exc:
                        self._log(
                            logging.WARNING,
                            "chat.image_retrieve_error",
                            conversation_id=conversation_id,
                            museum_slug=payload.museum_slug,
                            museum_id=self._resolve_museum_id(payload),
                            reason=f"artifact_image_fetch_failed: {exc}",
                        )
                        artifact_image_hits = []
                    image_matches = self._build_image_matches(
                        image_hits=artifact_image_hits,
                        artifact_docs=retrieved_docs,
                    )
                    image_matches = await self._filter_image_matches_with_llm(
                        image_matches=image_matches,
                        user_message=payload.message,
                        museum_slug=payload.museum_slug,
                        intent=str(router_decision.get("intent", "")),
                        model_override=payload.model_override,
                        system_prompt=payload.system_prompt,
                    )
                    if not self._intent_requires_recall_guardrail(
                        str(router_decision.get("intent", ""))
                    ):
                        retrieved_docs = self._filter_docs_by_image_matches(
                            docs=retrieved_docs,
                            image_matches=image_matches,
                        )
                        image_matches = self._filter_image_matches_by_docs(
                            image_matches=image_matches,
                            docs=retrieved_docs,
                        )
        else:
            self._log(
                logging.DEBUG,
                "chat.retrieve_skipped",
                conversation_id=conversation_id,
                mode=router_decision.get("mode"),
                rag_enabled=self.settings.CHAT_ENABLE_RAG,
            )
        artifact_results = await self._build_artifact_results(
            museum_slug=payload.museum_slug,
            museum_id=self._resolve_museum_id(payload),
            artifact_docs=retrieved_docs,
            max_images_per_artifact=1,
            artifact_image_hits=artifact_image_hits,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=self._resolve_museum_id(payload),
            docs=retrieved_docs,
        )
        image_matches = self._enrich_image_matches(
            context="text",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        if navigation_targets:
            self._log(
                logging.INFO,
                "chat.navigation_targets",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                targets=[target.model_dump(exclude_none=True) for target in navigation_targets],
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

        final_message = self._build_final_prompt(
            payload=payload,
            state=state,
            router_decision=router_decision,
            retrieval_context=retrieval_context,
            effective_filters=effective_filters,
            effective_sort=effective_sort,
            use_history_for_answer=use_history_for_answer,
        )
        await self._emit_status(status_cb, "status.generating_final_answer", language=language)

        try:
            llm_response = await self.llm_service.generate(
                message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
            )
        except LLMServiceError as exc:
            self._log(
                logging.WARNING,
                "chat.llm_error",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                error=str(exc),
            )
            # Soft-fail in dev so frontend keeps moving while LLM infra is still unstable.
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            return ChatMessageResponse(
                conversation_id=conversation_id,
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
            )

        self._log(
            logging.INFO,
            "chat.reply",
            conversation_id=conversation_id,
            model=llm_response.model,
            reply_chars=len(llm_response.text),
        )
        final_reply_text = self._sanitize_assistant_reply(
            llm_response.text,
            docs=retrieved_docs,
            language=language,
        )
        if final_reply_text != llm_response.text:
            self._log(
                logging.INFO,
                "chat.reply_sanitized",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
            )
        selected_from_reply = self._infer_selected_artifact_from_reply(
            reply_text=final_reply_text,
            docs=retrieved_docs,
        )

        if not carry_filters:
            state.filters = {}
        if not carry_sort:
            state.sort = {}

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_result_selection_state(
            state=state,
            artifact_docs=retrieved_docs,
            effective_filters=effective_filters,
            hinted_selected_artifact_id=selected_from_reply,
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
        self._log_state_after_reply(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        return ChatMessageResponse(
            conversation_id=conversation_id,
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
        self._log(
            logging.INFO,
            "chat.receive_image",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            response_format=requested_format.type,
            image_filename=image_filename,
            image_content_type=image_content_type,
            image_bytes=len(image_bytes),
        )
        if self.settings.LOG_CHAT_MESSAGES:
            self._log(
                logging.DEBUG,
                "chat.receive_message",
                conversation_id=conversation_id,
                text=user_message,
            )

        self.session_store.append_turn(state, role="user", text=user_message)
        await self._emit_status(status_cb, "status.analyzing_image", language=language)

        image_matches: list[ImageMatchResult] = []
        artifact_docs: list[dict[str, object]] = []
        artifact_results: list[ArtifactResult] = []
        navigation_targets: list[TourNavigationTarget] = []
        image_results_total = 0
        try:
            image_embedding = await self.embedding_provider.embed_multimodal_image_bytes(
                image_bytes=image_bytes,
                text=None,
            )
            self._log(
                logging.INFO,
                "chat.image_embedding_ready",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                image_embedding_dim=len(image_embedding),
            )
        except Exception as exc:
            self._log(
                logging.WARNING,
                "chat.image_retrieve_error",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                reason=f"image_embedding_failed: {exc}",
            )
            fallback = translate("error.image_processing_failed", language)
            return ChatMessageResponse(
                conversation_id=conversation_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
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
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                image_embedding=image_embedding,
                from_offset=0,
                page_size=image_candidates_k,
                retrieval_window_size=image_retrieval_window_size,
            )
            image_hits = image_page.results
            image_results_total = self._bounded_retrieval_total(
                image_page.total,
                image_retrieval_window_size,
            )
        except Exception as exc:
            self._log(
                logging.ERROR,
                "chat.image_retrieve_error",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                reason=f"image_search_failed: {exc}",
            )
            logger.exception("chat.image_retrieve_error exception")
            image_hits = []

        if image_hits:
            try:
                artifact_docs = await self._fetch_artifact_docs_for_image_hits(
                    museum_slug=payload.museum_slug,
                    museum_id=museum_id,
                    artifact_museum_id=explicit_museum_id,
                    image_hits=image_hits,
                    top_k=image_candidates_k,
                )
            except Exception as exc:
                self._log(
                    logging.ERROR,
                    "chat.image_retrieve_error",
                    conversation_id=conversation_id,
                    museum_slug=payload.museum_slug,
                    museum_id=museum_id,
                    reason=f"artifact_fetch_failed: {exc}",
                )
                logger.exception("chat.image_retrieve_error exception")
                artifact_docs = []

            image_matches = self._build_image_matches(image_hits=image_hits, artifact_docs=artifact_docs)
            image_matches = await self._filter_image_matches_with_llm(
                image_matches=image_matches,
                user_message=user_message,
                museum_slug=payload.museum_slug,
                intent="image_search",
                model_override=payload.model_override,
                system_prompt=payload.system_prompt,
            )
            filtered_artifact_docs = await self._filter_docs_with_llm(
                docs=artifact_docs,
                user_message=user_message,
                museum_slug=payload.museum_slug,
                intent="image_search",
                model_override=payload.model_override,
                system_prompt=payload.system_prompt,
            )
            artifact_docs = filtered_artifact_docs or artifact_docs
            if not self._intent_requires_recall_guardrail("image_search"):
                artifact_docs = self._filter_docs_by_image_matches(
                    docs=artifact_docs,
                    image_matches=image_matches,
                )
                image_matches = self._filter_image_matches_by_docs(
                    image_matches=image_matches,
                    docs=artifact_docs,
                )
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=len(artifact_docs),
            )
            self._log(
                logging.INFO,
                "chat.image_retrieve",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                image_hits=len(image_hits),
                artifact_docs=len(artifact_docs),
            )
        else:
            self._log(
                logging.INFO,
                "chat.image_retrieve_empty",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
            )
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=0,
            )

        artifact_results = await self._build_artifact_results(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context="image",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        if navigation_targets:
            self._log(
                logging.INFO,
                "chat.navigation_targets",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                targets=[target.model_dump(exclude_none=True) for target in navigation_targets],
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
                "museum_id": museum_id,
                "artifact_museum_id": explicit_museum_id,
                "image_embedding": image_embedding,
                "retrieval_window_size": image_retrieval_window_size,
                "results_total": image_results_total,
            },
        )

        retrieval_context_sections: list[str] = []
        if image_matches:
            retrieval_context_sections.append("image_retrieval_matches:")
            retrieval_context_sections.append(
                json.dumps(
                    [
                        match.model_dump(
                            exclude_none=True,
                            exclude={"artifact", "navigation_target"},
                        )
                        for match in image_matches
                    ],
                    ensure_ascii=True,
                )
            )
        docs_context = self._format_docs_for_prompt(
            docs=artifact_docs,
            top_k=max(self.settings.CHAT_IMAGE_ARTIFACT_TOP_K, 1),
        )
        if docs_context:
            retrieval_context_sections.append("artifact_docs_from_image_retrieval:")
            retrieval_context_sections.append(docs_context)
        retrieval_context = "\n".join(retrieval_context_sections).strip()

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
            museum_slug=payload.museum_slug,
            museum_name=payload.museum_name,
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

        try:
            llm_response = await self.llm_service.generate(
                message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
            )
        except LLMServiceError as exc:
            self._log(
                logging.WARNING,
                "chat.llm_error",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                error=str(exc),
            )
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            return ChatMessageResponse(
                conversation_id=conversation_id,
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
            )

        self._log(
            logging.INFO,
            "chat.reply",
            conversation_id=conversation_id,
            model=llm_response.model,
            reply_chars=len(llm_response.text),
        )
        final_reply_text = self._sanitize_assistant_reply(
            llm_response.text,
            docs=artifact_docs,
            language=language,
        )
        if final_reply_text != llm_response.text:
            self._log(
                logging.INFO,
                "chat.reply_sanitized",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
            )
        selected_from_reply = self._infer_selected_artifact_from_reply(
            reply_text=final_reply_text,
            docs=artifact_docs,
        )

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_result_selection_state(
            state=state,
            artifact_docs=artifact_docs,
            effective_filters=None,
            hinted_selected_artifact_id=selected_from_reply,
        )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=user_message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        self._log_state_after_reply(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        return ChatMessageResponse(
            conversation_id=conversation_id,
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
        self._log(
            logging.INFO,
            "chat.receive_model",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            response_format=requested_format.type,
            model_filename=model_filename,
            model_content_type=model_content_type,
            model_bytes=len(model_bytes),
        )
        if self.settings.LOG_CHAT_MESSAGES:
            self._log(
                logging.DEBUG,
                "chat.receive_message",
                conversation_id=conversation_id,
                text=user_message,
            )

        self.session_store.append_turn(state, role="user", text=user_message)
        await self._emit_status(status_cb, "status.preparing_model", language=language)

        image_matches: list[ImageMatchResult] = []
        artifact_docs: list[dict[str, object]] = []
        artifact_results: list[ArtifactResult] = []
        navigation_targets: list[TourNavigationTarget] = []
        file_name = (model_filename or "model.glb").strip() or "model.glb"
        model_results_total = 0
        model_image_embeddings: list[list[float]] = []

        try:
            retrieval_result = await self.model_retrieval_service.retrieve(
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
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
            image_matches = self._build_image_matches(
                image_hits=retrieval_result.image_hits,
                artifact_docs=artifact_docs,
            )
            image_matches = await self._filter_image_matches_with_llm(
                image_matches=image_matches,
                user_message=user_message,
                museum_slug=payload.museum_slug,
                intent="model_search",
                model_override=payload.model_override,
                system_prompt=payload.system_prompt,
            )
            filtered_artifact_docs = await self._filter_docs_with_llm(
                docs=artifact_docs,
                user_message=user_message,
                museum_slug=payload.museum_slug,
                intent="model_search",
                model_override=payload.model_override,
                system_prompt=payload.system_prompt,
            )
            artifact_docs = filtered_artifact_docs or artifact_docs
            if not self._intent_requires_recall_guardrail("model_search"):
                artifact_docs = self._filter_docs_by_image_matches(
                    docs=artifact_docs,
                    image_matches=image_matches,
                )
                image_matches = self._filter_image_matches_by_docs(
                    image_matches=image_matches,
                    docs=artifact_docs,
                )
            if retrieval_result.image_hits:
                self._log(
                    logging.INFO,
                    "chat.model_retrieve",
                    conversation_id=conversation_id,
                    museum_slug=payload.museum_slug,
                    museum_id=museum_id,
                    image_hits=len(retrieval_result.image_hits),
                    image_hits_total=retrieval_result.image_hits_total,
                    artifact_docs=len(artifact_docs),
                    extra_views_used=retrieval_result.extra_views_used,
                    top_score=retrieval_result.top_score,
                )
            else:
                self._log(
                    logging.INFO,
                    "chat.model_retrieve_empty",
                    conversation_id=conversation_id,
                    museum_slug=payload.museum_slug,
                    museum_id=museum_id,
                    extra_views_used=retrieval_result.extra_views_used,
                )
        except Exception as exc:
            self._log(
                logging.ERROR,
                "chat.model_retrieve_error",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                reason=str(exc),
            )
            logger.exception("chat.model_retrieve_error exception")
            fallback = translate("error.model_processing_failed", language)
            return ChatMessageResponse(
                conversation_id=conversation_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
            )

        await self._emit_status(
            status_cb,
            "status.artifacts_found",
            language=language,
            artifact_count=len(artifact_docs),
        )

        artifact_results = await self._build_artifact_results(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context="model",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        if navigation_targets:
            self._log(
                logging.INFO,
                "chat.navigation_targets",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                targets=[target.model_dump(exclude_none=True) for target in navigation_targets],
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
                "museum_id": museum_id,
                "artifact_museum_id": explicit_museum_id,
                "image_embeddings": model_image_embeddings,
                "retrieval_window_size": retrieval_result.retrieval_window_size,
                "results_total": model_results_total,
            },
        )

        retrieval_context_sections: list[str] = []
        if image_matches:
            retrieval_context_sections.append("model_view_matches:")
            retrieval_context_sections.append(
                json.dumps(
                    [
                        match.model_dump(
                            exclude_none=True,
                            exclude={"artifact", "navigation_target"},
                        )
                        for match in image_matches
                    ],
                    ensure_ascii=True,
                )
            )
        docs_context = self._format_docs_for_prompt(
            docs=artifact_docs,
            top_k=max(self.settings.CHAT_IMAGE_ARTIFACT_TOP_K, 1),
        )
        if docs_context:
            retrieval_context_sections.append("artifact_docs_from_model_retrieval:")
            retrieval_context_sections.append(docs_context)
        retrieval_context = "\n".join(retrieval_context_sections).strip()

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
            museum_slug=payload.museum_slug,
            museum_name=payload.museum_name,
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

        try:
            llm_response = await self.llm_service.generate(
                message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
            )
        except LLMServiceError as exc:
            self._log(
                logging.WARNING,
                "chat.llm_error",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                error=str(exc),
            )
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            return ChatMessageResponse(
                conversation_id=conversation_id,
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
            )

        self._log(
            logging.INFO,
            "chat.reply",
            conversation_id=conversation_id,
            model=llm_response.model,
            reply_chars=len(llm_response.text),
        )
        final_reply_text = self._sanitize_assistant_reply(
            llm_response.text,
            docs=artifact_docs,
            language=language,
        )
        if final_reply_text != llm_response.text:
            self._log(
                logging.INFO,
                "chat.reply_sanitized",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
            )
        selected_from_reply = self._infer_selected_artifact_from_reply(
            reply_text=final_reply_text,
            docs=artifact_docs,
        )

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_result_selection_state(
            state=state,
            artifact_docs=artifact_docs,
            effective_filters=None,
            hinted_selected_artifact_id=selected_from_reply,
        )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=user_message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        self._log_state_after_reply(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        return ChatMessageResponse(
            conversation_id=conversation_id,
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
        )

    def _build_structured_query_schema(self, *, museum_id: str | None) -> QuerySchema:
        fields = {
            "artifact_id": QuerySchemaField(type="keyword", facetable=True),
            "museum_id": QuerySchemaField(type="keyword", facetable=True),
            "inventory_number": QuerySchemaField(type="keyword", facetable=True, text=True),
            "inventory_number.text": QuerySchemaField(type="text", text=True),
            "title": QuerySchemaField(type="text", text=True, semantic=True),
            "title.keyword": QuerySchemaField(type="keyword", facetable=True),
            "description": QuerySchemaField(type="text", text=True, semantic=True),
            "search_text": QuerySchemaField(type="text", text=True, semantic=True),
            "category": QuerySchemaField(type="keyword", facetable=True, text=True),
            "category.text": QuerySchemaField(type="text", text=True),
            "super_category": QuerySchemaField(type="keyword", facetable=True, text=True),
            "super_category.text": QuerySchemaField(type="text", text=True),
            "support_or_material": QuerySchemaField(type="keyword", facetable=True, text=True),
            "support_or_material.text": QuerySchemaField(type="text", text=True),
            "technique": QuerySchemaField(type="keyword", facetable=True, text=True),
            "technique.text": QuerySchemaField(type="text", text=True),
            "creator": QuerySchemaField(type="keyword", facetable=True, text=True),
            "creator.text": QuerySchemaField(type="text", text=True),
            "date_or_period": QuerySchemaField(type="keyword", facetable=True, text=True),
            "date_or_period.text": QuerySchemaField(type="text", text=True),
            "origin_history": QuerySchemaField(type="text", text=True),
            "production_center": QuerySchemaField(type="keyword", facetable=True, text=True),
            "production_center.text": QuerySchemaField(type="text", text=True),
            "incorporation": QuerySchemaField(type="keyword", facetable=True, text=True),
            "incorporation.text": QuerySchemaField(type="text", text=True),
            "museum": QuerySchemaField(type="keyword", facetable=True, text=True),
            "museum.text": QuerySchemaField(type="text", text=True),
            "detail_type": QuerySchemaField(type="keyword", facetable=True),
        }
        default_filters: list[TermFilter] = []
        if museum_id:
            default_filters.append(TermFilter(kind="term", field="museum_id", value=museum_id))

        return QuerySchema(
            index_name=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            fields=fields,
            facetable_fields=[
                "museum_id",
                "museum",
                "category",
                "super_category",
                "support_or_material",
                "technique",
                "creator",
                "date_or_period",
                "production_center",
                "incorporation",
                "detail_type",
                "title.keyword",
                "inventory_number",
            ],
            text_fields=[
                "search_text",
                "title",
                "description",
                "inventory_number",
                "inventory_number.text",
                "category.text",
                "super_category.text",
                "support_or_material.text",
                "technique.text",
                "creator.text",
                "date_or_period.text",
                "origin_history",
                "production_center.text",
                "incorporation.text",
                "museum.text",
            ],
            semantic_fields=["search_text", "title", "description"],
            default_filters=default_filters,
        )

    async def _try_handle_structured_query(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        conversation_id: str,
        requested_format: ResponseFormatObject,
        status_cb: StatusCallback | None,
    ) -> ChatMessageResponse | None:
        if not self.settings.CHAT_ENABLE_STRUCTURED_QUERY_PLANNING:
            return None

        museum_id = self._resolve_museum_id(payload)
        schema = self._build_structured_query_schema(museum_id=museum_id)
        mode = classify_query(payload.message, schema)
        self._log(
            logging.INFO,
            "chat.structured.classify",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            mode=mode,
            question=payload.message,
        )
        if mode != "structured":
            return None

        language = normalize_language(state.language)
        await self._emit_status(
            status_cb,
            "status.interpreting_analytics_request",
            language=language,
        )
        try:
            plan = await plan_query(
                payload.message,
                schema,
                llm_service=self.llm_service,
                model_override=payload.model_override,
            )
        except QueryPlanningError as exc:
            self._log(
                logging.WARNING,
                "chat.structured.fallback",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                reason=f"planner_error: {exc}",
            )
            return None

        self._log(
            logging.INFO,
            "chat.structured.plan",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            operation=plan.operation,
            confidence=plan.confidence,
            plan=plan.model_dump(mode="json"),
        )
        if plan.confidence < self.settings.CHAT_ANALYTICS_PLANNER_MIN_CONFIDENCE:
            self._log(
                logging.INFO,
                "chat.structured.fallback",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                reason=f"low_confidence:{plan.confidence}",
            )
            return None

        if plan.operation == "list":
            if plan.list_spec is None:
                plan.list_spec = ListSpec(limit=self.settings.CHAT_ANALYTICS_LIST_TOP_K)
            else:
                plan.list_spec.limit = min(
                    plan.list_spec.limit,
                    max(self.settings.CHAT_ANALYTICS_LIST_TOP_K, 1),
                )

        try:
            dsl = compile_query(plan, schema)
        except QueryCompileError as exc:
            self._log(
                logging.WARNING,
                "chat.structured.fallback",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                reason=f"compile_error: {exc}",
            )
            return None

        if plan.operation == "list":
            dsl = dsl.model_copy(deep=True)
            dsl.body["track_total_hits"] = True

        await self._emit_status(status_cb, "status.querying_collection", language=language)
        try:
            result = await self.opensearch_gateway.execute_structured_query(
                plan=plan,
                dsl=dsl,
            )
        except Exception as exc:
            self._log(
                logging.WARNING,
                "chat.structured.fallback",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                reason=f"execute_error: {exc}",
            )
            return None

        self._log(
            logging.INFO,
            "chat.structured.execute",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            operation=plan.operation,
            total=result.total,
            count=result.count,
            exists=result.exists,
            items=len(result.items),
            groups=len(result.groups),
        )

        final_reply_text = self._format_structured_reply(
            plan=plan,
            result=result,
            language=language,
        )
        self._log(
            logging.INFO,
            "chat.structured.reply",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            operation=plan.operation,
            reply_chars=len(final_reply_text),
        )

        docs_for_navigation = result.items if plan.operation == "list" else []
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            docs=docs_for_navigation,
        )
        if navigation_targets:
            self._log(
                logging.INFO,
                "chat.navigation_targets",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                targets=[target.model_dump(exclude_none=True) for target in navigation_targets],
            )

        router_decision: dict[str, object] = {
            "mode": "structured",
            "intent": f"analytics_{plan.operation}",
            "filters_delta": {},
            "sort_delta": {},
        }
        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        if plan.operation == "list":
            self._update_result_selection_state(
                state=state,
                artifact_docs=result.items,
                effective_filters=None,
            )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=payload.message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        self._log_state_after_reply(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        reply_json_payload: dict[str, object] | None = None
        if requested_format.type == "json_object":
            reply_json_payload = {
                "operation": result.operation,
                "count": result.count,
                "exists": result.exists,
                "total": result.total,
                "items": result.items,
                "groups": [bucket.model_dump(mode="json") for bucket in result.groups],
            }
        artifact_results: list[ArtifactResult] = []
        if plan.operation == "list":
            artifact_results = await self._build_artifact_results(
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                artifact_docs=result.items,
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
            image_matches=[],
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=max(
                self.settings.CHAT_ANALYTICS_LIST_TOP_K,
                self.settings.CHAT_RETRIEVAL_TOP_K,
                1,
            ),
            total_override=(int(result.total or 0) if plan.operation == "list" else None),
            retrieval_request=(
                {
                    "kind": "structured_list",
                    "museum_id": museum_id,
                    "plan": plan.model_dump(mode="json"),
                    "dsl": dsl.model_dump(mode="json"),
                    "results_total": int(result.total or 0),
                }
                if plan.operation == "list"
                else None
            ),
        )
        self.session_store.save(state)

        return ChatMessageResponse(
            conversation_id=conversation_id,
            response_format=requested_format,
            reply=final_reply_text,
            reply_json=reply_json_payload,
            model_hint="structured_query_executor",
            image_matches=paged_image_matches,
            artifact_results=paged_artifact_results,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
        )

    def _format_structured_reply(
        self,
        *,
        plan: QueryPlan,
        result: QueryExecutionResult,
        language: str | None = None,
    ) -> str:
        if plan.operation == "count":
            count = int(result.count or 0)
            return translate("structured.count", language, count=count)

        if plan.operation == "exists":
            if result.exists:
                return translate("structured.exists.yes", language)
            return translate("structured.exists.no", language)

        if plan.operation == "list":
            if not result.items:
                return translate("structured.list.empty", language)
            lines = [translate("structured.list.header", language)]
            inventory_label = translate("structured.list.inventory_prefix", language)
            for index, item in enumerate(result.items, start=1):
                title = str(item.get("title") or "").strip()
                inventory = self._doc_inventory(item)
                if title and inventory:
                    lines.append(f"{index}. {title} ({inventory})")
                elif title:
                    lines.append(f"{index}. {title}")
                elif inventory:
                    lines.append(f"{index}. {inventory_label} {inventory}")
                else:
                    lines.append(f"{index}. {translate('structured.list.untitled', language)}")
            return "\n".join(lines)

        if plan.operation == "group_by":
            if not result.groups:
                return translate("structured.group.empty", language)
            lines = [translate("structured.group.header", language)]
            for bucket in result.groups:
                key = bucket.key.strip() or translate("structured.group.empty_key", language)
                lines.append(f"- {key}: {bucket.doc_count}")
            return "\n".join(lines)

        return translate("structured.fallback", language)

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
            "selected_artifact_id": state.selected_artifact_id,
            "last_result_ids": state.last_result_ids[:3],
            "history_size": len(state.history),
        }
        router_prompt = build_router_user_prompt(
            museum_slug=payload.museum_slug,
            museum_name=payload.museum_name,
            rolling_summary=state.rolling_summary if include_aux_context else "",
            filters_state=state.filters,
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
        if mode == "llm_only":
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

        # Referential follow-up should stay in retrieval mode so state filters can be applied.
        guarded["mode"] = "rag"
        guarded["needs_retrieval"] = True

        rewritten_query = str(guarded.get("rewritten_query", "")).strip()
        if not rewritten_query:
            guarded["rewritten_query"] = user_message

        reason = str(guarded.get("reason", "")).strip()
        guardrail_tag = "guardrail_context_policy_follow_up"
        if reason:
            if guardrail_tag not in reason:
                guarded["reason"] = f"{reason} | {guardrail_tag}"
        else:
            guarded["reason"] = guardrail_tag
        return guarded

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
        lexical_query_fallback = self._build_lexical_query(
            query=raw_query,
            museum_slug=museum_slug,
            museum_id=museum_id,
        )
        lexical_query = lexical_query_fallback
        embedding_query = (lexical_query_fallback or raw_query).strip()
        query_rewrite_source = "heuristic"

        llm_lexical_query, llm_embedding_query = await self._rewrite_retrieval_query_with_llm(
            query=raw_query,
            museum_slug=museum_slug,
            museum_id=museum_id,
            filters=filters,
            sort=sort,
        )
        if llm_lexical_query:
            lexical_query = llm_lexical_query
        if llm_embedding_query:
            embedding_query = llm_embedding_query
        if llm_lexical_query or llm_embedding_query:
            query_rewrite_source = "llm"

        self._log(
            logging.INFO,
            "chat.retrieve_query_rewrite",
            museum_slug=museum_slug,
            museum_id=museum_id,
            source=query_rewrite_source,
            original_query=raw_query,
            lexical_query=lexical_query,
            embedding_query=embedding_query,
        )
        self._log(
            logging.INFO,
            "chat.retrieve_prepare",
            museum_slug=museum_slug,
            museum_id=museum_id,
            original_query=raw_query,
            embedding_query=embedding_query,
            lexical_query=lexical_query,
            filters=filters,
            sort=sort,
        )
        if self.settings.CHAT_USE_QUERY_EMBEDDINGS:
            try:
                query_embedding = await self.embedding_provider.embed_text(embedding_query)
                self._log(
                    logging.INFO,
                    "chat.retrieve_embedding_ready",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    query_embedding_dim=len(query_embedding),
                )
            except NotImplementedError:
                self._log(logging.DEBUG, "chat.retrieve_not_implemented", museum_slug=museum_slug)
                return "", 0, [], {}
            except Exception as exc:
                self._log(
                    logging.WARNING,
                    "chat.retrieve_embedding_error",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    reason=str(exc),
                )
                return "", 0, [], {}
        else:
            self._log(
                logging.INFO,
                "chat.retrieve_embedding_disabled",
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

        try:
            page_result = await self.opensearch_gateway.search_relevant_context_page(
                museum_slug=museum_slug,
                museum_id=museum_id,
                query_text=embedding_query,
                lexical_query=lexical_query,
                query_embedding=query_embedding,
                from_offset=0,
                page_size=retrieval_page_size,
                filters=filters,
                sort=sort,
                retrieval_window_size=retrieval_window_size,
            )
        except Exception:
            self._log(logging.ERROR, "chat.retrieve_error", museum_slug=museum_slug)
            logger.exception("chat.retrieve_error exception")
            return "", 0, [], {}

        docs = page_result.results

        if not docs:
            return "", 0, [], {}

        docs_for_context = docs[:final_top_k]
        context = self._format_docs_for_prompt(docs=docs_for_context, top_k=final_top_k)
        results_total = self._bounded_retrieval_total(page_result.total, retrieval_window_size)
        retrieval_request = {
            "kind": "text",
            "museum_id": museum_id,
            "query_text": embedding_query,
            "lexical_query": lexical_query,
            "query_embedding": query_embedding,
            "filters": dict(filters),
            "sort": dict(sort),
            "retrieval_window_size": retrieval_window_size,
            "results_total": results_total,
        }
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

    def _has_query_language_mismatch(self, source: str, candidate: str) -> bool:
        source_pt = self._is_probably_portuguese_query(source)
        source_en = self._is_probably_english_query(source)
        candidate_pt = self._is_probably_portuguese_query(candidate)
        candidate_en = self._is_probably_english_query(candidate)
        if source_pt and candidate_en and not candidate_pt:
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
            museum_slug=museum_slug,
            museum_id=museum_id,
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
        embedding_query = str(payload.get("embedding_query") or "").strip()
        if lexical_query and self._has_query_language_mismatch(raw_query, lexical_query):
            self._log(
                logging.INFO,
                "chat.retrieve_query_rewrite_guardrail",
                museum_slug=museum_slug,
                museum_id=museum_id,
                original_query=raw_query,
                rejected_field="lexical_query",
                rejected_value=lexical_query,
                reason="language_mismatch",
            )
            lexical_query = ""
        if embedding_query and self._has_query_language_mismatch(raw_query, embedding_query):
            self._log(
                logging.INFO,
                "chat.retrieve_query_rewrite_guardrail",
                museum_slug=museum_slug,
                museum_id=museum_id,
                original_query=raw_query,
                rejected_field="embedding_query",
                rejected_value=embedding_query,
                reason="language_mismatch",
            )
            embedding_query = ""
        return lexical_query, embedding_query

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
        for index, doc in enumerate(docs[: max(top_k, 1)], start=1):
            prompt_doc = {
                key: value
                for key, value in doc.items()
                if key != "artifact_id" and value not in (None, "")
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
                self._log(
                    logging.WARNING,
                    "chat.image_retrieve_error",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    reason=f"artifact_modal_image_fetch_failed: {exc}",
                )
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
                        local_path=normalized_local_path,
                        source_url=normalized_source_url,
                        caption=str(caption or "").strip() or None,
                        alt_text=str(alt_text or "").strip() or None,
                    )
                )

            for hit in images_by_artifact.get(artifact_id, []):
                _append_image(
                    image_id=str(hit.get("image_id") or hit.get("id") or "").strip() or None,
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

            results.append(
                ArtifactResult(
                    artifact_id=artifact_id,
                    inventory_number=self._doc_inventory(doc) or None,
                    title=str(doc.get("title") or "").strip() or None,
                    museum_id=str(doc.get("museum_id") or "").strip() or None,
                    museum=str(doc.get("museum") or doc.get("museum_name") or "").strip() or None,
                    category=str(doc.get("category") or "").strip() or None,
                    super_category=str(doc.get("super_category") or "").strip() or None,
                    creator=str(doc.get("creator") or "").strip() or None,
                    date_or_period=str(doc.get("date_or_period") or "").strip() or None,
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
                    image_count=image_count_value,
                    images=images,
                )
            )

        return results

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

    def _filter_docs_by_image_matches(
        self,
        *,
        docs: list[dict[str, object]],
        image_matches: list[ImageMatchResult],
    ) -> list[dict[str, object]]:
        if not docs or not image_matches:
            return docs

        inventories = {
            str(match.inventory or "").strip().casefold()
            for match in image_matches
            if str(match.inventory or "").strip()
        }
        if inventories:
            filtered = [doc for doc in docs if self._doc_inventory(doc).casefold() in inventories]
            return filtered

        artifact_ids = {
            str(match.artifact_id or "").strip()
            for match in image_matches
            if str(match.artifact_id or "").strip()
        }
        if artifact_ids:
            return [
                doc
                for doc in docs
                if str(doc.get("artifact_id") or "").strip() in artifact_ids
            ]

        return docs

    def _filter_image_matches_by_docs(
        self,
        *,
        image_matches: list[ImageMatchResult],
        docs: list[dict[str, object]],
    ) -> list[ImageMatchResult]:
        if not image_matches:
            return image_matches
        if not docs:
            return []

        inventories = {
            self._doc_inventory(doc).casefold()
            for doc in docs
            if self._doc_inventory(doc)
        }
        if inventories:
            return [
                match
                for match in image_matches
                if str(match.inventory or "").strip().casefold() in inventories
            ]

        artifact_ids = {
            str(doc.get("artifact_id") or "").strip()
            for doc in docs
            if str(doc.get("artifact_id") or "").strip()
        }
        if artifact_ids:
            return [
                match
                for match in image_matches
                if str(match.artifact_id or "").strip() in artifact_ids
            ]

        return image_matches

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
        debug_rows: list[dict[str, object]] = []
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
            debug_rows.append(
                {
                    "idx": len(debug_rows) + 1,
                    "match_image": match.original_image_name,
                    "match_artifact_id": artifact_id or None,
                    "match_inventory": inventory or None,
                    "match_title": match.title,
                    "artifact_found": artifact is not None,
                    "artifact_id": artifact.artifact_id if artifact else None,
                    "artifact_inventory": artifact.inventory_number if artifact else None,
                    "artifact_title": artifact.title if artifact else None,
                    "artifact_image_count": len(artifact.images) if artifact else 0,
                    "navigation_found": target is not None,
                    "navigation_inventory": target.inventory_id if target else None,
                    "navigation_overlay_id": target.overlay_id if target else None,
                    "navigation_panorama_key": target.panorama_key if target else None,
                }
            )

        self._log(
            logging.INFO,
            "chat.image_match_enrichment",
            context=context,
            conversation_id=conversation_id,
            museum_slug=museum_slug,
            image_matches=len(image_matches),
            artifact_results=len(artifact_results),
            navigation_targets=len(navigation_targets),
            rows=debug_rows,
        )

        return enriched

    async def _filter_docs_with_llm(
        self,
        *,
        docs: list[dict[str, object]],
        user_message: str,
        museum_slug: str,
        intent: str,
        model_override: str | None,
        system_prompt: str | None,
    ) -> list[dict[str, object]]:
        if len(docs) <= 1:
            return docs
        if self._intent_requires_recall_guardrail(intent):
            self._log(
                logging.INFO,
                "chat.docs_llm_filter_skipped",
                museum_slug=museum_slug,
                intent=intent,
                reason="intent_requires_recall",
                docs=len(docs),
            )
            return docs

        candidates = []
        for index, doc in enumerate(docs, start=1):
            candidates.append(
                {
                    "idx": index,
                    "title": str(doc.get("title") or "").strip() or None,
                    "inventory": self._doc_inventory(doc) or None,
                    "description": str(doc.get("description") or "").strip()[:280] or None,
                }
            )

        selector_prompt = (
            "Seleciona apenas os artefactos relevantes para responder ao pedido do utilizador.\n"
            "Responde estritamente em JSON no formato:\n"
            '{"keep_indexes":[1,2]}\n'
            "Regras:\n"
            "- Usa apenas indices presentes em candidates.\n"
            "- Mantem todos os relevantes (0..N).\n"
            "- Se nenhum for relevante, devolve keep_indexes vazio.\n"
            "- Nao inventes dados fora de candidates.\n\n"
            f"museum_slug: {museum_slug}\n"
            f"intent: {intent}\n"
            f"user_message: {user_message}\n"
            f"candidates: {json.dumps(candidates, ensure_ascii=True)}\n"
        )

        try:
            response = await self.llm_service.generate(
                message=selector_prompt,
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=system_prompt,
                model_override=model_override,
            )
            payload = response.parsed_json
            if not isinstance(payload, dict):
                return docs
            raw_indexes = payload.get("keep_indexes")
            if not isinstance(raw_indexes, list):
                return docs
            keep_positions: set[int] = set()
            for value in raw_indexes:
                if isinstance(value, int):
                    keep_positions.add(value)
                elif isinstance(value, str) and value.isdigit():
                    keep_positions.add(int(value))

            filtered = [doc for idx, doc in enumerate(docs, start=1) if idx in keep_positions]
            self._log(
                logging.INFO,
                "chat.docs_llm_filter",
                museum_slug=museum_slug,
                intent=intent,
                before=len(docs),
                after=len(filtered),
            )
            return filtered
        except Exception as exc:
            self._log(
                logging.WARNING,
                "chat.docs_llm_filter_error",
                museum_slug=museum_slug,
                intent=intent,
                reason=str(exc),
            )
            return docs

    async def _filter_image_matches_with_llm(
        self,
        *,
        image_matches: list[ImageMatchResult],
        user_message: str,
        museum_slug: str,
        intent: str,
        model_override: str | None,
        system_prompt: str | None,
    ) -> list[ImageMatchResult]:
        if len(image_matches) <= 1:
            return image_matches
        if self._intent_requires_recall_guardrail(intent):
            self._log(
                logging.INFO,
                "chat.image_matches_llm_filter_skipped",
                museum_slug=museum_slug,
                intent=intent,
                reason="intent_requires_recall",
                image_matches=len(image_matches),
            )
            return image_matches

        candidates = [
            {
                "idx": index,
                "original_image_name": match.original_image_name,
                "title": match.title,
                "inventory": match.inventory,
                "score": match.score,
            }
            for index, match in enumerate(image_matches, start=1)
        ]
        selector_prompt = (
            "Seleciona apenas os candidatos de imagem que sao relevantes para a pergunta do utilizador.\n"
            "Responde estritamente em JSON com o formato:\n"
            '{"keep_indexes":[1,2]}\n'
            "Regras:\n"
            "- Usa apenas os indices presentes em candidates.\n"
            "- Mantem todos os que fizerem sentido; nao limites a 1 por defeito.\n"
            "- Se nenhum fizer sentido, devolve keep_indexes vazio.\n"
            "- Nao inventes informacao fora de candidates.\n\n"
            f"museum_slug: {museum_slug}\n"
            f"intent: {intent}\n"
            f"user_message: {user_message}\n"
            f"candidates: {json.dumps(candidates, ensure_ascii=True)}\n"
        )

        try:
            response = await self.llm_service.generate(
                message=selector_prompt,
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=system_prompt,
                model_override=model_override,
            )
            payload = response.parsed_json
            if not isinstance(payload, dict):
                return image_matches
            raw_indexes = payload.get("keep_indexes")
            if not isinstance(raw_indexes, list):
                return image_matches
            keep_positions: set[int] = set()
            for value in raw_indexes:
                if isinstance(value, int):
                    keep_positions.add(value)
                elif isinstance(value, str) and value.isdigit():
                    keep_positions.add(int(value))
            filtered = [
                match for idx, match in enumerate(image_matches, start=1) if idx in keep_positions
            ]
            self._log(
                logging.INFO,
                "chat.image_matches_llm_filter",
                museum_slug=museum_slug,
                intent=intent,
                before=len(image_matches),
                after=len(filtered),
            )
            return filtered
        except Exception as exc:
            self._log(
                logging.WARNING,
                "chat.image_matches_llm_filter_error",
                museum_slug=museum_slug,
                intent=intent,
                reason=str(exc),
            )
            return image_matches

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
            limit=6,
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
            r"\b(?:retrieval_context|current_message|explicit_state|recent_history_aux|rolling_summary_aux)\b",
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
        return cleaned.strip()

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
    ) -> str:
        mode = str(router_decision.get("mode", "llm_only"))
        rewritten_query = str(router_decision.get("rewritten_query", payload.message))
        return build_final_answer_prompt(
            museum_slug=payload.museum_slug,
            museum_name=payload.museum_name,
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

    def _extract_single_artifact_id_from_filters(
        self,
        filters: dict[str, object] | None,
    ) -> str | None:
        if not isinstance(filters, dict):
            return None

        value = filters.get("artifact_id")
        if isinstance(value, (str, int, float)):
            cleaned = str(value).strip()
            return cleaned or None

        if isinstance(value, list):
            cleaned_values = [str(item).strip() for item in value if str(item).strip()]
            if len(cleaned_values) == 1:
                return cleaned_values[0]

        return None

    def _infer_selected_artifact_from_reply(
        self,
        *,
        reply_text: str,
        docs: list[dict[str, object]],
    ) -> str | None:
        reply_normalized = (reply_text or "").strip().casefold()
        if not reply_normalized or not docs:
            return None

        first_mention: tuple[int, str] | None = None
        matched_ids: list[str] = []
        for doc in docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            inventory = self._doc_inventory(doc)
            title = str(doc.get("title") or "").strip()
            inventory_match = bool(inventory) and inventory.casefold() in reply_normalized
            title_match = len(title) >= 8 and title.casefold() in reply_normalized
            if inventory_match or title_match:
                matched_ids.append(artifact_id)
                mention_positions = []
                if inventory_match:
                    mention_positions.append(reply_normalized.find(inventory.casefold()))
                if title_match:
                    mention_positions.append(reply_normalized.find(title.casefold()))
                mention_positions = [idx for idx in mention_positions if idx >= 0]
                if mention_positions:
                    mention_idx = min(mention_positions)
                    if first_mention is None or mention_idx < first_mention[0]:
                        first_mention = (mention_idx, artifact_id)

        unique_ids: list[str] = []
        seen: set[str] = set()
        for artifact_id in matched_ids:
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            unique_ids.append(artifact_id)

        if len(unique_ids) == 1:
            return unique_ids[0]
        if first_mention is not None:
            return first_mention[1]
        return None

    def _update_result_selection_state(
        self,
        *,
        state: ChatSessionState,
        artifact_docs: list[dict[str, object]],
        effective_filters: dict[str, object] | None,
        hinted_selected_artifact_id: str | None = None,
        preserve_existing_when_empty: bool = False,
    ) -> None:
        artifact_ids = self._extract_artifact_ids_from_docs(artifact_docs)
        if artifact_ids:
            state.last_result_ids = artifact_ids
        elif not preserve_existing_when_empty:
            state.last_result_ids = []

        selected_artifact_id = self._extract_single_artifact_id_from_filters(effective_filters)
        if (
            not selected_artifact_id
            and hinted_selected_artifact_id
            and (not artifact_ids or hinted_selected_artifact_id in artifact_ids)
        ):
            selected_artifact_id = hinted_selected_artifact_id
        if not selected_artifact_id and artifact_ids:
            # Keep deterministic focus for follow-up singular references.
            selected_artifact_id = artifact_ids[0]

        if selected_artifact_id:
            state.selected_artifact_id = selected_artifact_id
        elif artifact_ids:
            state.selected_artifact_id = None
        elif not preserve_existing_when_empty:
            state.selected_artifact_id = None

    def _apply_follow_up_artifact_scope(
        self,
        *,
        message: str,
        state: ChatSessionState,
        router_decision: dict[str, object],
        filters: dict[str, object],
    ) -> dict[str, object]:
        scoped_filters = dict(filters)
        if str(router_decision.get("mode", "")) != "rag":
            return scoped_filters
        if not bool(router_decision.get("use_history_for_query", False)):
            return scoped_filters

        existing_single = self._extract_single_artifact_id_from_filters(scoped_filters)
        existing_value = scoped_filters.get("artifact_id")
        if existing_single or (isinstance(existing_value, list) and existing_value):
            return scoped_filters

        text = (message or "").strip().lower()
        singular_referential_patterns = (
            r"\b(nele|nela|dele|dela|nisto|nisso|disto|disso)\b",
            r"\b(deste|desta|desse|dessa|este|esta|esse|essa)\b",
            r"\b(neste|nesta|nesse|nessa)\b",
            r"\b(o mesmo|a mesma)\b",
            r"\b(este azulejo|esta peca|este objeto|essa peca|esse objeto|neste azulejo|nessa peca)\b",
            r"\b(fala mais sobre isso|detalha isso|explica isso)\b",
        )
        plural_referential_patterns = (
            r"\b(desses|dessas|esses|essas|deles|delas|neles|nelas)\b",
            r"\b(nestes|nestas|nesses|nessas)\b",
            r"\b(os mesmos|as mesmas)\b",
            r"\b(esses azulejos|desses azulejos|esses objetos|desses objetos)\b",
        )
        has_plural_reference = any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in plural_referential_patterns
        )
        has_singular_reference = any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in singular_referential_patterns
        )
        if not has_singular_reference and not has_plural_reference:
            return scoped_filters

        if has_plural_reference:
            candidate_ids = [artifact_id.strip() for artifact_id in state.last_result_ids if artifact_id.strip()]
            if not candidate_ids:
                return scoped_filters
            scoped_filters["artifact_id"] = candidate_ids
            return scoped_filters

        candidate_artifact_id = (state.selected_artifact_id or "").strip() or None
        if not candidate_artifact_id and state.last_result_ids:
            candidate_artifact_id = state.last_result_ids[0]
        if not candidate_artifact_id:
            return scoped_filters

        scoped_filters["artifact_id"] = candidate_artifact_id
        return scoped_filters

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

    def _log_state_after_reply(self, state: ChatSessionState) -> None:
        history_preview = [
            {"role": turn.role, "text": turn.text[:160]}
            for turn in state.history[-self.settings.CHAT_HISTORY_WINDOW :]
        ]
        payload: dict[str, object] = {
            "conversation_id": state.conversation_id,
            "museum_slug": state.museum_slug,
            "language": state.language,
            "intent": state.intent,
            "filters": state.filters,
            "sort": state.sort,
            "selected_artifact_id": state.selected_artifact_id,
            "last_result_ids": state.last_result_ids,
            "rolling_summary": state.rolling_summary,
            "history_size": len(state.history),
            "history_preview": history_preview,
        }

        if self.settings.LOG_CHAT_STATE_HISTORY:
            payload["history_full"] = [
                {"role": turn.role, "text": turn.text}
                for turn in state.history
            ]

        self._log(logging.INFO, "chat.state_after_reply", **payload)


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
    )
