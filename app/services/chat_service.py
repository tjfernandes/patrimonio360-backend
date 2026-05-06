from functools import lru_cache
import json
import logging
import re
import unicodedata
from typing import Any, Awaitable, Callable
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.prompts import (
    build_lexical_query_prompt,
    build_final_answer_prompt,
    build_router_user_prompt,
    get_lexical_query_system_prompt,
    get_router_system_prompt,
)
from app.query_planning import (
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
    ChatHealthResponse,
    ChatImageMessageRequest,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatModelMessageRequest,
    ChatRegenerateRequest,
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
from app.services.llm_service import LLMService, LLMServiceError, get_llm_service
from app.services.opensearch_client import OpenSearchGateway, get_opensearch_gateway
from app.services.model_retrieval import ModelRetrievalService, get_model_retrieval_service
from app.services.reranker import (
    PairwiseRerankerService,
    RerankerError,
    get_reranker_service,
)
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
    "chat.lexical_query": "Query lexical limpa gerada para OpenSearch.",
    "chat.lexical_query_error": "Erro ao gerar query lexical por LLM; fallback local aplicado.",
    "chat.retrieve_embedding_ready": "Embedding de query gerado para retrieval.",
    "chat.retrieve": "Tentativa de retrieval no RAG.",
    "chat.retrieve_embedding_disabled": "Embeddings de query desativados; retrieval ignorado.",
    "chat.retrieve_embedding_error": "Embedding indisponivel; retrieval cancelado.",
    "chat.retrieve_skipped": "Retrieval ignorado nesta mensagem.",
    "chat.image_embedding_ready": "Embedding multimodal da imagem gerado.",
    "chat.image_retrieve": "Image retrieval executado com sucesso.",
    "chat.image_matches_filtered": "Image matches filtrados apos resposta final.",
    "chat.image_retrieve_empty": "Image retrieval sem resultados.",
    "chat.image_retrieve_error": "Erro inesperado durante image retrieval.",
    "chat.model_retrieve": "Model retrieval executado com sucesso.",
    "chat.model_retrieve_empty": "Model retrieval sem resultados.",
    "chat.model_retrieve_error": "Erro inesperado durante model retrieval.",
    "chat.rerank": "Reranking aplicado aos candidatos de retrieval.",
    "chat.rerank_skipped": "Reranking ignorado para esta mensagem.",
    "chat.rerank_error": "Erro durante reranking; fallback para ordem do retrieval.",
    "chat.reply_sanitized": "Resposta final sanitizada para remover IDs internos.",
    "chat.llm_error": "Erro ao chamar o LLM.",
    "chat.reply": "Resposta final gerada pelo LLM.",
    "chat.state_after_reply": "Estado da conversa apos resposta.",
    "chat.retrieve_not_implemented": "Retrieval ainda nao implementado neste ambiente.",
    "chat.retrieve_error": "Erro inesperado durante retrieval.",
    "chat.navigation_targets": "Alvos de navegacao em tour resolvidos para esta resposta.",
    "chat.context_policy": "Politica de reutilizacao de contexto para o turno atual.",
    "chat.context_artifact_scope": "Filtro contextual por artifact_id aplicado para follow-up referencial.",
    "chat.structured.classify": "Classificacao da query para modo estruturado.",
    "chat.structured.plan": "Plano estruturado gerado para query analitica.",
    "chat.structured.fallback": "Fallback do modo estruturado para fluxo RAG.",
    "chat.structured.execute": "Query estruturada executada no OpenSearch.",
    "chat.structured.reply": "Resposta final gerada por executor estruturado.",
}

_SUPPORTED_LANGUAGES = {"pt", "en"}
_DEFAULT_LANGUAGE = "pt"
_STATUS_MESSAGES: dict[str, dict[str, str]] = {
    "analyzing_request": {
        "pt": "A analisar pedido",
        "en": "Analyzing request",
    },
    "searching_collection": {
        "pt": "A procurar artefactos no acervo",
        "en": "Searching artifacts in collection",
    },
    "found_artifacts": {
        "pt": "Encontrados {artifact_count} artefactos",
        "en": "Found {artifact_count} artifacts",
    },
    "generating_final_answer": {
        "pt": "A gerar resposta final",
        "en": "Generating final answer",
    },
    "answer_ready": {
        "pt": "Resposta pronta",
        "en": "Answer ready",
    },
    "analyzing_uploaded_image": {
        "pt": "A analisar imagem enviada",
        "en": "Analyzing uploaded image",
    },
    "preparing_3d_model": {
        "pt": "A preparar modelo 3D",
        "en": "Preparing 3D model",
    },
    "interpreting_analytics_request": {
        "pt": "A interpretar pedido analitico",
        "en": "Interpreting analytical request",
    },
    "querying_collection": {
        "pt": "A consultar o acervo",
        "en": "Querying collection",
    },
    "rendering_model_views": {
        "pt": "A gerar vistas do modelo 3D",
        "en": "Generating 3D model views",
    },
}
_MODEL_PROGRESS_STAGE_TO_STATUS: dict[str, str] = {
    "render_first_pass": "rendering_model_views",
    "search_first_pass": "searching_collection",
}
_MODEL_PROGRESS_LEGACY_TO_STATUS: dict[str, str] = {
    "A gerar vistas do modelo 3D": "rendering_model_views",
    "A procurar artefactos no acervo": "searching_collection",
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
        reranker_service: PairwiseRerankerService,
        tour_navigation_service: TourNavigationService,
        llm_service: LLMService,
        session_store: ChatSessionStore,
    ) -> None:
        self.settings = settings
        self.opensearch_gateway = opensearch_gateway
        self.embedding_provider = embedding_provider
        self.model_retrieval_service = model_retrieval_service
        self.reranker_service = reranker_service
        self.tour_navigation_service = tour_navigation_service
        self.llm_service = llm_service
        self.session_store = session_store

    def _resolve_language(self, language: str | None, *, fallback: str | None = None) -> str:
        candidate = (language or "").strip().lower()
        if candidate in _SUPPORTED_LANGUAGES:
            return candidate
        fallback_candidate = (fallback or "").strip().lower()
        if fallback_candidate in _SUPPORTED_LANGUAGES:
            return fallback_candidate
        return _DEFAULT_LANGUAGE

    def _status_text(self, key: str, *, language: str, **kwargs: object) -> str:
        templates = _STATUS_MESSAGES.get(key)
        if not templates:
            return key
        template = templates.get(language) or templates[_DEFAULT_LANGUAGE]
        if kwargs:
            try:
                return template.format(**kwargs)
            except Exception:
                return template
        return template

    def _translate_model_progress_message(
        self,
        *,
        message: str,
        language: str,
        stage: object,
    ) -> str:
        stage_key = str(stage or "").strip()
        status_key = _MODEL_PROGRESS_STAGE_TO_STATUS.get(stage_key)
        if status_key:
            return self._status_text(status_key, language=language)

        legacy_key = _MODEL_PROGRESS_LEGACY_TO_STATUS.get((message or "").strip())
        if legacy_key:
            return self._status_text(legacy_key, language=language)
        return message

    def _llm_unavailable_fallback(self, *, language: str, error: str) -> str:
        if language == "en":
            return (
                f"LLM unavailable in dev: {error}. "
                "Check LLM_PROVIDER/LLM_BASE_URL/model settings in backend .env."
            )
        return (
            f"LLM indisponivel no ambiente de desenvolvimento: {error}. "
            "Verifica LLM_PROVIDER/LLM_BASE_URL/model no .env do backend."
        )

    def _image_processing_fallback(self, *, language: str) -> str:
        if language == "en":
            return (
                "I could not process the uploaded image. "
                "Try again with another image or a different format."
            )
        return (
            "Nao consegui processar a imagem enviada. "
            "Tenta novamente com outra imagem ou formato diferente."
        )

    def _model_processing_fallback(self, *, language: str) -> str:
        if language == "en":
            return (
                "I could not process the uploaded 3D model. "
                "Try again with a valid .glb, .gltf, or .obj file."
            )
        return (
            "Nao consegui processar o modelo 3D enviado. "
            "Tenta novamente com um ficheiro .glb, .gltf ou .obj valido."
        )

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

    def health(self) -> ChatHealthResponse:
        return ChatHealthResponse(
            llm_provider=self.settings.LLM_PROVIDER,
            llm_base_url=self.settings.llm_base_url_resolved,
            llm_text_model=self.settings.llm_model_resolved,
            llm_json_model=self.settings.llm_model_resolved,
            text_embedding_model=self.settings.text_embedding_model_resolved,
            multimodal_embedding_model=self.settings.multimodal_embedding_model_resolved,
            reranking_enabled=self.settings.CHAT_ENABLE_RERANKING,
            reranker_model=self.settings.reranker_model_resolved,
        )

    async def _emit_status(
        self,
        status_cb: StatusCallback | None,
        message: str,
        **fields: object,
    ) -> None:
        if status_cb is None:
            return
        payload: dict[str, Any] = {"type": "status", "message": message, **fields}
        try:
            await status_cb(payload)
        except Exception:
            logger.debug("status callback failed", exc_info=True)

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
        language = self._resolve_language(payload.language, fallback=state.language)
        state.language = language
        self.session_store.append_turn(state, role="user", text=payload.message)
        await self._emit_status(
            status_cb,
            self._status_text("analyzing_request", language=language),
        )

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
            language=language,
        )
        if structured_response is not None:
            return structured_response

        try:
            router_decision = await self._route_message(
                payload=payload,
                state=state,
                context_policy=context_policy,
                language=language,
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
        image_matches: list[ImageMatchResult] = []
        if router_decision["mode"] == "rag" and self.settings.CHAT_ENABLE_RAG:
            # Retrieval must stay strict to user wording (no expansion/rewrite additions).
            retrieval_query = payload.message
            await self._emit_status(
                status_cb,
                self._status_text("searching_collection", language=language),
            )
            retrieval_context, retrieved_docs_count, retrieved_docs = await self._retrieve_context(
                museum_slug=payload.museum_slug,
                museum_id=self._resolve_museum_id(payload),
                query=retrieval_query,
                filters=effective_filters,
                sort=effective_sort,
                language=language,
            )
            await self._emit_status(
                status_cb,
                self._status_text(
                    "found_artifacts",
                    language=language,
                    artifact_count=retrieved_docs_count,
                ),
                artifact_count=retrieved_docs_count,
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
                        image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
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
                        image_hits = []
                    image_matches = self._build_image_matches(
                        image_hits=image_hits,
                        artifact_docs=retrieved_docs,
                    )
        else:
            self._log(
                logging.DEBUG,
                "chat.retrieve_skipped",
                conversation_id=conversation_id,
                mode=router_decision.get("mode"),
                rag_enabled=self.settings.CHAT_ENABLE_RAG,
            )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=self._resolve_museum_id(payload),
            docs=retrieved_docs,
        )
        if navigation_targets:
            self._log(
                logging.INFO,
                "chat.navigation_targets",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                targets=[target.model_dump(exclude_none=True) for target in navigation_targets],
            )

        final_message = self._build_final_prompt(
            payload=payload,
            state=state,
            router_decision=router_decision,
            retrieval_context=retrieval_context,
            effective_filters=effective_filters,
            effective_sort=effective_sort,
            use_history_for_answer=use_history_for_answer,
            language=language,
        )
        await self._emit_status(
            status_cb,
            self._status_text("generating_final_answer", language=language),
        )

        try:
            llm_response = await self.llm_service.generate(
                message=final_message,
                response_format=requested_format,
                system_prompt=payload.system_prompt,
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
            fallback = self._llm_unavailable_fallback(language=language, error=str(exc))
            return ChatMessageResponse(
                conversation_id=conversation_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override
                or self.settings.llm_model_resolved,
                image_matches=image_matches,
                navigation_targets=navigation_targets,
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
        image_matches = self._filter_image_matches_by_relevance(
            reply_text=final_reply_text,
            image_matches=image_matches,
            docs=retrieved_docs,
            intent=str(router_decision.get("intent", "")),
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
        await self._emit_status(
            status_cb,
            self._status_text("answer_ready", language=language),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
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
        language = self._resolve_language(payload.language, fallback=state.language)
        state.language = language
        if not state.history:
            if language == "en":
                raise ValueError("Conversation has no history to regenerate.")
            raise ValueError("Conversa sem historico para regenerar.")

        while state.history and state.history[-1].role == "assistant":
            state.history.pop()

        if not state.history or state.history[-1].role != "user":
            if language == "en":
                raise ValueError("No user message available to regenerate the reply.")
            raise ValueError("Nao existe mensagem de utilizador para regenerar resposta.")

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
        museum_id = self._resolve_museum_id_values(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._resolve_language(payload.language, fallback=state.language)
        state.language = language
        default_image_message = (
            "Analyze this image and identify the most likely artifact in the museum."
            if language == "en"
            else self.settings.CHAT_IMAGE_DEFAULT_MESSAGE
        )
        user_message = (payload.message or "").strip() or default_image_message
        self._log(
            logging.INFO,
            "chat.receive_image",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            response_format=requested_format.type,
            language=language,
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
        await self._emit_status(
            status_cb,
            self._status_text("analyzing_uploaded_image", language=language),
        )

        image_matches: list[ImageMatchResult] = []
        artifact_docs: list[dict[str, object]] = []
        navigation_targets: list[TourNavigationTarget] = []
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
            fallback = self._image_processing_fallback(language=language)
            return ChatMessageResponse(
                conversation_id=conversation_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
            )

        await self._emit_status(
            status_cb,
            self._status_text("searching_collection", language=language),
        )
        image_candidates_k = max(
            self.settings.CHAT_IMAGE_RETRIEVAL_TOP_K,
            self.settings.CHAT_IMAGE_ARTIFACT_TOP_K,
            1,
        )
        try:
            image_hits = await self.opensearch_gateway.search_similar_images(
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                image_embedding=image_embedding,
                top_k=image_candidates_k,
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
            artifact_ids = [
                str(hit.get("artifact_id") or "").strip()
                for hit in image_hits
                if str(hit.get("artifact_id") or "").strip()
            ]
            try:
                artifact_docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
                    museum_slug=payload.museum_slug,
                    museum_id=museum_id,
                    artifact_ids=artifact_ids,
                    top_k=max(self.settings.CHAT_IMAGE_ARTIFACT_TOP_K, 1),
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
            await self._emit_status(
                status_cb,
                self._status_text(
                    "found_artifacts",
                    language=language,
                    artifact_count=len(artifact_docs),
                ),
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
                self._status_text("found_artifacts", language=language, artifact_count=0),
                artifact_count=0,
            )

        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            docs=artifact_docs,
        )
        if navigation_targets:
            self._log(
                logging.INFO,
                "chat.navigation_targets",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                targets=[target.model_dump(exclude_none=True) for target in navigation_targets],
            )

        retrieval_context_sections: list[str] = []
        if image_matches:
            retrieval_context_sections.append("image_retrieval_matches:")
            retrieval_context_sections.append(
                json.dumps(
                    [match.model_dump(exclude_none=True) for match in image_matches],
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
        await self._emit_status(
            status_cb,
            self._status_text("generating_final_answer", language=language),
        )

        try:
            llm_response = await self.llm_service.generate(
                message=final_message,
                response_format=requested_format,
                system_prompt=payload.system_prompt,
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
            fallback = self._llm_unavailable_fallback(language=language, error=str(exc))
            return ChatMessageResponse(
                conversation_id=conversation_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                image_matches=image_matches,
                navigation_targets=navigation_targets,
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
        image_matches = self._filter_image_matches_by_relevance(
            reply_text=final_reply_text,
            image_matches=image_matches,
            docs=artifact_docs,
            intent="image_search",
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
        await self._emit_status(
            status_cb,
            self._status_text("answer_ready", language=language),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
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
        museum_id = self._resolve_museum_id_values(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._resolve_language(payload.language, fallback=state.language)
        state.language = language
        default_model_message = (
            "Analyze this 3D model and identify the most likely artifact in the museum."
            if language == "en"
            else self.settings.CHAT_MODEL_DEFAULT_MESSAGE
        )
        user_message = (payload.message or "").strip() or default_model_message
        self._log(
            logging.INFO,
            "chat.receive_model",
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            response_format=requested_format.type,
            language=language,
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
        await self._emit_status(
            status_cb,
            self._status_text("preparing_3d_model", language=language),
        )

        image_matches: list[ImageMatchResult] = []
        artifact_docs: list[dict[str, object]] = []
        navigation_targets: list[TourNavigationTarget] = []
        file_name = (model_filename or "model.glb").strip() or "model.glb"

        try:
            retrieval_result = await self.model_retrieval_service.retrieve(
                museum_slug=payload.museum_slug,
                museum_id=museum_id,
                model_bytes=model_bytes,
                file_name=file_name,
                progress_cb=lambda message, fields: self._emit_status(
                    status_cb,
                    self._translate_model_progress_message(
                        message=message,
                        language=language,
                        stage=fields.get("stage"),
                    ),
                    **fields,
                ),
            )
            artifact_docs = retrieval_result.artifact_docs
            image_matches = self._build_image_matches(
                image_hits=retrieval_result.image_hits,
                artifact_docs=artifact_docs,
            )
            if retrieval_result.image_hits:
                self._log(
                    logging.INFO,
                    "chat.model_retrieve",
                    conversation_id=conversation_id,
                    museum_slug=payload.museum_slug,
                    museum_id=museum_id,
                    image_hits=len(retrieval_result.image_hits),
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
            fallback = self._model_processing_fallback(language=language)
            return ChatMessageResponse(
                conversation_id=conversation_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
            )

        await self._emit_status(
            status_cb,
            self._status_text(
                "found_artifacts",
                language=language,
                artifact_count=len(artifact_docs),
            ),
            artifact_count=len(artifact_docs),
        )

        navigation_targets = self._resolve_navigation_targets(
            museum_slug=payload.museum_slug,
            museum_id=museum_id,
            docs=artifact_docs,
        )
        if navigation_targets:
            self._log(
                logging.INFO,
                "chat.navigation_targets",
                conversation_id=conversation_id,
                museum_slug=payload.museum_slug,
                targets=[target.model_dump(exclude_none=True) for target in navigation_targets],
            )

        retrieval_context_sections: list[str] = []
        if image_matches:
            retrieval_context_sections.append("model_view_matches:")
            retrieval_context_sections.append(
                json.dumps(
                    [match.model_dump(exclude_none=True) for match in image_matches],
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
        await self._emit_status(
            status_cb,
            self._status_text("generating_final_answer", language=language),
        )

        try:
            llm_response = await self.llm_service.generate(
                message=final_message,
                response_format=requested_format,
                system_prompt=payload.system_prompt,
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
            fallback = self._llm_unavailable_fallback(language=language, error=str(exc))
            return ChatMessageResponse(
                conversation_id=conversation_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                image_matches=image_matches,
                navigation_targets=navigation_targets,
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
        image_matches = self._filter_image_matches_by_relevance(
            reply_text=final_reply_text,
            image_matches=image_matches,
            docs=artifact_docs,
            intent="model_search",
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
        await self._emit_status(
            status_cb,
            self._status_text("answer_ready", language=language),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
        )

    def _build_structured_query_schema(self, *, museum_id: str | None) -> QuerySchema:
        fields = {
            "artifact_id": QuerySchemaField(type="keyword", facetable=True),
            "museum_id": QuerySchemaField(type="keyword", facetable=True),
            "inventory": QuerySchemaField(type="keyword", facetable=True, text=True),
            "title": QuerySchemaField(type="text", text=True, semantic=True),
            "title.keyword": QuerySchemaField(type="keyword", facetable=True),
            "description": QuerySchemaField(type="text", text=True, semantic=True),
            "full_text": QuerySchemaField(type="text", text=True, semantic=True),
            "category": QuerySchemaField(type="keyword", facetable=True, text=True),
            "support": QuerySchemaField(type="keyword", facetable=True, text=True),
            "location": QuerySchemaField(type="keyword", facetable=True, text=True),
            "historical_origin": QuerySchemaField(type="keyword", facetable=True, text=True),
            "manufacturer_location": QuerySchemaField(type="keyword", facetable=True, text=True),
            "incorporation": QuerySchemaField(type="keyword", facetable=True, text=True),
            "initial_year": QuerySchemaField(type="integer", facetable=True),
            "final_year": QuerySchemaField(type="integer", facetable=True),
            "initial_century": QuerySchemaField(type="integer", facetable=True),
            "final_century": QuerySchemaField(type="integer", facetable=True),
            "in_tour": QuerySchemaField(type="boolean", facetable=True),
            "museum_name": QuerySchemaField(type="text", text=True),
        }
        default_filters: list[TermFilter] = []
        if museum_id:
            default_filters.append(TermFilter(kind="term", field="museum_id", value=museum_id))

        return QuerySchema(
            index_name=self.settings.OPENSEARCH_INDEX_ARTIFACT,
            fields=fields,
            facetable_fields=[
                "museum_id",
                "category",
                "support",
                "location",
                "historical_origin",
                "manufacturer_location",
                "incorporation",
                "initial_year",
                "final_year",
                "initial_century",
                "final_century",
                "title.keyword",
                "inventory",
                "in_tour",
            ],
            text_fields=[
                "full_text",
                "title",
                "description",
                "inventory",
                "category",
                "support",
                "location",
                "historical_origin",
                "manufacturer_location",
                "museum_name",
            ],
            semantic_fields=["full_text", "title", "description"],
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
        language: str,
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

        await self._emit_status(
            status_cb,
            self._status_text("interpreting_analytics_request", language=language),
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

        await self._emit_status(
            status_cb,
            self._status_text("querying_collection", language=language),
        )
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
        await self._emit_status(
            status_cb,
            self._status_text("answer_ready", language=language),
        )

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

        return ChatMessageResponse(
            conversation_id=conversation_id,
            response_format=requested_format,
            reply=final_reply_text,
            reply_json=reply_json_payload,
            model_hint="structured_query_executor",
            navigation_targets=navigation_targets,
        )

    def _format_structured_reply(
        self,
        *,
        plan: QueryPlan,
        result: QueryExecutionResult,
        language: str,
    ) -> str:
        if language == "en":
            if plan.operation == "count":
                count = int(result.count or 0)
                if count == 1:
                    return "There is 1 artifact in the collection that matches the request."
                return f"There are {count} artifacts in the collection that match the request."

            if plan.operation == "exists":
                if result.exists:
                    return "Yes, there is at least 1 artifact in the collection that matches the request."
                return "I did not find artifacts in the collection that match the request."

            if plan.operation == "list":
                if not result.items:
                    return "I did not find artifacts in the collection that match the request."
                lines = ["Artifacts found:"]
                for index, item in enumerate(result.items, start=1):
                    title = str(item.get("title") or "").strip()
                    inventory = str(item.get("inventory") or "").strip()
                    if title and inventory:
                        lines.append(f"{index}. {title} ({inventory})")
                    elif title:
                        lines.append(f"{index}. {title}")
                    elif inventory:
                        lines.append(f"{index}. Inventory {inventory}")
                    else:
                        lines.append(f"{index}. Artifact without an available title")
                return "\n".join(lines)

            if plan.operation == "group_by":
                if not result.groups:
                    return "I did not find collection results to group."
                lines = ["Distribution found:"]
                for bucket in result.groups:
                    key = bucket.key.strip() or "No value"
                    lines.append(f"- {key}: {bucket.doc_count}")
                return "\n".join(lines)

            return "I could not generate an answer for this analytical query."

        if plan.operation == "count":
            count = int(result.count or 0)
            if count == 1:
                return "Existe 1 artefacto no acervo que corresponde ao pedido."
            return f"Existem {count} artefactos no acervo que correspondem ao pedido."

        if plan.operation == "exists":
            if result.exists:
                return "Sim, existe pelo menos 1 artefacto no acervo que corresponde ao pedido."
            return "Nao encontrei artefactos no acervo que correspondam ao pedido."

        if plan.operation == "list":
            if not result.items:
                return "Nao encontrei artefactos no acervo que correspondam ao pedido."
            lines = ["Artefactos encontrados:"]
            for index, item in enumerate(result.items, start=1):
                title = str(item.get("title") or "").strip()
                inventory = str(item.get("inventory") or "").strip()
                if title and inventory:
                    lines.append(f"{index}. {title} ({inventory})")
                elif title:
                    lines.append(f"{index}. {title}")
                elif inventory:
                    lines.append(f"{index}. Inventario {inventory}")
                else:
                    lines.append(f"{index}. Artefacto sem titulo disponivel")
            return "\n".join(lines)

        if plan.operation == "group_by":
            if not result.groups:
                return "Nao encontrei resultados no acervo para agrupar."
            lines = ["Distribuicao encontrada:"]
            for bucket in result.groups:
                key = bucket.key.strip() or "Sem valor"
                lines.append(f"- {key}: {bucket.doc_count}")
            return "\n".join(lines)

        return "Nao foi possivel gerar resposta para esta query analitica."

    async def _route_message(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        context_policy: dict[str, object],
        language: str,
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
            language=language,
        )

        router_response = await self.llm_service.generate(
            message=router_prompt,
            response_format=ResponseFormatObject(type="json_object"),
            system_prompt=get_router_system_prompt(language=language),
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
        language: str,
    ) -> tuple[str, int, list[dict[str, object]]]:
        raw_query = (query or "").strip()
        lexical_query = await self._build_lexical_query(
            query=raw_query,
            museum_slug=museum_slug,
            museum_id=museum_id,
            language=language,
        )
        embedding_query = (lexical_query or raw_query).strip()
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
                return "", 0, []
            except Exception as exc:
                self._log(
                    logging.WARNING,
                    "chat.retrieve_embedding_error",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    reason=str(exc),
                )
                return "", 0, []
        else:
            self._log(
                logging.INFO,
                "chat.retrieve_embedding_disabled",
                museum_slug=museum_slug,
                museum_id=museum_id,
            )
            return "", 0, []

        final_top_k = max(self.settings.CHAT_RETRIEVAL_TOP_K, 1)
        retrieval_candidates_k = (
            max(self.settings.CHAT_RETRIEVAL_CANDIDATES, final_top_k, 1)
            if self.settings.CHAT_ENABLE_RERANKING
            else final_top_k
        )

        try:
            docs = await self.opensearch_gateway.search_relevant_context(
                museum_slug=museum_slug,
                museum_id=museum_id,
                query_text=embedding_query,
                lexical_query=lexical_query,
                query_embedding=query_embedding,
                top_k=retrieval_candidates_k,
                filters=filters,
                sort=sort,
            )
        except Exception:
            self._log(logging.ERROR, "chat.retrieve_error", museum_slug=museum_slug)
            logger.exception("chat.retrieve_error exception")
            return "", 0, []

        if not docs:
            return "", 0, []

        retrieved_candidates = len(docs)
        if self.settings.CHAT_ENABLE_RERANKING:
            top_before = [doc.get("artifact_id") for doc in docs[:3] if doc.get("artifact_id")]
            try:
                reranked_docs = await self.reranker_service.rerank(
                    query_text=query,
                    documents=docs,
                    top_k=final_top_k,
                )
            except RerankerError as exc:
                self._log(
                    logging.WARNING,
                    "chat.rerank_error",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    reason=str(exc),
                )
                docs = docs[:final_top_k]
            else:
                if reranked_docs:
                    docs = reranked_docs
                else:
                    docs = docs[:final_top_k]
                top_after = [doc.get("artifact_id") for doc in docs[:3] if doc.get("artifact_id")]
                self._log(
                    logging.INFO,
                    "chat.rerank",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    reranker_model=self.settings.reranker_model_resolved,
                    candidates=retrieved_candidates,
                    returned=len(docs),
                    requested_candidates=retrieval_candidates_k,
                    final_top_k=final_top_k,
                    top_before=top_before,
                    top_after=top_after,
                )
        else:
            self._log(
                logging.DEBUG,
                "chat.rerank_skipped",
                museum_slug=museum_slug,
                museum_id=museum_id,
                reason="CHAT_ENABLE_RERANKING=false",
                candidates=retrieved_candidates,
                requested_candidates=retrieval_candidates_k,
                final_top_k=final_top_k,
            )
            docs = docs[:final_top_k]

        context = self._format_docs_for_prompt(docs=docs, top_k=final_top_k)
        return context, len(docs[:final_top_k]), docs[:final_top_k]

    async def _build_lexical_query(
        self,
        *,
        query: str,
        museum_slug: str,
        museum_id: str | None,
        language: str,
    ) -> str:
        raw_query = (query or "").strip()
        if not raw_query:
            return ""

        fallback_query = self._build_lexical_query_fallback(
            query=raw_query,
            museum_slug=museum_slug,
            museum_id=museum_id,
        )
        if not self.settings.CHAT_ENABLE_LLM_LEXICAL_QUERY:
            return fallback_query

        generate = getattr(self.llm_service, "generate", None)
        if not callable(generate):
            return fallback_query

        try:
            llm_response = await generate(
                message=build_lexical_query_prompt(
                    query=raw_query,
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    language=language,
                ),
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=get_lexical_query_system_prompt(language=language),
            )
        except Exception as exc:
            self._log(
                logging.WARNING,
                "chat.lexical_query_error",
                museum_slug=museum_slug,
                museum_id=museum_id,
                original_query=raw_query,
                fallback_query=fallback_query,
                reason=str(exc),
            )
            return fallback_query

        candidate = ""
        parsed_json = getattr(llm_response, "parsed_json", None)
        if isinstance(parsed_json, dict):
            candidate = str(
                parsed_json.get("lexical_query")
                or parsed_json.get("query")
                or parsed_json.get("clean_query")
                or ""
            )
        if not candidate:
            text = str(getattr(llm_response, "text", "") or "").strip()
            try:
                parsed_text = json.loads(text)
            except Exception:
                parsed_text = None
            if isinstance(parsed_text, dict):
                candidate = str(
                    parsed_text.get("lexical_query")
                    or parsed_text.get("query")
                    or parsed_text.get("clean_query")
                    or ""
                )

        lexical_query = self._sanitize_lexical_query_candidate(candidate)
        if not lexical_query:
            self._log(
                logging.WARNING,
                "chat.lexical_query_error",
                museum_slug=museum_slug,
                museum_id=museum_id,
                original_query=raw_query,
                fallback_query=fallback_query,
                reason="empty_or_invalid_llm_output",
            )
            return fallback_query

        self._log(
            logging.INFO,
            "chat.lexical_query",
            museum_slug=museum_slug,
            museum_id=museum_id,
            original_query=raw_query,
            lexical_query=lexical_query,
            fallback_query=fallback_query,
            query_changed=lexical_query != raw_query,
        )
        return lexical_query

    def _build_lexical_query_fallback(
        self,
        *,
        query: str,
        museum_slug: str,
        museum_id: str | None,
    ) -> str:
        text = (query or "").strip().lower()
        if not text:
            return ""
        normalized = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        tokens = [token for token in normalized.split() if token]

        museum_tokens = {
            self._fold_lexical_token(value)
            for value in (museum_slug, museum_id or "")
            if str(value or "").strip()
        }
        filtered: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            folded = self._fold_lexical_token(token)
            if len(folded) <= 1 or folded in museum_tokens or folded in seen:
                continue
            seen.add(folded)
            filtered.append(token)
        return " ".join(filtered[:12])

    def _sanitize_lexical_query_candidate(self, value: str) -> str:
        text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
        if not text:
            return ""
        text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[`´'\"“”‘’<>()[\]{}^~*?:\\/+=!|&]", " ", text)
        text = re.sub(r"[^\w\s.-]", " ", text, flags=re.UNICODE)
        tokens = [token.strip(".-") for token in text.split() if token.strip(".-")]

        filtered: list[str] = []
        seen: set[str] = set()
        boolean_operator_tokens = {"and", "or", "not"}
        for token in tokens:
            folded = self._fold_lexical_token(token)
            if len(folded) <= 1 or folded in boolean_operator_tokens or folded in seen:
                continue
            seen.add(folded)
            filtered.append(token)
        return " ".join(filtered[:12])[:160].strip()

    def _fold_lexical_token(self, value: str) -> str:
        normalized_value = unicodedata.normalize("NFKD", value.casefold())
        return "".join(char for char in normalized_value if not unicodedata.combining(char))

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

    def _build_image_matches(
        self,
        *,
        image_hits: list[dict[str, object]],
        artifact_docs: list[dict[str, object]],
    ) -> list[ImageMatchResult]:
        doc_by_id: dict[str, dict[str, object]] = {}
        for doc in artifact_docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if artifact_id:
                doc_by_id[artifact_id] = doc

        matches: list[ImageMatchResult] = []
        for hit in image_hits:
            original_image_name = str(hit.get("original_image_name") or "").strip()
            if not original_image_name:
                continue
            artifact_id = str(hit.get("artifact_id") or "").strip()
            if not artifact_id or artifact_id not in doc_by_id:
                continue
            doc = doc_by_id.get(artifact_id, {})
            score_raw = hit.get("score")
            score: float | None = None
            if isinstance(score_raw, (int, float)):
                score = float(score_raw)
            matches.append(
                ImageMatchResult(
                    original_image_name=original_image_name,
                    score=score,
                    title=str(doc.get("title") or "").strip() or None,
                    inventory=str(doc.get("inventory") or "").strip() or None,
                )
            )
        return matches

    def _resolve_navigation_targets(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        docs: list[dict[str, object]],
    ) -> list[TourNavigationTarget]:
        inventories: list[str] = []
        for doc in docs:
            inventory = str(doc.get("inventory") or "").strip()
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
        language: str = _DEFAULT_LANGUAGE,
    ) -> str:
        cleaned = text or ""
        doc_label_map = self._build_doc_label_map(docs or [], language=language)
        if doc_label_map:
            cleaned = self._replace_doc_refs_with_labels(
                text=cleaned,
                doc_label_map=doc_label_map,
                language=language,
            )

        artifact_label = "the artifact" if language == "en" else "a peca"
        document_label = "document" if language == "en" else "documento"
        cleaned = re.sub(
            r"`?\bartifact[_-]?\d+\b`?",
            artifact_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            rf"\b{document_label}\s+{re.escape(artifact_label)}\b",
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
        language: str,
    ) -> dict[int, str]:
        labels: dict[int, str] = {}
        for index, doc in enumerate(docs, start=1):
            title = str(doc.get("title") or "").strip()
            inventory = str(doc.get("inventory") or "").strip()
            if language == "en":
                if title and inventory:
                    labels[index] = f'the artifact "{title}" ({inventory})'
                elif title:
                    labels[index] = f'the artifact "{title}"'
                elif inventory:
                    labels[index] = f"the artifact with inventory {inventory}"
                else:
                    labels[index] = "the artifact from the collection"
                continue

            if title and inventory:
                labels[index] = f'a peca "{title}" ({inventory})'
            elif title:
                labels[index] = f'a peca "{title}"'
            elif inventory:
                labels[index] = f"a peca de inventario {inventory}"
            else:
                labels[index] = "a peca do acervo"
        return labels

    def _filter_image_matches_by_relevance(
        self,
        *,
        reply_text: str,
        image_matches: list[ImageMatchResult],
        docs: list[dict[str, object]],
        intent: str,
    ) -> list[ImageMatchResult]:
        if not image_matches:
            return image_matches

        reply_compact = self._compact_for_match(reply_text)
        if not reply_compact:
            return image_matches

        relevant_inventories: set[str] = set()
        relevant_titles: set[str] = set()
        for doc in docs:
            inventory = str(doc.get("inventory") or "").strip()
            title = str(doc.get("title") or "").strip()
            inventory_compact = self._compact_for_match(inventory)
            title_compact = self._compact_for_match(title)

            if inventory_compact and inventory_compact in reply_compact:
                relevant_inventories.add(inventory_compact)
            if title_compact and len(title_compact) >= 8 and title_compact in reply_compact:
                relevant_titles.add(title_compact)

        if not relevant_inventories and not relevant_titles:
            if intent in {"image_search", "model_search"}:
                filtered = image_matches[:1]
                if len(filtered) != len(image_matches):
                    self._log(
                        logging.INFO,
                        "chat.image_matches_filtered",
                        intent=intent,
                        before=len(image_matches),
                        after=len(filtered),
                        reason="media_fallback_top1",
                    )
                return filtered
            return image_matches

        filtered: list[ImageMatchResult] = []
        for match in image_matches:
            inventory_compact = self._compact_for_match(match.inventory or "")
            title_compact = self._compact_for_match(match.title or "")
            if (
                (inventory_compact and inventory_compact in relevant_inventories)
                or (title_compact and title_compact in relevant_titles)
            ):
                filtered.append(match)

        if not filtered:
            return image_matches

        if len(filtered) != len(image_matches):
            self._log(
                logging.INFO,
                "chat.image_matches_filtered",
                intent=intent,
                before=len(image_matches),
                after=len(filtered),
                reason="reply_relevance",
            )
        return filtered

    def _compact_for_match(self, value: str) -> str:
        lowered = (value or "").casefold()
        compact = re.sub(r"[\W_]+", "", lowered, flags=re.UNICODE)
        return compact

    def _replace_doc_refs_with_labels(
        self,
        *,
        text: str,
        doc_label_map: dict[int, str],
        language: str,
    ) -> str:
        fallback_label = (
            "the artifact from the collection" if language == "en" else "a peca do acervo"
        )

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
        language: str,
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
            language=language,
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
            inventory = str(doc.get("inventory") or "").strip()
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
        reranker_service=get_reranker_service(),
        tour_navigation_service=get_tour_navigation_service(),
        llm_service=get_llm_service(),
        session_store=get_chat_session_store(),
    )
