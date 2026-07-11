from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.services.chat_service import ChatService
from app.services.chat_session_store import ChatSessionStore
from app.services.embeddings import EmbeddingProvider
from app.services.llm_service import LLMService, LLMServiceError
from app.services.model_retrieval import ModelRetrievalService
from app.services.multiview_renderer import PersistentMultiviewRenderer
from app.services.opensearch_client import OpenSearchGateway
from app.services.reranker import PairwiseRerankerService
from app.services.tour_navigation import TourNavigationService
from app.services.warmup import warmup_chat_stack
from benchmarks.variants import VariantSpec



class BenchmarkLLMService:
    def __init__(self, inner: LLMService | None) -> None:
        self.inner = inner
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.inner is not None

    def reset_tracking(self) -> None:
        self.last_error = None

    async def generate(self, **kwargs: Any) -> Any:
        self.last_error = None
        if self.inner is None:
            self.last_error = "assistant_selection_disabled"
            raise LLMServiceError(self.last_error)
        try:
            return await self.inner.generate(**kwargs)
        except LLMServiceError as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            self.last_error = str(exc)
            raise LLMServiceError(str(exc)) from exc


@dataclass(slots=True)
class BenchmarkServiceBundle:
    settings: Settings
    opensearch_gateway: OpenSearchGateway
    embedding_provider: EmbeddingProvider
    renderer: PersistentMultiviewRenderer
    model_retrieval_service: ModelRetrievalService
    reranker_service: PairwiseRerankerService
    session_store: ChatSessionStore
    llm_service: BenchmarkLLMService
    chat_service: ChatService

    def close(self) -> None:
        self.renderer.close()

    async def warmup(
        self,
        *,
        include_multimodal: bool,
        include_multiview_worker: bool,
    ) -> None:
        await warmup_chat_stack(
            service=self.chat_service,
            include_multimodal=include_multimodal,
            include_multiview_worker=include_multiview_worker,
        )


def _clone_settings() -> Settings:
    settings = get_settings()
    if hasattr(settings, "model_dump"):
        return Settings(**settings.model_dump())
    return Settings()


def build_service_bundle(
    variant: VariantSpec,
    *,
    enable_assistant_selection: bool = True,
) -> BenchmarkServiceBundle:
    settings = _clone_settings()

    opensearch_gateway = OpenSearchGateway(settings)
    embedding_provider = EmbeddingProvider(settings)
    renderer = PersistentMultiviewRenderer(settings)
    model_retrieval_service = ModelRetrievalService(
        settings=settings,
        renderer=renderer,
        embedding_provider=embedding_provider,
        opensearch_gateway=opensearch_gateway,
    )
    reranker_service = PairwiseRerankerService(settings)
    session_store = ChatSessionStore(settings)
    llm_service = BenchmarkLLMService(
        LLMService(settings) if enable_assistant_selection else None
    )
    chat_service = ChatService(
        settings=settings,
        opensearch_gateway=opensearch_gateway,
        embedding_provider=embedding_provider,
        model_retrieval_service=model_retrieval_service,
        tour_navigation_service=TourNavigationService(settings),
        llm_service=llm_service,
        session_store=session_store,
    )

    return BenchmarkServiceBundle(
        settings=settings,
        opensearch_gateway=opensearch_gateway,
        embedding_provider=embedding_provider,
        renderer=renderer,
        model_retrieval_service=model_retrieval_service,
        reranker_service=reranker_service,
        session_store=session_store,
        llm_service=llm_service,
        chat_service=chat_service,
    )
