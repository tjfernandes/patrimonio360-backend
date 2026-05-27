from dataclasses import dataclass, field
from functools import lru_cache
from time import time
from typing import Any, Literal

from app.core.config import Settings, get_settings

ChatRole = Literal["user", "assistant"]


@dataclass
class ChatTurn:
    role: ChatRole
    text: str


@dataclass
class ChatSessionState:
    conversation_id: str
    museum_slug: str
    language: str = "pt"
    intent: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    sort: dict[str, Any] = field(default_factory=dict)
    selected_artifact_id: str | None = None
    last_result_ids: list[str] = field(default_factory=list)
    last_paged_artifact_results: list[dict[str, Any]] = field(default_factory=list)
    last_paged_image_matches: list[dict[str, Any]] = field(default_factory=list)
    last_paged_navigation_targets: list[dict[str, Any]] = field(default_factory=list)
    last_paged_results_default_page_size: int = 0
    last_paged_retrieval_request: dict[str, Any] = field(default_factory=dict)
    rolling_summary: str = ""
    history: list[ChatTurn] = field(default_factory=list)
    updated_at: float = field(default_factory=time)


class ChatSessionStore:
    """In-memory chat session store with TTL eviction for dev use."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sessions: dict[str, ChatSessionState] = {}

    def load_or_create(self, *, conversation_id: str, museum_slug: str) -> ChatSessionState:
        self._evict_expired()

        existing = self._sessions.get(conversation_id)
        if existing is None:
            state = ChatSessionState(conversation_id=conversation_id, museum_slug=museum_slug)
            self._sessions[conversation_id] = state
            return state

        # If museum changes on same conversation id, keep conversation id but reset museum-bound state.
        if existing.museum_slug != museum_slug:
            existing.museum_slug = museum_slug
            existing.intent = None
            existing.filters = {}
            existing.sort = {}
            existing.selected_artifact_id = None
            existing.last_result_ids = []
            existing.last_paged_artifact_results = []
            existing.last_paged_image_matches = []
            existing.last_paged_navigation_targets = []
            existing.last_paged_results_default_page_size = 0
            existing.last_paged_retrieval_request = {}
            existing.rolling_summary = ""
            existing.history = []

        existing.updated_at = time()
        return existing

    def save(self, state: ChatSessionState) -> None:
        state.updated_at = time()
        self._sessions[state.conversation_id] = state

    def append_turn(self, state: ChatSessionState, *, role: ChatRole, text: str) -> None:
        state.history.append(ChatTurn(role=role, text=text))
        window = max(self.settings.CHAT_HISTORY_WINDOW, 1)
        if len(state.history) > window:
            state.history = state.history[-window:]
        state.updated_at = time()

    def get(self, conversation_id: str) -> ChatSessionState | None:
        self._evict_expired()
        state = self._sessions.get(conversation_id)
        if state is not None:
            state.updated_at = time()
        return state

    def _evict_expired(self) -> None:
        ttl = max(self.settings.CHAT_SESSION_TTL_SECONDS, 1)
        now = time()
        expired_keys = [
            conversation_id
            for conversation_id, state in self._sessions.items()
            if now - state.updated_at > ttl
        ]
        for conversation_id in expired_keys:
            self._sessions.pop(conversation_id, None)


@lru_cache(maxsize=1)
def get_chat_session_store() -> ChatSessionStore:
    return ChatSessionStore(get_settings())
