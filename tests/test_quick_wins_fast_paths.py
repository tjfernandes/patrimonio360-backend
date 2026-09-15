"""Quick wins pós-submissão: fast-path de saudação, template de paginação e
grounding em modo observação. Tudo atrás de flags; com os flags desligados o
comportamento tem de ser o histórico.

    python -m unittest tests.test_quick_wins_fast_paths
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ChatMessageRequest, ChatResultsPageRequest, ResponseFormatObject  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.chat_session_store import ChatSessionStore  # noqa: E402
from app.services.llm_service import LLMServiceError  # noqa: E402


class _RouterCalled(RuntimeError):
    """Sentinela: o teste chegou ao LLM (router) quando não devia."""


class _FailingLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **_kwargs):
        self.calls += 1
        raise _RouterCalled("LLM chamado")


class _CapturingLogger:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def log_event(self, event: dict) -> None:
        self.events.append(event)


def _settings(**overrides) -> Settings:
    base = dict(QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False)
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _service(settings: Settings, llm=None, logger=None) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = settings
    service.session_store = ChatSessionStore(settings)
    service.llm_service = llm or _FailingLLM()
    service.query_logger = logger or _CapturingLogger()
    service.opensearch_gateway = None
    service.embedding_provider = None
    service.model_retrieval_service = None
    service.tour_navigation_service = None
    service._reranker_service = None
    service._vl_reranker_service = None
    return service


def _request(message: str, **kwargs) -> ChatMessageRequest:
    payload = dict(museum_slug="mnt", museum_id="mnt", museum_name="Museu Nacional do Traje",
                   language="pt", message=message)
    payload.update(kwargs)
    return ChatMessageRequest(**payload)


class TestSmalltalkFastPath(unittest.TestCase):
    def test_flag_off_keeps_router_call(self) -> None:
        service = _service(_settings(CHAT_DETERMINISTIC_SMALLTALK=False))
        with self.assertRaises(_RouterCalled):
            asyncio.run(service.handle_message(_request("Olá")))

    def test_pure_greeting_skips_llm_and_logs_event(self) -> None:
        llm, logger = _FailingLLM(), _CapturingLogger()
        service = _service(_settings(CHAT_DETERMINISTIC_SMALLTALK=True), llm=llm, logger=logger)
        statuses: list[str] = []

        async def status_cb(event):
            statuses.append(str(event.get("message")))

        for text in ["Olá", "olá!", "Bom dia", "hello", "Tudo bem?"]:
            response = asyncio.run(service.handle_message(_request(text), status_cb=status_cb))
            self.assertEqual(llm.calls, 0, text)
            self.assertTrue(response.reply, text)
            self.assertEqual(response.model_hint, "deterministic:smalltalk")
            self.assertEqual(response.artifact_results, [])
        event = logger.events[-1]
        self.assertEqual(event["route"], "llm_only")
        self.assertEqual(event["retrieval_mode"], "none")
        self.assertEqual(event["fast_path"], "smalltalk")
        self.assertEqual(event["latency_llm_ms"], 0.0)
        self.assertEqual(event["status"], "ok")
        self.assertTrue(statuses)

    def test_greeting_reply_follows_language(self) -> None:
        service = _service(_settings(CHAT_DETERMINISTIC_SMALLTALK=True))
        pt = asyncio.run(service.handle_message(_request("Olá", language="pt")))
        en = asyncio.run(service.handle_message(_request("Hello", language="en")))
        self.assertIn("Olá", pt.reply)
        self.assertIn("Hello", en.reply)

    def test_fast_path_updates_session_history(self) -> None:
        service = _service(_settings(CHAT_DETERMINISTIC_SMALLTALK=True))
        response = asyncio.run(service.handle_message(_request("Olá", conversation_id="c1")))
        state = service.session_store.get("c1")
        self.assertIsNotNone(state)
        roles = [turn.role for turn in state.history]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(state.history[-1].text, response.reply)
        self.assertIn("intent=fallback", state.rolling_summary)

    def test_composed_greeting_still_goes_to_router(self) -> None:
        service = _service(_settings(CHAT_DETERMINISTIC_SMALLTALK=True))
        for text in ["Olá, encontra vestidos", "bom dia, há azulejos azuis?", "olá quero ver esculturas"]:
            with self.assertRaises(_RouterCalled, msg=text):
                asyncio.run(service.handle_message(_request(text)))

    def test_json_format_request_is_not_short_circuited(self) -> None:
        service = _service(_settings(CHAT_DETERMINISTIC_SMALLTALK=True))
        with self.assertRaises(_RouterCalled):
            asyncio.run(service.handle_message(
                _request("Olá", response_format=ResponseFormatObject(type="json_object"))
            ))


class TestResultsPageTemplate(unittest.TestCase):
    def _docs(self, n: int) -> list[dict]:
        return [{"artifact_id": f"a{i}", "title": f"Peça {i}", "inventory_number": f"MNT {i}"} for i in range(n)]

    def test_flag_off_calls_llm(self) -> None:
        llm = _FailingLLM()
        service = _service(_settings(CHAT_TEMPLATE_RESULTS_PAGE_REPLY=False), llm=llm)
        payload = ChatResultsPageRequest(conversation_id="c", museum_slug="mnt", language="pt", results_page=2)
        reply = asyncio.run(service._generate_results_page_reply(
            payload=payload, request={"museum_slug": "mnt"}, docs=self._docs(3),
            artifact_results=[], page=2, total=11,
        ))
        self.assertEqual(llm.calls, 1)
        self.assertTrue(reply)  # fallback traduzido quando o LLM falha

    def test_flag_on_returns_template_without_llm(self) -> None:
        llm = _FailingLLM()
        service = _service(_settings(CHAT_TEMPLATE_RESULTS_PAGE_REPLY=True), llm=llm)
        payload = ChatResultsPageRequest(conversation_id="c", museum_slug="mnt", language="pt", results_page=2)
        reply = asyncio.run(service._generate_results_page_reply(
            payload=payload, request={"museum_slug": "mnt"}, docs=self._docs(3),
            artifact_results=[], page=2, total=11,
        ))
        self.assertEqual(llm.calls, 0)
        self.assertIn("Página 2", reply)
        self.assertIn("3 peças", reply)
        self.assertIn("11", reply)
        one = asyncio.run(service._generate_results_page_reply(
            payload=ChatResultsPageRequest(conversation_id="c", museum_slug="mnt", language="en", results_page=3),
            request={"museum_slug": "mnt"}, docs=self._docs(1), artifact_results=[], page=3, total=9,
        ))
        self.assertIn("Page 3", one)
        self.assertIn("1 more object", one)


class TestGroundingReport(unittest.TestCase):
    def _cards(self) -> list[dict]:
        return [
            {"title": "Vestido de noiva", "inventory_number": "MNT 1234"},
            {"title": "Capa de asperges", "inventory_number": "793 Mob MNSR"},
        ]

    def test_off_returns_none(self) -> None:
        service = _service(_settings(CHAT_GROUNDING_CHECK="off"))
        self.assertIsNone(service._grounding_report(
            reply='Veja «Peça inventada»', docs=self._cards(), conversation_id="c", query_id="q"))

    def test_matching_citations_have_no_violations(self) -> None:
        service = _service(_settings(CHAT_GROUNDING_CHECK="log"))
        report = service._grounding_report(
            reply='Encontrei o «Vestido de noiva» (MNT 1234) e a **Capa de asperges**.',
            docs=self._cards(), conversation_id="c", query_id="q")
        self.assertEqual(report["violations"], 0)
        self.assertEqual(report["quoted_titles"], 2)
        self.assertEqual(report["cards"], 2)

    def test_invented_title_and_inventory_are_reported(self) -> None:
        service = _service(_settings(CHAT_GROUNDING_CHECK="log"))
        report = service._grounding_report(
            reply='Recomendo «A Dança do Tempo» e a peça MNT 9999.',
            docs=self._cards(), conversation_id="c", query_id="q")
        self.assertEqual(report["unmatched_titles"], ["A Dança do Tempo"])
        self.assertEqual(report["unmatched_inventories"], ["mnt 9999"])
        self.assertEqual(report["violations"], 2)

    def test_no_cards_flags_every_citation(self) -> None:
        service = _service(_settings(CHAT_GROUNDING_CHECK="log"))
        report = service._grounding_report(
            reply='Temos «Reflexos Urbanos» e «Memórias da Terra».',
            docs=[], conversation_id="c", query_id="q")
        self.assertEqual(report["cards"], 0)
        self.assertEqual(report["violations"], 2)

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(Exception):
            _settings(CHAT_GROUNDING_CHECK="enforce")


if __name__ == "__main__":
    unittest.main()
