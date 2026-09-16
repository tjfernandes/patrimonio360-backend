"""Quick win LLM-04/LLM-03: orçamentos de tokens por tipo de chamada,
finish_reason/usage no LLMResponse e geração em stream.

    python -m unittest tests.test_llm_budgets
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ResponseFormatObject  # noqa: E402
from app.services.llm_service import LLMService, LLMServiceError  # noqa: E402


def _completion(text: str, finish_reason: str = "stop", usage: dict | None = None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(**(usage or {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})),
    )


class _FakeCompletions:
    def __init__(self, response=None, chunks=None) -> None:
        self.response = response
        self.chunks = chunks or []
        self.payloads: list[dict] = []

    async def parse(self, **payload):
        self.payloads.append(payload)
        return self.response

    async def create(self, **payload):
        self.payloads.append(payload)
        if payload.get("stream"):
            chunks = self.chunks

            async def _iter():
                for chunk in chunks:
                    yield chunk
            return _iter()
        return self.response


def _service(completions: _FakeCompletions, **overrides) -> LLMService:
    settings = Settings(_env_file=None, LLM_BASE_URL="http://localhost:1/v1", **overrides)
    service = LLMService(settings)
    service.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return service


class TestTokenBudgets(unittest.TestCase):
    def test_default_sends_no_max_tokens(self) -> None:
        fake = _FakeCompletions(_completion("ok"))
        service = _service(fake)
        response = asyncio.run(service.generate(message="olá", response_format=ResponseFormatObject(type="text")))
        self.assertNotIn("max_tokens", fake.payloads[0])
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage["completion_tokens"], 3)

    def test_json_and_text_budgets_are_independent(self) -> None:
        fake = _FakeCompletions(_completion('{"a": 1}'))
        service = _service(fake, LLM_MAX_TOKENS_JSON=200, LLM_MAX_TOKENS_TEXT=450)
        asyncio.run(service.generate(message="x", response_format=ResponseFormatObject(type="json_object")))
        asyncio.run(service.generate(message="x", response_format=ResponseFormatObject(type="text")))
        self.assertEqual(fake.payloads[0]["max_tokens"], 200)
        self.assertEqual(fake.payloads[1]["max_tokens"], 450)

    def test_global_budget_is_fallback(self) -> None:
        fake = _FakeCompletions(_completion("ok"))
        service = _service(fake, LLM_MAX_TOKENS=300)
        asyncio.run(service.generate(message="x", response_format=ResponseFormatObject(type="text")))
        self.assertEqual(fake.payloads[0]["max_tokens"], 300)

    def test_length_finish_reason_is_surfaced(self) -> None:
        fake = _FakeCompletions(_completion("truncado…", finish_reason="length"))
        service = _service(fake, LLM_MAX_TOKENS_TEXT=5)
        response = asyncio.run(service.generate(message="x", response_format=ResponseFormatObject(type="text")))
        self.assertEqual(response.finish_reason, "length")


class TestGenerateStream(unittest.TestCase):
    def _chunks(self):
        def chunk(content, finish=None, usage=None):
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)],
                usage=SimpleNamespace(**usage) if usage else None,
            )
        return [
            chunk("Olá"),
            chunk(", encontrei"),
            chunk(" 3 peças.", finish="stop"),
            SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=40, completion_tokens=7, total_tokens=47)),
        ]

    def test_stream_delivers_deltas_and_full_text(self) -> None:
        fake = _FakeCompletions(chunks=self._chunks())
        service = _service(fake, LLM_MAX_TOKENS_TEXT=100)
        deltas: list[str] = []

        async def on_delta(delta: str) -> None:
            deltas.append(delta)

        response = asyncio.run(service.generate_stream(
            message="x", response_format=ResponseFormatObject(type="text"), on_delta=on_delta))
        self.assertEqual(deltas, ["Olá", ", encontrei", " 3 peças."])
        self.assertEqual(response.text, "Olá, encontrei 3 peças.")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage["completion_tokens"], 7)
        self.assertTrue(fake.payloads[0]["stream"])
        self.assertEqual(fake.payloads[0]["max_tokens"], 100)

    def test_empty_stream_raises(self) -> None:
        fake = _FakeCompletions(chunks=[])
        service = _service(fake)
        with self.assertRaises(LLMServiceError):
            asyncio.run(service.generate_stream(message="x", response_format=ResponseFormatObject(type="text")))


if __name__ == "__main__":
    unittest.main()


class _RefusingCompletions:
    """Primário em baixo: recusa a ligação como o httpx faria."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **_payload):
        self.calls += 1
        import httpx
        from openai import APIConnectionError
        raise APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8004/v1/chat/completions"))


class TestFallback(unittest.TestCase):
    def _service(self, primary, fallback, **overrides) -> LLMService:
        settings = Settings(_env_file=None, LLM_BASE_URL="http://127.0.0.1:8004/v1",
                            LLM_FALLBACK_BASE_URL="https://remote.example.test/v1/chat/completions",
                            LLM_FALLBACK_MODEL="remote/model", LLM_MODEL="local/model", **overrides)
        service = LLMService(settings)
        service.client = SimpleNamespace(chat=SimpleNamespace(completions=primary))
        service.fallback_client = SimpleNamespace(chat=SimpleNamespace(completions=fallback))
        return service

    def test_settings_resolve_fallback_base_url(self) -> None:
        settings = Settings(_env_file=None, LLM_FALLBACK_BASE_URL="https://x.test/amalia-llm/v1/chat/completions")
        self.assertEqual(settings.llm_fallback_openai_base_url_resolved, "https://x.test/amalia-llm/v1")
        self.assertIsNone(Settings(_env_file=None).llm_fallback_openai_base_url_resolved)
        self.assertIsNone(LLMService(Settings(_env_file=None, LLM_BASE_URL="http://localhost:1/v1")).fallback_client)

    def test_connection_error_goes_to_fallback_with_fallback_model(self) -> None:
        primary, fallback = _RefusingCompletions(), _FakeCompletions(_completion("resposta remota"))
        service = self._service(primary, fallback)
        response = asyncio.run(service.generate(message="olá", response_format=ResponseFormatObject(type="text")))
        self.assertEqual(primary.calls, 1)
        self.assertEqual(response.text, "resposta remota")
        self.assertEqual(response.endpoint, "fallback")
        self.assertEqual(fallback.payloads[0]["model"], "remote/model")

    def test_primary_success_does_not_touch_fallback(self) -> None:
        primary, fallback = _FakeCompletions(_completion("local")), _FakeCompletions(_completion("remoto"))
        service = self._service(primary, fallback)
        response = asyncio.run(service.generate(message="olá", response_format=ResponseFormatObject(type="text")))
        self.assertEqual((response.text, response.endpoint), ("local", "primary"))
        self.assertEqual(fallback.payloads, [])

    def test_without_fallback_connection_error_raises(self) -> None:
        settings = Settings(_env_file=None, LLM_BASE_URL="http://127.0.0.1:8004/v1")
        service = LLMService(settings)
        service.client = SimpleNamespace(chat=SimpleNamespace(completions=_RefusingCompletions()))
        with self.assertRaises(LLMServiceError):
            asyncio.run(service.generate(message="olá", response_format=ResponseFormatObject(type="text")))

    def test_stream_falls_back_before_any_delta(self) -> None:
        def chunk(content, finish=None):
            return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)], usage=None)
        primary = _RefusingCompletions()
        fallback = _FakeCompletions(chunks=[chunk("Olá"), chunk(" remoto", finish="stop")])
        service = self._service(primary, fallback)
        deltas: list[str] = []

        async def on_delta(delta: str) -> None:
            deltas.append(delta)

        response = asyncio.run(service.generate_stream(message="x", response_format=ResponseFormatObject(type="text"), on_delta=on_delta))
        self.assertEqual(deltas, ["Olá", " remoto"])
        self.assertEqual(response.endpoint, "fallback")
        self.assertEqual(fallback.payloads[0]["model"], "remote/model")
