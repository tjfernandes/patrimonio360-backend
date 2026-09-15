"""Quick win LLM-03 (CHAT_STREAM_FINAL_ANSWER): tokens da resposta final
emitidos pelo canal de estado, `result` continua autoritativo, e a ordem
dos eventos SSE é token…, result, done.

    python -m unittest tests.test_stream_final_answer
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.routes.chat import _build_streaming_response  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ChatMessageResponse, ResponseFormatObject  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.llm_service import LLMResponse, LLMServiceError  # noqa: E402


class _StreamingLLM:
    def __init__(self, pieces: list[str], fail_first_with: str | None = None) -> None:
        self.pieces = pieces
        self.fail_first_with = fail_first_with
        self.stream_calls = 0
        self.plain_calls = 0

    async def generate_stream(self, *, message, response_format, system_prompt=None, model_override=None, on_delta=None):
        self.stream_calls += 1
        if self.fail_first_with and self.stream_calls == 1:
            raise LLMServiceError(self.fail_first_with)
        for piece in self.pieces:
            await on_delta(piece)
        return LLMResponse(text="".join(self.pieces), model="fake", response_format=response_format,
                           finish_reason="stop")

    async def generate(self, **_kwargs):
        self.plain_calls += 1
        return LLMResponse(text="plain", model="fake", response_format=ResponseFormatObject(type="text"))


def _service(llm) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = Settings(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False)
    service.llm_service = llm
    return service


class TestTokenEmitter(unittest.TestCase):
    def test_deltas_are_forwarded_with_sequence_and_query_id(self) -> None:
        events: list[dict] = []

        async def status_cb(event):
            events.append(event)

        llm = _StreamingLLM(["Olá", ", ", "mundo"])
        service = _service(llm)
        response = asyncio.run(service._generate_final_answer_with_context_retries(
            initial_message="prompt", response_format=ResponseFormatObject(type="text"), system_prompt=None,
            model_override=None, visible_result_count=3, build_message_for_result_count=lambda n: f"prompt {n}",
            stream_cb_factory=service._make_token_emitter(status_cb, query_id="q-1"),
        ))
        self.assertEqual(response.text, "Olá, mundo")
        self.assertEqual([e["type"] for e in events], ["token"] * 3)
        self.assertEqual([e["seq"] for e in events], [0, 1, 2])
        self.assertEqual([e["delta"] for e in events], ["Olá", ", ", "mundo"])
        self.assertTrue(all(e["query_id"] == "q-1" for e in events))
        self.assertEqual(llm.plain_calls, 0)

    def test_retry_restarts_sequence(self) -> None:
        events: list[dict] = []

        async def status_cb(event):
            events.append(event)

        llm = _StreamingLLM(["ok"], fail_first_with="prompt is too long (context length)")
        service = _service(llm)
        response = asyncio.run(service._generate_final_answer_with_context_retries(
            initial_message="prompt", response_format=ResponseFormatObject(type="text"), system_prompt=None,
            model_override=None, visible_result_count=2, build_message_for_result_count=lambda n: f"prompt {n}",
            stream_cb_factory=service._make_token_emitter(status_cb, query_id="q-2"),
        ))
        self.assertEqual(response.text, "ok")
        self.assertEqual(llm.stream_calls, 2)
        self.assertEqual([e["seq"] for e in events], [0])

    def test_without_factory_uses_plain_generate(self) -> None:
        llm = _StreamingLLM(["nunca"])
        service = _service(llm)
        response = asyncio.run(service._generate_final_answer_with_context_retries(
            initial_message="prompt", response_format=ResponseFormatObject(type="text"), system_prompt=None,
            model_override=None, visible_result_count=0, build_message_for_result_count=lambda n: "",
        ))
        self.assertEqual(response.text, "plain")
        self.assertEqual(llm.stream_calls, 0)

    def test_channel_failure_does_not_break_generation(self) -> None:
        async def broken_cb(_event):
            raise RuntimeError("cliente foi-se embora")

        llm = _StreamingLLM(["a", "b"])
        service = _service(llm)
        response = asyncio.run(service._generate_final_answer_with_context_retries(
            initial_message="prompt", response_format=ResponseFormatObject(type="text"), system_prompt=None,
            model_override=None, visible_result_count=0, build_message_for_result_count=lambda n: "",
            stream_cb_factory=service._make_token_emitter(broken_cb, query_id="q-3"),
        ))
        self.assertEqual(response.text, "ab")


class TestSseOrder(unittest.TestCase):
    def test_token_events_precede_result_and_done(self) -> None:
        async def run(status_cb):
            await status_cb({"type": "status", "message": "a procurar"})
            await status_cb({"type": "token", "query_id": "q", "seq": 0, "delta": "Olá"})
            await status_cb({"type": "token", "query_id": "q", "seq": 1, "delta": " mundo"})
            return ChatMessageResponse(conversation_id="c", query_id="q", reply="Olá mundo")

        async def collect():
            response = _build_streaming_response(run)
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
            return "".join(chunks)

        raw = asyncio.run(collect())
        blocks = [block for block in raw.split("\n\n") if block.strip()]
        types = [block.split("\n", 1)[0].removeprefix("event: ") for block in blocks]
        self.assertEqual(types, ["status", "token", "token", "result", "done"])
        result_payload = json.loads(blocks[3].split("\ndata: ", 1)[1])
        self.assertEqual(result_payload["payload"]["reply"], "Olá mundo")
        token_payload = json.loads(blocks[1].split("\ndata: ", 1)[1])
        self.assertEqual(token_payload["delta"], "Olá")


if __name__ == "__main__":
    unittest.main()
