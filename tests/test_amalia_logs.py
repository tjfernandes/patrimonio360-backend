"""O visualizador dos passos do RAG (tools/registos/amalia_logs.py) lê o que o
backend escreve: as linhas são produzidas aqui pelas funções de registo do
próprio backend, com o formato do app/core/logging.py, e lidas de volta.

    python -m unittest tests.test_amalia_logs
"""
from __future__ import annotations

import asyncio
import io
import logging
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "registos"))

import amalia_logs as V  # noqa: E402
from app.api.routes import chat as chat_routes  # noqa: E402
from app.core import logging as backend_logging  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.logging import log_event, log_json_event, reset_request_id, set_request_id  # noqa: E402
from app.schemas.chat import ArtifactResult, ChatMessageResponse, ResponseFormatObject  # noqa: E402
from app.services.llm_service import LLMService, LLMServiceError  # noqa: E402

OLD_LOG = """\
2026-11-22 17:07:42 INFO     [app.main] [rid=aaa111] http.request.start method=POST path=/api/v1/chat/messages/stream client=10.0.0.1
INFO:     10.0.0.1:0 - "POST /api/v1/chat/messages/stream HTTP/1.1" 200 OK
2026-11-22 17:07:42 INFO     [app.services.chat_service] [rid=aaa111] chat.message.start conversation_id=c1 query_id=q1 museum_slug=mnaa language=pt response_format=text message_chars=24
2026-11-22 17:07:42 INFO     [app.services.llm_service] [rid=aaa111] llm.generate.start provider=openai_compatible model=Qwen/Q response_format=json_object prompt_chars=772 max_tokens=200
Generating embeddings:   0%|          | 0/1 [00:00<?, ?batch/s]2026-11-22 17:07:42 INFO     [app.main] [rid=bbb222] http.request.start method=GET path=/api/v1/chat/images/x.jpg client=10.0.0.1
2026-11-22 17:07:42 INFO     [app.services.llm_service] [rid=aaa111] llm.generate.finish model=Qwen/Q response_format=json_object response_chars=75 finish_reason=stop prompt_tokens=262 completion_tokens=27 endpoint=primary duration_ms=373.9
2026-11-22 17:07:42 INFO     [app.services.chat_service] [rid=aaa111] chat.request_reading conversation_id=c1 query_id=q1 follow_up=false request='tentações de santo antão' used_request=false
2026-11-22 17:07:42 INFO     [app.services.chat_service] [rid=aaa111] rag.router_decision
{
  "input": {
    "message": "tentações de santo antão"
  },
  "router_decision": {
    "intent": "search",
    "mode": "rag"
  }
}
2026-11-22 17:07:43 INFO     [app.services.opensearch_client] [rid=aaa111] opensearch.search_relevant_context_page.start museum_slug=mnaa page_size=15
2026-11-22 17:07:43 INFO     [app.services.opensearch_client] [rid=aaa111] opensearch.search_relevant_context_page.finish duration_ms=338.1 museum_slug=mnaa page_size=15 result_count=15 total=1000
2026-11-22 17:07:44 INFO     [app.services.llm_service] [rid=aaa111] llm.generate.start provider=openai_compatible model=carminho/AMALIA response_format=text prompt_chars=843 stream=true
2026-11-22 17:07:45 ERROR    [app.services.llm_service] [rid=aaa111] llm.generate.error model=carminho/AMALIA duration_ms=14.6 stream=true error='Error code: 400 - contexto a mais'
2026-11-22 17:07:45 INFO     [app.services.chat_service] [rid=aaa111] chat.message.finish conversation_id=c1 query_id=q1 status=ok shown_artifacts=2 duration_ms=3374.2
"""


def _load(text: str):
    requests, system = {}, []
    for event in V.read_events(text.splitlines(keepends=True)):
        V.file_event(event, requests, system)
    return requests, system


def _render(request, *, full: bool = False) -> str:
    return "\n".join(V.Renderer(V.Style(False), full=full, width=120).request(request))


class _Capture:
    """As linhas tal como o backend as escreve, com um rid por pedido."""

    def __init__(self, rid: str = "rid123") -> None:
        self.stream = io.StringIO()
        self.rid = rid

    def __enter__(self) -> "_Capture":
        self.handler = logging.StreamHandler(self.stream)
        self.handler.addFilter(backend_logging._RequestIdFilter())
        self.handler.setFormatter(
            logging.Formatter(fmt=backend_logging.BACKEND_LOG_FORMAT, datefmt=backend_logging.BACKEND_LOG_DATEFMT)
        )
        self.logger = logging.getLogger("app")
        self.previous = (self.logger.level, self.logger.propagate, list(self.logger.handlers))
        self.logger.handlers = [self.handler]
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.token = set_request_id(self.rid)
        return self

    def __exit__(self, *_exc) -> None:
        reset_request_id(self.token)
        self.logger.level, self.logger.propagate, self.logger.handlers = self.previous

    @property
    def text(self) -> str:
        return self.stream.getvalue()


class TestParser(unittest.TestCase):
    def test_groups_by_rid_and_skips_noise(self) -> None:
        requests, _ = _load(OLD_LOG)
        self.assertEqual(sorted(requests), ["aaa111", "bbb222"])  # a linha colada à barra de progresso conta
        request = requests["aaa111"]
        self.assertTrue(request.is_chat)
        self.assertFalse(requests["bbb222"].is_chat)
        self.assertEqual(request.question, "tentações de santo antão")
        self.assertEqual(request.status, "ok")
        self.assertEqual(request.duration_ms, 3374.2)

    def test_quoted_values_and_json_block(self) -> None:
        request = _load(OLD_LOG)[0]["aaa111"]
        reading = request.first("chat.request_reading")
        self.assertEqual(reading.fields["request"], "tentações de santo antão")
        self.assertEqual(request.first("rag.router_decision").payload["router_decision"]["intent"], "search")

    def test_start_and_finish_become_one_line(self) -> None:
        items = V.build_items(_load(OLD_LOG)[0]["aaa111"].events)
        spans = [item for item in items if item.is_span]
        self.assertEqual([item.name for item in spans], ["llm.generate", "opensearch.search_relevant_context_page", "llm.generate"])
        self.assertEqual(spans[0].duration_ms, 373.9)
        self.assertEqual(spans[0].fields["prompt_chars"], "772")  # do início
        self.assertFalse(spans[0].failed)
        self.assertTrue(spans[2].failed)

    def test_render_has_steps_and_error(self) -> None:
        text = _render(_load(OLD_LOG)[0]["aaa111"])
        for expected in ("«tentações de santo antão»", "LEITURA DO PEDIDO", "ENCAMINHAMENTO", "PESQUISA", "RESPOSTA",
                         "262→27 tokens", "15 resultados de 1000", "contexto a mais", "TEMPOS",
                         "BACKEND_RAG_DEBUG_LLM_IO"):
            self.assertIn(expected, text)

    def test_old_log_without_end_is_not_an_error(self) -> None:
        log = "\n".join(line for line in OLD_LOG.splitlines() if "chat.message.finish" not in line and "ERROR" not in line)
        request = _load(log + "\n")[0]["aaa111"]
        self.assertEqual(request.status, "—")
        self.assertFalse(V.matches(request, V.build_parser().parse_args(["-e"])))

    def test_parallel_calls_pair_by_step(self) -> None:
        log = (
            "2026-11-22 17:00:00.100 INFO     [a] [rid=r] llm.generate.start step=rascunho_a model=m prompt_chars=1\n"
            "2026-11-22 17:00:00.101 INFO     [a] [rid=r] llm.generate.start step=rascunho_b model=m prompt_chars=2\n"
            "2026-11-22 17:00:00.300 INFO     [a] [rid=r] llm.generate.finish step=rascunho_b model=m duration_ms=199\n"
            "2026-11-22 17:00:00.400 INFO     [a] [rid=r] llm.generate.finish step=rascunho_a model=m duration_ms=300\n"
        )
        items = V.build_items(_load(log)[0]["r"].events)
        self.assertEqual([(i.fields["step"], i.fields["prompt_chars"]) for i in items], [("rascunho_b", "2"), ("rascunho_a", "1")])
        self.assertEqual(V.assign_phases(items), [V.P_RESPOSTA, V.P_RESPOSTA])


class TestLive(unittest.TestCase):
    """-f: os passos saem à medida que chegam, cada um uma só vez."""

    def test_steps_come_out_once_as_they_arrive(self) -> None:
        renderer = V.Renderer(V.Style(False), full=False, width=120)
        requests, system, seen = {}, [], []
        now = 1000.0
        for event in V.read_events(OLD_LOG.splitlines(keepends=True)):
            request = V.file_event(event, requests, system)
            if request is not None and request.rid == "aaa111" and request.is_chat:  # como no follow()
                request.touched = now
                seen.append("\n".join(renderer.live(request, now, final=False)))
                now += 0.05
        request = requests["aaa111"]
        before_end = "\n".join(seen)
        self.assertIn("pedido novo", seen[0])  # o cabeçalho sai com a primeira linha do pedido
        self.assertIn("pedido lido: «tentações de santo antão»", before_end)
        self.assertIn("262→27 tokens", before_end)
        self.assertNotIn("TEMPOS", before_end)
        closing = "\n".join(renderer.live(request, now + 5, final=True))
        self.assertIn("FIM «tentações de santo antão» · ok · 3,37 s", closing)
        everything = before_end + "\n" + closing
        for once in ("pedido novo", "262→27 tokens", "15 resultados de 1000", "contexto a mais", "fim · estado ok"):
            self.assertEqual(everything.count(once), 1, once)
        self.assertEqual(renderer.live(request, now + 9, final=False), [])

    def test_slow_call_is_announced_before_it_ends(self) -> None:
        renderer = V.Renderer(V.Style(False), full=False, width=120)
        log = "2026-11-22 17:00:00.100 INFO     [a] [rid=r] llm.generate.start step=resposta_final model=org/AMALIA stream=true\n"
        request = _load(log)[0]["r"]
        start = request.events[0].arrived
        self.assertNotIn("em curso", "\n".join(renderer.live(request, start + 0.1, final=False)))
        waiting = "\n".join(renderer.live(request, start + 1.0, final=False))
        self.assertIn("⏳ modelo «resposta final» AMALIA em curso…", waiting)
        self.assertEqual(renderer.live(request, start + 1.5, final=False), [])
        self.assertIn("não há registo do fim", "\n".join(renderer.live(request, start + 200, final=True)))


class TestLenientJson(unittest.TestCase):
    def test_truncated_block_keeps_what_is_whole(self) -> None:
        import json

        payload = {"artifact_results": [{"inventory_number": f"{n} Pint", "title": "t" * 50} for n in range(40)], "z": 1}
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        cut = rendered[:900] + "\n... truncated terminal JSON: 999 chars omitted ..."
        parsed, truncated = V.parse_json_lenient(cut)
        self.assertTrue(truncated)
        self.assertGreaterEqual(len(parsed["artifact_results"]), 5)
        self.assertEqual(parsed["artifact_results"][0]["inventory_number"], "0 Pint")
        self.assertEqual(V.parse_json_lenient(rendered), (payload, False))


class TestBackendWritesWhatTheViewerReads(unittest.TestCase):
    def _llm(self, **overrides) -> LLMService:
        settings = Settings(_env_file=None, LLM_BASE_URL="http://localhost:1/v1", **overrides)
        service = LLMService(settings)
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"follow_up": false}'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=262, completion_tokens=27, total_tokens=289),
        )

        async def create(**_payload):
            return completion

        service.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return service

    def test_llm_call_with_step_prompt_and_response(self) -> None:
        service = self._llm(BACKEND_RAG_DEBUG_LLM_IO=True)
        with _Capture() as capture:
            asyncio.run(service.generate(
                message="Lê o pedido:\n«e as dele?»", response_format=ResponseFormatObject(type="json_object"),
                system_prompt="És um leitor de pedidos.", step="leitura_do_pedido",
            ))
        request = _load(capture.text)[0]["rid123"]
        self.assertTrue(all(event.has_ms for event in request.events))
        (item,) = V.build_items(request.events)
        self.assertEqual(item.fields["step"], "leitura_do_pedido")
        self.assertEqual(item.io.payload["prompt"], "Lê o pedido:\n«e as dele?»")
        self.assertEqual(V.assign_phases([item]), [V.P_LEITURA])
        brief, full = _render(request), _render(request, full=True)
        self.assertIn("modelo «leitura do pedido»", brief)
        self.assertIn('↳ {"follow_up": false}', brief)
        self.assertNotIn("És um leitor de pedidos.", brief)
        for expected in ("És um leitor de pedidos.", "«e as dele?»", '{"follow_up": false}'):
            self.assertIn(expected, full)

    def test_without_the_flag_no_text_is_logged(self) -> None:
        service = self._llm()
        with _Capture() as capture:
            asyncio.run(service.generate(message="segredo", response_format=ResponseFormatObject(type="json_object"), step="router"))
        self.assertNotIn("rag.llm_call", capture.text)
        self.assertNotIn("segredo", capture.text)
        self.assertIn("step=router", capture.text)

    def test_failed_call_keeps_the_prompt(self) -> None:
        service = self._llm(BACKEND_RAG_DEBUG_LLM_IO=True)

        async def create(**_payload):
            raise RuntimeError("servidor em baixo")

        service.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with _Capture() as capture:
            with self.assertRaises(LLMServiceError):
                asyncio.run(service.generate(message="o prompt", response_format=ResponseFormatObject(type="text"), step="resposta_final"))
        request = _load(capture.text)[0]["rid123"]
        (item,) = V.build_items(request.events)
        self.assertTrue(item.failed)
        self.assertEqual(item.io.payload["prompt"], "o prompt")
        self.assertIn("servidor em baixo", _render(request))

    def test_long_text_is_cut_but_the_block_stays_json(self) -> None:
        with _Capture() as capture:
            log_json_event(logging.getLogger("app.x"), logging.INFO, "rag.llm_call",
                           {"model": "m", "prompt": "p" * 5000, "response": "r"}, max_chars=0, max_string_chars=1000)
        event = _load(capture.text)[0]["rid123"].events[0]
        self.assertFalse(event.cut)
        self.assertIn("5000 caracteres ao todo", V.text_of(event.payload["prompt"]))

    def test_chat_response_closes_any_request(self) -> None:
        response = ChatMessageResponse(
            conversation_id="c1", query_id="q1", reply="O tríptico é de **Bosch**.", results_total=7,
            artifact_results=[ArtifactResult(artifact_id="raiz:movel:247559", inventory_number="1498 Pint",
                                             title="Tentações de Santo Antão", creator="Jheronymus Bosch",
                                             date_or_period="1505–1506")],
        )
        original = chat_routes.get_settings
        chat_routes.get_settings = lambda: Settings(_env_file=None, BACKEND_RAG_DEBUG_LLM_IO=True)
        try:
            with _Capture() as capture:
                log_event(logging.getLogger("app.x"), logging.INFO, "chat.message.route", mode="rag", intent="plan_route")
                returned = chat_routes._log_chat_response(response, started_at=0.0, message="quem pintou as tentações?")
        finally:
            chat_routes.get_settings = original
        self.assertIs(returned, response)
        request = _load(capture.text)[0]["rid123"]
        self.assertTrue(request.ended)
        self.assertEqual(request.status, "ok")
        self.assertEqual(request.question, "quem pintou as tentações?")
        self.assertEqual(request.answer, "O tríptico é de **Bosch**.")
        text = _render(request)
        for expected in ("resposta enviada ao visitante", "O tríptico é de **Bosch**.", "1498 Pint", "Jheronymus Bosch"):
            self.assertIn(expected, text)

    def test_new_log_without_end_is_flagged(self) -> None:
        with _Capture() as capture:
            log_event(logging.getLogger("app.x"), logging.INFO, "chat.message.start", museum_slug="mnaa")
        request = _load(capture.text)[0]["rid123"]
        self.assertEqual(request.status, "sem fim")
        self.assertTrue(V.matches(request, V.build_parser().parse_args(["-e"])))


if __name__ == "__main__":
    unittest.main()
