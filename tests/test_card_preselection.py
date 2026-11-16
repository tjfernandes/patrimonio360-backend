"""Pré-processamento noutro modelo (CHAT_PREPROCESS_LLM_ENABLED) e escolha dos
cartões antes da resposta (CHAT_PRESELECT_CARDS): o modelo de pré-processamento
diz que cartões correspondem ao pedido, a AMÁLIA só vê esses e deixa de
escrever o bloco «[[cartões: …]]».

    python -m unittest tests.test_card_preselection
"""
from __future__ import annotations

import asyncio
import json

from app.core.config import Settings
from app.schemas.chat import ResponseFormatObject
from app.services.chat_service import ChatService
from app.services.llm_service import LLMResponse, LLMService, preprocess_llm_settings
from tests.test_model_selects_cards import MNAZ_DOCS, TestEndToEnd, _FinalLLM, _SearchGateway
from tests.test_tour_location import _Base

DOCS = [dict(d) for d in MNAZ_DOCS]
DOCS[4]["incorporation"] = "Convento da Madre de Deus, Lisboa"
DOCS[5]["incorporation"] = "Palácio da Mitra, Santo Antão do Tojal"


class _PreLLM:
    """O modelo de pré-processamento: devolve o JSON dado (ou lança o erro)."""

    def __init__(self, reply: object) -> None:
        self.reply, self.prompts = reply, []

    async def generate(self, *, message, response_format=ResponseFormatObject(type="json_object"), **_kwargs):
        self.prompts.append(message)
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(text=json.dumps(self.reply), model="pre", response_format=response_format,
                           parsed_json=self.reply, finish_reason="stop")


class TestPreselectCards(_Base):
    POI_BY_MUSEUM = TestEndToEnd.POI_BY_MUSEUM

    def _search_service(self, llm, pre, *, preselect: bool = True) -> ChatService:
        settings = self._settings(
            CHAT_MODEL_SELECTS_CARDS=True, CHAT_PRESELECT_CARDS=preselect, CHAT_COMPACT_ANSWER_CONTEXT=True,
            CHAT_ROUTER_MODE="deterministic", CHAT_STREAM_FINAL_ANSWER=False, CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8,
        )
        service = self._service(settings, llm=llm)
        service.preprocess_llm_service = pre
        service.opensearch_gateway = _SearchGateway()

        async def _retrieve_context(**_kwargs):
            return "ctx", 20, [dict(d) for d in DOCS], {"kind": "text", "results_total": 20, "filters": {}, "sort": {}}

        service._retrieve_context = _retrieve_context
        return service

    def _ask(self, service, message):
        request = self._request(message, museum_slug="mnaz", museum_id="mnaz", museum_name="Museu Nacional do Azulejo")
        return asyncio.run(service.handle_message(request))

    def test_preprocess_model_chooses_and_final_model_only_writes(self) -> None:
        llm, pre = _FinalLLM("É a Fuga para o Egipto que veio da Madre de Deus."), _PreLLM({"cartoes": ["doc_5"]})
        service = self._search_service(llm, pre)
        response = self._ask(service, "mostra-me a fuga para o egipto da madre de deus")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d5"])
        self.assertEqual([m.artifact_id for m in response.image_matches], ["d5"])
        self.assertEqual([t.overlay_id for t in response.navigation_targets], ["ov-1690"])
        # o pré-processamento viu os oito cartões, com a proveniência
        self.assertIn("[doc_8]", pre.prompts[-1])
        self.assertIn("Proveniência: Convento da Madre de Deus, Lisboa", pre.prompts[-1])
        # a resposta final só viu o escolhido e não lhe pedem o bloco
        self.assertIn("MNAz 1690 Az", llm.prompts[0])
        self.assertNotIn("MNAz 865 Az", llm.prompts[0])
        self.assertNotIn("[[cartões", llm.prompts[0])
        event = service.query_logger.events[-1]
        self.assertEqual(event["card_selection"]["rule"], "preselect")
        self.assertEqual(event["card_selection"]["model_cards"], [5])
        self.assertEqual(len(event["retrieved_artifacts"]), 8)
        self.assertEqual([a["artifact_id"] for a in event["shown_artifacts"]], ["d5"])

    def test_search_order_is_kept_and_several_cards_stay(self) -> None:
        llm, pre = _FinalLLM("Encontrei três martírios."), _PreLLM({"cartoes": ["doc_3", "doc_1", "doc_2", "doc_3"]})
        response = self._ask(self._search_service(llm, pre), "encontra martírios de santos")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d1", "d2", "d3"])
        self.assertTrue(response.results_has_more)
        self.assertNotIn("[[cartões", llm.prompts[0])

    def test_no_usable_choice_falls_back_to_the_block(self) -> None:
        for reply in ({"cartoes": []}, {"cartoes": ["doc_99"]}, {"outra": 1}, RuntimeError("em baixo")):
            with self.subTest(reply=reply):
                llm = _FinalLLM("Encontrei a peça MNAz 1690 Az.\n\n[[cartões: doc_5]]")
                service = self._search_service(llm, _PreLLM(reply))
                response = self._ask(service, "encontra cenas da vida de cristo")
                self.assertIn("[[cartões: nenhum]]", llm.prompts[0])
                self.assertEqual([a.artifact_id for a in response.artifact_results], ["d5"])
                self.assertEqual(service.query_logger.events[-1]["card_selection"]["rule"], "cited")

    def test_flag_off_never_calls_the_preprocess_model(self) -> None:
        llm, pre = _FinalLLM("Texto.\n\n[[cartões: doc_1]]"), _PreLLM({"cartoes": ["doc_5"]})
        response = self._ask(self._search_service(llm, pre, preselect=False), "encontra cenas da vida de cristo")
        self.assertEqual(pre.prompts, [])
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d1"])


class TestPreprocessService(_Base):
    def test_default_is_the_main_llm(self) -> None:
        llm = _FinalLLM("x")
        self.assertIs(self._service(self._settings(), llm=llm).preprocess_llm_service, llm)

    def test_settings_point_at_the_preprocess_model_with_the_main_llm_as_fallback(self) -> None:
        settings = Settings(
            LLM_BASE_URL="http://127.0.0.1:8004/v1/chat/completions", LLM_MODEL="amalia", LLM_API_KEY="k",
            CHAT_PREPROCESS_LLM_BASE_URL="http://127.0.0.1:8005/v1", CHAT_PREPROCESS_LLM_MODEL="qwen",
        )
        pre = preprocess_llm_settings(settings)
        self.assertEqual(pre.llm_openai_base_url_resolved, "http://127.0.0.1:8005/v1")
        self.assertEqual(pre.llm_model_resolved, "qwen")
        self.assertEqual(pre.llm_fallback_openai_base_url_resolved, "http://127.0.0.1:8004/v1")
        self.assertEqual(pre.LLM_FALLBACK_MODEL, "amalia")
        service = LLMService(pre)
        self.assertIsNotNone(service.fallback_client)
        self.assertEqual(service.fallback_model, "amalia")
        # o LLM principal fica como estava
        self.assertEqual(settings.llm_model_resolved, "amalia")
