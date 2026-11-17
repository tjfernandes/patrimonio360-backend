"""Pesquisa do chat, Bloco C: leitura do pedido pelo modelo de pré-processamento
(CHAT_REQUEST_READING). O modelo é simulado; a medição com o modelo real usa os
casos de tests/request_reading_cases.py (docs/pesquisa-bloco-c.md).
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.prompts.query_planner_prompts import build_request_reading_prompt  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.chat_session_store import ChatSessionState, ChatTurn  # noqa: E402
from tests.request_reading_cases import CASES  # noqa: E402

CARDS = [
    {"title": "Tentações de Santo Antão", "inventory_number": "1498 Pint", "creators": ["Bosch, Hieronymus"]},
    {"title": "Tentações de Santo Antão", "inventory_number": "1470 Pint", "creators": []},
]


class _LLM:
    def __init__(self, parsed=None, error: Exception | None = None) -> None:
        self.parsed, self.error, self.calls = parsed, error, []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(parsed_json=self.parsed)


def _service(llm: _LLM | None = None, **flags) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = Settings(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False, **flags)
    service._preprocess_llm_service = llm
    service.llm_service = llm
    return service


def _state(*turns: tuple[str, str], cards=None) -> ChatSessionState:
    state = ChatSessionState(conversation_id="c", museum_slug="mnaa")
    state.history = [ChatTurn(role=role, text=text) for role, text in turns]
    state.last_paged_artifact_results = list(cards or [])
    return state


FOLLOW_UP = _state(("user", "Onde estão as Tentações de Santo Antão?"), ("assistant", "Estão na sala 61."),
                   ("user", "Quem o pintou e quando?"), cards=CARDS)


class TestFlagAndPrompt(unittest.TestCase):
    def test_off_by_default(self) -> None:
        self.assertIs(Settings(_env_file=None).CHAT_REQUEST_READING, False)

    def test_prompt_has_the_three_fields_the_museum_and_the_conversation(self) -> None:
        prompt = build_request_reading_prompt(museum_name="Museu Nacional de Arte Antiga",
                                              history_lines=["Visitante: Olá", "Guia: mostrou 1 peça: Custódia (740 Our)"], message="Quem a fez?")
        self.assertIn('{"lingua": "pt|en|es|outra", "seguimento": true|false, "pedido": "..."}', prompt)
        self.assertIn("visita virtual 360° ao Museu Nacional de Arte Antiga", prompt)
        self.assertIn("CONVERSA ANTERIOR:\nVisitante: Olá\nGuia: mostrou 1 peça: Custódia (740 Our)\n\nMENSAGEM: Quem a fez?", prompt)

    def test_prompt_without_conversation(self) -> None:
        self.assertIn("CONVERSA ANTERIOR:\n(nenhuma)", build_request_reading_prompt(museum_name=None, history_lines=[], message="Olá"))

    def test_the_measured_cases_are_kept(self) -> None:
        self.assertEqual(len(CASES), 120)
        self.assertEqual(sum(1 for case in CASES if case[4] is True), 16)   # seguimentos


class TestHistory(unittest.TestCase):
    def test_last_guide_turn_lists_the_cards_with_inventory_and_author(self) -> None:
        lines = _service()._request_reading_history(FOLLOW_UP)
        self.assertEqual(lines, [
            "Visitante: Onde estão as Tentações de Santo Antão?",
            "Guia: mostrou 2 peças: Tentações de Santo Antão (1498 Pint, Bosch, Hieronymus); Tentações de Santo Antão (1470 Pint)",
        ])

    def test_without_cards_the_guide_turn_is_the_start_of_its_text(self) -> None:
        state = _state(("user", "Olá"), ("assistant", "Olá!\n\nSou o guia. " + "x" * 400), ("user", "E agora?"))
        lines = _service()._request_reading_history(state)
        self.assertEqual(lines[0], "Visitante: Olá")
        self.assertTrue(lines[1].startswith("Guia: Olá! Sou o guia. xxx"))
        self.assertLessEqual(len(lines[1]), 206)

    def test_current_message_is_not_part_of_the_history(self) -> None:
        self.assertEqual(_service()._request_reading_history(_state(("user", "Olá"))), [])


class TestReading(unittest.TestCase):
    def _read(self, llm, state=FOLLOW_UP):
        return asyncio.run(_service(llm)._read_request_with_llm(message="Quem o pintou e quando?", state=state, museum_name="MNAA"))

    def test_valid_answer(self) -> None:
        llm = _LLM({"lingua": "PT", "seguimento": True, "pedido": " Quem pintou as Tentações de Santo Antão e quando? "})
        self.assertEqual(self._read(llm), {"language": "pt", "follow_up": True, "request": "Quem pintou as Tentações de Santo Antão e quando?"})
        (call,) = llm.calls
        self.assertEqual(call["response_format"].type, "json_object")
        self.assertIn("MENSAGEM: Quem o pintou e quando?", call["message"])
        self.assertIn("Bosch, Hieronymus", call["message"])

    def test_unknown_language_is_none(self) -> None:
        self.assertIsNone(self._read(_LLM({"lingua": "outra", "seguimento": False, "pedido": "x"}))["language"])

    def test_malformed_answer_or_error_changes_nothing(self) -> None:
        self.assertIsNone(self._read(_LLM({"lingua": "pt", "pedido": "x"})))
        self.assertIsNone(self._read(_LLM({"lingua": "pt", "seguimento": "sim", "pedido": "x"})))
        self.assertIsNone(self._read(_LLM(["x"])))
        self.assertIsNone(self._read(_LLM(error=RuntimeError("sem modelo"))))


class TestApply(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _service()

    def _apply(self, reading, state=FOLLOW_UP, message="Quem o pintou e quando?"):
        policy = self.service._derive_context_policy(message=message, state=state)
        return policy, self.service._apply_request_reading(reading=reading, context_policy=policy, message=message, state=state)

    def test_follow_up_the_regex_missed(self) -> None:
        before, (policy, request) = self._apply({"language": "pt", "follow_up": True, "request": "Quem pintou as Tentações de Santo Antão e quando?"})
        self.assertFalse(before["use_history_for_query"])
        self.assertEqual(request, "Quem pintou as Tentações de Santo Antão e quando?")
        self.assertEqual((policy["is_follow_up"], policy["use_history_for_query"], policy["use_history_for_answer"], policy["reason"]),
                         (True, True, True, "request_reading_follow_up"))

    def test_follow_up_the_regex_also_saw_keeps_its_policy(self) -> None:
        state = _state(("user", "Que peças de ourivesaria tens?"), ("assistant", "Estas."), ("user", "E de prata?"))
        before, (policy, request) = self._apply({"language": "pt", "follow_up": True, "request": "Peças de ourivesaria de prata"}, state, "E de prata?")
        self.assertTrue(before["use_history_for_query"])
        self.assertEqual(policy, before)
        self.assertEqual(request, "Peças de ourivesaria de prata")

    def test_standalone_message_overrides_a_regex_follow_up(self) -> None:
        state = _state(("user", "Que peças de ourivesaria tens?"), ("assistant", "Estas."), ("user", "E agora mostra-me os coches."))
        before, (policy, request) = self._apply({"language": "pt", "follow_up": False, "request": "E agora mostra-me os coches."}, state,
                                                "E agora mostra-me os coches.")
        self.assertTrue(before["use_history_for_query"])
        self.assertEqual((policy["use_history_for_query"], policy["carry_filters"], policy["reason"]), (False, False, "request_reading_standalone"))
        self.assertEqual(request, "")

    def test_no_previous_conversation_is_never_a_follow_up(self) -> None:
        state = _state(("user", "Quem o pintou?"))
        before, (policy, request) = self._apply({"language": "pt", "follow_up": True, "request": "Quem pintou o quadro?"}, state, "Quem o pintou?")
        self.assertEqual((policy, request), (before, ""))

    def test_follow_up_without_a_real_rewrite_changes_nothing(self) -> None:
        before, (policy, request) = self._apply({"language": "pt", "follow_up": True, "request": "quem o pintou e quando"})
        self.assertEqual((policy, request), (before, ""))
        before, (policy, request) = self._apply({"language": "pt", "follow_up": True, "request": ""})
        self.assertEqual((policy, request), (before, ""))


class TestGuardrail(unittest.TestCase):
    def _resolve(self, service, rewritten, original="And who made it?", from_reading=True):
        state = _state(("user", "Mostra-me as Tentações."), ("assistant", "Aqui estão."), ("user", original), cards=CARDS)
        return service._resolve_contextual_retrieval_query(original_query=original, rewritten_query=rewritten, state=state, requested_museum=None,
                                                           museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga",
                                                           from_request_reading=from_reading)

    def test_other_rewrites_keep_the_old_rules(self) -> None:
        on = _service(CHAT_REQUEST_READING=True)
        self.assertEqual(self._resolve(on, "Who made the Tentações de Santo Antão?", from_reading=False), ("And who made it?", "user_message"))
        self.assertEqual(self._resolve(on, "obras de Bosch", "E desse autor?", from_reading=False)[1], "user_message")

    def test_a_full_sentence_may_bring_short_function_words(self) -> None:
        on = _service(CHAT_REQUEST_READING=True)
        self.assertEqual(self._resolve(on, "Quem pintou as Tentações de Santo Antão e por que razão?", "Quem o pintou e por que razão?")[1], "router_rewrite")

    def test_on_the_titles_and_authors_of_the_shown_cards_are_accepted(self) -> None:
        on = _service(CHAT_REQUEST_READING=True)
        self.assertEqual(self._resolve(on, "Who made the Tentações de Santo Antão?"), ("Who made the Tentações de Santo Antão?", "router_rewrite"))

    def test_on_invented_words_are_still_rejected(self) -> None:
        on = _service(CHAT_REQUEST_READING=True)
        self.assertEqual(self._resolve(on, "Who made the Mona Lisa?")[1], "user_message")

    def test_last_shown_card_texts(self) -> None:
        self.assertEqual(ChatService._last_shown_card_texts(_state(cards=CARDS + ["lixo", {}])),
                         ["Tentações de Santo Antão 1498 Pint Bosch, Hieronymus", "Tentações de Santo Antão 1470 Pint"])


class _FakeClient:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def search(self, *, index, body):
        self.bodies.append(body)
        return {"hits": {"total": {"value": 0}, "hits": []}, "aggregations": {}}

    def count(self, *, index, body):
        return {"count": 0}


class TestAnalyticsSubject(unittest.TestCase):
    """«Qual delas é a mais antiga?» chega à estatística como «…obras de Gregório Lopes…»."""

    @staticmethod
    def _subject_match(**flags):
        from app.services.opensearch_client import OpenSearchGateway

        gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ARTIFACT="arts", **flags))
        gateway._client = _FakeClient()
        asyncio.run(gateway.aggregate_artifacts(museum_slug="mnaa", museum_id="mnaa", subject="gregorio lopes", order="asc", size=8))
        return gateway._client.bodies[0]["query"]["bool"]["must"][0]["multi_match"]

    def test_off_by_default_and_unchanged(self) -> None:
        self.assertIs(Settings(_env_file=None).CHAT_CATALOGUE_ANALYTICS_STRICT_SUBJECT, False)
        match = self._subject_match()
        self.assertEqual((match["type"], match["minimum_should_match"]), ("best_fields", "75%"))
        self.assertFalse(any(field.startswith("creators") for field in match["fields"]))

    def test_on_every_word_is_required_and_the_author_counts(self) -> None:
        match = self._subject_match(CHAT_CATALOGUE_ANALYTICS_STRICT_SUBJECT=True)
        self.assertEqual((match["type"], match["operator"]), ("cross_fields", "and"))
        self.assertNotIn("minimum_should_match", match)
        self.assertIn("creators.text^3", match["fields"])
        self.assertIn("title^3", match["fields"])


if __name__ == "__main__":
    unittest.main()
