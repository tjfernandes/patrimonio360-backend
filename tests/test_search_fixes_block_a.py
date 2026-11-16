"""Correção da pesquisa do chat, Bloco A (docs/plano-correcao-pesquisa-mnaa.md).

Uma flag por correção, todas desligadas por omissão:

- F1  CHAT_PRESELECT_EMPTY_IS_NONE: a lista vazia do pré-processamento vale (zero cartões).
- F2  CHAT_NO_CARDS_ANSWER_PROMPT: instrução própria para a pesquisa sem cartões.
- F3  CHAT_IN_TOUR_FROM_NAVIGATION: «Na visita virtual: sim» pelos alvos de navegação.
- F7  alias «jeronimo» retirado (dado, sem flag).
- F9  CHAT_ROOM_BEFORE_ROUTE: o nome completo de uma sala ganha à regra dos percursos.
- F11 CHAT_STRIP_CARD_LABELS: rótulos dos cartões fora da resposta.
- F12 ROUTE_THEME_STRICT_TIME_MENTION: séculos do LLM do tema só com menção de época.

    python -m unittest tests.test_search_fixes_block_a
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.core.config import Settings
from app.prompts.chat_prompts import build_final_answer_prompt_v2
from app.schemas.chat import ResponseFormatObject
from app.services.chat_service import ChatService
from app.services.llm_service import LLMResponse
from app.services.route_planning.store import RouteStore
from app.services.route_planning.theme import merge_llm_facets, rule_facets
from tests import test_model_selects_cards as cards_tests
from tests import test_route_planning as route_tests
from tests.test_tour_location import _Base

MNAZ_POI = cards_tests.TestEndToEnd.POI_BY_MUSEUM
FLAGS = (
    "CHAT_PRESELECT_EMPTY_IS_NONE", "CHAT_NO_CARDS_ANSWER_PROMPT", "CHAT_IN_TOUR_FROM_NAVIGATION",
    "CHAT_ROOM_BEFORE_ROUTE", "CHAT_STRIP_CARD_LABELS", "ROUTE_THEME_STRICT_TIME_MENTION",
)


class _PreLLM:
    """O modelo de pré-processamento: devolve o JSON dado."""

    def __init__(self, reply: object) -> None:
        self.reply, self.prompts = reply, []

    async def generate(self, *, message, response_format=ResponseFormatObject(type="json_object"), **_kwargs):
        self.prompts.append(message)
        return LLMResponse(text=json.dumps(self.reply), model="pre", response_format=response_format,
                           parsed_json=self.reply, finish_reason="stop")


def _card(prompt: str, index: int) -> str:
    """O texto do cartão [doc_N] dentro de um prompt."""
    start = prompt.index(f"[doc_{index}]")
    end = prompt.find(f"[doc_{index + 1}]", start)
    return prompt[start:end if end != -1 else len(prompt)]


class TestFlagsDefaultOff(_Base):
    def test_every_flag_is_off_by_default(self) -> None:
        settings = Settings(_env_file=None)
        self.assertEqual({flag: getattr(settings, flag) for flag in FLAGS}, {flag: False for flag in FLAGS})


class _SearchBase(_Base):
    POI_BY_MUSEUM = MNAZ_POI
    DOCS = cards_tests.MNAZ_DOCS

    def _search_service(self, llm, pre, **flags) -> ChatService:
        values = dict(
            CHAT_MODEL_SELECTS_CARDS=True, CHAT_PRESELECT_CARDS=True, CHAT_COMPACT_ANSWER_CONTEXT=True,
            CHAT_ROUTER_MODE="deterministic", CHAT_STREAM_FINAL_ANSWER=False, CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8,
        )
        values.update(flags)
        service = self._service(self._settings(**values), llm=llm)
        service.preprocess_llm_service = pre
        service.opensearch_gateway = cards_tests._SearchGateway()
        docs = self.DOCS

        async def _retrieve_context(**_kwargs):
            return "ctx", 20, [dict(d) for d in docs], {"kind": "text", "results_total": 20, "filters": {}, "sort": {}}

        service._retrieve_context = _retrieve_context
        return service

    def _ask(self, service, message, **kwargs):
        request = self._request(message, museum_slug="mnaz", museum_id="mnaz", museum_name="Museu Nacional do Azulejo", **kwargs)
        return asyncio.run(service.handle_message(request))


class TestEmptyPreselection(_SearchBase):
    """F1: «cartoes»: [] é a resposta «nenhum corresponde», não uma falha."""

    def test_empty_list_shows_no_cards(self) -> None:
        llm, pre = cards_tests._FinalLLM("Não encontrei quadros do Picasso neste museu."), _PreLLM({"cartoes": []})
        service = self._search_service(llm, pre, CHAT_PRESELECT_EMPTY_IS_NONE=True)
        response = self._ask(service, "mostra-me quadros do picasso", conversation_id="vazio")
        self.assertEqual(response.artifact_results, [])
        self.assertEqual(response.image_matches, [])
        self.assertEqual(response.navigation_targets, [])
        self.assertEqual(response.results_total, 0)
        self.assertFalse(response.results_has_more)
        self.assertEqual(response.reply, "Não encontrei quadros do Picasso neste museu.")
        # a AMÁLIA não vê nenhuma ficha (nem as da pesquisa) e não lhe pedem o bloco
        prompt = llm.prompts[0]
        self.assertTrue(prompt.endswith("CARTÕES CONFIRMADOS:" + chr(10) + "(nenhum)"))
        self.assertNotIn("[doc_2]", prompt)
        self.assertNotIn("Martírio", prompt)
        self.assertNotIn("ctx", prompt.split("PERGUNTA:")[-1])
        self.assertNotIn("[[cartões", prompt)
        event = service.query_logger.events[-1]
        self.assertEqual(event["card_selection"]["rule"], "preselect_none")
        self.assertEqual(event["card_selection"]["shown_after"], 0)
        self.assertEqual(event["shown_artifacts"], [])
        self.assertEqual(len(event["retrieved_artifacts"]), 8)   # o que a pesquisa devolveu continua no registo
        state = service.session_store.load_or_create(conversation_id="vazio", museum_slug="mnaz")
        self.assertEqual(state.last_result_ids, [])

    def test_invalid_labels_are_still_a_failure(self) -> None:
        for reply in ({"cartoes": ["doc_99"]}, {"cartoes": ["nenhum"]}, {"cartoes": None}, {"outra": 1}):
            with self.subTest(reply=reply):
                llm = cards_tests._FinalLLM("Encontrei a peça MNAz 1690 Az.\n\n[[cartões: doc_5]]")
                service = self._search_service(llm, _PreLLM(reply), CHAT_PRESELECT_EMPTY_IS_NONE=True)
                response = self._ask(service, "encontra cenas da vida de cristo")
                self.assertIn("[[cartões: nenhum]]", llm.prompts[0])
                self.assertEqual([a.artifact_id for a in response.artifact_results], ["d5"])
                self.assertEqual(service.query_logger.events[-1]["card_selection"]["rule"], "cited")

    def test_flag_off_keeps_the_eight_cards(self) -> None:
        llm, pre = cards_tests._FinalLLM("Não encontrei."), _PreLLM({"cartoes": []})
        service = self._search_service(llm, pre)
        response = self._ask(service, "mostra-me quadros do picasso")
        self.assertEqual(len(response.artifact_results), 8)
        self.assertIn("[doc_8]", llm.prompts[0])
        self.assertEqual(response.results_total, 20)

    def test_a_real_choice_is_unchanged_by_the_flag(self) -> None:
        llm, pre = cards_tests._FinalLLM("É a Fuga para o Egipto."), _PreLLM({"cartoes": ["doc_5"]})
        service = self._search_service(llm, pre, CHAT_PRESELECT_EMPTY_IS_NONE=True)
        response = self._ask(service, "mostra-me a fuga para o egipto")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d5"])
        self.assertEqual(service.query_logger.events[-1]["card_selection"]["rule"], "preselect")


class TestNoCardsPrompt(_SearchBase):
    """F2: sem cartões numa pesquisa, a instrução é «não encontrei», não «destaca três peças»."""

    BASE = dict(museum_name="Museu Nacional de Arte Antiga", intent="search", mode="rag", rolling_summary="",
                filters_state={}, user_message="Há azulejos?", rewritten_query="Há azulejos?", national_treasures=True)

    def test_search_without_cards_gets_the_short_instruction(self) -> None:
        prompt = build_final_answer_prompt_v2(**self.BASE, retrieval_context="", language="pt", no_cards_instruction=True)
        lines = prompt.split("\n")
        self.assertEqual(lines[0], "És a Amália, guia virtual de uma visita 360° ao museu Museu Nacional de Arte Antiga.")
        self.assertEqual(lines[1], "Responde em português europeu, com frases curtas e tom cordial.")
        self.assertTrue(lines[2].startswith("A pesquisa não devolveu nenhuma peça que corresponda a este pedido."))
        self.assertIn("Não nomeies peças, autores, salas nem obras", lines[2])
        self.assertEqual(lines[3:], ["PERGUNTA: Há azulejos?"])
        for gone in ("CARTÕES CONFIRMADOS", "destaca até três peças", "Tesouro Nacional", "TIPO DE RESPOSTA"):
            self.assertNotIn(gone, prompt)

    def test_interpreted_request_is_kept(self) -> None:
        args = dict(self.BASE, user_message="E do Picasso?", rewritten_query="Quadros do Picasso")
        prompt = build_final_answer_prompt_v2(**args, retrieval_context=" ", language="pt", no_cards_instruction=True)
        self.assertTrue(prompt.endswith("PERGUNTA: E do Picasso?\nPEDIDO INTERPRETADO: Quadros do Picasso"))

    def test_english(self) -> None:
        prompt = build_final_answer_prompt_v2(**self.BASE, retrieval_context="", language="en", no_cards_instruction=True)
        self.assertIn("Answer in English", prompt)
        self.assertIn("The search returned no object that matches this request.", prompt)
        self.assertTrue(prompt.endswith("QUESTION: Há azulejos?"))
        self.assertNotIn("CONFIRMED CARDS", prompt)

    def test_nothing_changes_with_context_or_another_kind_of_answer(self) -> None:
        cases = [
            dict(retrieval_context="[doc_1] Título: Painel | Inventário: 1"),          # há cartões
            dict(retrieval_context="Lista dos Tesouros Nacionais: 18 peças."),          # nota sem cartões
            dict(retrieval_context="", mode="llm_only"),                                # conversa
            dict(retrieval_context="", intent="selected_artifact_question"),            # peça selecionada
            dict(retrieval_context="", input_modality="image"),                         # pesquisa por imagem
        ]
        for case in cases:
            with self.subTest(case=case):
                args = dict(self.BASE, language="pt")
                args.update(case)
                self.assertEqual(
                    build_final_answer_prompt_v2(**args, no_cards_instruction=True),
                    build_final_answer_prompt_v2(**args),
                )

    def test_flag_off_keeps_the_search_prompt(self) -> None:
        prompt = build_final_answer_prompt_v2(**self.BASE, retrieval_context="", language="pt")
        self.assertIn("CARTÕES CONFIRMADOS:\n(nenhum)", prompt)
        self.assertNotIn("A pesquisa não devolveu nenhuma peça", prompt)

    def test_end_to_end_with_the_empty_preselection(self) -> None:
        llm, pre = cards_tests._FinalLLM("Não encontrei isso no acervo."), _PreLLM({"cartoes": []})
        service = self._search_service(llm, pre, CHAT_PRESELECT_EMPTY_IS_NONE=True, CHAT_NO_CARDS_ANSWER_PROMPT=True)
        self._ask(service, "quem ganhou o jogo do benfica?")
        self.assertIn("A pesquisa não devolveu nenhuma peça que corresponda a este pedido.", llm.prompts[0])
        self.assertNotIn("CARTÕES CONFIRMADOS", llm.prompts[0])
        self.assertTrue(llm.prompts[0].endswith("PERGUNTA: quem ganhou o jogo do benfica?"))

    def test_end_to_end_cards_keep_the_search_prompt(self) -> None:
        llm, pre = cards_tests._FinalLLM("É a Fuga para o Egipto."), _PreLLM({"cartoes": ["doc_5"]})
        service = self._search_service(llm, pre, CHAT_PRESELECT_EMPTY_IS_NONE=True, CHAT_NO_CARDS_ANSWER_PROMPT=True)
        self._ask(service, "mostra-me a fuga para o egipto")
        self.assertIn("CARTÕES CONFIRMADOS:", llm.prompts[0])
        self.assertNotIn("A pesquisa não devolveu nenhuma peça", llm.prompts[0])


class TestInTourFromNavigation(_SearchBase):
    """F3: uma peça com alvo de navegação está na visita, diga o índice o que disser."""

    # d5 tem hotspot na visita (ov-1690) mas o índice diz in_tour=false, como 712 peças do MNAA
    DOCS = [dict(d, in_tour=False) if d["artifact_id"] == "d5" else dict(d) for d in cards_tests.MNAZ_DOCS]

    def _prompts(self, **flags):
        llm, pre = cards_tests._FinalLLM("Texto."), _PreLLM({"cartoes": ["doc_1", "doc_5", "doc_6"]})
        service = self._search_service(llm, pre, **flags)
        response = self._ask(service, "encontra cenas da vida de cristo")
        return llm.prompts[0], pre.prompts[0], response

    def test_card_says_yes_when_there_is_a_target(self) -> None:
        final, pre, response = self._prompts(CHAT_IN_TOUR_FROM_NAVIGATION=True)
        # a entrada do pré-processamento não muda: a redação dele é sensível e foi medida assim
        self.assertIn("Na visita virtual: não", _card(pre, 5))
        self.assertIn("Na visita virtual: sim", _card(pre, 1))
        self.assertIn("MNAz 1690 Az", _card(final, 2))
        self.assertIn("Na visita virtual: sim", _card(final, 2))
        self.assertIn("Na visita virtual: não", _card(final, 3))
        # o cartão devolvido ao frontend não muda
        self.assertEqual([(a.artifact_id, a.in_tour) for a in response.artifact_results], [("d1", True), ("d5", False), ("d6", False)])

    def test_flag_off_keeps_the_index_value(self) -> None:
        final, pre, _response = self._prompts()
        self.assertIn("Na visita virtual: não", _card(pre, 5))
        self.assertIn("Na visita virtual: não", _card(final, 2))

    def test_never_downgrades_and_ignores_missing_targets(self) -> None:
        service = self._service(self._settings(CHAT_IN_TOUR_FROM_NAVIGATION=True))
        results = cards_tests._cards(self.DOCS)
        docs = service._artifact_results_as_prompt_docs(results, navigation_targets=cards_tests._targets())
        self.assertEqual([d["in_tour"] for d in docs], [True, False, False, False, True, False, False, False])
        without = service._artifact_results_as_prompt_docs(results)
        self.assertEqual([d["in_tour"] for d in without], [True, False, False, False, False, False, False, False])
        self.assertEqual(service._artifact_results_as_prompt_docs(results, navigation_targets=[]), without)


class TestJeronimoAlias(_Base):
    """F7: «São Jerónimo» é uma pintura do MNAA, não o Mosteiro dos Jerónimos."""

    def test_saint_jerome_is_not_a_museum(self) -> None:
        service = self._service()
        for message in ("Quem pintou o São Jerónimo?", "mostra-me o são jeronimo do dürer", "Who painted Saint Jerome? jeronimo"):
            self.assertIsNone(service._resolve_requested_search_museum(message), message)

    def test_the_monastery_is_still_recognised(self) -> None:
        service = self._service()
        for message in ("peças do Mosteiro dos Jerónimos", "e nos Jerónimos?", "túmulos no mosteiro dos jeronimos"):
            museum = service._resolve_requested_search_museum(message)
            self.assertEqual(museum.museum_id if museum else None, "mj", message)


class TestRoomBeforeRoute(_Base):
    """F9: «Leva-me à Sala da Música» vai à sala; só um pedido de percurso cria um percurso."""

    def setUp(self) -> None:
        super().setUp()
        (Path(self._tmp.name) / "tour-graph-mnt.json").write_text(json.dumps(route_tests.GRAPH), encoding="utf-8")
        self.stops = [route_tests._stop("a", "p4", "Sala C"), route_tests._stop("b", "p1", "Sala A", kind="object")]

    def _route_service(self, **flags) -> ChatService:
        values = {"CHAT_TOUR_LOCATION_ANSWERS": True, "CHAT_ROUTE_PLANNING": True, "ROUTE_THEME_LLM": False, "ROUTE_MIN_FUSION_SCORE": 0.0}
        values.update(flags)
        service = self._service(self._settings(**values))
        service.embedding_provider = route_tests._Embeddings()
        service._route_search = route_tests._FakeSearch(self.stops)
        from app.services.route_planning.graph import TourGraphStore

        service._route_graph_store = TourGraphStore(self._tmp.name, check_interval=0.0)
        service._route_store = RouteStore(600)
        return service

    def _ask(self, service, message, **kwargs):
        return asyncio.run(service.handle_message(self._request(message, **kwargs)))

    def test_full_room_name_goes_to_the_room(self) -> None:
        service = self._route_service(CHAT_ROOM_BEFORE_ROUTE=True)
        for message in ("Leva-me à Sala da Música.", "leva-me a sala cantonal", "Take me to the Sala Cantonal"):
            with self.subTest(message=message):
                response = self._ask(service, message)
                self.assertIsNone(response.route)
                self.assertEqual(response.model_hint, "deterministic:tour_room")
                self.assertIsNotNone(response.tour_room)
                self.assertEqual(service.query_logger.events[-1]["retrieval_mode"], "tour_room")

    def test_flag_off_still_plans_a_route(self) -> None:
        service = self._route_service()
        response = self._ask(service, "Leva-me à Sala da Música.")
        self.assertIsNone(response.tour_room)
        self.assertIn("percurso", response.reply)
        self.assertEqual(service.query_logger.events[-1]["retrieval_mode"], "route")

    def test_route_requests_stay_routes(self) -> None:
        service = self._route_service(CHAT_ROOM_BEFORE_ROUTE=True)
        cases = [
            "Faz-me um percurso pela Sala da Música",      # palavras a mais: é um percurso
            "Leva-me a ver azulejos com bichos",           # sem sala
            "Leva-me à Música",                            # nome só meio dito
            "Roteiro pela sala cantonal e pelo pátio",
        ]
        for message in cases:
            with self.subTest(message=message):
                self.assertFalse(service._room_request_precedes_route(self._request(message), {}))

    def test_explicit_route_intent_wins(self) -> None:
        service = self._route_service(CHAT_ROOM_BEFORE_ROUTE=True)
        request = self._request("Sala da Música")
        self.assertTrue(service._room_request_precedes_route(request, {}))
        self.assertFalse(service._room_request_precedes_route(request, {"intent": "plan_route"}))

    def test_room_match_reports_whether_the_name_is_complete(self) -> None:
        service = self._route_service()
        match = lambda text: service._match_tour_room_query(text, museum_slug="mnt", museum_id="mnt")
        self.assertTrue(match("Leva-me à sala da música")["full_match"])
        self.assertFalse(match("Leva-me à música")["full_match"])


class TestStripCardLabels(_Base):
    """F11: os rótulos que o backend escreve nos cartões não chegam ao visitante."""

    strip = staticmethod(ChatService._strip_card_labels)

    def test_parenthetical_labels(self) -> None:
        text = "1. «São Jerónimo» (Tesouro Nacional: sim)\n2. «Painéis de São Vicente» (Tesouro Nacional: sim, Na visita virtual: sim)"
        self.assertEqual(self.strip(text), "1. «São Jerónimo»\n2. «Painéis de São Vicente»")
        self.assertEqual(self.strip("«Biombo Namban» (Na visita virtual: não)."), "«Biombo Namban».")
        self.assertEqual(self.strip("«Biombo» [na visita virtual: nao] e mais."), "«Biombo» e mais.")
        self.assertEqual(self.strip("«Custódia» (inv. 740 Our, Tesouro Nacional: sim)"), "«Custódia» (inv. 740 Our)")

    def test_label_after_a_separator_or_as_a_loose_sentence(self) -> None:
        self.assertEqual(self.strip("«São Jerónimo» — Tesouro Nacional: sim."), "«São Jerónimo».")
        self.assertEqual(self.strip("«São Jerónimo», Tesouro Nacional: sim\n«Salomé»"), "«São Jerónimo»\n«Salomé»")
        self.assertEqual(self.strip("É de Dürer. Tesouro Nacional: sim. Está na sala 3."), "É de Dürer. Está na sala 3.")

    def test_quoted_label_takes_its_clause_with_it(self) -> None:
        # as três formas vistas nas repetições da checklist de 2026-11-16
        self.assertEqual(
            self.strip('São, de facto, um Tesouro Nacional. Confirmado pelo cartão que indica "Tesouro Nacional: sim". Esta obra é de Bosch.'),
            "São, de facto, um Tesouro Nacional. Esta obra é de Bosch.",
        )
        self.assertEqual(
            self.strip('Todas estão representadas na visita 360°, conforme indicado no cartão "Na visita virtual: sim". O Porta-Paz é de prata.'),
            "Todas estão representadas na visita 360°. O Porta-Paz é de prata.",
        )
        self.assertEqual(
            self.strip("Estão disponíveis na visita virtual, embora nem todos os cartões indiquem «Na visita virtual: sim»."),
            "Estão disponíveis na visita virtual.",
        )
        self.assertEqual(
            self.strip('O cartão diz "Tesouro Nacional: sim", por isso é um tesouro.'), "Por isso é um tesouro.",
        )
        self.assertEqual(self.strip("1. **Última Ceia** – óleo sobre madeira. *(Na visita virtual: sim)*\n2. **Outra**"),
                         "1. **Última Ceia** – óleo sobre madeira.\n2. **Outra**")

    def test_label_on_its_own_line(self) -> None:
        text = "«Biombo Namban»\n    Na visita virtual: sim\n«Outro biombo»\n- Tesouro Nacional: sim"
        self.assertEqual(self.strip(text), "«Biombo Namban»\n«Outro biombo»\n")

    def test_real_sentences_are_left_alone(self) -> None:
        for text in (
            "A peça está na visita virtual: sim, na sala 3.",
            "Na visita virtual: sim.",                                   # a resposta inteira
            "Na visita virtual: sim. Fica na Sala 12.",                  # abre a resposta: é a resposta à pergunta
            "Esta custódia é um Tesouro Nacional e está na visita virtual.",
            "A peça está na visita virtual: sim.",
        ):
            self.assertEqual(self.strip(text), text)

    def test_sanitizer_uses_it_only_with_the_flag(self) -> None:
        text = "Encontrei «São Jerónimo» (Tesouro Nacional: sim)."
        on = self._service(self._settings(CHAT_STRIP_CARD_LABELS=True))
        self.assertEqual(on._sanitize_assistant_reply(text, docs=[], language="pt"), "Encontrei «São Jerónimo».")
        self.assertEqual(self._service()._sanitize_assistant_reply(text, docs=[], language="pt"), text)


class TestStrictTimeMention(_Base):
    """F12: «30 minutos» e «5 peças» não autorizam um século proposto pelo LLM do tema."""

    VOCAB = route_tests.VOCAB

    def _merge(self, message: str, parsed_facets: dict, *, strict: bool):
        facets = rule_facets(message, vocabulary=self.VOCAB)
        return merge_llm_facets(facets, {"facets": parsed_facets}, vocabulary=self.VOCAB, message=message, strict_time=strict)

    def test_durations_and_counts_are_not_a_period(self) -> None:
        for message in ("Faz um percurso de 30 minutos sobre louça", "Encurta o percurso para 5 peças", "percurso com 12 paragens de prata"):
            with self.subTest(message=message):
                loose = self._merge(message, {"centuries": [17], "year_range": [1600, 1699]}, strict=False)
                self.assertEqual(loose.centuries, [17])            # o defeito, com a flag desligada
                strict = self._merge(message, {"centuries": [17], "year_range": [1600, 1699]}, strict=True)
                self.assertEqual(strict.centuries, [])
                self.assertIsNone(strict.year_range)

    def test_a_period_in_the_request_still_lets_the_llm_help(self) -> None:
        for message in ("percurso sobre louça antiga", "percurso sobre louça de 1650", "percurso sobre louça dos anos vinte"):
            with self.subTest(message=message):
                merged = self._merge(message, {"centuries": [17]}, strict=True)
                self.assertEqual(merged.centuries, [17])

    def test_centuries_written_in_the_request_come_from_the_rules(self) -> None:
        merged = self._merge("percurso de 30 minutos sobre louça do século XVIII", {"centuries": [17]}, strict=True)
        self.assertEqual(merged.centuries, [18])


if __name__ == "__main__":
    import unittest

    unittest.main()
