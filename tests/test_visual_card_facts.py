"""Perguntas pelo aspeto: o que as legendas dizem vai nos cartões (CHAT_VISUAL_CARD_FACTS),
as peças que só as legendas encontraram entram na escolha (CHAT_VISUAL_CARD_CANDIDATES)
e o modelo de pré-processamento diz se o pedido é visual (CHAT_VISUAL_INTENT_MODEL).

    python -m unittest tests.test_visual_card_facts
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from app.schemas.chat import ResponseFormatObject
from app.services.chat_service import ChatService
from app.services.llm_service import LLMResponse
from app.services.retrieval.fusion import BranchProvenance, FusedResult
from tests.test_card_preselection import _PreLLM
from tests.test_model_selects_cards import _FinalLLM, _Gateway
from tests.test_tour_location import _Base


def _doc(artifact_id: str, inventory: str, title: str) -> dict:
    return {"artifact_id": artifact_id, "inventory_number": inventory, "title": title, "museum_id": "mnaa",
            "in_tour": False, "image_count": 1, "category": "Escultura"}


TEXT_DOCS = [_doc(f"d{n}", f"{n} Esc", title) for n, title in enumerate(
    ["Diana", "Minerva", "Gratidão", "Fé", "Devoção", "Efebo"], start=1)]
CAPTION_DOCS = [_doc("c1", "58 Esc", "Flora"), _doc("c2", "761 Esc", "Apolo"), _doc("c3", "762 Esc", "Danaide")]
ALL_DOCS = TEXT_DOCS + CAPTION_DOCS
STRUCTS = {
    "d2": {"tipo_objeto": "modelo de estátua", "materiais": ["barro cozido"], "cores_principais": ["castanho"],
           "cores": ["castanho", "ocre"], "motivos": ["figura feminina", "capacete"], "forma": "figura de pé"},
    "c1": {"tipo_objeto": "escultura", "materiais": ["mármore"], "cores_principais": ["branco"],
           "cores": ["branco", "cinzento"], "motivos": ["grinalda", "fruta"], "forma": "estátua feminina de pé"},
    "c2": {"tipo_objeto": "busto", "materiais": ["mármore"], "cores_principais": ["branco"], "cores": ["branco"]},
}


def _outcome():
    """A pesquisa de texto (d1…d6) fundida com o ramo das legendas (c1, c2, c3 e d2)."""
    caption_rank = {"c1": 1, "c2": 2, "d2": 3, "c3": 4}
    fused = []
    for position, doc in enumerate(ALL_DOCS, start=1):
        sources = {}
        if doc["artifact_id"].startswith("d"):
            sources["artifact_search"] = BranchProvenance(rank=position)
        if doc["artifact_id"] in caption_rank:
            sources["text_to_caption"] = BranchProvenance(rank=caption_rank[doc["artifact_id"]])
        fused.append(FusedResult(artifact_id=doc["artifact_id"], fusion_score=1.0 / position, sources=sources))
    return SimpleNamespace(
        docs=[dict(d) for d in ALL_DOCS], total=len(ALL_DOCS), fused=fused, matched_image_hits=[],
        router=SimpleNamespace(rule="color_pt"), diagnostics={}, debug_payload=lambda: {},
    )


class _CaptionGateway(_Gateway):
    def __init__(self, *, structs=None, error=None) -> None:
        super().__init__()
        self.structs, self.error, self.caption_calls = STRUCTS if structs is None else structs, error, []

    async def fetch_images_by_artifact_ids(self, *, artifact_ids, **_kwargs):
        by_id = {d["artifact_id"]: d for d in ALL_DOCS}
        return [
            {"artifact_id": i, "image_id": f"img-{i}", "local_path": f"Images/x/{i}.jpg",
             "inventory_number": by_id[i]["inventory_number"], "museum_id": "mnaa", "score": 1.0}
            for i in artifact_ids if i in by_id
        ]

    async def fetch_caption_structs(self, *, index, artifact_ids):
        self.caption_calls.append((index, list(artifact_ids)))
        if self.error:
            raise self.error
        return {i: self.structs[i] for i in artifact_ids if i in self.structs}


class _TwoLLM:
    """O modelo de pré-processamento nas duas perguntas: a escolha dos cartões e a escolha pelo aspeto."""

    def __init__(self, cards: object, picks: object = None) -> None:
        self.cards, self.picks, self.card_prompts, self.pick_prompts = cards, picks, [], []

    async def generate(self, *, message, response_format=ResponseFormatObject(type="json_object"), **_kwargs):
        is_pick = "PEÇAS (título" in message
        (self.pick_prompts if is_pick else self.card_prompts).append(message)
        reply = self.picks if is_pick else self.cards
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(text=json.dumps(reply), model="pre", response_format=response_format,
                           parsed_json=reply, finish_reason="stop")


class TestVisualCardFacts(_Base):
    def _search_service(self, llm, pre, *, facts=True, extra=2, visual=True, gateway=None, candidates=0,
                        answer_rule=False) -> ChatService:
        settings = self._settings(
            CHAT_VISUAL_ANSWER_RULE=answer_rule,
            CHAT_MODEL_SELECTS_CARDS=True, CHAT_PRESELECT_CARDS=True, CHAT_PRESELECT_EMPTY_IS_NONE=True,
            CHAT_PRESELECT_CARDS_CANDIDATES=candidates, CHAT_COMPACT_ANSWER_CONTEXT=True,
            CHAT_ROUTER_MODE="deterministic", CHAT_STREAM_FINAL_ANSWER=False, CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=4,
            CAPTION_TEXT_INDEX="caps", CHAT_VISUAL_CARD_FACTS=facts, CHAT_VISUAL_CARD_CANDIDATES=extra,
        )
        service = self._service(settings, llm=llm)
        service.preprocess_llm_service = pre
        service.opensearch_gateway = gateway or _CaptionGateway()

        async def _retrieve_context(**_kwargs):
            return "ctx", 6, [dict(d) for d in TEXT_DOCS], {"kind": "text", "results_total": 6, "filters": {}, "sort": {}}

        async def _maybe_fuse_text_to_image(**_kwargs):
            return _outcome() if visual else None

        service._retrieve_context = _retrieve_context
        service._maybe_fuse_text_to_image = _maybe_fuse_text_to_image
        return service

    def _ask(self, service, message="encontra estatuas brancas"):
        request = self._request(message, museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga")
        return asyncio.run(service.handle_message(request))

    def test_cards_carry_the_look_and_caption_only_pieces_are_judged_apart(self) -> None:
        llm = _FinalLLM("Encontrei duas estátuas brancas.")
        pre = _TwoLLM({"cartoes": ["doc_3"]}, {"doc_1": "sim", "doc_2": "não"})
        service = self._search_service(llm, pre)
        response = self._ask(service)
        # o aspeto das quatro da página e das duas primeiras do ramo das legendas, por essa ordem
        self.assertEqual(service.opensearch_gateway.caption_calls, [("caps", ["d1", "d2", "d3", "d4", "c1", "c2"])])
        cards = pre.card_prompts[-1]
        self.assertIn("[doc_4]", cards)
        self.assertNotIn("[doc_5]", cards)
        self.assertIn("Aspeto na fotografia: objeto: modelo de estátua; materiais: barro cozido", cards)
        self.assertNotIn("Flora", cards)
        # as que só as legendas encontraram: uma linha cada, com o título e o aspeto
        picks = pre.pick_prompts[-1]
        self.assertIn("«encontra estatuas brancas»", picks)
        self.assertIn(
            "doc_1 | Flora | objeto: escultura; materiais: mármore; cores principais: branco; "
            "outras cores: cinzento; motivos: grinalda, fruta; forma: estátua feminina de pé", picks)
        self.assertIn("doc_2 | Apolo | objeto: busto", picks)
        self.assertNotIn("doc_3 |", picks)
        self.assertNotIn("Diana", picks)
        # primeiro a escolhida entre as da pesquisa, depois a do aspeto
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d3", "c1"])
        self.assertEqual([m.artifact_id for m in response.image_matches], ["d3", "c1"])
        # a resposta final vê os mesmos factos, só das escolhidas
        self.assertIn("cores principais: branco", llm.prompts[0])
        self.assertNotIn("barro cozido", llm.prompts[0])
        selection = service.query_logger.events[-1]["card_selection"]
        self.assertEqual((selection["rule"], selection["model_cards"]), ("preselect", [3]))
        self.assertEqual(
            (selection["candidates"], selection["visual_facts"], selection["visual_candidates"], selection["visual_picked"]),
            (4, 3, 2, 1),
        )
        self.assertEqual(selection["beyond_page"], 1)

    def test_answer_rule_hides_the_request_from_the_final_model(self) -> None:
        llm, pre = _FinalLLM("Encontrei estas peças."), _TwoLLM({"cartoes": ["doc_2"]}, {"doc_1": "sim"})
        self._ask(self._search_service(llm, pre, answer_rule=True))
        prompt = llm.prompts[0]
        self.assertNotIn("estatuas brancas", prompt)
        self.assertNotIn("PEDIDO INTERPRETADO", prompt)
        self.assertIn("PERGUNTA: o visitante descreveu o aspeto das peças que procura", prompt)
        self.assertIn("Não vês o que ele escreveu", prompt)
        self.assertLess(prompt.index("Não vês o que ele escreveu"), prompt.index("CARTÕES CONFIRMADOS:"))
        self.assertIn("cores principais: branco", prompt)
        # a escolha dos cartões continua a ver o pedido
        self.assertIn("encontra estatuas brancas", pre.card_prompts[-1])
        self.assertIn("«encontra estatuas brancas»", pre.pick_prompts[-1])

    def test_answer_rule_only_when_the_cards_carry_the_look(self) -> None:
        cases = (
            dict(answer_rule=False),                                   # flag desligada
            dict(answer_rule=True, visual=False),                      # pesquisa que não é visual
            dict(answer_rule=True, facts=False),                       # sem o aspeto nos cartões
            dict(answer_rule=True, gateway=_CaptionGateway(structs={})),  # legendas em falta
        )
        for kwargs in cases:
            with self.subTest(**{k: v for k, v in kwargs.items() if k != "gateway"}):
                llm, pre = _FinalLLM("Encontrei."), _TwoLLM({"cartoes": ["doc_2"]}, {"doc_1": "sim"})
                self._ask(self._search_service(llm, pre, **kwargs))
                self.assertIn("PERGUNTA: encontra estatuas brancas", llm.prompts[0])
                self.assertNotIn("Não vês o que ele escreveu", llm.prompts[0])

    def test_answer_rule_in_english(self) -> None:
        llm, pre = _FinalLLM("I found these objects."), _TwoLLM({"cartoes": ["doc_2"]}, {"doc_1": "sim"})
        service = self._search_service(llm, pre, answer_rule=True)
        request = self._request("find white statues", museum_slug="mnaa", museum_id="mnaa",
                                museum_name="Museu Nacional de Arte Antiga", language="en")
        asyncio.run(service.handle_message(request))
        self.assertIn("You cannot see what they wrote", llm.prompts[0])
        self.assertNotIn("white statues", llm.prompts[0])

    def test_pieces_picked_by_look_show_even_when_no_search_card_matches(self) -> None:
        llm = _FinalLLM("Encontrei duas estátuas brancas.")
        pre = _TwoLLM({"cartoes": []}, {"doc_1": "Sim", "doc_2": "sim", "doc_9": "sim"})
        service = self._search_service(llm, pre)
        response = self._ask(service)
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["c1", "c2"])
        selection = service.query_logger.events[-1]["card_selection"]
        self.assertEqual((selection["rule"], selection["model_cards"], selection["visual_picked"]), ("preselect", [], 2))
        self.assertIn("Flora", llm.prompts[0])
        self.assertNotIn("Diana", llm.prompts[0])

    def test_nothing_chosen_shows_no_cards(self) -> None:
        llm, pre = _FinalLLM("Não encontrei estátuas brancas."), _TwoLLM({"cartoes": []}, {"doc_1": "não", "doc_2": "não"})
        service = self._search_service(llm, pre)
        response = self._ask(service)
        self.assertEqual(response.artifact_results, [])
        self.assertEqual(service.query_logger.events[-1]["card_selection"]["rule"], "preselect_none")

    def test_look_question_failure_keeps_the_search_choice(self) -> None:
        for picks in (RuntimeError("em baixo"), ["doc_1"], {"doc_1": "talvez"}):
            with self.subTest(picks=picks):
                llm, pre = _FinalLLM("Encontrei uma."), _TwoLLM({"cartoes": ["doc_1"]}, picks)
                response = self._ask(self._search_service(llm, pre))
                self.assertEqual([a.artifact_id for a in response.artifact_results], ["d1"])

    def test_caption_only_candidates_come_after_the_preselection_window(self) -> None:
        llm, pre = _FinalLLM("Encontrei uma."), _TwoLLM({"cartoes": []}, {"doc_1": "sim"})
        service = self._search_service(llm, pre, candidates=6, extra=1)
        response = self._ask(service)
        # janela de 6 (d1…d6) e a primeira das legendas que não está nela
        self.assertEqual(service.opensearch_gateway.caption_calls[0][1], ["d1", "d2", "d3", "d4", "d5", "d6", "c1"])
        self.assertIn("[doc_6]", pre.card_prompts[-1])
        self.assertNotIn("doc_2 |", pre.pick_prompts[-1])
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["c1"])

    def test_piece_without_a_caption_is_not_offered(self) -> None:
        llm, pre = _FinalLLM("Encontrei uma."), _TwoLLM({"cartoes": []}, {"doc_1": "sim", "doc_2": "sim", "doc_3": "sim"})
        service = self._search_service(llm, pre, extra=3)
        response = self._ask(service)
        self.assertNotIn("Danaide", pre.pick_prompts[-1])
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["c1", "c2"])

    def test_flag_off_keeps_the_cards_as_they_were(self) -> None:
        llm, pre = _FinalLLM("Encontrei."), _TwoLLM({"cartoes": ["doc_1"]}, {"doc_1": "sim"})
        service = self._search_service(llm, pre, facts=False)
        response = self._ask(service)
        self.assertEqual(service.opensearch_gateway.caption_calls, [])
        self.assertEqual(pre.pick_prompts, [])
        self.assertNotIn("Aspeto na fotografia", pre.card_prompts[-1])
        self.assertNotIn("Aspeto na fotografia", llm.prompts[0])
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d1"])

    def test_search_that_is_not_visual_never_reads_the_captions(self) -> None:
        llm, pre = _FinalLLM("Encontrei."), _TwoLLM({"cartoes": ["doc_1"]}, {"doc_1": "sim"})
        service = self._search_service(llm, pre, visual=False)
        self._ask(service, "encontra estatuas")
        self.assertEqual(service.opensearch_gateway.caption_calls, [])
        self.assertEqual(pre.pick_prompts, [])
        self.assertNotIn("Aspeto na fotografia", pre.card_prompts[-1])

    def test_no_caption_only_candidates_by_default(self) -> None:
        llm, pre = _FinalLLM("Encontrei."), _TwoLLM({"cartoes": ["doc_2"]}, {"doc_1": "sim"})
        service = self._search_service(llm, pre, extra=0)
        response = self._ask(service)
        self.assertEqual(service.opensearch_gateway.caption_calls[0][1], ["d1", "d2", "d3", "d4"])
        self.assertEqual(pre.pick_prompts, [])
        self.assertIn("barro cozido", pre.card_prompts[-1])
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d2"])

    def test_caption_lookup_failure_leaves_the_search_as_it_was(self) -> None:
        for gateway in (_CaptionGateway(error=RuntimeError("em baixo")), _CaptionGateway(structs={})):
            with self.subTest(error=gateway.error):
                llm, pre = _FinalLLM("Encontrei."), _TwoLLM({"cartoes": ["doc_1"]}, {"doc_1": "sim"})
                service = self._search_service(llm, pre, gateway=gateway)
                response = self._ask(service)
                self.assertEqual(pre.pick_prompts, [])
                self.assertNotIn("Aspeto na fotografia", pre.card_prompts[-1])
                self.assertEqual([a.artifact_id for a in response.artifact_results], ["d1"])


class TestVisualFactsLine(_Base):
    def test_line_from_caption(self) -> None:
        line = ChatService._visual_facts_from_caption({
            "tipo_objeto": " Escultura. ", "materiais": ["mármore", "Mármore", ""], "cores_principais": ["branco"],
            "cores": ["branco", "cinzento"], "motivos": [], "forma": "x " * 80, "descricao": "não entra", "estado": "bom",
        })
        self.assertTrue(line.startswith("objeto: Escultura; materiais: mármore; cores principais: branco; outras cores: cinzento; forma: x x"))
        self.assertTrue(line.endswith("…"))
        self.assertNotIn("não entra", line)
        self.assertEqual(ChatService._visual_facts_from_caption({}), "")

    def test_caption_branch_ranks(self) -> None:
        self.assertEqual(ChatService._caption_branch_ranks(_outcome()), {"d2": 3, "c1": 1, "c2": 2, "c3": 4})
        self.assertEqual(ChatService._caption_branch_ranks(None), {})


class TestVisualIntentFromModel(_Base):
    def _decide(self, reply, *, flag=True):
        service = self._service(self._settings(CHAT_VISUAL_INTENT_MODEL=flag))
        service.preprocess_llm_service = pre = _PreLLM(reply)
        decision = asyncio.run(service._visual_intent_from_model(message="pinturas com barcos", conversation_id="c", query_id="q"))
        return decision, pre

    def test_only_answer_a_turns_the_visual_branch_on(self) -> None:
        decision, pre = self._decide({"resposta": "A"})
        self.assertTrue(decision.use_visual)
        self.assertEqual(decision.rule, "model_visual")
        self.assertIn("«pinturas com barcos»", pre.prompts[0])
        for reply in ({"resposta": "B"}, {"resposta": "C"}, {"outra": 1}, RuntimeError("em baixo")):
            with self.subTest(reply=reply):
                self.assertIsNone(self._decide(reply)[0])

    def test_flag_off_never_asks(self) -> None:
        decision, pre = self._decide({"resposta": "A"}, flag=False)
        self.assertIsNone(decision)
        self.assertEqual(pre.prompts, [])

    def test_rules_first_and_the_model_only_when_they_say_no(self) -> None:
        calls = []

        class _Retrieval:
            async def fuse_text_search(self, **kwargs):
                calls.append(kwargs)
                return "fundido"

        def run(query, message, reply, flag=True):
            service = self._service(self._settings(MULTIMODAL_RETRIEVAL_MODE="intent", CHAT_VISUAL_INTENT_MODEL=flag))
            service.preprocess_llm_service = pre = _PreLLM(reply)
            service._multimodal_text_retrieval = _Retrieval()
            outcome = asyncio.run(service._maybe_fuse_text_to_image(
                query_text=query, message_text=message, museum_slug="mnaa", museum_id="mnaa",
                artifact_docs=[], conversation_id="c", query_id="q"))
            return outcome, pre

        # as regras veem a cor: o modelo não é consultado
        outcome, pre = run("estatuas brancas", "encontra estatuas brancas", {"resposta": "B"})
        self.assertEqual((outcome, pre.prompts, calls[-1]["router_decision"].rule), ("fundido", [], "color_pt"))
        # as regras não veem nada: decide o modelo, com a mensagem do visitante
        outcome, pre = run("quadros barcos", "há quadros com barcos?", {"resposta": "A"})
        self.assertEqual((outcome, calls[-1]["router_decision"].rule), ("fundido", "model_visual"))
        self.assertIn("«há quadros com barcos?»", pre.prompts[0])
        self.assertEqual(calls[-1]["query_text"], "quadros barcos")
        outcome, _ = run("obras de Bosch", "há obras de Bosch?", {"resposta": "B"})
        self.assertIsNone(outcome)
        outcome, pre = run("quadros barcos", "há quadros com barcos?", {"resposta": "A"}, flag=False)
        self.assertEqual((outcome, pre.prompts), (None, []))
