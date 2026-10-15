"""O modelo escolhe os cartões (CHAT_MODEL_SELECTS_CARDS): bloco final
«[[cartões: doc_N, …]]» retirado do texto e dos tokens, cartões reduzidos aos
que ele usou (pela ordem dele) e, num pedido por nome ou n.º, só as peças que
correspondem.

    python -m unittest tests.test_model_selects_cards
"""
from __future__ import annotations

import asyncio
import unittest

from app.prompts.chat_prompts import build_final_answer_prompt_v2
from app.schemas.chat import ArtifactResult, ImageMatchResult, ResponseFormatObject, TourNavigationTarget
from app.services.chat_service import ChatService
from app.services.llm_service import LLMResponse
from tests.test_tour_location import _Base, _Gateway

MNAZ_DOCS = [
    {"artifact_id": "d1", "inventory_number": "MC.AZU.0490", "title": "Martírio de São Lourenço", "in_tour": True},
    {"artifact_id": "d2", "inventory_number": "MNAz 593 Proj", "title": "Martírio de São Sebastião"},
    {"artifact_id": "d3", "inventory_number": "MNAz 731 Proj", "title": "Martírio de São Sebastião"},
    {"artifact_id": "d4", "inventory_number": "MNAz 1025/12 Cer", "title": "Martírio"},
    {"artifact_id": "d5", "inventory_number": "MNAz 1690 Az", "title": "Fuga para o Egipto", "in_tour": True},
    {"artifact_id": "d6", "inventory_number": "MNAz 865 Az", "title": "Fuga para o Egipto"},
    {"artifact_id": "d7", "inventory_number": "MNAz 158 Esc", "title": "Regresso do Egipto"},
    {"artifact_id": "d8", "inventory_number": "MNAz 730 Az", "title": "Cena de Caça"},
]
for _doc in MNAZ_DOCS:
    _doc.setdefault("museum_id", "mnaz")
    _doc.setdefault("in_tour", False)
    _doc.setdefault("image_count", 1)
    _doc.setdefault("category", "Azulejaria")


def _cards(docs=MNAZ_DOCS) -> list[ArtifactResult]:
    return [ArtifactResult(**{k: v for k, v in d.items() if k in ArtifactResult.model_fields}) for d in docs]


def _images(docs=MNAZ_DOCS) -> list[ImageMatchResult]:
    return [ImageMatchResult(original_image_name=f"{d['artifact_id']}.jpg", artifact_id=d["artifact_id"],
                             inventory=d["inventory_number"]) for d in docs]


def _targets() -> list[TourNavigationTarget]:
    return [
        TourNavigationTarget(overlay_id="ov-490", panorama_key="pan_a", inventory_id="MC.AZU.0490"),
        TourNavigationTarget(overlay_id="ov-1690", panorama_key="pan_b", inventory_id="MNAz 1690 Az"),
    ]


class TestSplitCardSelection(unittest.TestCase):
    split = staticmethod(ChatService._split_card_selection)

    def test_block_is_removed_and_labels_kept_in_order(self) -> None:
        self.assertEqual(self.split("Encontrei duas peças.\n\n[[cartões: doc_2, doc_5]]"), ("Encontrei duas peças.", [2, 5]))
        self.assertEqual(self.split("Found two.\n[[cards: doc_5; doc_2; doc_5]]"), ("Found two.", [5, 2]))
        self.assertEqual(self.split("Texto. [[cartoes: 4, 1]]"), ("Texto.", [4, 1]))

    def test_none_and_missing(self) -> None:
        self.assertEqual(self.split("Texto. [[cartões: nenhum]]"), ("Texto.", []))
        self.assertEqual(self.split("Texto. [[cards: none]]"), ("Texto.", []))
        self.assertEqual(self.split("Texto sem bloco"), ("Texto sem bloco", None))
        self.assertEqual(self.split(""), ("", None))

    def test_truncated_block_still_counts(self) -> None:
        self.assertEqual(self.split("Texto.\n[[cartões: doc_3, doc_"), ("Texto.", [3]))

    def test_inventory_inside_the_block_is_not_a_card(self) -> None:
        self.assertEqual(self.split("Texto. [[cartões: doc_1 (MNAz 12 Az)]]"), ("Texto.", [1]))


class TestStreamFilter(unittest.TestCase):
    def _run(self, deltas: list[str]) -> list[str]:
        out: list[str] = []

        async def emit(delta: str) -> None:
            out.append(delta)

        filtered = ChatService._card_selection_stream_factory(lambda: emit)()

        async def go():
            for delta in deltas:
                await filtered(delta)

        asyncio.run(go())
        return out

    def test_block_never_reaches_the_client(self) -> None:
        out = self._run(["Encontrei ", "a peça.", "\n\n[[", "cartões: doc_1", "]]"])
        self.assertEqual("".join(out), "Encontrei a peça.\n\n")
        self.assertNotIn("[[", "".join(out))

    def test_single_bracket_is_held_until_resolved(self) -> None:
        self.assertEqual("".join(self._run(["ver [", "1] e [", "[cartões: doc_1]]"])), "ver [1] e ")

    def test_long_non_block_text_is_released(self) -> None:
        out = "".join(self._run(["[[nota]] " + "x" * 250]))
        self.assertTrue(out.startswith("[[nota]] "))


class TestNamedInQuery(unittest.TestCase):
    def setUp(self) -> None:
        self.service = object.__new__(ChatService)
        self.cards = _cards()

    def named(self, message: str, request: dict | None = None) -> list[str]:
        return self.service._artifacts_named_in_query(
            user_message=message, artifact_results=self.cards, retrieval_request=request or {"kind": "text"},
        )

    def test_by_title_with_tour_scope_and_typos(self) -> None:
        self.assertEqual(self.named("encontra o martirio de sao lourenço nesta visita"), ["d1"])
        self.assertEqual(self.named("encontra a fuga para o egito"), ["d5", "d6"])
        self.assertEqual(self.named("Onde está a Fuga para o Egipto?"), ["d5", "d6"])
        self.assertEqual(self.named("find the martírio de são lourenço"), ["d1"])

    def test_generic_requests_name_nothing(self) -> None:
        self.assertEqual(self.named("Mostra-me painéis com cenas de caça"), [])
        self.assertEqual(self.named("mostra-me azulejos"), [])
        self.assertEqual(self.named("quem pintou a fuga para o egito?"), [])
        self.assertEqual(self.named("encontra o martirio"), [])   # uma palavra só não é um nome
        self.assertEqual(self.named("peças de 1690"), [])

    def test_by_inventory(self) -> None:
        self.assertEqual(self.named("encontra MNAz 1690 Az"), ["d5"])
        self.assertEqual(self.named("qualquer coisa", {"kind": "inventory", "inventory_candidates": ["mc.azu.0490"]}), ["d1"])


class TestSelectCards(unittest.TestCase):
    def setUp(self) -> None:
        self.service = object.__new__(ChatService)

    def select(self, message: str, indices, *, total=20, has_more=True, request=None, reply="", docs=MNAZ_DOCS):
        return self.service._select_cards(
            user_message=message, retrieval_request=request or {"kind": "text"},
            artifact_results=_cards(docs), image_matches=_images(docs), navigation_targets=_targets(),
            model_indices=indices, results_total=total, results_has_more=has_more, reply_text=reply,
        )

    # medido em produção a 2026-09-30: «encontra botões com aranhas», bloco [2, 5],
    # texto a descrever o doc_2 (11472) e o doc_6 (13913)
    BOTOES = [
        {"artifact_id": f"b{i}", "inventory_number": inv, "title": title, "museum_id": "mnt", "in_tour": False, "image_count": 1}
        for i, (inv, title) in enumerate([
            ("1704", "Botão"), ("11472", "Botões (dois)"), ("21429", "Botão"), ("14988", "Botões (três)"),
            ("13682", "Cartela com botões"), ("13913", "Botões (oito)"), ("10288", "Botão"), ("8250", "Botão"),
        ], start=1)
    ]
    BOTOES_REPLY = (
        "Encontrei dois botões.\n1. Botões (dois) – Inventário: 11472. Dois botões facetados de tartaruga.\n"
        "2. Botões (oito) – Inventário: 13913. Oito botões de plástico transparente."
    )

    def test_pieces_cited_in_the_text_win_over_a_wrong_block(self) -> None:
        artifacts, images, _t, total, has_more, report = self.select(
            "encontra botões com aranhas", [2, 5], reply=self.BOTOES_REPLY, docs=self.BOTOES,
        )
        self.assertEqual([a.inventory_number for a in artifacts], ["11472", "13913"])
        self.assertEqual([m.inventory for m in images], ["11472", "13913"])
        self.assertEqual((total, has_more, report["rule"]), (20, True, "cited"))
        self.assertEqual((report["cited_artifacts"], report["block_only_dropped"]), (2, 1))

    def test_citation_by_title_and_mention_order(self) -> None:
        reply = "Destaco a Cena de Caça (MNAz 730 Az) e ainda o Martírio de São Lourenço."
        artifacts, _i, _t, _total, _more, report = self.select("Mostra-me painéis", [1, 8], reply=reply)
        self.assertEqual([a.artifact_id for a in artifacts], ["d8", "d1"])
        self.assertEqual(report["rule"], "cited")

    def test_single_word_title_is_not_a_citation(self) -> None:
        # «Martírio» (d4) aparece em qualquer resposta sobre martírios; só conta pelo n.º
        reply = "O martírio aparece em duas peças: MNAz 593 Proj e MNAz 1025/12 Cer."
        artifacts, *_rest, report = self.select("Mostra-me martírios", None, reply=reply)
        self.assertEqual([a.artifact_id for a in artifacts], ["d2", "d4"])
        self.assertEqual(report["rule"], "cited")

    def test_block_decides_only_without_citations(self) -> None:
        artifacts, *_rest, report = self.select("Mostra-me painéis", [8, 5], reply="Há cenas de caça em duas peças.")
        self.assertEqual([a.artifact_id for a in artifacts], ["d8", "d5"])
        self.assertEqual(report["rule"], "model")

    def test_model_order_and_paging_kept(self) -> None:
        artifacts, images, targets, total, has_more, report = self.select("Mostra-me painéis com cenas de caça", [8, 5, 8, 99])
        self.assertEqual([a.artifact_id for a in artifacts], ["d8", "d5"])
        self.assertEqual([m.artifact_id for m in images], ["d8", "d5"])
        self.assertEqual([t.inventory_id for t in targets], ["MNAz 1690 Az"])
        self.assertEqual((total, has_more, report["rule"]), (20, True, "model"))

    def test_named_piece_restricts_and_closes_paging(self) -> None:
        artifacts, images, targets, total, has_more, report = self.select("encontra a fuga para o egito", [6, 7])
        self.assertEqual([a.artifact_id for a in artifacts], ["d6", "d5"])   # citada pelo modelo primeiro, d7 fora
        self.assertEqual([m.artifact_id for m in images], ["d6", "d5"])
        self.assertEqual([t.inventory_id for t in targets], ["MNAz 1690 Az"])
        self.assertEqual((total, has_more, report["rule"]), (2, False, "named"))

    def test_named_piece_applies_even_without_a_block(self) -> None:
        artifacts, _i, targets, total, has_more, report = self.select("encontra o martirio de sao lourenço nesta visita", None)
        self.assertEqual([a.artifact_id for a in artifacts], ["d1"])
        self.assertEqual([t.inventory_id for t in targets], ["MC.AZU.0490"])
        self.assertEqual((total, has_more, report["rule"]), (1, False, "named"))

    def test_no_block_and_no_name_keeps_everything(self) -> None:
        for indices in (None, []):
            artifacts, images, targets, total, has_more, report = self.select("Mostra-me painéis com cenas de caça", indices)
            self.assertEqual(len(artifacts), 8)
            self.assertEqual(len(images), 8)
            self.assertEqual(len(targets), 2)
            self.assertEqual((total, has_more, report["rule"]), (20, True, "unchanged"))


class TestPrompt(unittest.TestCase):
    def _prompt(self, select_cards: bool, language: str = "pt") -> str:
        return build_final_answer_prompt_v2(
            museum_name="Museu Nacional do Azulejo", intent="search", mode="rag", rolling_summary="",
            filters_state={}, user_message="encontra a fuga para o egito", rewritten_query="fuga para o egito",
            retrieval_context="[doc_1] Título: Fuga para o Egipto | Inventário: MNAz 1690 Az",
            language=language, select_cards=select_cards,
        )

    def test_instruction_only_when_selecting(self) -> None:
        off, on = self._prompt(False), self._prompt(True)
        self.assertNotIn("[[cartões", off)
        self.assertIn("Os restantes cartões ficam visíveis", off)
        self.assertTrue(on.rstrip().endswith("não a expliques nem a repitas."))
        self.assertIn("[[cartões: nenhum]]", on)
        self.assertNotIn("Os restantes cartões ficam visíveis", on)
        self.assertIn("só ficam visíveis os cartões que indicares", on)
        self.assertIn("[[cards: none]]", self._prompt(True, "en"))


# ----- de ponta a ponta, pelo caminho normal de pesquisa ----- #


class _SearchGateway(_Gateway):
    async def fetch_images_by_artifact_ids(self, *, artifact_ids, **_kwargs):
        by_id = {d["artifact_id"]: d for d in MNAZ_DOCS}
        return [
            {"artifact_id": i, "image_id": f"img-{i}", "local_path": f"Images/x/{i}.jpg",
             "inventory_number": by_id[i]["inventory_number"], "museum_id": "mnaz", "score": 1.0}
            for i in artifact_ids if i in by_id
        ]


class _FinalLLM:
    """A resposta final, por generate ou generate_stream; guarda os prompts."""

    def __init__(self, text: str) -> None:
        self.text, self.prompts = text, []

    async def generate(self, *, message, response_format=ResponseFormatObject(type="text"), **_kwargs):
        self.prompts.append(message)
        return LLMResponse(text=self.text, model="fake", response_format=response_format, finish_reason="stop")

    async def generate_stream(self, *, message, response_format, on_delta=None, **_kwargs):
        self.prompts.append(message)
        for start in range(0, len(self.text), 7):
            await on_delta(self.text[start:start + 7])
        return LLMResponse(text=self.text, model="fake", response_format=response_format, finish_reason="stop")


class TestEndToEnd(_Base):
    POI_BY_MUSEUM = {"mnaz": [
        {"overlayId": "ov-490", "panoramaKey": "pan_a", "inventoryIds": ["MC.AZU.0490"], "location": "Sala A", "title": "Martírio"},
        {"overlayId": "ov-1690", "panoramaKey": "pan_b", "inventoryIds": ["MNAz 1690 Az"], "location": "Sala B", "title": "Fuga"},
    ]}

    def _search_service(self, llm, *, select: bool = True, total: int = 20, stream: bool = False) -> ChatService:
        settings = self._settings(
            CHAT_MODEL_SELECTS_CARDS=select, CHAT_COMPACT_ANSWER_CONTEXT=True, CHAT_ROUTER_MODE="deterministic",
            CHAT_STREAM_FINAL_ANSWER=stream, CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8,
        )
        service = self._service(settings, llm=llm)
        service.opensearch_gateway = _SearchGateway()

        async def _retrieve_context(**_kwargs):
            docs = [dict(d) for d in MNAZ_DOCS]
            return "ctx", total, docs, {"kind": "text", "results_total": total, "filters": {}, "sort": {}}

        service._retrieve_context = _retrieve_context
        return service

    def _ask(self, service, message, status_cb=None):
        request = self._request(message, museum_slug="mnaz", museum_id="mnaz", museum_name="Museu Nacional do Azulejo")
        return asyncio.run(service.handle_message(request, status_cb=status_cb))

    def test_named_piece_in_the_tour(self) -> None:
        llm = _FinalLLM("Encontrei o Martírio de São Lourenço (MC.AZU.0490), que está nesta visita.\n\n[[cartões: doc_1]]")
        service = self._search_service(llm)
        response = self._ask(service, "encontra o martirio de sao lourenço nesta visita")
        self.assertEqual(response.reply, "Encontrei o Martírio de São Lourenço (MC.AZU.0490), que está nesta visita.")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d1"])
        self.assertEqual([m.artifact_id for m in response.image_matches], ["d1"])
        self.assertEqual([t.overlay_id for t in response.navigation_targets], ["ov-490"])
        self.assertFalse(response.results_has_more)
        self.assertEqual(response.results_total, 1)
        self.assertIn("[[cartões: nenhum]]", llm.prompts[0])
        event = service.query_logger.events[-1]
        self.assertEqual([a["artifact_id"] for a in event["shown_artifacts"]], ["d1"])
        self.assertEqual(len(event["retrieved_artifacts"]), 8)
        self.assertEqual(event["card_selection"]["rule"], "named")
        self.assertIsNone(event["participant_id"])

    def test_model_choice_keeps_order_and_paging(self) -> None:
        llm = _FinalLLM("Há cenas de caça em duas peças.\n[[cartões: doc_8, doc_5]]")
        response = self._ask(self._search_service(llm), "Mostra-me painéis com cenas de caça")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d8", "d5"])
        self.assertEqual([t.overlay_id for t in response.navigation_targets], ["ov-1690"])
        self.assertTrue(response.results_has_more)
        self.assertEqual(response.results_total, 20)

    def test_without_block_nothing_changes(self) -> None:
        llm = _FinalLLM("Encontrei várias peças.")
        service = self._search_service(llm)
        response = self._ask(service, "Mostra-me painéis com cenas de caça")
        self.assertEqual(len(response.artifact_results), 8)
        self.assertEqual(len(response.navigation_targets), 2)
        self.assertEqual(service.query_logger.events[-1]["card_selection"]["rule"], "unchanged")

    def test_flag_off_keeps_old_prompt_and_cards(self) -> None:
        llm = _FinalLLM("Encontrei várias peças.")
        service = self._search_service(llm, select=False)
        response = self._ask(service, "encontra a fuga para o egito")
        self.assertEqual(len(response.artifact_results), 8)
        self.assertNotIn("[[cartões", llm.prompts[0])
        self.assertNotIn("card_selection", service.query_logger.events[-1])

    def test_streaming_hides_the_block(self) -> None:
        events: list[dict] = []

        async def status_cb(event: dict) -> None:
            events.append(event)

        llm = _FinalLLM("A Fuga para o Egipto está nesta visita.\n\n[[cartões: doc_5, doc_6]]")
        service = self._search_service(llm, stream=True)
        response = self._ask(service, "encontra a fuga para o egito", status_cb=status_cb)
        streamed = "".join(e["delta"] for e in events if e["type"] == "token")
        self.assertEqual(streamed.strip(), "A Fuga para o Egipto está nesta visita.")
        self.assertEqual(response.reply, "A Fuga para o Egipto está nesta visita.")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d5", "d6"])
        self.assertFalse(response.results_has_more)


# ----- encher a página 1 (CHAT_MODEL_SELECTS_CARDS_FILL) ----- #

MORE_DOCS = [
    {"artifact_id": f"m{i}", "inventory_number": f"MNAz {900 + i} Az", "title": title, "museum_id": "mnaz",
     "in_tour": i % 2 == 0, "image_count": 1, "category": "Azulejaria"}
    for i, title in enumerate(
        ["Cena de Caça ao Javali", "Painel de Jardim", "Cena de Caça ao Veado", "Albarrada", "Cena Galante",
         "Caçada Real", "Frontal de Altar", "Padrão de Ponta de Diamante",
         "Montaria", "Macaco Caçador", "Painel Avulso", "Cenas de Caça (par)", "Fonte", "Vaso", "Brasão", "Anjo"],
        start=1,
    )
]


class _PagedGateway(_SearchGateway):
    """Serve os candidatos 9+ a partir de MORE_DOCS e guarda os offsets pedidos."""

    def __init__(self) -> None:
        super().__init__()
        self.page_calls: list[tuple[int, int]] = []

    async def search_relevant_context_page(self, *, from_offset, page_size, **_kwargs):
        self.page_calls.append((from_offset, page_size))
        everything = [dict(d) for d in MNAZ_DOCS] + [dict(d) for d in MORE_DOCS]
        docs = everything[from_offset: from_offset + page_size]
        return type("Page", (), {"results": docs, "total": len(everything), "query_body": {}})()

    async def fetch_images_by_artifact_ids(self, *, artifact_ids, **_kwargs):
        by_id = {d["artifact_id"]: d for d in MNAZ_DOCS + MORE_DOCS}
        return [
            {"artifact_id": i, "image_id": f"img-{i}", "local_path": f"Images/x/{i}.jpg",
             "inventory_number": by_id[i]["inventory_number"], "museum_id": "mnaz", "score": 1.0}
            for i in artifact_ids if i in by_id
        ]


class _ScriptedLLM(_FinalLLM):
    """Uma resposta por chamada, pela ordem: a final e depois as de enchimento."""

    def __init__(self, *texts: str) -> None:
        super().__init__(texts[0])
        self.texts = list(texts)

    async def generate(self, *, message, response_format=ResponseFormatObject(type="text"), **_kwargs):
        self.prompts.append(message)
        text = self.texts[min(len(self.prompts), len(self.texts)) - 1]
        return LLMResponse(text=text, model="fake", response_format=response_format, finish_reason="stop")


class TestParseFillSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.service = object.__new__(ChatService)

    def parse(self, text: str, inventories=None, candidates=None):
        return self.service._parse_fill_selection(text, inventories, candidates)

    def test_cited_pieces_come_first_and_negated_lines_do_not_count(self) -> None:
        cards = _cards()
        text = ("Fuga para o Egipto (Inventário: MNAz 1690 Az) – cena bíblica\n"
                "Cena de Caça (Inventário: MNAz 730 Az) – caçada com cavaleiros\n"
                "Martírio de São Lourenço (MC.AZU.0490) – não corresponde (tema diferente)")
        self.assertEqual(self.parse(text, None, cards), [5, 8])
        # medido a 2026-10-15: «Nenhuma. … Apenas o cartão [doc_6] ("Interior…") … mas não é uma escultura»
        text = 'Nenhuma. Apenas o cartão [doc_6] ("Regresso do Egipto") tem mulheres, mas não é uma escultura.'
        self.assertEqual(self.parse(text, None, cards), [])
        self.assertEqual(self.parse("Nenhuma.", None, cards), [])
        self.assertEqual(self.parse("None.", None, cards), [])
        # título partilhado por dois cartões («Martírio de São Sebastião», d2 e d3): só o n.º decide
        self.assertIsNone(self.parse("Martírio de São Sebastião – cena de martírio", None, cards))
        self.assertEqual(self.parse("Martírio de São Sebastião (MNAz 731 Proj) – cena de martírio", None, cards), [3])
        # com um n.º na linha, um título de outro cartão no motivo não conta
        self.assertEqual(self.parse("Regresso do Egipto (MNAz 158 Esc) – segue-se à fuga para o egipto", None, cards), [7])

    def test_lines_decide_by_inventory_then_block_then_bare_labels(self) -> None:
        inventories = ["MNAz 730 Az", "MNAz 7883 Az", "MNAz 391 Az"]
        # o rótulo está errado (doc_2) mas o n.º copiado é o do 3.º cartão: vale o n.º
        self.assertEqual(self.parse("doc_1 (MNAz 730 Az): não\ndoc_2 (MNAz 391 Az): sim\n[[cartões: doc_2]]", inventories), [3])
        self.assertEqual(self.parse("doc_1 (mnaz 730 az): sim\ndoc_3 (xxx): sim", inventories), [1, 3])
        self.assertEqual(self.parse("doc_1: não\ndoc_2: sim\ndoc_3: sim"), [2, 3])     # sem n.ºs: o rótulo
        self.assertEqual(self.parse("doc_1: no\ndoc_2: yes"), [2])
        self.assertEqual(self.parse("doc_1: não\ndoc_2: não\n[[cartões: nenhum]]"), [])
        self.assertEqual(self.parse("Texto.\n[[cartões: doc_3, doc_2]]"), [3, 2])
        # medido a 2026-10-15: o 9B omitia o «cartões:»
        self.assertEqual(self.parse("[doc_3, doc_6]"), [3, 6])
        self.assertEqual(self.parse("[[doc_7]]"), [7])
        self.assertEqual(self.parse("Nenhum."), [])
        self.assertEqual(self.parse("[[cartões: nenhum]]"), [])

    def test_unreadable_is_none(self) -> None:
        self.assertIsNone(self.parse("Não sei."))
        self.assertIsNone(self.parse(""))
        self.assertIsNone(self.parse("x" * 200 + " doc_2"))


class TestPageOffset(unittest.TestCase):
    offset = staticmethod(ChatService._text_page_offset)

    def test_without_fill_pages_are_contiguous(self) -> None:
        self.assertEqual([self.offset({}, p, 8) for p in (1, 2, 3)], [0, 8, 16])

    def test_after_fill_page_two_starts_after_the_last_candidate_seen(self) -> None:
        request = {"candidates_consumed": 24}
        self.assertEqual([self.offset(request, p, 8) for p in (1, 2, 3)], [0, 24, 32])
        self.assertEqual(self.offset({"candidates_consumed": 8}, 2, 8), 8)
        self.assertEqual(self.offset({"candidates_consumed": "x"}, 2, 8), 8)


class TestFill(_Base):
    POI_BY_MUSEUM = {"mnaz": TestEndToEnd.POI_BY_MUSEUM["mnaz"] + [
        {"overlayId": "ov-902", "panoramaKey": "pan_c", "inventoryIds": ["MNAz 902 Az"], "location": "Sala C", "title": "Jardim"},
    ]}
    _search_service = TestEndToEnd._search_service
    _ask = TestEndToEnd._ask

    def _fill_service(self, llm, *, fill: bool = True, rounds: int = 2, total: int = 24) -> ChatService:
        service = self._search_service(llm, total=total)
        service.settings = self._settings(
            CHAT_MODEL_SELECTS_CARDS=True, CHAT_MODEL_SELECTS_CARDS_FILL=fill, CHAT_MODEL_SELECTS_CARDS_FILL_ROUNDS=rounds,
            CHAT_COMPACT_ANSWER_CONTEXT=True, CHAT_ROUTER_MODE="deterministic", CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8,
        )
        service.opensearch_gateway = _PagedGateway()
        return service

    def test_short_page_is_filled_from_the_next_candidates(self) -> None:
        llm = _ScriptedLLM(
            "Há cenas de caça numa peça: Cena de Caça (MNAz 730 Az).\n[[cartões: doc_8]]",
            "[[cartões: doc_3, doc_1, doc_6]]",     # candidatos 9–16: m3, m1, m6
            "[[cartões: doc_4]]",                   # candidatos 17–24: m12
        )
        service = self._fill_service(llm)
        response = self._ask(service, "Mostra-me painéis com cenas de caça")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d8", "m3", "m1", "m6", "m12"])
        self.assertEqual([m.artifact_id for m in response.image_matches], ["d8", "m3", "m1", "m6", "m12"])
        self.assertEqual(response.reply, "Há cenas de caça numa peça: Cena de Caça (MNAz 730 Az).")
        self.assertFalse(response.results_has_more)            # viu os 24 candidatos
        self.assertEqual(service.opensearch_gateway.page_calls, [(8, 8), (16, 8)])
        self.assertEqual(len(llm.prompts), 3)
        self.assertIn("MAIS CANDIDATOS:", llm.prompts[1])
        self.assertIn("[doc_1] Título: Cena de Caça ao Javali", llm.prompts[1])
        self.assertIn("PERGUNTA: Mostra-me painéis com cenas de caça", llm.prompts[1])
        self.assertNotIn("Cena de Caça ao Javali", llm.prompts[0])
        fill = service.query_logger.events[-1]["card_selection"]["fill"]
        self.assertEqual((fill["rounds"], fill["candidates_seen"], fill["added"], fill["candidates_consumed"]), (2, 16, 4, 24))

    def test_stops_when_the_page_is_full_and_see_more_continues_after_the_last_seen(self) -> None:
        from app.schemas.chat import ChatResultsPageRequest

        llm = _ScriptedLLM(
            "Há cenas de caça em duas peças.\n[[cartões: doc_8, doc_5]]",
            "[[cartões: doc_1, doc_2, doc_3, doc_4, doc_5, doc_6, doc_7, doc_8]]",
        )
        service = self._fill_service(llm, total=40)
        response = self._ask(service, "Mostra-me painéis com cenas de caça")
        self.assertEqual(len(response.artifact_results), 8)
        self.assertEqual([a.artifact_id for a in response.artifact_results][:2], ["d8", "d5"])
        self.assertTrue(response.results_has_more)
        self.assertEqual(len(llm.prompts), 2)                   # uma ronda chegou
        self.assertEqual([t.overlay_id for t in response.navigation_targets], ["ov-1690", "ov-902"])
        page2 = asyncio.run(service.get_results_page(ChatResultsPageRequest(
            museum_slug="mnaz", museum_id="mnaz", language="pt", conversation_id=response.conversation_id,
            results_page=2, results_page_size=8, results_request_id=response.results_request_id,
        )))
        self.assertEqual(service.opensearch_gateway.page_calls[-1], (16, 8))   # a seguir ao último candidato visto
        self.assertEqual([a.artifact_id for a in page2.artifact_results], [f"m{i}" for i in range(9, 17)])

    def test_no_block_in_the_fill_adds_nothing(self) -> None:
        llm = _ScriptedLLM("Há cenas de caça numa peça.\n[[cartões: doc_8]]", "Não sei.", "[[cartões: nenhum]]")
        service = self._fill_service(llm)
        response = self._ask(service, "Mostra-me painéis com cenas de caça")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d8"])
        fill = service.query_logger.events[-1]["card_selection"]["fill"]
        self.assertEqual((fill["rounds"], fill["added"], fill["no_block"]), (2, 0, 1))

    def test_not_applied_without_the_flag_for_named_pieces_or_full_pages(self) -> None:
        llm = _ScriptedLLM("Há cenas de caça numa peça.\n[[cartões: doc_8]]", "[[cartões: doc_1]]")
        service = self._fill_service(llm, fill=False)
        self.assertEqual(len(self._ask(service, "Mostra-me painéis com cenas de caça").artifact_results), 1)
        self.assertEqual(len(llm.prompts), 1)
        llm = _ScriptedLLM("A Fuga para o Egipto.\n[[cartões: doc_5]]", "[[cartões: doc_1]]")
        service = self._fill_service(llm)
        response = self._ask(service, "encontra a fuga para o egito")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["d5", "d6"])
        self.assertEqual(len(llm.prompts), 1)                   # pedido por nome: não enche
        llm = _ScriptedLLM("Várias peças.", "[[cartões: doc_1]]")
        service = self._fill_service(llm)
        self.assertEqual(len(self._ask(service, "Mostra-me painéis").artifact_results), 8)
        self.assertEqual(len(llm.prompts), 1)                   # sem bloco, a página já está cheia


if __name__ == "__main__":
    unittest.main()
