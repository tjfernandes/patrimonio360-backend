"""Tesouros Nacionais: lista por museu na configuração, campo national_treasure
nos cartões e pedidos sobre tesouros na pesquisa (CHAT_NATIONAL_TREASURES).

    python -m unittest tests.test_national_treasures
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.prompts.chat_prompts import build_final_answer_prompt_v2  # noqa: E402
from app.schemas.chat import ArtifactResult  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.national_treasures import (  # noqa: E402
    LIST,
    NO_MATCH_ID,
    QUESTION,
    NationalTreasureRegistry,
    inventory_key,
    parse_treasure_intent,
    strip_treasure_words,
    subject_matches,
)

REAL = NationalTreasureRegistry()   # app/config/national_treasures.json

MNAA_TREASURES = [
    {"artifact_id": "a-1874", "inventory_number": "1874 Pint", "title": "Fuga para o Egipto", "category": "Pintura", "museum_id": "mnaa"},
    {"artifact_id": "a-2", "inventory_number": "2 Pint", "title": "Aparição de Cristo à Virgem", "category": "Pintura", "museum_id": "mnaa"},
    {"artifact_id": "a-740", "inventory_number": "740 Our", "title": "Custódia", "category": "Ourivesaria", "museum_id": "mnaa"},
    {"artifact_id": "a-1915", "inventory_number": "1915 Tec", "title": "Pluvial", "category": "Têxteis", "museum_id": "mnaa"},
    {"artifact_id": "a-51", "inventory_number": "51 Mov", "title": "Cadeira de braços", "category": "Mobiliário", "museum_id": "mnaa"},
]


class TestRegistry(unittest.TestCase):
    def test_real_file(self) -> None:
        self.assertEqual(len(REAL.inventories("mnaa")), 34)
        self.assertEqual(len(REAL.inventories("mnc")), 18)            # «A 3776 a A 3784» são 9
        self.assertIn("A 3780", REAL.inventories("mnc"))
        self.assertTrue(REAL.is_treasure("mnaa", ["1874 Pint"]))
        self.assertFalse(REAL.is_treasure("mnaa", ["1874 Des"]))
        self.assertFalse(REAL.is_treasure("mnaa", ["47 Pint"]))        # outra «Fuga para o Egipto»
        self.assertFalse(REAL.is_treasure("mnt", ["1874 Pint"]))       # outro museu

    def test_cmag_number_needs_the_prefix(self) -> None:
        self.assertTrue(REAL.is_treasure("cmag", ["CMAG 32"]))
        self.assertFalse(REAL.is_treasure("cmag", ["32"]))

    def test_comparison_ignores_case_spaces_and_punctuation(self) -> None:
        self.assertTrue(REAL.is_treasure("mnc", ["A. 3908"]))
        self.assertTrue(REAL.is_treasure("mnc", ["a3908"]))
        self.assertTrue(REAL.is_treasure("mnsr", ["114/34 Pin MNSR"]))
        self.assertTrue(REAL.is_treasure("MNSR", ["114/ 34 pin mnsr"]))
        self.assertEqual(inventory_key("114/ 34 Pin MNSR"), "11434pinmnsr")

    def test_group_is_treasure_if_any_member_is(self) -> None:
        self.assertTrue(REAL.is_treasure("mnaa", ["1 Pint", "2 Pint"]))
        self.assertFalse(REAL.is_treasure("mnaa", [None, ""]))

    def test_file_changes_are_picked_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json"
            path.write_text(json.dumps({"_x": "y", "mnt": ["100"]}), encoding="utf-8")
            reg = NationalTreasureRegistry(path)
            self.assertTrue(reg.is_treasure("mnt", ["100"]))
            path.write_text(json.dumps({"mnt": ["200"]}), encoding="utf-8")
            os.utime(path, (time.time() + 5, time.time() + 5))
            self.assertFalse(reg.is_treasure("mnt", ["100"]))
            self.assertTrue(reg.is_treasure("mnt", ["200"]))

    def test_broken_or_missing_file_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "x.json"
            broken.write_text("{", encoding="utf-8")
            self.assertEqual(NationalTreasureRegistry(broken).inventories("mnaa"), [])
            self.assertEqual(NationalTreasureRegistry(Path(tmp) / "nada.json").inventories("mnaa"), [])


class TestIntent(unittest.TestCase):
    def test_list(self) -> None:
        cases = {
            "tesouros nacionais": (),
            "Que tesouros nacionais há de pintura?": ("pintura",),
            "mostra-me os tesouros nacionais de pintura": ("pintura",),
            "peças classificadas do século XVI": (),
            "quais são os tesouros nacionais desta visita?": (),
            "show me the national treasures": (),
            "tesouros nacionais de ourivesaria importantes": ("ourivesaria",),
        }
        for message, tokens in cases.items():
            intent = parse_treasure_intent(message)
            self.assertEqual((intent.kind, intent.subject_tokens), (LIST, tokens), message)

    def test_question(self) -> None:
        for message in ["isto é tesouro nacional?", "A Fuga para o Egipto é tesouro nacional?", "é um tesouro nacional?",
                        "está classificada?", "Is this a national treasure?", "is the custody a national treasure?"]:
            self.assertEqual(parse_treasure_intent(message).kind, QUESTION, message)

    def test_not_about_treasures(self) -> None:
        for message in ["tesouro de dez sólidos", "mostra-me pinturas", "o que é a Custódia de Belém?", "peças do século XVI"]:
            self.assertIsNone(parse_treasure_intent(message), message)

    def test_strip(self) -> None:
        self.assertEqual(strip_treasure_words("A Fuga para o Egipto é tesouro nacional?"), "A Fuga para o Egipto é ?")
        self.assertEqual(strip_treasure_words("Que tesouros nacionais há de pintura?"), "Que há de pintura?")

    def test_subject(self) -> None:
        self.assertTrue(subject_matches(("pintura",), MNAA_TREASURES[0]))
        self.assertTrue(subject_matches(("pinturas",), MNAA_TREASURES[0]))
        self.assertTrue(subject_matches(("quadros",), MNAA_TREASURES[0]))
        self.assertFalse(subject_matches(("pintura",), MNAA_TREASURES[2]))
        self.assertTrue(subject_matches(("joias",), MNAA_TREASURES[2]))
        self.assertTrue(subject_matches(("tecidos",), MNAA_TREASURES[3]))
        self.assertTrue(subject_matches(("moveis",), MNAA_TREASURES[4]))
        self.assertTrue(subject_matches((), MNAA_TREASURES[4]))
        self.assertFalse(subject_matches(("vidro",), MNAA_TREASURES[0]))


class _Gateway:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []
        self.key_calls: list[tuple] = []

    async def fetch_images_by_artifact_ids(self, **_kwargs):
        return []

    async def fetch_set_object_counts(self, *, set_ids):
        return {s: 0 for s in set_ids}

    async def fetch_artifacts_by_inventory_keys(self, *, museum_id, keys):
        self.key_calls.append((museum_id, tuple(keys)))
        wanted = set(keys)
        return [d for d in MNAA_TREASURES if d["museum_id"] == museum_id and inventory_key(d["inventory_number"]) in wanted]

    async def search_relevant_context_page(self, **kwargs):
        self.search_calls.append(kwargs)
        ids = (kwargs.get("filters") or {}).get("artifact_id")
        docs = [dict(d) for d in MNAA_TREASURES if ids is None or d["artifact_id"] in ids]
        return SimpleNamespace(results=docs, total=len(docs), query_body={})

    async def search_artifacts_by_inventory_candidates(self, **_kwargs):
        return []


class _Embedder:
    async def embed_text(self, _text):
        return [0.1, 0.2]


def _service(**flags) -> ChatService:
    service = object.__new__(ChatService)
    base = dict(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False, CHAT_NATIONAL_TREASURES=True,
                CHAT_COMPACT_ANSWER_CONTEXT=True, CHAT_ENABLE_LLM_LEXICAL_QUERY=False, CHAT_LAY_VOCABULARY_EXPANSION=False,
                CHAT_RETRIEVAL_TOP_K=8, CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8)
    base.update(flags)
    service.settings = Settings(**base)
    service.opensearch_gateway = _Gateway()
    service.embedding_provider = _Embedder()
    service.llm_service = None
    service.tour_navigation_service = SimpleNamespace(resolve_targets=lambda **_k: [], corrected_inventories=lambda **_k: [])
    return service


class TestCards(unittest.TestCase):
    def test_field_on_results_and_groups(self) -> None:
        service = _service(CHAT_NATIONAL_TREASURES=False)   # o campo não depende da flag
        docs = [
            {"artifact_id": "a-1874", "inventory_number": "1874 Pint", "title": "Fuga para o Egipto", "museum_id": "mnaa"},
            {"artifact_id": "a-47", "inventory_number": "47 Pint", "title": "Fuga para o Egipto", "museum_id": "mnaa"},
            {"artifact_id": "g", "inventory_number": "3 Pint", "title": "x", "museum_id": "mnaa",
             "group_size": 2, "group_inventory_numbers": ["3 Pint", "2 Pint"], "group_artifact_ids": ["g", "h"]},
        ]
        results = asyncio.run(service._build_artifact_results(museum_slug="mnaa", museum_id="mnaa", artifact_docs=docs))
        self.assertEqual([r.national_treasure for r in results], [True, False, True])
        self.assertIs(results[0].model_dump(mode="json")["national_treasure"], True)

    def test_compact_card_marks_only_with_the_flag(self) -> None:
        card = ArtifactResult(artifact_id="a", inventory_number="1874 Pint", title="Fuga para o Egipto", museum_id="mnaa", national_treasure=True)
        plain = ArtifactResult(artifact_id="b", inventory_number="47 Pint", title="Fuga para o Egipto", museum_id="mnaa")
        on = _service()
        text = on._format_docs_compact(docs=on._artifact_results_as_prompt_docs([card, plain]))
        self.assertEqual(text.count("Tesouro Nacional: sim"), 1)
        self.assertNotIn("Tesouro Nacional: não", text)
        off = _service(CHAT_NATIONAL_TREASURES=False)
        self.assertNotIn("Tesouro Nacional", off._format_docs_compact(docs=off._artifact_results_as_prompt_docs([card])))

    def test_prompt_rule_only_with_the_flag(self) -> None:
        def prompt(flag):
            return build_final_answer_prompt_v2(museum_name="MNAA", intent="search", mode="rag", rolling_summary="", filters_state={},
                                                user_message="x", rewritten_query="x", retrieval_context="", national_treasures=flag)
        self.assertIn("Só dizes que uma peça é Tesouro Nacional se o cartão indicar «Tesouro Nacional: sim»", prompt(True))
        self.assertNotIn("Tesouro Nacional", prompt(False))


class TestQueryBody(unittest.TestCase):
    def _body(self, filters):
        from app.services.opensearch_client import OpenSearchGateway

        gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ARTIFACT="arts"))
        return gateway._build_query_body(museum_slug="mnaa", museum_id="mnaa", query_text="pintura", lexical_query="pintura",
                                         query_embedding=[0.1, 0.2], top_k=8, filters=filters, sort={})

    @staticmethod
    def _knn_specs(body):
        found = []

        def walk(node):
            if isinstance(node, dict):
                if "knn" in node and isinstance(node["knn"], dict) and "text_embedding" in node["knn"]:
                    found.append(node["knn"]["text_embedding"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(body)
        return found

    def test_closed_list_filters_inside_the_knn(self) -> None:
        specs = self._knn_specs(self._body({"artifact_id": ["a-1874", "a-2"]}))
        self.assertTrue(specs)
        clauses = specs[0]["filter"]["bool"]["filter"]
        self.assertIn({"terms": {"artifact_id": ["a-1874", "a-2"]}}, clauses)
        self.assertIn({"term": {"museum_id": "mnaa"}}, clauses)

    def test_other_searches_keep_the_old_knn(self) -> None:
        for filters in ({}, {"in_tour": True}):
            specs = self._knn_specs(self._body(filters))
            self.assertTrue(specs)
            self.assertNotIn("filter", specs[0])


class TestSearch(unittest.TestCase):
    def _retrieve(self, service, message):
        return asyncio.run(service._retrieve_context(museum_slug="mnaa", museum_id="mnaa", query=message, filters={}, sort={},
                                                     result_window_size=8, inventory_query=message))

    def test_list_filters_by_the_list_and_the_subject(self) -> None:
        service = _service()
        _ctx, total, docs, request = self._retrieve(service, "Que tesouros nacionais há de pintura?")
        call = service.opensearch_gateway.search_calls[0]
        self.assertEqual(sorted(call["filters"]["artifact_id"]), ["a-1874", "a-2"])
        self.assertNotIn("tesouro", call["lexical_query"].lower())
        self.assertEqual(sorted(d["artifact_id"] for d in docs), ["a-1874", "a-2"])
        info = request["national_treasure"]
        self.assertEqual((info["kind"], info["museum_total"], info["matched"], info["subject"]), (LIST, 5, 2, ["pintura"]))
        self.assertEqual(request["filters"]["artifact_id"], call["filters"]["artifact_id"])   # «ver mais» repete o filtro

    def test_list_without_subject_is_the_whole_list(self) -> None:
        service = _service()
        self._retrieve(service, "mostra-me os tesouros nacionais")
        self.assertEqual(len(service.opensearch_gateway.search_calls[0]["filters"]["artifact_id"]), 5)

    def test_list_with_era_keeps_the_temporal_filter(self) -> None:
        service = _service()
        self._retrieve(service, "peças classificadas do século XVI")
        filters = service.opensearch_gateway.search_calls[0]["filters"]
        self.assertEqual(len(filters["artifact_id"]), 5)
        self.assertIn("_temporal_interval", filters)

    def test_nothing_matches_gives_the_sentinel(self) -> None:
        service = _service()
        _ctx, _total, docs, request = self._retrieve(service, "tesouros nacionais de vidro")
        self.assertEqual(service.opensearch_gateway.search_calls[0]["filters"]["artifact_id"], [NO_MATCH_ID])
        self.assertEqual(docs, [])
        self.assertEqual(request["national_treasure"]["matched"], 0)

    def test_question_does_not_filter(self) -> None:
        service = _service()
        _ctx, _t, _docs, request = self._retrieve(service, "A Fuga para o Egipto é tesouro nacional?")
        call = service.opensearch_gateway.search_calls[0]
        self.assertNotIn("artifact_id", call["filters"])
        self.assertNotIn("tesouro", call["lexical_query"].lower())
        self.assertEqual(request["national_treasure"]["kind"], QUESTION)

    def test_flag_off_and_other_questions_untouched(self) -> None:
        service = _service(CHAT_NATIONAL_TREASURES=False)
        _ctx, _t, _d, request = self._retrieve(service, "Que tesouros nacionais há de pintura?")
        self.assertNotIn("artifact_id", service.opensearch_gateway.search_calls[0]["filters"])
        self.assertNotIn("national_treasure", request)
        service = _service()
        _ctx, _t, _d, request = self._retrieve(service, "mostra-me pinturas")
        self.assertNotIn("artifact_id", service.opensearch_gateway.search_calls[0]["filters"])
        self.assertNotIn("national_treasure", request)
        self.assertEqual(service.opensearch_gateway.key_calls, [])

    def test_treasure_docs_are_cached(self) -> None:
        service = _service()
        self._retrieve(service, "tesouros nacionais")
        self._retrieve(service, "tesouros nacionais de pintura")
        self.assertEqual(len(service.opensearch_gateway.key_calls), 1)

    def test_visual_fusion_is_skipped_only_for_the_list(self) -> None:
        self.assertTrue(ChatService._is_treasure_list_request({"national_treasure": {"kind": LIST}}))
        self.assertFalse(ChatService._is_treasure_list_request({"national_treasure": {"kind": QUESTION}}))
        self.assertFalse(ChatService._is_treasure_list_request({}))

    def test_note(self) -> None:
        service = _service()
        note = service._national_treasure_note({"kind": LIST, "museum_total": 34, "matched": 18, "subject": ["pintura"],
                                                "categories": [("Pintura", 18), ("Ourivesaria", 6)]}, "pt")
        self.assertEqual(note, "Este museu tem 18 Tesouros Nacionais de «pintura» (lista oficial: 34 — Pintura 18, Ourivesaria 6). Todos os cartões abaixo são Tesouro Nacional.")
        visible = [ArtifactResult(artifact_id=str(i), national_treasure=True) for i in range(8)]
        note = service._national_treasure_note({"kind": LIST, "museum_total": 34, "matched": 34, "categories": [("Pintura", 18)]}, "pt", visible=visible)
        self.assertEqual(note, "Este museu tem 34 Tesouros Nacionais (lista oficial: 34 — Pintura 18). Os 8 cartões abaixo são alguns deles; ao todo são 34.")
        # «do século XVI»: a época reduz a lista a 25
        note = service._national_treasure_note({"kind": LIST, "museum_total": 34, "matched": 34}, "pt", visible=visible, found=25)
        self.assertEqual(note, "Este museu tem 25 Tesouros Nacionais que correspondem ao pedido (lista oficial: 34). Os 8 cartões abaixo são alguns deles; ao todo são 25.")
        note = service._national_treasure_note({"kind": LIST, "museum_total": 34, "matched": 34}, "pt", visible=[], found=0)
        self.assertTrue(note.startswith("Este museu tem 0 Tesouros Nacionais que correspondem ao pedido") and note.endswith("Não há nenhum a mostrar."))
        self.assertIn("Este museu não tem peças na lista", service._national_treasure_note({"kind": LIST, "museum_total": 0}, "pt"))
        self.assertIn("«Tesouro Nacional: sim»", service._national_treasure_note({"kind": QUESTION, "museum_total": 34}, "pt"))

    def test_question_note_names_the_cards_on_the_list(self) -> None:
        service = _service()
        visible = [
            ArtifactResult(artifact_id="a", inventory_number="14 Pint", title="Fuga para o Egipto", national_treasure=False),
            ArtifactResult(artifact_id="b", inventory_number="1874 Pint", title="Fuga para o Egipto", national_treasure=True),
        ]
        note = service._national_treasure_note({"kind": QUESTION, "museum_total": 34}, "pt", visible=visible)
        self.assertTrue(note.endswith("Dos cartões abaixo, estão na lista: «Fuga para o Egipto» (1874 Pint); os outros não."))
        note = service._national_treasure_note({"kind": QUESTION, "museum_total": 34}, "pt", visible=visible[:1])
        self.assertTrue(note.endswith("Nenhum dos cartões abaixo está na lista."))


if __name__ == "__main__":
    unittest.main()
