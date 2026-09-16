"""Vocabulário leigo -> catálogo (CHAT_LAY_VOCABULARY_EXPANSION).

    python -m unittest tests.test_lay_vocabulary
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.opensearch_client import (  # noqa: E402
    NO_STEM_PHRASE_FIELDS,
    OpenSearchGateway,
    _LAY_VOCABULARY_PATH,
    _load_lay_vocabulary_rules,
)


def _gateway(enabled: bool) -> OpenSearchGateway:
    return OpenSearchGateway(Settings(_env_file=None, CHAT_LAY_VOCABULARY_EXPANSION=enabled,
                                      QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False))


def _rule_ids(gateway: OpenSearchGateway, museum: str, text: str) -> list[str]:
    return [m["rule"] for m in gateway.matched_lay_vocabulary(museum_slug=museum, museum_id=museum,
                                                              query_text=text, lexical_query=None)]


def _lexical_should(body: dict) -> list[dict]:
    found: list[list[dict]] = []

    def visit(node):
        if isinstance(node, dict):
            inner = node.get("bool")
            if isinstance(inner, dict) and isinstance(inner.get("should"), list) and any(
                isinstance(c, dict) and "multi_match" in c for c in inner["should"]
            ):
                found.append(inner["should"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(body)
    assert len(found) == 1, found
    return found[0]


def _body(gateway: OpenSearchGateway, museum: str, text: str) -> dict:
    return gateway._build_query_body(museum_slug=museum, museum_id=museum, query_text=text, lexical_query=text,
                                     query_embedding=[0.1, 0.2], top_k=8, filters={}, sort={})


class TestRulesFile(unittest.TestCase):
    def test_every_rule_documents_evidence_and_is_museum_scoped(self) -> None:
        payload = json.loads(_LAY_VOCABULARY_PATH.read_text(encoding="utf-8"))
        self.assertTrue(payload["rules"])
        for rule in payload["rules"]:
            self.assertTrue(rule.get("museums"), rule["id"])
            self.assertTrue(str(rule.get("evidencia") or "").strip(), rule["id"])
        self.assertEqual(len(_load_lay_vocabulary_rules()), len(payload["rules"]))


class TestMatching(unittest.TestCase):
    def test_real_study_queries_trigger_expected_rules(self) -> None:
        gateway = _gateway(True)
        self.assertEqual(_rule_ids(gateway, "mnsr", "consegues encontrar-me um túmulo de pedra com gravuras de pessoas na lateral"),
                         ["mnsr-sarcofago"])
        self.assertEqual(_rule_ids(gateway, "mnsr", "estou à procura de um caixão em pedra com imagens humanas esculpidas"),
                         ["mnsr-sarcofago"])
        self.assertEqual(_rule_ids(gateway, "mnsr", "onde posso encontrar este jarro com bacia a condizer em metal?"),
                         ["mnsr-gomil-e-bacia"])

    def test_directional_and_conservative(self) -> None:
        gateway = _gateway(True)
        # Sem «pedra» o túmulo pode ser a pintura «Deposição no túmulo».
        self.assertEqual(_rule_ids(gateway, "mnsr", "quadros com um túmulo"), [])
        # Prato não é bacia: «jarro e prato» é outra peça do catálogo.
        self.assertEqual(_rule_ids(gateway, "mnsr", "jarro e prato de faiança"), [])
        self.assertEqual(_rule_ids(gateway, "mnsr", "gomil"), [])
        # Regras presas ao museu onde o termo de catálogo existe.
        self.assertEqual(_rule_ids(gateway, "mnt", "túmulo de pedra"), [])
        self.assertEqual(_rule_ids(gateway, "mnsr", "tumulos de PEDRA"), ["mnsr-sarcofago"])

    def test_flag_off_never_matches(self) -> None:
        self.assertEqual(_rule_ids(_gateway(False), "mnsr", "túmulo de pedra"), [])


class TestQueryBody(unittest.TestCase):
    def test_expansion_adds_phrase_clauses_to_lexical_branch_only(self) -> None:
        text = "túmulo de pedra com figuras"
        off = _body(_gateway(False), "mnsr", text)
        on = _body(_gateway(True), "mnsr", text)
        off_should, on_should = _lexical_should(off), _lexical_should(on)
        self.assertEqual(on_should[: len(off_should)], off_should)
        added = on_should[len(off_should):]
        self.assertEqual([c["multi_match"]["query"] for c in added], ["sarcófago", "arca tumular"])
        for clause in added:
            self.assertEqual(clause["multi_match"]["type"], "phrase")
            self.assertEqual(clause["multi_match"]["fields"], NO_STEM_PHRASE_FIELDS)
            self.assertLess(clause["multi_match"]["boost"], 4)  # abaixo da phrase da query original
        # O ramo denso não muda.
        self.assertEqual(json.dumps(off["query"]["hybrid"]["queries"][0], sort_keys=True),
                         json.dumps(on["query"]["hybrid"]["queries"][0], sort_keys=True))

    def test_no_rule_keeps_body_identical(self) -> None:
        text = "vestidos de noiva"
        self.assertEqual(_body(_gateway(False), "mnt", text), _body(_gateway(True), "mnt", text))


class TestLogTrace(unittest.TestCase):
    def test_chat_service_reports_applied_rules(self) -> None:
        service = object.__new__(ChatService)
        service.settings = Settings(_env_file=None, CHAT_LAY_VOCABULARY_EXPANSION=True,
                                    QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False)
        service.opensearch_gateway = _gateway(True)
        trace = service._matched_lay_vocabulary_for_log(museum_slug="mnsr", museum_id="mnsr",
                                                         query_text="caixão em pedra", lexical_query=None)
        self.assertEqual([(t["group"], t["rule"]) for t in trace], [("lay_vocabulary", "mnsr-sarcofago")])


if __name__ == "__main__":
    unittest.main()
