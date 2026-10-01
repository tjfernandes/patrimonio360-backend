"""Fast-path de perguntas gerais sobre o museu.

Detecao deterministica (regex ancoradas PT/EN sobre texto folded), override
do router e contexto a partir de app/data/museum_profiles.json.
    python -m unittest tests.test_museum_overview
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chat_service import ChatService, _load_museum_profiles  # noqa: E402


def _service() -> ChatService:
    return object.__new__(ChatService)


MNT = "Museu Nacional do Traje"


class TestMuseumOverviewDetection(unittest.TestCase):
    def test_positive_pt_variants(self) -> None:
        service = _service()
        for text in [
            "Fala-me sobre o museu",
            "fala me do museu",
            "Conta-me mais sobre este museu",
            "O que é este museu?",
            "qual é a história do museu",
            "História deste museu",
            "Apresenta o museu",
            "sobre o museu",
            "que museu é este?",
            "informações sobre o museu",
            "Fala-me sobre o Museu Nacional do Traje",
        ]:
            self.assertTrue(
                service._is_museum_overview_query(text, museum_name=MNT),
                text,
            )

    def test_positive_en_variants(self) -> None:
        service = _service()
        for text in [
            "Tell me about this museum",
            "tell me more about the museum",
            "What is this museum?",
            "history of the museum",
            "about this museum",
        ]:
            self.assertTrue(
                service._is_museum_overview_query(text, museum_name=MNT),
                text,
            )

    def test_monastery_wording_for_mj(self) -> None:
        service = _service()
        self.assertTrue(
            service._is_museum_overview_query(
                "fala-me sobre o mosteiro",
                museum_name="Mosteiro dos Jerónimos",
            )
        )

    def test_negatives_are_not_matched(self) -> None:
        service = _service()
        for text in [
            "vestidos no museu",
            "o museu tem vestidos de seda?",
            "fala-me sobre os vestidos do museu",
            "encontra brincos",
            "Olá",
            "quais as peças mais importantes do museu?",
            "fala-me sobre o museu do louvre",
        ]:
            self.assertFalse(
                service._is_museum_overview_query(text, museum_name=MNT),
                text,
            )


class TestMuseumOverviewOverrideAndContext(unittest.TestCase):
    def test_profiles_load_for_all_museums(self) -> None:
        profiles = _load_museum_profiles()
        self.assertEqual(set(profiles.keys()), {"mnt", "mnaz", "mnsr", "mj", "cmag", "mnaa"})
        for key, profile in profiles.items():
            self.assertTrue(profile.get("name"), key)
            self.assertTrue(profile.get("description"), key)

    def test_override_forces_rag_for_overview(self) -> None:
        service = _service()
        decision = {"mode": "llm_only", "intent": "fallback", "reason": "router"}
        updated = service._maybe_override_for_museum_overview(
            router_decision=decision,
            query_text="fala-me sobre o museu",
            museum_id="mnt",
        )
        self.assertEqual(updated["mode"], "rag")
        self.assertTrue(updated["needs_retrieval"])
        self.assertEqual(updated["intent"], "overview")
        self.assertIn("deterministic_museum_overview", str(updated["reason"]))

    def test_override_noop_for_normal_search(self) -> None:
        service = _service()
        decision = {"mode": "rag", "intent": "search", "reason": "router"}
        updated = service._maybe_override_for_museum_overview(
            router_decision=decision,
            query_text="vestidos de seda",
            museum_id="mnt",
        )
        self.assertEqual(updated, decision)

    def test_fast_path_returns_profile_context(self) -> None:
        service = _service()
        result = service._retrieve_museum_overview_context(
            museum_slug="museu_nacional_do_traje",
            museum_id="mnt",
            raw_query="fala-me sobre o museu",
        )
        self.assertIsNotNone(result)
        context, total, docs, request = result
        self.assertIn("Museu Nacional do Traje", context)
        self.assertIn("Palácio Angeja-Palmela", context)
        self.assertEqual(total, 0)
        self.assertEqual(docs, [])
        self.assertEqual(request["kind"], "museum_overview")

    def test_fast_path_none_for_unknown_museum_or_query(self) -> None:
        service = _service()
        self.assertIsNone(
            service._retrieve_museum_overview_context(
                museum_slug="outro",
                museum_id="xxx",
                raw_query="fala-me sobre o museu",
            )
        )
        self.assertIsNone(
            service._retrieve_museum_overview_context(
                museum_slug="museu_nacional_do_traje",
                museum_id="mnt",
                raw_query="vestidos de seda",
            )
        )


if __name__ == "__main__":
    unittest.main()
