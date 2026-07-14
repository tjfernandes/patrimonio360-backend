"""Router de intenção visual (Fase 3, Etapa 6) — PT e EN, determinístico.

    python -m unittest tests.test_visual_intent
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.retrieval.visual_intent import (  # noqa: E402
    TEXT_AND_VISUAL,
    TEXT_ONLY,
    decide_visual_intent,
)


class TestModes(unittest.TestCase):
    def test_always_never_blocks(self) -> None:
        decision = decide_visual_intent("quem foi o diretor do museu?", mode="always")
        self.assertEqual(decision.decision, TEXT_AND_VISUAL)
        self.assertTrue(decision.use_visual)
        self.assertEqual(decision.rule, "mode_always")

    def test_off_is_defensive_text_only(self) -> None:
        decision = decide_visual_intent("peças azuis", mode="off")
        self.assertEqual(decision.decision, TEXT_ONLY)
        self.assertFalse(decision.use_visual)


class TestVisualQueriesPT(unittest.TestCase):
    VISUAL = [
        "mostra-me peças parecidas com esta",
        "objetos com flores",
        "peças azuis",
        "imagens de santos",
        "objetos com formato circular",
        "azulejos com padrões geométricos",
        "pratos com decoração floral",
        "esculturas douradas",
        "painéis com motivos vegetalistas",
        "objetos visualmente semelhantes a um cálice",
    ]

    def test_visual_queries_activate_branch(self) -> None:
        for query in self.VISUAL:
            decision = decide_visual_intent(query, mode="intent")
            self.assertTrue(decision.use_visual, f"{query!r} -> {decision}")
            self.assertEqual(decision.decision, TEXT_AND_VISUAL)
            self.assertTrue(decision.reason)
            self.assertGreater(decision.confidence, 0)


class TestFactualQueriesPT(unittest.TestCase):
    FACTUAL = [
        "quem foi Amadeo de Souza-Cardoso?",
        "quando foi fundado o museu?",
        "quantas peças tem a coleção?",
        "qual o horário de abertura?",
        "onde nasceu o autor da peça?",
        "biografia de Rafael Bordalo Pinheiro",
        "qual o preço do bilhete?",
        "qual é o número de inventário desta peça?",
    ]

    def test_factual_queries_do_not_activate_branch(self) -> None:
        for query in self.FACTUAL:
            decision = decide_visual_intent(query, mode="intent")
            self.assertFalse(decision.use_visual, f"{query!r} -> {decision}")
            self.assertEqual(decision.decision, TEXT_ONLY)


class TestEnglishQueries(unittest.TestCase):
    def test_visual_english(self) -> None:
        for query in [
            "show me similar pieces",
            "blue ceramic vases",
            "objects with floral patterns",
            "images of saints",
            "round shaped objects",
        ]:
            self.assertTrue(
                decide_visual_intent(query, mode="intent").use_visual, query
            )

    def test_factual_english(self) -> None:
        for query in [
            "who painted this artwork?",
            "when was the museum founded?",
            "how many pieces are in the collection?",
            "what are the opening hours?",
        ]:
            self.assertFalse(
                decide_visual_intent(query, mode="intent").use_visual, query
            )


class TestEdgeCases(unittest.TestCase):
    def test_mixed_factual_with_strong_visual_signal_activates(self) -> None:
        # Sinal visual forte coexiste com marcador factual -> visual vence.
        decision = decide_visual_intent(
            "quem fez os azulejos azuis com padrões florais?", mode="intent"
        )
        self.assertTrue(decision.use_visual)

    def test_weak_show_me_with_factual_stays_text_only(self) -> None:
        decision = decide_visual_intent(
            "mostra-me quando foi fundado o museu", mode="intent"
        )
        self.assertFalse(decision.use_visual)

    def test_neutral_query_defaults_to_text_only(self) -> None:
        decision = decide_visual_intent("colchas de seda do museu", mode="intent")
        self.assertFalse(decision.use_visual)
        self.assertEqual(decision.rule, "default_no_visual_signal")

    def test_decision_payload_shape(self) -> None:
        payload = decide_visual_intent("peças azuis", mode="intent").as_dict()
        self.assertEqual(
            set(payload), {"decision", "use_visual", "reason", "rule", "confidence"}
        )


if __name__ == "__main__":
    unittest.main()
