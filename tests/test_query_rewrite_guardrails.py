"""Guardrails da reescrita contextual de queries de retrieval.

A rewritten_query do router so e usada quando resolve contexto de forma
plausivel; reescritas alucinadas, noutra lingua ou com o museu injetado
caem para a mensagem do utilizador. O extrator lexical fica limitado a
EXTRAIR tokens da fonte.
    python -m unittest tests.test_query_rewrite_guardrails
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chat_service import ChatService  # noqa: E402


def _service() -> ChatService:
    return object.__new__(ChatService)


def _state(*user_messages: str) -> SimpleNamespace:
    history = [SimpleNamespace(role="user", text=text) for text in user_messages]
    return SimpleNamespace(history=history)


def _resolve(service, *, original, rewritten, state, museum_name="Museu Nacional do Traje"):
    return service._resolve_contextual_retrieval_query(
        original_query=original,
        rewritten_query=rewritten,
        state=state,
        requested_museum=None,
        museum_slug="mnt",
        museum_id="mnt",
        museum_name=museum_name,
    )


class TestResolveContextualRetrievalQuery(unittest.TestCase):
    def test_follow_up_resolves_with_context(self) -> None:
        service = _service()
        state = _state("Encontra vestidos de seda", "e de linho?")

        query, source = _resolve(
            service,
            original="e de linho?",
            rewritten="vestidos de linho",
            state=state,
        )

        self.assertEqual(query, "vestidos de linho")
        self.assertEqual(source, "router_rewrite")

    def test_hallucinated_rewrite_is_rejected(self) -> None:
        service = _service()
        state = _state("Olá")

        query, source = _resolve(
            service,
            original="Olá",
            rewritten="Qual é o nome do museu?",
            state=state,
        )

        self.assertEqual(query, "Olá")
        self.assertEqual(source, "user_message")

    def test_injected_museum_name_is_stripped(self) -> None:
        service = _service()
        state = _state("vestidos pretos")

        query, source = _resolve(
            service,
            original="vestidos pretos",
            rewritten="vestidos pretos Museu Nacional do Traje",
            state=state,
        )

        self.assertEqual(query, "vestidos pretos")
        self.assertEqual(source, "user_message")

    def test_injected_slug_stripped_but_context_kept(self) -> None:
        service = _service()
        state = _state("encontra vestidos", "e pretos?")

        query, source = _resolve(
            service,
            original="e pretos?",
            rewritten="vestidos pretos mnt",
            state=state,
        )

        self.assertEqual(query, "vestidos pretos")
        self.assertEqual(source, "router_rewrite")

    def test_museum_mentioned_by_user_is_kept(self) -> None:
        service = _service()
        state = _state("vestidos do Museu Nacional do Traje")

        query, _source = _resolve(
            service,
            original="vestidos do Museu Nacional do Traje",
            rewritten="vestidos do Museu Nacional do Traje antigos",
            state=state,
        )

        self.assertIn("Museu Nacional do Traje", query)

    def test_unrelated_language_or_terms_rejected(self) -> None:
        service = _service()
        state = _state("vestidos de seda")

        query, source = _resolve(
            service,
            original="vestidos de seda",
            rewritten="find the silk garments",
            state=state,
        )

        self.assertEqual(query, "vestidos de seda")
        self.assertEqual(source, "user_message")

    def test_empty_rewrite_falls_back(self) -> None:
        service = _service()
        state = _state("brincos")

        query, source = _resolve(
            service,
            original="brincos",
            rewritten="",
            state=state,
        )

        self.assertEqual(query, "brincos")
        self.assertEqual(source, "user_message")

    def test_term_substitution_without_context_is_rejected(self) -> None:
        # 1a mensagem da conversa: o router nao tem contexto que justifique
        # trocar "seda" por "linho" — rejeitar e usar a mensagem original.
        service = _service()
        state = _state("Encontra vestidos de seda")

        query, source = _resolve(
            service,
            original="Encontra vestidos de seda",
            rewritten="vestidos de linho",
            state=state,
        )

        self.assertEqual(query, "Encontra vestidos de seda")
        self.assertEqual(source, "user_message")


class _FakeLLMService:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = 0

    async def generate(self, **_kwargs):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return SimpleNamespace(parsed_json=self.payload)


class TestFollowUpResolver(unittest.TestCase):
    def _service_with_llm(self, payload) -> ChatService:
        service = _service()
        service.llm_service = _FakeLLMService(payload)
        return service

    def test_resolves_elliptical_follow_up(self) -> None:
        import asyncio

        service = self._service_with_llm({"resolved_query": "brincos de prata"})
        state = _state("encontra brincos de ouro", "e de prata?")

        result = asyncio.run(
            service._resolve_follow_up_query_with_llm(
                current_message="e de prata?",
                state=state,
            )
        )

        self.assertEqual(result, "brincos de prata")

    def test_no_previous_user_turns_skips_llm(self) -> None:
        import asyncio

        service = self._service_with_llm({"resolved_query": "nunca chamado"})
        state = _state("e de prata?")

        result = asyncio.run(
            service._resolve_follow_up_query_with_llm(
                current_message="e de prata?",
                state=state,
            )
        )

        self.assertEqual(result, "")
        self.assertEqual(service.llm_service.calls, 0)

    def test_llm_error_returns_empty(self) -> None:
        import asyncio

        service = self._service_with_llm(RuntimeError("llm down"))
        state = _state("encontra brincos de ouro", "e de prata?")

        result = asyncio.run(
            service._resolve_follow_up_query_with_llm(
                current_message="e de prata?",
                state=state,
            )
        )

        self.assertEqual(result, "")

    def test_resolver_output_still_subject_to_guardrails(self) -> None:
        # Output do resolver com termos fora do vocabulario do utilizador
        # e rejeitado pelo mesmo guardrail de subset.
        service = _service()
        state = _state("encontra brincos de ouro", "e de prata?")

        query, source = _resolve(
            service,
            original="e de prata?",
            rewritten="colares de diamantes",
            state=state,
        )

        self.assertEqual(query, "e de prata?")
        self.assertEqual(source, "user_message")


class TestStripUnrequestedMuseumTerms(unittest.TestCase):
    def test_strips_full_name_and_dangling_preposition(self) -> None:
        service = _service()
        result = service._strip_unrequested_museum_terms(
            text="vestidos no Museu Nacional do Traje",
            original="vestidos",
            museum_slug="mnt",
            museum_id="mnt",
            museum_name="Museu Nacional do Traje",
        )
        self.assertEqual(result, "vestidos")

    def test_strips_slug(self) -> None:
        service = _service()
        result = service._strip_unrequested_museum_terms(
            text="vestidos pretos mnt",
            original="vestidos pretos",
            museum_slug="mnt",
            museum_id="mnt",
            museum_name="Museu Nacional do Traje",
        )
        self.assertEqual(result, "vestidos pretos")

    def test_keeps_terms_present_in_original(self) -> None:
        service = _service()
        result = service._strip_unrequested_museum_terms(
            text="azulejos do MNAZ",
            original="azulejos do mnaz",
            museum_slug="mnaz",
            museum_id="mnaz",
            museum_name="Museu Nacional do Azulejo",
        )
        self.assertEqual(result, "azulejos do MNAZ")


class TestFilterLexicalToSource(unittest.TestCase):
    def test_keeps_subset_tokens(self) -> None:
        service = _service()
        self.assertEqual(
            service._filter_lexical_to_source("túmulos de reis", "há túmulos de reis?"),
            "túmulos de reis",
        )

    def test_drops_injected_museum(self) -> None:
        service = _service()
        self.assertEqual(
            service._filter_lexical_to_source("vestidos pretos mnt", "vestidos pretos"),
            "vestidos pretos",
        )

    def test_invented_query_becomes_empty(self) -> None:
        service = _service()
        self.assertEqual(
            service._filter_lexical_to_source("nome do museu", "Olá"),
            "",
        )


if __name__ == "__main__":
    unittest.main()
