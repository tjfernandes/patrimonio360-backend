"""Peça implícita: a pergunta resolve-se para a peça certa sem botões nem
etiquetas. Frases de afinação (TUNE) e de validação (VALID, escritas antes de
afinar). Regra: `ChatService._resolve_implicit_artifact`.
"""
from __future__ import annotations

import asyncio
import unittest

from app.schemas.chat import ChatMessageResponse
from tests.test_tour_location import _Base

VISIBLE = [
    {"artifact_id": "a1", "inventory_number": "19663", "title": "Barrete de fumador/Masculino"},
    {"artifact_id": "a2", "inventory_number": "5622", "title": "Vestido (duas peças)/Feminino"},
    {"artifact_id": "a3", "inventory_number": "28519/1", "title": "Vestido/Criança"},
    {"artifact_id": "a4", "inventory_number": "26510", "title": "Retrato de família"},
    {"artifact_id": "a5", "inventory_number": "5262", "title": "Leque/Feminino"},
]
FIRST = {**VISIBLE[0], "source": "first_result"}
HOTSPOT = {"artifact_id": "h1", "inventory_number": "11459", "title": "Retrato de senhora", "source": "tour_hotspot"}

# (mensagem, artifact_id esperado ou None, resolved_by esperado ou None)
TUNE = [
    ("Fala-me do segundo", "a2", "ordinal"), ("E o terceiro?", "a3", "ordinal"), ("O último é de quando?", "a5", "ordinal"),
    ("the second one", "a2", "ordinal"), ("E o 5622?", "a2", "inventory"), ("Fala-me do 28519/1", "a3", "inventory"),
    ("26510", "a4", "inventory"), ("O retrato de família é de quando?", "a4", "title"), ("Esse vestido é de seda?", "a2", "title"),
    ("E o leque, de que é feito?", "a5", "title"), ("De que é feito?", "a1", "context"), ("Quem o usava?", "a1", "context"),
    ("Mostra peças parecidas", "a1", "context"), ("Conta-me a história", "a1", "context"), ("Quero saber mais", "a1", "context"),
    ("Encontra chapéus de mulher", None, None), ("Vestidos de noiva", None, None), ("Encontra o vestido de noiva", None, None),
    ("Retratos de família", None, None), ("Onde estou?", None, None), ("Que salas existem?", None, None), ("Olá", None, None),
    ("E o 12045?", None, None), ("Peças de 1850", None, None), ("Qual é o horário do museu?", None, None),
    ("O que é um tournure?", None, None), ("Quem foi o marquês de Angeja?", None, None),
]
VALID = [
    ("E a primeira?", "a1", "ordinal"), ("Fala do penúltimo", "a4", "ordinal"), ("O 3.º é de que época?", "a3", "ordinal"),
    ("E o 19663, quem o fez?", "a1", "inventory"), ("Número 5262", "a5", "inventory"), ("Do que é feito o barrete?", "a1", "title"),
    ("Esse leque é pintado?", "a5", "title"), ("E o vestido de criança?", "a3", "title"), ("De que ano é?", "a1", "context"),
    ("Tem assinatura?", "a1", "context"), ("Há mais parecidos?", "a1", "context"), ("É de seda?", "a1", "context"),
    ("Procura leques", None, None), ("Mostra retratos", None, None), ("E sapatos?", None, None), ("Obrigado", None, None),
    ("Leva-me ao pátio", None, None), ("Quero ver pinturas do século XIX", None, None), ("Há vestidos de 1900?", None, None),
    ("Trajes de criança", None, None),
]


class TestImplicitArtifact(_Base):
    def _resolve(self, message: str, *, visible=VISIBLE, context=FIRST):
        metadata = {"visible_artifacts": visible, "context_artifact": context}
        found = self._service()._resolve_implicit_artifact(message, metadata, museum_slug="mnt", museum_id="mnt")
        return (found or {}).get("artifact_id"), (found or {}).get("resolved_by")

    def _check(self, cases) -> None:
        wrong = [(m, self._resolve(m), (a, how)) for m, a, how in cases if self._resolve(m) != (a, how)]
        self.assertEqual(wrong, [])

    def test_tune(self) -> None:
        self._check(TUNE)

    def test_valid(self) -> None:
        self._check(VALID)

    def test_hotspot_context_wins_for_questions_without_subject(self) -> None:
        self.assertEqual(self._resolve("De que é feito?", context=HOTSPOT), ("h1", "context"))
        # uma referência explícita a um cartão continua a mandar
        self.assertEqual(self._resolve("E o segundo?", context=HOTSPOT), ("a2", "ordinal"))

    def test_without_cards_or_context_nothing_is_resolved(self) -> None:
        self.assertEqual(self._resolve("De que é feito?", visible=[], context=None), (None, None))


class TestImplicitArtifactWiring(_Base):
    def _run(self, flag: bool, message: str):
        service = self._service(self._settings(CHAT_IMPLICIT_ARTIFACT_CONTEXT=flag))
        seen = {}

        async def fake(payload, *, status_cb=None):
            seen["metadata"] = payload.metadata
            return ChatMessageResponse(reply="ok")

        service._handle_message_with_selection = fake
        request = self._request(message)
        request.metadata.update({"visible_artifacts": VISIBLE, "context_artifact": FIRST})
        response = asyncio.run(service.handle_message(request))
        return response, seen["metadata"]

    def test_resolved_piece_is_injected_and_reported(self) -> None:
        response, metadata = self._run(True, "Fala-me do segundo")
        self.assertEqual(metadata["selected_artifact"]["artifact_id"], "a2")
        self.assertEqual(response.subject_artifact.artifact_id, "a2")
        self.assertEqual(response.subject_artifact.resolved_by, "ordinal")

    def test_new_search_goes_without_piece(self) -> None:
        response, metadata = self._run(True, "Encontra chapéus de mulher")
        self.assertNotIn("selected_artifact", metadata)
        self.assertIsNone(response.subject_artifact)

    def test_flag_off(self) -> None:
        response, metadata = self._run(False, "De que é feito?")
        self.assertNotIn("selected_artifact", metadata)
        self.assertIsNone(response.subject_artifact)


if __name__ == "__main__":
    unittest.main()
