"""Autor da peça em foco (CHAT_FOCUSED_ARTIFACT_AUTHOR): com uma peça à frente no
minichat, «Que mais obras existem deste autor?» passa a ser «obras de <autor>».

    python -m unittest tests.test_focused_author
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.core.config import Settings
from app.schemas.chat import ArtifactResult, ChatMessageResponse, ResponseFormatObject
from app.services.author_cards import build_author_directory, other_author_words
from app.services.chat_service import ChatService
from tests.test_author_card import ROWS, _AuthorGateway
from tests.test_tour_location import _Base

PIECE = {"artifact_id": "p1", "inventory_number": "30 Pint", "title": "A Virgem, o Menino e Anjos"}
FOLLOW_UP = "Que mais obras existem deste autor?"


class _Gateway(_AuthorGateway):
    def __init__(self, creators=("Gregório Lopes",), fail: bool = False) -> None:
        super().__init__()
        self.creators, self.fail, self.piece_calls = list(creators), fail, []

    async def fetch_artifacts_by_ids(self, *, artifact_ids, **kwargs):
        self.piece_calls.append(list(artifact_ids))
        if self.fail:
            raise RuntimeError("OpenSearch em baixo")
        return [{"artifact_id": artifact_ids[0], "title": PIECE["title"], "creators": list(self.creators)}]


class _Model:
    def __init__(self, letter: str | None = "A") -> None:
        self.letter, self.prompts = letter, []

    async def generate(self, *, message, **_kwargs):
        self.prompts.append(message)
        if self.letter is None:
            raise RuntimeError("modelo em baixo")
        return SimpleNamespace(parsed_json={"resposta": self.letter})


class TestFocusedAuthor(_Base):
    def _service_with(self, model: _Model, *, flag: bool = True, gateway=None, author_card: bool = False):
        service = self._service(self._settings(CHAT_FOCUSED_ARTIFACT_AUTHOR=flag, CHAT_AUTHOR_CARD=author_card))
        service._preprocess_llm_service = model
        service.opensearch_gateway = gateway or _Gateway()
        service.seen = []

        async def routed(payload, *, status_cb=None):
            service.seen.append(payload)
            return ChatMessageResponse(reply="texto", artifact_results=[ArtifactResult(artifact_id="p2", creator_ids=["a:lopes"])],
                                       response_format=ResponseFormatObject(type="text"))

        service._handle_message_routed = routed
        return service

    def _ask(self, service, message: str = FOLLOW_UP, *, metadata=None, language: str = "pt"):
        metadata = {"context_artifact": dict(PIECE), "visible_artifacts": [dict(PIECE)]} if metadata is None else metadata
        request = self._request(message, museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga",
                                language=language, metadata=metadata)
        return asyncio.run(service.handle_message(request))

    def test_off_by_default(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_FOCUSED_ARTIFACT_AUTHOR)
        model = _Model("A")
        service = self._service_with(model, flag=False)
        self._ask(service)
        self.assertEqual(service.seen[0].message, FOLLOW_UP)
        self.assertEqual((model.prompts, service.opensearch_gateway.piece_calls), ([], []))

    def test_the_request_becomes_the_works_of_the_focused_pieces_author(self) -> None:
        model = _Model("A")
        service = self._service_with(model)
        self._ask(service)
        seen = service.seen[0]
        self.assertEqual(seen.message, "obras de Gregório Lopes")
        # a peça em foco sai do pedido: já não é uma pergunta sobre ela
        self.assertNotIn("context_artifact", seen.metadata)
        self.assertNotIn("visible_artifacts", seen.metadata)
        self.assertEqual(seen.metadata["focused_author_request"],
                         {"artifact_id": "p1", "author": "Gregório Lopes", "message": FOLLOW_UP})
        self.assertEqual(service.opensearch_gateway.piece_calls, [["p1"]])
        self.assertIn(f"«{FOLLOW_UP}»", model.prompts[0])
        self.assertIn("«A Virgem, o Menino e Anjos», de Gregório Lopes", model.prompts[0])

    def test_the_model_sees_the_previous_request(self) -> None:
        # «e japonesas?» depois de «Encontra peças chinesas» continua a pesquisa: o modelo tem de a ver
        from app.services.chat_session_store import ChatTurn

        model = _Model("D")
        service = self._service_with(model)
        state = service.session_store.load_or_create(conversation_id="c1", museum_slug="mnaa")
        state.history.extend([ChatTurn(role="user", text="Encontra peças chinesas"), ChatTurn(role="assistant", text="Aqui estão.")])
        service.session_store.save(state)
        request = self._request("e japonesas?", museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga",
                                conversation_id="c1", metadata={"context_artifact": dict(PIECE)})
        asyncio.run(service.handle_message(request))
        self.assertEqual(service.seen[0].message, "e japonesas?")
        self.assertIn("de Gregório Lopes. O pedido anterior do visitante foi «Encontra peças chinesas». O que pede a mensagem?", model.prompts[0])
        # sem conversa anterior, a frase não vai
        other = _Model("D")
        self._ask(self._service_with(other))
        self.assertIn("de Gregório Lopes. O que pede a mensagem?", other.prompts[0])
        self.assertNotIn("pedido anterior do visitante foi", other.prompts[0])

    def test_in_english_the_request_is_in_english(self) -> None:
        service = self._service_with(_Model("A"))
        self._ask(service, "what else did this artist make?", language="en")
        self.assertEqual(service.seen[0].message, "works by Gregório Lopes")

    def test_the_author_card_follows_the_new_request(self) -> None:
        service = self._service_with(_Model("A"), author_card=True)
        response = self._ask(service)
        self.assertEqual(response.author.name, "Gregório Lopes")

    def test_any_other_answer_leaves_the_request_alone(self) -> None:
        for letter in ("B", "C", "D", "?", None):
            with self.subTest(letter=letter):
                service = self._service_with(_Model(letter))
                self._ask(service)
                self.assertEqual(service.seen[0].message, FOLLOW_UP)
                self.assertIn("context_artifact", service.seen[0].metadata)

    def test_a_message_that_names_another_author_is_never_asked_about(self) -> None:
        model = _Model("A")
        service = self._service_with(model)
        self._ask(service, "Há obras de Domingos Sequeira?")
        self.assertEqual(service.seen[0].message, "Há obras de Domingos Sequeira?")
        self.assertEqual(model.prompts, [])

    def test_only_with_a_focused_piece_and_no_selected_one(self) -> None:
        cases = (
            {},                                                                   # sem peça em foco
            {"context_artifact": dict(PIECE), "selected_artifact": dict(PIECE)},  # pergunta sobre a peça selecionada
            {"context_artifact": dict(PIECE), "intent": "describe_artifact"},     # «Ver na tour»
        )
        for metadata in cases:
            with self.subTest(metadata=sorted(metadata)):
                model = _Model("A")
                service = self._service_with(model)
                self._ask(service, metadata=metadata)
                self.assertEqual(service.seen[0].message, FOLLOW_UP)
                self.assertEqual((model.prompts, service.opensearch_gateway.piece_calls), ([], []))

    def test_a_piece_without_a_known_author_is_never_asked_about(self) -> None:
        for creators in ((), ("Autor desconhecido",), ("Não identificado",)):
            with self.subTest(creators=creators):
                model = _Model("A")
                service = self._service_with(model, gateway=_Gateway(creators=creators))
                self._ask(service)
                self.assertEqual(service.seen[0].message, FOLLOW_UP)
                self.assertEqual(model.prompts, [])

    def test_a_failure_leaves_the_request_as_it_came(self) -> None:
        service = self._service_with(_Model("A"), gateway=_Gateway(fail=True))
        self._ask(service)
        self.assertEqual(service.seen[0].message, FOLLOW_UP)


class TestFocusedAuthorFunctions(unittest.TestCase):
    def test_the_name_as_it_is_said_in_a_request(self) -> None:
        name = ChatService._author_name_for_request
        self.assertEqual(name("Machado de Castro, Joaquim"), "Joaquim Machado de Castro")
        self.assertEqual(name("Keil, Maria (Silves, 1914 - Lisboa 2012)"), "Maria Keil")
        self.assertEqual(name("Gregório Lopes"), "Gregório Lopes")
        self.assertEqual(name("Josefa de Óbidos (1630-1684)"), "Josefa de Óbidos")

    def test_words_of_another_authors_name(self) -> None:
        directory = build_author_directory([*ROWS, {"id": "a:x", "name": "Autor desconhecido", "count": 900},
                                            {"id": "a:s", "name": "António dos Santos", "count": 3}])
        self.assertEqual(other_author_words("Que mais obras existem deste autor?", directory), [])
        self.assertEqual(other_author_words("Há obras de Gregório Lopes?", directory), ["gregorio", "lopes"])
        self.assertEqual(other_author_words("mais obras de Gregório Lopes", directory, except_name="Gregório Lopes"), [])
        # larga de propósito: uma palavra comum que também é apelido conta
        self.assertEqual(other_author_words("mais obras dele com santos", directory), ["santos"])
        self.assertEqual(other_author_words("Leva-me à Sala dos Presépios", directory), [])


if __name__ == "__main__":
    unittest.main()
