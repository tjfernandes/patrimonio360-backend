"""Ficha do autor (CHAT_AUTHOR_CARD): a resposta leva `author` quando a mensagem diz
o nome de um autor das peças mostradas.

O modelo de pré-processamento aqui é um guião; a redação do prompt mede-se com o
modelo real (tests/author_card_cases.py).

    python -m unittest tests.test_author_card
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.core.config import Settings
from app.prompts.query_planner_prompts import build_author_name_prompt
from app.schemas.chat import ArtifactResult, ChatMessageResponse, ResponseFormatObject, TourNavigationTarget
from app.services.author_cards import (
    author_candidates,
    build_author_directory,
    clean_life_date,
    life_year,
    merged_author_fields,
    name_tokens,
    words_in_message,
)
from app.services.author_portraits import AuthorPortraitStore, absolute_http_url
from app.services.opensearch_client import OpenSearchGateway
from tests.test_tour_location import _Base
from tests.test_tour_object_targets import _WithObjects

ROWS = [
    {"id": "a:sequeira-1", "name": "Sequeira, Domingos António de", "count": 63},
    {"id": "a:sequeira-2", "name": "Sequeira, Domingos António de (1768-1837)", "count": 46},
    {"id": "a:lopes", "name": "Gregório Lopes", "count": 18},
    {"id": "a:queiros", "name": "QUEIRÓS, Gregório Francisco de (1768-1845)", "count": 3},
    {"id": "a:porto", "name": "Silva Porto, António Carvalho de", "count": 47},
    {"id": "a:keil", "name": "Keil, Maria (Silves, 1914 - Lisboa 2012)", "count": 119},
    {"id": "a:reis", "name": "Reis, António Soares dos", "count": 84},
]
AUTHOR_DOCS = {
    "a:sequeira-1": {"entity_id": "a:sequeira-1", "name": "Sequeira, Domingos António de", "activity": "", "birth_date": "",
                     "biography": "", "url": "https://catalogo.example/autor/1"},
    "a:sequeira-2": {"entity_id": "a:sequeira-2", "name": "Sequeira, Domingos António de (1768-1837)", "activity": "Pintor",
                     "birth_date": "1768-03-10 00:00:00", "death_date": "1837", "birth_place": "Lisboa", "death_place": "Roma",
                     "biography": "Pintor da corte.", "url": "https://catalogo.example/autor/2"},
    "a:lopes": {"entity_id": "a:lopes", "name": "Gregório Lopes", "activity": "", "birth_date": "", "death_date": "", "biography": "",
                "url": "https://catalogo.example/autor/3"},
}


class _Preprocess:
    """Responde à pergunta da ficha do autor pelo que o teste dita."""

    def __init__(self, resposta: str | None = "A") -> None:
        self.resposta, self.prompts = resposta, []

    async def generate(self, *, message, **_kwargs):
        self.prompts.append(message)
        if self.resposta is None:
            raise RuntimeError("modelo em baixo")
        return SimpleNamespace(parsed_json={"resposta": self.resposta})


class _AuthorGateway:
    def __init__(self, *, fail_authors: bool = False, inventories: list[str] | None = None) -> None:
        self.creator_calls: list[dict] = []
        self.author_calls: list[list[str]] = []
        self.inventory_calls: list[dict] = []
        self.fail_authors = fail_authors
        # os n.ºs de inventário das peças do autor; None = a leitura falha
        self.inventories = [] if inventories is None else inventories
        self.fail_inventories = False

    async def fetch_inventories_by_creator_ids(self, **kwargs):
        self.inventory_calls.append(kwargs)
        if self.fail_inventories:
            raise RuntimeError("OpenSearch em baixo")
        return list(self.inventories)

    async def aggregate_creators(self, **kwargs):
        self.creator_calls.append(kwargs)
        return [dict(row) for row in ROWS]

    async def fetch_authors_by_ids(self, *, author_ids):
        self.author_calls.append(list(author_ids))
        if self.fail_authors:
            raise RuntimeError("índice de autores em baixo")
        return [AUTHOR_DOCS[i] for i in author_ids if i in AUTHOR_DOCS]


def _card(artifact_id: str, *creator_ids: str) -> ArtifactResult:
    return ArtifactResult(artifact_id=artifact_id, creator_ids=list(creator_ids))


class TestAuthorCard(_Base):
    def _service_with(self, preprocess: _Preprocess, cards: list[ArtifactResult], *, flag: bool = True, gateway=None,
                      response_format: str = "text"):
        service = self._service(self._settings(CHAT_AUTHOR_CARD=flag))
        service._preprocess_llm_service = preprocess
        service.opensearch_gateway = gateway or _AuthorGateway()
        answered = ChatMessageResponse(reply="texto da resposta", artifact_results=cards,
                                       response_format=ResponseFormatObject(type=response_format))

        async def routed(payload, *, status_cb=None):
            return answered

        service._handle_message_routed = routed
        return service

    def _ask(self, service, message: str):
        return asyncio.run(service.handle_message(self._request(message, museum_slug="mnaa", museum_id="mnaa",
                                                                museum_name="Museu Nacional de Arte Antiga")))

    def test_off_by_default_nothing_is_asked(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_AUTHOR_CARD)
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, [_card("p1", "a:lopes")], flag=False)
        response = self._ask(service, "Mostra-me obras de Gregório Lopes.")
        self.assertIsNone(response.author)
        self.assertEqual(preprocess.prompts, [])
        self.assertEqual(service.opensearch_gateway.creator_calls, [])

    def test_the_author_named_in_the_message_gets_a_card(self) -> None:
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, [_card("p1", "a:lopes"), _card("p2", "a:lopes"), _card("p3", "a:queiros")])
        response = self._ask(service, "Mostra-me obras de Gregório Lopes.")
        self.assertEqual(response.reply, "texto da resposta")
        self.assertEqual(len(response.artifact_results), 3)
        author = response.author
        self.assertEqual((author.name, author.entity_ids, author.artifact_count, author.in_tour_count), ("Gregório Lopes", ["a:lopes"], 18, 0))
        self.assertEqual(author.url, "https://catalogo.example/autor/3")
        self.assertIsNone(author.biography)
        self.assertIsNone(author.image_url)
        # ao modelo vão as palavras do visitante, não o nome do catálogo
        self.assertEqual(len(preprocess.prompts), 1)
        self.assertIn("Na mensagem, «Gregório Lopes» é o nome de quê?", preprocess.prompts[0])
        self.assertEqual(service.opensearch_gateway.creator_calls, [{"museum_slug": "mnaa", "museum_id": "mnaa"}])

    def test_two_spellings_of_the_same_author_are_one_card(self) -> None:
        service = self._service_with(_Preprocess("A"), [_card("p1", "a:sequeira-2")])
        author = self._ask(service, "Tens obras do Domingos Sequeira?").author
        self.assertEqual(author.name, "Sequeira, Domingos António de")           # a grafia com mais peças
        self.assertEqual(author.entity_ids, ["a:sequeira-1", "a:sequeira-2"])
        self.assertEqual((author.artifact_count, author.in_tour_count), (109, 0))
        self.assertEqual(service.opensearch_gateway.inventory_calls,
                         [{"museum_slug": "mnaa", "museum_id": "mnaa", "creator_ids": ["a:sequeira-1", "a:sequeira-2"]}])
        self.assertEqual((author.activity, author.birth_date, author.birth_year, author.death_year), ("Pintor", "1768-03-10", 1768, 1837))
        self.assertEqual((author.birth_place, author.death_place, author.biography), ("Lisboa", "Roma", "Pintor da corte."))
        self.assertEqual(author.url, "https://catalogo.example/autor/1")

    def test_the_model_says_the_words_are_a_place(self) -> None:
        preprocess = _Preprocess("D")
        service = self._service_with(preprocess, [_card("p1", "a:porto")])
        response = self._ask(service, "Há paisagens do Porto?")
        self.assertIsNone(response.author)
        self.assertIn("«Porto»", preprocess.prompts[0])
        self.assertEqual(service.opensearch_gateway.author_calls, [])

    def test_an_author_who_is_not_on_the_cards_is_not_asked_about(self) -> None:
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, [_card("p1", "a:keil"), _card("p2")])
        self.assertIsNone(self._ask(service, "Mostra-me obras de Gregório Lopes.").author)
        self.assertEqual(preprocess.prompts, [])

    def test_no_cards_no_question(self) -> None:
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, [])
        self.assertIsNone(self._ask(service, "Mostra-me obras de Gregório Lopes.").author)
        self.assertEqual(preprocess.prompts, [])
        self.assertEqual(service.opensearch_gateway.creator_calls, [])

    def test_a_message_without_words_of_the_name_is_not_asked_about(self) -> None:
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, [_card("p1", "a:lopes")])
        self.assertIsNone(self._ask(service, "Quem pintou este retábulo?").author)
        self.assertEqual(preprocess.prompts, [])

    def test_json_answers_are_left_alone(self) -> None:
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, [_card("p1", "a:lopes")], response_format="json_object")
        self.assertIsNone(self._ask(service, "Mostra-me obras de Gregório Lopes.").author)
        self.assertEqual(preprocess.prompts, [])

    def test_a_failing_model_leaves_the_answer_as_it_was(self) -> None:
        service = self._service_with(_Preprocess(None), [_card("p1", "a:lopes")])
        response = self._ask(service, "Mostra-me obras de Gregório Lopes.")
        self.assertIsNone(response.author)
        self.assertEqual(response.reply, "texto da resposta")

    def test_a_failing_directory_leaves_the_answer_as_it_was(self) -> None:
        class _Broken(_AuthorGateway):
            async def aggregate_creators(self, **kwargs):
                raise RuntimeError("OpenSearch em baixo")

        service = self._service_with(_Preprocess("A"), [_card("p1", "a:lopes")], gateway=_Broken())
        response = self._ask(service, "Mostra-me obras de Gregório Lopes.")
        self.assertIsNone(response.author)
        self.assertEqual(len(response.artifact_results), 1)

    def test_without_the_authors_index_the_card_has_the_name_and_the_counts(self) -> None:
        service = self._service_with(_Preprocess("A"), [_card("p1", "a:lopes")], gateway=_AuthorGateway(fail_authors=True))
        author = self._ask(service, "Mostra-me obras de Gregório Lopes.").author
        self.assertEqual((author.name, author.artifact_count, author.url, author.activity), ("Gregório Lopes", 18, None, None))

    def test_the_directory_is_read_once(self) -> None:
        service = self._service_with(_Preprocess("A"), [_card("p1", "a:lopes")])
        self._ask(service, "Mostra-me obras de Gregório Lopes.")
        self._ask(service, "Quem foi Gregório Lopes?")
        self.assertEqual(len(service.opensearch_gateway.creator_calls), 1)


def _piece(inventory: str, *creator_ids: str) -> ArtifactResult:
    return ArtifactResult(artifact_id=f"a-{inventory}", inventory_number=inventory, creator_ids=list(creator_ids))


class TestAuthorInTourCount(_WithObjects):
    """in_tour_count sai dos alvos de navegação, com o critério do «Ver na tour»
    dos cartões. A visita dos testes (museu mnt) tem hotspots para 19663, 5262,
    38090 e 4703/4704, e peças localizadas no panorama: 777 (confirmada) e 999
    (cosseno acima do limiar); a 888 fica abaixo do limiar e a 0000 não está."""

    def _answer(self, cards: list[ArtifactResult], *, inventories: list[str] | None, objects: bool = True,
                targets: list[TourNavigationTarget] | None = None):
        settings = self._settings(CHAT_AUTHOR_CARD=True, TOUR_OBJECT_TARGETS_ENABLED=objects)
        service = self._service(settings)
        service._panorama_object_store = self.store
        service._preprocess_llm_service = _Preprocess("A")
        gateway = service.opensearch_gateway = _AuthorGateway(inventories=inventories)
        gateway.fail_inventories = inventories is None
        if targets is None:  # os alvos que a resposta traria: os dos cartões, pelo código dos cartões
            targets = service._resolve_navigation_targets(
                museum_slug="mnt", museum_id="mnt", docs=[{"inventory_number": card.inventory_number} for card in cards])
        answered = ChatMessageResponse(reply="texto", artifact_results=cards, navigation_targets=targets)

        async def routed(payload, *, status_cb=None):
            return answered

        service._handle_message_routed = routed
        response = asyncio.run(service.handle_message(self._request("Mostra-me obras de Gregório Lopes.")))
        self.assertIsNotNone(response.author)
        return response, gateway

    @staticmethod
    def _own_cards_with_target(response) -> int:
        with_target = {target.inventory_id.strip().casefold() for target in response.navigation_targets}
        ids = set(response.author.entity_ids)
        return len({card.inventory_number.strip().casefold() for card in response.artifact_results
                    if ids.intersection(card.creator_ids) and card.inventory_number.strip().casefold() in with_target})

    def test_counts_hotspots_and_located_objects_of_the_whole_museum(self) -> None:
        # só um cartão na resposta; o autor tem seis peças no museu, quatro com alvo
        response, gateway = self._answer([_piece("19663", "a:lopes")], inventories=["19663", "5262", "777", "999", "888", "0000"])
        self.assertEqual(response.author.in_tour_count, 4)
        self.assertEqual(response.author.artifact_count, 18)   # a contagem das fichas não muda
        self.assertEqual(gateway.inventory_calls, [{"museum_slug": "mnt", "museum_id": "mnt", "creator_ids": ["a:lopes"]}])

    def test_without_object_targets_only_hotspots_count(self) -> None:
        response, _ = self._answer([_piece("19663", "a:lopes")], inventories=["19663", "5262", "777", "999"], objects=False)
        self.assertEqual(response.author.in_tour_count, 2)

    def test_a_piece_with_two_hotspots_counts_once(self) -> None:
        # 5262 e 38090 têm dois hotspots cada (Sala da Música e Sala Cantonal)
        response, _ = self._answer([_piece("5262", "a:lopes")], inventories=["5262", "38090", " 5262 "])
        self.assertEqual(response.author.in_tour_count, 2)

    def test_cards_of_other_authors_do_not_count(self) -> None:
        cards = [_piece("19663", "a:lopes"), _piece("38090", "a:queiros"), _piece("999", "a:porto")]
        response, _ = self._answer(cards, inventories=["19663", "0000"])
        self.assertEqual(len(response.navigation_targets), 3)   # 19663, 38090 (dois hotspots, um alvo) e 999
        self.assertEqual(response.author.in_tour_count, 1)

    def test_never_less_than_the_authors_cards_with_a_target(self) -> None:
        cards = [_piece("19663", "a:lopes"), _piece("5262", "a:lopes", "a:queiros"), _piece("777", "a:lopes"),
                 _piece("999", "a:lopes"), _piece("888", "a:lopes"), _piece("38090", "a:queiros")]
        casos = {
            "todas as peças do autor lidas": ["19663", "5262", "777", "999", "888", "4703"],
            "a leitura das peças falha": None,
            "a leitura vem vazia": [],
            "a leitura vem incompleta": ["0000", "4703"],
        }
        esperado = {"todas as peças do autor lidas": 5, "a leitura das peças falha": 4, "a leitura vem vazia": 4,
                    "a leitura vem incompleta": 5}
        for nome, inventories in casos.items():
            with self.subTest(nome):
                response, _ = self._answer(cards, inventories=inventories)
                self.assertEqual(self._own_cards_with_target(response), 4)
                self.assertGreaterEqual(response.author.in_tour_count, self._own_cards_with_target(response))
                self.assertEqual(response.author.in_tour_count, esperado[nome])

    def test_a_target_in_the_answer_counts_even_if_the_map_no_longer_has_it(self) -> None:
        # a resposta traz um alvo para a peça 0000 que o mapa não conhece: o cartão mostra «Ver na tour», a ficha conta-a
        targets = [TourNavigationTarget(overlay_id="ov-x", panorama_key="pan_x", inventory_id="0000")]
        response, _ = self._answer([_piece("0000", "a:lopes")], inventories=["0000"], targets=targets)
        self.assertEqual(response.author.in_tour_count, 1)


PORTRAITS = {
    "a:lopes": {"image_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/a1/Lopes.jpg/330px-Lopes.jpg",
                "page_url": "https://commons.wikimedia.org/wiki/File:Lopes.jpg", "license": "Public domain", "credit": "Autor desconhecido",
                "wikidata": "Q724514", "name": "Gregório Lopes"},
    "a:sequeira-2": {"image_url": "https://upload.wikimedia.org/wikipedia/commons/b/b2/Sequeira.jpg",
                     "page_url": "https://commons.wikimedia.org/wiki/File:Sequeira.jpg", "license": "CC BY-SA 4.0", "credit": "Maria Silva"},
    "a:porto": {"image_url": "/imagens/porto.jpg", "page_url": "commons.wikimedia.org/wiki/File:Porto.jpg", "license": "Public domain"},
    "a:keil": {"image_url": "data:image/png;base64,AAAA", "license": "Public domain"},
}


class TestAuthorPortraits(_Base):
    """CHAT_AUTHOR_PORTRAITS: o retrato vem do ficheiro dos retratos, por id do autor."""

    def _ask(self, message: str, card: ArtifactResult, *, flag: bool = True, portraits=PORTRAITS, write: bool = True):
        path = Path(self._tmp.name) / "author_portraits.json"
        if write:
            path.write_text(portraits if isinstance(portraits, str) else json.dumps({"portraits": portraits}), encoding="utf-8")
        service = self._service(self._settings(CHAT_AUTHOR_CARD=True, CHAT_AUTHOR_PORTRAITS=flag, AUTHOR_PORTRAITS_PATH=str(path)))
        service._preprocess_llm_service = _Preprocess("A")
        service.opensearch_gateway = _AuthorGateway()
        answered = ChatMessageResponse(reply="texto", artifact_results=[card])

        async def routed(payload, *, status_cb=None):
            return answered

        service._handle_message_routed = routed
        author = asyncio.run(service.handle_message(self._request(message, museum_slug="mnaa", museum_id="mnaa"))).author
        self.assertIsNotNone(author)
        return author

    def test_off_by_default(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_AUTHOR_PORTRAITS)
        author = self._ask("Mostra-me obras de Gregório Lopes.", _card("p1", "a:lopes"), flag=False)
        self.assertEqual((author.image_url, author.image_credit, author.image_license, author.image_source_url), (None, None, None, None))

    def test_the_portrait_comes_with_its_credits(self) -> None:
        author = self._ask("Mostra-me obras de Gregório Lopes.", _card("p1", "a:lopes"))
        self.assertEqual(author.image_url, "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/a1/Lopes.jpg/330px-Lopes.jpg")
        self.assertEqual((author.image_credit, author.image_license), ("Autor desconhecido", "Public domain"))
        self.assertEqual(author.image_source_url, "https://commons.wikimedia.org/wiki/File:Lopes.jpg")
        self.assertEqual(author.name, "Gregório Lopes")   # o resto da ficha não muda

    def test_any_spelling_of_the_author_brings_the_portrait(self) -> None:
        # o retrato está na segunda ficha do autor («Sequeira, Domingos António de (1768-1837)»)
        author = self._ask("Tens obras do Domingos Sequeira?", _card("p1", "a:sequeira-1"))
        self.assertEqual(author.entity_ids, ["a:sequeira-1", "a:sequeira-2"])
        self.assertEqual((author.image_url, author.image_license), ("https://upload.wikimedia.org/wikipedia/commons/b/b2/Sequeira.jpg", "CC BY-SA 4.0"))

    def test_only_absolute_http_addresses_are_sent(self) -> None:
        # o frontend ignora caminhos relativos e outros esquemas: não se enviam
        self.assertIsNone(self._ask("Quadros de Silva Porto.", _card("p1", "a:porto")).image_url)
        self.assertIsNone(self._ask("Fala-me da Maria Keil.", _card("p1", "a:keil")).image_url)
        for value, expected in [("https://a.example/x.jpg", "https://a.example/x.jpg"), ("http://a.example/x.jpg", "http://a.example/x.jpg"),
                                ("//a.example/x.jpg", None), ("/x.jpg", None), ("x.jpg", None), ("javascript:alert(1)", None),
                                ("https://a.example/x y.jpg", None), ("", None), (None, None)]:
            self.assertEqual(absolute_http_url(value), expected, value)

    def test_an_author_without_portrait_keeps_the_card(self) -> None:
        author = self._ask("Há obras de Soares dos Reis?", _card("p1", "a:reis"))
        self.assertEqual((author.name, author.image_url, author.image_credit), ("Reis, António Soares dos", None, None))

    def test_a_missing_or_broken_file_leaves_the_card_without_portrait(self) -> None:
        self.assertIsNone(self._ask("Mostra-me obras de Gregório Lopes.", _card("p1", "a:lopes"), write=False).image_url)
        self.assertIsNone(self._ask("Mostra-me obras de Gregório Lopes.", _card("p1", "a:lopes"), portraits="{isto não é JSON").image_url)
        self.assertIsNone(self._ask("Mostra-me obras de Gregório Lopes.", _card("p1", "a:lopes"), portraits="[1, 2]").image_url)

    def test_the_file_is_read_again_when_it_changes(self) -> None:
        path = Path(self._tmp.name) / "retratos.json"
        path.write_text(json.dumps({"portraits": {}}), encoding="utf-8")
        store = AuthorPortraitStore(path)
        self.assertIsNone(store.portrait_for(["a:lopes"]))
        path.write_text(json.dumps({"portraits": PORTRAITS}), encoding="utf-8")
        os.utime(path, (1_900_000_000, 1_900_000_000))
        self.assertEqual(store.portrait_for(["a:outro", "a:lopes"])["image_source_url"], "https://commons.wikimedia.org/wiki/File:Lopes.jpg")

    def test_the_shipped_file_only_has_usable_portraits(self) -> None:
        path = Settings(_env_file=None).author_portraits_path_resolved
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreater(len(data["portraits"]), 0)
        for author_id, entry in data["portraits"].items():
            with self.subTest(author_id):
                self.assertTrue(entry["image_url"].startswith(("https://upload.wikimedia.org/", "https://thumb.wikimedia.org/")), entry["image_url"])
                self.assertEqual(absolute_http_url(entry["image_url"]), entry["image_url"])
                self.assertTrue(entry["page_url"].startswith("https://commons.wikimedia.org/wiki/File:"), entry["page_url"])
                self.assertTrue(entry["license"])
                self.assertTrue(entry["wikidata"].startswith("Q"))


class TestAuthorFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = build_author_directory(ROWS)

    def test_what_is_in_brackets_is_not_part_of_the_name(self) -> None:
        self.assertEqual(name_tokens("Keil, Maria (Silves, 1914 - Lisboa 2012)"), ["keil", "maria"])
        self.assertEqual(name_tokens("Zagallo, Raquel [1932 - 1998]"), ["zagallo", "raquel"])
        self.assertEqual(author_candidates("Há azulejos de Lisboa?", self.directory), [])

    def test_the_closest_name_comes_first(self) -> None:
        self.assertEqual(author_candidates("Mostra-me obras de Gregório Lopes.", self.directory, limit=2)[0].name, "Gregório Lopes")
        self.assertEqual(author_candidates("Há alguma obra do sequeira?", self.directory)[0].ids, ["a:sequeira-1", "a:sequeira-2"])
        self.assertEqual(author_candidates("Que esculturas fez o Soares dos Reis?", self.directory)[0].name, "Reis, António Soares dos")

    def test_only_among_the_authors_given(self) -> None:
        keil = [group for group in self.directory.groups if "keil" in group.tokens]
        self.assertEqual(author_candidates("obras de Gregório Lopes", self.directory, among=keil), [])
        self.assertEqual(author_candidates("azulejos da Maria Keil", self.directory, among=keil)[0].name, "Keil, Maria (Silves, 1914 - Lisboa 2012)")

    def test_words_as_the_visitor_wrote_them(self) -> None:
        by_token = lambda token: next(group for group in self.directory.groups if token in group.tokens)  # noqa: E731
        self.assertEqual(words_in_message("Há paisagens do Porto?", by_token("porto")), "Porto")
        self.assertEqual(words_in_message("mostra-me as obras de soares dos reis", by_token("reis")), "soares dos reis")
        self.assertEqual(words_in_message("Quadros de Silva Porto.", by_token("porto")), "Silva Porto")
        self.assertEqual(words_in_message("o Sequeira que se chamava Domingos", by_token("sequeira")), "Sequeira Domingos")
        self.assertEqual(words_in_message("Qual é a peça mais antiga?", by_token("porto")), "")

    def test_life_dates(self) -> None:
        self.assertEqual(clean_life_date("1922-03-05 00:00:00"), "1922-03-05")
        self.assertEqual(clean_life_date("09/01/1884"), "09/01/1884")
        self.assertIsNone(clean_life_date("  "))
        self.assertEqual((life_year("09/01/1884"), life_year("c. 1510"), life_year("1922-03-05")), (1884, 1510, 1922))
        self.assertIsNone(life_year("Século XV (?)"))
        self.assertIsNone(life_year("1510 ou 1512"))

    def test_fields_come_from_the_first_record_that_has_them(self) -> None:
        fields = merged_author_fields([AUTHOR_DOCS["a:sequeira-1"], AUTHOR_DOCS["a:sequeira-2"]])
        self.assertEqual((fields["activity"], fields["death_date"], fields["url"]), ("Pintor", "1837", "https://catalogo.example/autor/1"))
        self.assertEqual(merged_author_fields([])["biography"], None)
        self.assertEqual(merged_author_fields([{"atividade": "Escultor", "data_nascimento": "1847"}])["activity"], "Escultor")

    def test_the_prompt_shows_the_words_and_not_the_catalogue(self) -> None:
        prompt = build_author_name_prompt(message="Há paisagens do Porto?", words="Porto")
        self.assertIn("«Há paisagens do Porto?»", prompt)
        self.assertIn("Na mensagem, «Porto» é o nome de quê?", prompt)
        self.assertNotIn("Silva", prompt)
        self.assertNotIn("Museu Nacional", prompt)
        self.assertTrue(prompt.rstrip().endswith('{"resposta": "A" a "F"}'))


class _CreatorsClient:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def search(self, *, index, body):
        self.bodies.append(body)

        def bucket(key, count, ids, names):
            return {"key": key, "doc_count": count,
                    "ficha": {"hits": {"hits": [{"_source": {"creator_ids": ids, "creators": names}}]}}}

        return {"aggregations": {"autores": {"buckets": [
            bucket("a:lopes", 18, ["a:lopes"], ["Gregório Lopes"]),
            bucket("a:leal", 2, ["a:lopes", "a:leal"], ["Gregório Lopes", "Jorge Leal"]),   # peça com dois autores
            bucket("a:so", 5, "a:so", "Autor só"),                                            # o índice devolve texto simples
            bucket("a:torto", 3, ["a:torto", "a:outro"], ["Só um nome"]),                    # listas desalinhadas: fora
        ]}}}


class TestCreatorsGateway(unittest.TestCase):
    def test_creators_of_the_museum(self) -> None:
        gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ARTIFACT="arts"))
        fake = _CreatorsClient()
        gateway._client = fake
        rows = asyncio.run(gateway.aggregate_creators(museum_slug="mnaa", museum_id="mnaa"))
        self.assertEqual(rows, [
            {"id": "a:lopes", "name": "Gregório Lopes", "count": 18},
            {"id": "a:leal", "name": "Jorge Leal", "count": 2},
            {"id": "a:so", "name": "Autor só", "count": 5},
        ])
        body = fake.bodies[0]
        self.assertEqual(body["size"], 0)
        self.assertEqual(body["aggs"]["autores"]["terms"]["field"], "creator_ids")
        self.assertTrue(body["query"]["bool"]["filter"])
        # quantas peças estão na visita não sai do índice (campo in_tour desatualizado)
        self.assertNotIn("in_tour", str(body))

    def test_inventories_of_the_authors_pieces(self) -> None:
        class _Client:
            bodies: list[dict] = []

            def search(self, *, index, body):
                self.bodies.append(body)
                return {"hits": {"hits": [{"_source": {"inventory_number": "30 Pint"}}, {"_source": {"inventory": "80 Pint"}},
                                          {"_source": {"inventory_number": "30 Pint"}}, {"_source": {}}]}}

        gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ARTIFACT="arts"))
        gateway._client = _Client()
        found = asyncio.run(gateway.fetch_inventories_by_creator_ids(
            museum_slug="mnaa", museum_id="mnaa", creator_ids=["a:lopes-2", "a:lopes", "", "a:lopes"]))
        self.assertEqual(found, ["30 Pint", "80 Pint"])
        body = gateway._client.bodies[0]
        self.assertEqual(body["query"]["bool"]["filter"][-1], {"terms": {"creator_ids": ["a:lopes", "a:lopes-2"]}})
        self.assertGreater(len(body["query"]["bool"]["filter"]), 1)   # com o filtro do museu
        self.assertEqual(asyncio.run(gateway.fetch_inventories_by_creator_ids(museum_slug="mnaa", museum_id="mnaa", creator_ids=[])), [])


if __name__ == "__main__":
    unittest.main()
