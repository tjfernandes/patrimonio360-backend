"""Só peças do autor (CHAT_AUTHOR_ONLY_RESULTS): com a ficha do autor na resposta e o
visitante a querer as obras dele, os cartões são só de peças desse autor, na primeira
página e nas seguintes.

O modelo de pré-processamento aqui é um guião; a redação do prompt mede-se com o
modelo real (tests/author_card_cases.py).

    python -m unittest tests.test_author_only_results
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.core.config import Settings
from app.prompts.query_planner_prompts import build_author_only_prompt
from app.schemas.chat import (
    ArtifactResult,
    ChatMessageResponse,
    ChatResultsPageRequest,
    ImageMatchResult,
    ResponseFormatObject,
    TourNavigationTarget,
)
from tests.test_author_card import _AuthorGateway
from tests.test_tour_location import _Base


def _doc(artifact_id: str, *creator_ids: str) -> dict:
    return {"artifact_id": artifact_id, "inventory_number": f"inv {artifact_id}", "title": f"Peça {artifact_id}",
            "creator_ids": list(creator_ids), "creators": [f"nome de {c}" for c in creator_ids]}


# A ordem da pesquisa sem filtro: as 19 peças de Gregório Lopes misturadas com as de outros autores.
LOPES = [_doc(f"l{n}", "a:lopes") for n in range(1, 20)]
OUTROS = [_doc(f"o{n}", "a:queiros") for n in range(1, 12)]
CATALOGO = [doc for pair in zip(LOPES, OUTROS) for doc in pair] + LOPES[len(OUTROS):]
SEQUEIRA = [_doc("s1", "a:sequeira-1"), _doc("s2", "a:sequeira-2"), _doc("s3", "a:sequeira-1"), _doc("o1", "a:queiros")]
REQUEST = {"kind": "text", "museum_slug": "mnaa", "museum_id": "mnaa", "query_text": "obras de Gregório Lopes",
           "lexical_query": "gregório lopes", "query_embedding": [0.1, 0.2], "filters": {"category": "Pintura"}, "sort": {},
           "results_total": 1000, "retrieval_window_size": 1000, "preselected_beyond_page_ids": ["l9"]}


class _Preprocess:
    """Responde «A» à pergunta do nome e, à pergunta das peças, o que o teste dita."""

    def __init__(self, only: str | None = "A") -> None:
        self.only, self.prompts = only, []

    async def generate(self, *, message, **_kwargs):
        self.prompts.append(message)
        if "é o nome de quê?" in message:
            return SimpleNamespace(parsed_json={"resposta": "A"})
        if self.only is None:
            raise RuntimeError("modelo em baixo")
        return SimpleNamespace(parsed_json={"resposta": self.only})


class _SearchGateway(_AuthorGateway):
    """A pesquisa devolve o catálogo por fatias e respeita o filtro dos autores."""

    def __init__(self, docs: list[dict], *, fail_search: bool = False) -> None:
        super().__init__()
        self.docs, self.fail_search, self.search_calls = docs, fail_search, []

    async def search_relevant_context_page(self, *, from_offset, page_size, filters=None, **kwargs):
        self.search_calls.append({"from_offset": from_offset, "page_size": page_size, "filters": dict(filters or {}), **kwargs})
        if self.fail_search:
            raise RuntimeError("OpenSearch em baixo")
        wanted = set((filters or {}).get("creator_ids") or [])
        docs = [doc for doc in self.docs if not wanted or wanted.intersection(doc["creator_ids"])]
        return SimpleNamespace(results=[dict(doc) for doc in docs[from_offset:from_offset + page_size]], total=len(docs), query_body={})

    async def fetch_images_by_artifact_ids(self, **_kwargs):
        return []

    async def search_artifacts_by_inventory_candidates(self, **_kwargs):
        return []


def _card(doc: dict) -> ArtifactResult:
    return ArtifactResult(artifact_id=doc["artifact_id"], inventory_number=doc["inventory_number"], title=doc["title"],
                          creator_ids=doc["creator_ids"], creators=doc["creators"])


async def _no_reply(**_kwargs):
    return None


class TestAuthorOnlyResults(_Base):
    def _service_with(self, preprocess: _Preprocess, first_page: list[dict], *, catalogue=CATALOGO, flag: bool = True,
                      request: dict | None = REQUEST, fail_search: bool = False):
        service = self._service(self._settings(CHAT_AUTHOR_CARD=True, CHAT_AUTHOR_ONLY_RESULTS=flag, CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8))
        service._preprocess_llm_service = preprocess
        service.opensearch_gateway = _SearchGateway(catalogue, fail_search=fail_search)
        service._generate_results_page_reply = _no_reply
        cards = [_card(doc) for doc in first_page]
        images = [ImageMatchResult(original_image_name=f"{doc['artifact_id']}.jpg", artifact_id=doc["artifact_id"],
                                   inventory=doc["inventory_number"]) for doc in first_page]
        targets = [TourNavigationTarget(overlay_id=f"ov-{doc['artifact_id']}", panorama_key="p1", inventory_id=doc["inventory_number"])
                   for doc in first_page]

        async def routed(payload, *, status_cb=None):
            # como o serviço: a primeira página fica guardada com o pedido de pesquisa, para «ver mais resultados»
            state = service.session_store.load_or_create(conversation_id="c1", museum_slug=payload.museum_slug)
            request_id = None
            if request is not None:
                request_id = service._cache_last_retrieval_results(
                    state=state, artifact_results=cards, image_matches=images, navigation_targets=targets,
                    default_page_size=8, retrieval_request={key: (dict(value) if isinstance(value, dict) else value) for key, value in request.items()},
                )
                service.session_store.save(state)
            return ChatMessageResponse(conversation_id="c1", reply="texto da resposta", artifact_results=cards, image_matches=images,
                                       navigation_targets=targets, results_page=1, results_page_size=8, results_total=1000,
                                       results_has_more=True, results_request_id=request_id,
                                       response_format=ResponseFormatObject(type="text"))

        service._handle_message_routed = routed
        return service

    def _ask(self, service, message: str = "Mostra-me obras de Gregório Lopes."):
        return asyncio.run(service.handle_message(self._request(message, museum_slug="mnaa", museum_id="mnaa", conversation_id="c1",
                                                                museum_name="Museu Nacional de Arte Antiga")))

    def _page(self, service, response, page: int):
        payload = ChatResultsPageRequest(museum_slug="mnaa", museum_id="mnaa", conversation_id="c1", results_page=page,
                                         results_request_id=response.results_request_id)
        return asyncio.run(service.get_results_page(payload))

    def _stored_request(self, service, response) -> dict:
        return service.session_store.get("c1").paged_results_by_request_id[response.results_request_id]["retrieval_request"]

    def test_off_by_default_the_cards_stay(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_AUTHOR_ONLY_RESULTS)
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, CATALOGO[:8], flag=False)
        response = self._ask(service)
        self.assertEqual(response.author.name, "Gregório Lopes")
        self.assertEqual([card.artifact_id for card in response.artifact_results], [doc["artifact_id"] for doc in CATALOGO[:8]])
        self.assertEqual((response.results_total, response.results_has_more), (1000, True))
        self.assertEqual(len(preprocess.prompts), 1)
        self.assertEqual(service.opensearch_gateway.search_calls, [])
        self.assertNotIn("creator_ids", self._stored_request(service, response)["filters"])

    def test_only_the_authors_pieces_on_the_first_page(self) -> None:
        preprocess = _Preprocess("A")
        service = self._service_with(preprocess, CATALOGO[:8])   # l1 o1 l2 o2 l3 o3 l4 o4
        response = self._ask(service)
        # os quatro dele ficam pela ordem em que estavam; a página enche-se com as seguintes dele
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"])
        self.assertTrue(all(card.creator_ids == ["a:lopes"] for card in response.artifact_results))
        self.assertEqual((response.results_total, response.results_has_more, response.results_page_size), (19, True, 8))
        self.assertEqual(response.reply, "texto da resposta")
        self.assertEqual(response.author.name, "Gregório Lopes")
        # miniaturas e alvos de navegação dos cartões que saíram saem com eles
        self.assertEqual([match.artifact_id for match in response.image_matches], ["l1", "l2", "l3", "l4"])
        self.assertEqual([target.inventory_id for target in response.navigation_targets], ["inv l1", "inv l2", "inv l3", "inv l4"])
        # a pesquisa é a mesma, com os filtros que já tinha e mais o do autor
        call = service.opensearch_gateway.search_calls[0]
        self.assertEqual(call["filters"], {"category": "Pintura", "creator_ids": ["a:lopes"]})
        self.assertEqual((call["query_text"], call["lexical_query"], call["query_embedding"], call["from_offset"]),
                         ("obras de Gregório Lopes", "gregório lopes", [0.1, 0.2], 0))
        # ao modelo vão as palavras do visitante
        self.assertEqual(len(preprocess.prompts), 2)
        self.assertIn("«Gregório Lopes» é o nome de um artista.", preprocess.prompts[1])
        self.assertIn("«Mostra-me obras de Gregório Lopes.»", preprocess.prompts[1])

    def test_more_results_are_only_the_authors_too(self) -> None:
        service = self._service_with(_Preprocess("A"), CATALOGO[:8])
        response = self._ask(service)
        second = self._page(service, response, 2)
        self.assertEqual([card.artifact_id for card in second.artifact_results], [f"l{n}" for n in range(9, 17)])
        self.assertEqual((second.results_total, second.results_has_more, second.results_page), (19, True, 2))
        third = self._page(service, response, 3)
        self.assertEqual([card.artifact_id for card in third.artifact_results], ["l17", "l18", "l19"])
        self.assertEqual((third.results_total, third.results_has_more), (19, False))
        seen = [card.artifact_id for page in (response, second, third) for card in page.artifact_results]
        self.assertEqual(sorted(seen), sorted(doc["artifact_id"] for doc in LOPES))   # todas as dele, nenhuma repetida
        # a primeira página, pedida outra vez, é a que a resposta mostrou
        first = self._page(service, response, 1)
        self.assertEqual([card.artifact_id for card in first.artifact_results], [card.artifact_id for card in response.artifact_results])
        self.assertTrue(first.results_has_more)

    def test_cards_chosen_out_of_order_do_not_come_back(self) -> None:
        # a primeira página trazia peças dele que a pesquisa só daria mais à frente (l12 e l15)
        service = self._service_with(_Preprocess("A"), [LOPES[11], OUTROS[0], LOPES[14], LOPES[0]])
        response = self._ask(service)
        # eram quatro cartões: os três dele ficam e o lugar do outro é para a peça seguinte dele
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["l12", "l15", "l1", "l2"])
        self.assertEqual((response.results_total, response.results_has_more), (19, True))
        second = self._page(service, response, 2)
        self.assertEqual([card.artifact_id for card in second.artifact_results], [f"l{n}" for n in range(3, 11)])
        third = self._page(service, response, 3)
        self.assertEqual([card.artifact_id for card in third.artifact_results], ["l11", "l13", "l14", "l16", "l17", "l18", "l19"])
        self.assertFalse(third.results_has_more)

    def test_a_page_the_model_left_short_is_not_filled(self) -> None:
        # o modelo escolheu três cartões, todos dele: ficam os três, e as outras peças dele vêm em «ver mais»
        service = self._service_with(_Preprocess("A"), LOPES[:3])
        response = self._ask(service)
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["l1", "l2", "l3"])
        self.assertEqual((response.results_total, response.results_has_more), (19, True))
        self.assertEqual([card.artifact_id for card in self._page(service, response, 2).artifact_results], [f"l{n}" for n in range(4, 12)])

    def test_an_author_with_few_pieces_has_no_more_results(self) -> None:
        catalogue = [_doc("j1", "a:lopes"), _doc("o1", "a:queiros"), _doc("j2", "a:lopes"), *OUTROS[1:9]]
        service = self._service_with(_Preprocess("A"), catalogue[:8], catalogue=catalogue)
        response = self._ask(service)
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["j1", "j2"])
        self.assertEqual((response.results_total, response.results_has_more), (2, False))
        self.assertEqual(self._page(service, response, 2).artifact_results, [])

    def test_every_spelling_of_the_author_counts(self) -> None:
        service = self._service_with(_Preprocess("A"), SEQUEIRA, catalogue=SEQUEIRA)
        response = self._ask(service, "Tens obras do Domingos Sequeira?")
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["s1", "s2", "s3"])
        self.assertEqual(service.opensearch_gateway.search_calls[0]["filters"]["creator_ids"], ["a:sequeira-1", "a:sequeira-2"])
        self.assertEqual((response.results_total, response.results_has_more), (3, False))

    def test_when_the_visitor_wants_other_authors_too_nothing_changes(self) -> None:
        for answer in ("B", "C", "D", "", "X"):
            with self.subTest(answer):
                preprocess = _Preprocess(answer)
                service = self._service_with(preprocess, CATALOGO[:8])
                response = self._ask(service, "Obras de Gregório Lopes e de outros pintores do tempo dele")
                self.assertEqual(response.author.name, "Gregório Lopes")
                self.assertEqual([card.artifact_id for card in response.artifact_results], [doc["artifact_id"] for doc in CATALOGO[:8]])
                self.assertEqual((response.results_total, response.results_has_more), (1000, True))
                self.assertEqual(service.opensearch_gateway.search_calls, [])
                stored = self._stored_request(service, response)
                self.assertEqual(stored["filters"], {"category": "Pintura"})
                self.assertEqual(stored["preselected_beyond_page_ids"], ["l9"])
                self.assertNotIn("author_only_shown_ids", stored)

    def test_a_failure_leaves_the_answer_as_it_was(self) -> None:
        for preprocess, fail_search in ((_Preprocess(None), False), (_Preprocess("A"), True)):
            with self.subTest(fail_search=fail_search):
                service = self._service_with(preprocess, CATALOGO[:8], fail_search=fail_search)
                response = self._ask(service)
                self.assertEqual(response.author.name, "Gregório Lopes")
                self.assertEqual(len(response.artifact_results), 8)
                self.assertEqual((response.results_total, response.results_has_more), (1000, True))
                stored = self._stored_request(service, response)
                self.assertEqual(stored["filters"], {"category": "Pintura"})
                self.assertNotIn("author_only_shown_ids", stored)
                # e «ver mais resultados» continua como antes
                service.opensearch_gateway.fail_search = False
                self.assertEqual([card.artifact_id for card in self._page(service, response, 2).artifact_results],
                                 [doc["artifact_id"] for doc in CATALOGO[8:16]])

    def test_answers_that_are_not_a_text_search_are_left_alone(self) -> None:
        for request in (None, {**REQUEST, "kind": "image"}):
            with self.subTest(kind=(request or {}).get("kind")):
                preprocess = _Preprocess("A")
                service = self._service_with(preprocess, CATALOGO[:8], request=request)
                response = self._ask(service)
                self.assertEqual(len(response.artifact_results), 8)
                self.assertEqual(len(preprocess.prompts), 1)   # nem se pergunta ao modelo
                self.assertEqual(service.opensearch_gateway.search_calls, [])

    def test_the_prompt_carries_the_visitors_words(self) -> None:
        prompt = build_author_only_prompt(message="Há esculturas de Machado de Castro?", words="Machado de Castro")
        self.assertIn("«Há esculturas de Machado de Castro?»", prompt)
        self.assertEqual(prompt.count("«Machado de Castro»"), 5)
        self.assertIn('{"resposta": "A", "B", "C" ou "D"}', prompt)
