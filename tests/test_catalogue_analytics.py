"""Analítica do acervo por regras (CHAT_CATALOGUE_ANALYTICS): reconhecimento,
corpo da pesquisa e resposta por template com cartões.

    python -m unittest tests.test_catalogue_analytics
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from app.core.config import Settings
from app.services.catalogue_analytics import (
    CENTURIES, COUNT, NEWEST, OLDEST, TOP_CREATORS, century_label, century_of, parse_catalogue_analytics, roman,
)
from app.services.opensearch_client import OpenSearchGateway
from tests.test_tour_location import _Base, _RouterCalled

POSITIVES = {
    "Qual é a peça mais antiga?": (OLDEST, "", False),
    "qual o artefacto mais antigo nesta visita": (OLDEST, "", True),
    "qual é o vestido mais antigo do museu?": (OLDEST, "vestido", False),
    "Which is the oldest object in this tour?": (OLDEST, "", True),
    "e a peça mais recente?": (NEWEST, "", False),
    "quais são os azulejos mais recentes": (NEWEST, "azulejos", False),
    "What is the newest piece?": (NEWEST, "", False),
    "Quantas peças tem o museu?": (COUNT, "", False),
    "quantas peças há nesta visita?": (COUNT, "", True),
    "quantos vestidos de noiva existem?": (COUNT, "vestidos noiva", False),
    "How many objects are there in this tour?": (COUNT, "", True),
    "de que século é a maioria das peças?": (CENTURIES, "", False),
    "distribuição das peças por século": (CENTURIES, "", False),
    "which century are most of the objects from?": (CENTURIES, "", False),
    "que autor tem mais obras no museu?": (TOP_CREATORS, "", False),
    "quais os artistas com mais peças nesta visita": (TOP_CREATORS, "", True),
    "which artist has the most works?": (TOP_CREATORS, "", False),
}
NEGATIVES = [
    "encontra vestidos antigos", "mostra-me azulejos", "onde está a peça mais antiga desta sala", "que salas há na visita",
    "peças parecidas com esta", "qual a história deste museu", "quem foi Soares dos Reis", "o que é faiança",
    "há quantos anos foi fundado o museu", "Olá", "vestidos do século XVIII",
]


class TestParser(unittest.TestCase):
    def test_positives(self) -> None:
        wrong = {}
        for message, expected in POSITIVES.items():
            got = parse_catalogue_analytics(message)
            if got is None or (got.kind, got.subject, got.in_tour) != expected:
                wrong[message] = got
        self.assertEqual(wrong, {})

    def test_negatives(self) -> None:
        wrong = [m for m in NEGATIVES if parse_catalogue_analytics(m) is not None]
        self.assertEqual(wrong, [])

    def test_centuries_helpers(self) -> None:
        self.assertEqual([roman(c) for c in (1, 4, 9, 14, 18, 19, 20, 21)], ["I", "IV", "IX", "XIV", "XVIII", "XIX", "XX", "XXI"])
        self.assertEqual([century_of(y) for y in (1701, 1800, 1801, -1, -100, -101)], [18, 18, 19, -1, -1, -2])
        self.assertEqual(century_label(-5), "V a.C.")
        self.assertEqual(century_label(-5, english=True), "V BC")


class _FakeClient:
    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.counts: list[dict] = []

    def search(self, *, index, body):
        self.bodies.append(body)
        return {"hits": {"total": {"value": 2}, "hits": [
            {"_score": 1.0, "_source": {"artifact_id": "a1", "inventory_number": "17673", "title": "Pano de mesa", "museum_id": "mnt", "start_year": 1470, "end_year": 1480, "in_tour": False}},
            {"_score": 1.0, "_source": {"artifact_id": "a2", "inventory_number": "14409", "title": "Colcha", "museum_id": "mnt", "start_year": 1501, "end_year": 1700, "in_tour": True}},
        ]}, "aggregations": {"seculos": {"buckets": [{"key": 1701, "doc_count": 5}, {"key": 1801, "doc_count": 9}]},
                            "autores": {"buckets": [{"key": "raiz:autor:1", "doc_count": 12, "ficha": {"hits": {"hits": [
                                {"_source": {"creator_ids": ["raiz:autor:9", "raiz:autor:1"], "creators": ["Outro", "Malhoa, José"]}}]}}}]}}}

    def count(self, *, index, body):
        self.counts.append(body)
        return {"count": 10}


class TestGatewayBody(unittest.TestCase):
    def _gateway(self) -> tuple[OpenSearchGateway, _FakeClient]:
        gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ARTIFACT="arts"))
        fake = _FakeClient()
        gateway._client = fake
        return gateway, fake

    def test_oldest_sorts_by_date_and_requires_a_date(self) -> None:
        gateway, fake = self._gateway()
        result = asyncio.run(gateway.aggregate_artifacts(museum_slug="mnt", museum_id="mnt", in_tour=True, subject="vestido", order="asc", size=24))
        body = fake.bodies[0]
        self.assertEqual(body["sort"][0], {"start_year": "asc"})
        filters = body["query"]["bool"]["filter"]
        self.assertIn({"term": {"museum_id": "mnt"}}, filters)
        self.assertIn({"term": {"in_tour": True}}, filters)
        self.assertIn({"exists": {"field": "start_year"}}, filters)
        self.assertEqual(body["query"]["bool"]["must"][0]["multi_match"]["query"], "vestido")
        self.assertEqual([d["inventory_number"] for d in result["docs"]], ["17673", "14409"])
        self.assertEqual((result["total_dated"], result["total_all"]), (2, 10))

    def test_count_without_order_counts_all_and_dated(self) -> None:
        gateway, fake = self._gateway()
        result = asyncio.run(gateway.aggregate_artifacts(museum_slug="mnt", museum_id="mnt", size=0))
        self.assertNotIn("sort", fake.bodies[0])
        self.assertNotIn({"exists": {"field": "start_year"}}, fake.bodies[0]["query"]["bool"]["filter"])
        self.assertEqual(len(fake.counts), 2)
        self.assertEqual((result["total_all"], result["total_dated"]), (10, 10))

    def test_aggregations(self) -> None:
        gateway, fake = self._gateway()
        result = asyncio.run(gateway.aggregate_artifacts(museum_slug="mnt", museum_id="mnt", centuries=True, top_creators=5))
        self.assertEqual(fake.bodies[0]["aggs"]["seculos"]["histogram"], {"field": "start_year", "interval": 100, "offset": 1, "min_doc_count": 1})
        self.assertEqual(fake.bodies[0]["aggs"]["autores"]["terms"], {"field": "creator_ids", "size": 5})
        self.assertEqual(result["centuries"], [(1701, 5), (1801, 9)])
        self.assertEqual(result["creators"], [("Malhoa, José", 12)])   # nome original, não a chave normalizada
        self.assertEqual(result["total_attributed"], 10)
        self.assertNotIn("search_text", str(fake.bodies[0]))


class _AnalyticsGateway:
    def __init__(self, docs, *, total_all=120, total_dated=90, centuries=None, creators=None) -> None:
        self.docs, self.total_all, self.total_dated = docs, total_all, total_dated
        self.centuries, self.creators, self.calls = centuries or [], creators or [], []

    async def aggregate_artifacts(self, **kwargs):
        self.calls.append(kwargs)
        size = kwargs.get("size") or 0
        return {"docs": list(self.docs)[:size], "total_all": self.total_all, "total_dated": self.total_dated,
                "total_attributed": 40, "centuries": self.centuries, "creators": self.creators, "query_body": {}}

    async def fetch_images_by_artifact_ids(self, **_kwargs):
        return []

    async def fetch_artifacts_by_inventory_numbers(self, **_kwargs):
        return []


DOCS = [
    {"artifact_id": "a-19663", "inventory_number": "19663", "title": "Barrete de fumador/Masculino", "museum_id": "mnt", "start_year": 1701, "end_year": 1800, "in_tour": True, "image_count": 0},
    {"artifact_id": "a-5262", "inventory_number": "5262", "title": "Leque/Feminino", "museum_id": "mnt", "start_year": 1750, "end_year": 1750, "in_tour": True, "image_count": 0},
    {"artifact_id": "a-9", "inventory_number": "9", "title": "Vestido/Feminino", "museum_id": "mnt", "start_year": 1801, "end_year": 1900, "in_tour": False, "image_count": 0},
]


class TestFastPath(_Base):
    def _service_with(self, gateway, **flags):
        settings = self._settings(**{"CHAT_TOUR_LOCATION_ANSWERS": True, "CHAT_CATALOGUE_ANALYTICS": True, "CHAT_DETERMINISTIC_SMALLTALK": True, **flags})
        service = self._service(settings)      # _FailingLLM: o LLM nunca pode ser chamado
        service.opensearch_gateway = gateway
        return service

    def test_oldest_in_tour(self) -> None:
        gateway = _AnalyticsGateway(DOCS, total_all=93, total_dated=93)
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("Qual é a peça mais antiga nesta visita?")))
        call = gateway.calls[0]
        self.assertEqual((call["order"], call["in_tour"], call["subject"], call["size"]), ("asc", True, "", 24))
        self.assertEqual(response.reply, "Entre as 93 peças com data na ficha (100% das 93 desta visita virtual), a mais antiga é «Barrete de fumador/Masculino» (19663), do século XVIII. Os cartões abaixo mostram as 3 mais antigas, por ordem.")
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["19663", "5262", "9"])
        self.assertEqual(sorted(t.inventory_id for t in response.navigation_targets), ["19663", "5262", "5262"])
        self.assertEqual(response.model_hint, "deterministic:catalogue_analytics")
        event = service.query_logger.events[-1]
        self.assertEqual(event["retrieval_mode"], "catalogue_analytics")
        self.assertEqual(event["catalogue_analytics"]["kind"], "oldest")
        self.assertIsNone(event["participant_id"])

    def test_newest_with_subject_and_years(self) -> None:
        gateway = _AnalyticsGateway(list(reversed(DOCS)), total_all=500, total_dated=400)
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("qual é o vestido mais recente do museu?")))
        self.assertEqual((gateway.calls[0]["order"], gateway.calls[0]["subject"]), ("desc", "vestido"))
        self.assertTrue(response.reply.startswith("Entre as 400 peças com data na ficha (80% das 500 do museu que correspondem a «vestido»), a mais recente é «Vestido/Feminino» (9), do século XIX."))

    def test_count_without_subject_has_no_cards(self) -> None:
        gateway = _AnalyticsGateway(DOCS, total_all=9645, total_dated=9426)
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("Quantas peças tem o museu?")))
        self.assertEqual(response.reply, "Há 9645 peças do museu; 9426 com data na ficha.")
        self.assertEqual(response.artifact_results, [])
        self.assertEqual(gateway.calls[0]["size"], 0)

    def test_count_with_subject_and_century(self) -> None:
        gateway = _AnalyticsGateway(DOCS, total_all=3, total_dated=3)
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("quantos vestidos do século XVIII há nesta visita?")))
        call = gateway.calls[0]
        self.assertEqual((call["subject"], call["in_tour"], call["size"]), ("vestidos", True, 8))
        # o extrator de séculos do backend usa 1700–1799 para o século XVIII
        interval = call["filters"]["_temporal_interval"]
        self.assertEqual((interval["start_year"], interval["end_year"]), (1700, 1799))
        self.assertEqual(response.reply, "Há 3 peças desta visita virtual que correspondem a «vestidos» entre 1700 e 1799; 3 com data na ficha. Os cartões abaixo mostram as primeiras.")
        self.assertEqual(len(response.artifact_results), 3)

    def test_centuries(self) -> None:
        gateway = _AnalyticsGateway([], total_all=100, total_dated=90, centuries=[(1701, 30), (1801, 50), (1901, 10)])
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("de que século é a maioria das peças?")))
        self.assertEqual(response.reply, "Das 90 peças com data na ficha do museu, a maioria é do século XIX (50). Por século: XVIII 30, XIX 50, XX 10. Sem data na ficha: 10.")
        self.assertEqual(response.artifact_results, [])

    def test_top_creators(self) -> None:
        gateway = _AnalyticsGateway([], total_all=100, total_dated=90, creators=[("Malhoa, José", 12), ("Silva Porto", 7)])
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("que autor tem mais obras no museu?")))
        self.assertEqual(response.reply, "Os autores com mais obras do museu: Malhoa, José (12), Silva Porto (7). Entre 100 peças, 40 têm autor na ficha.")

    def test_english(self) -> None:
        gateway = _AnalyticsGateway(DOCS, total_all=93, total_dated=93)
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("Which is the oldest object in this tour?", language="en")))
        self.assertTrue(response.reply.startswith("Among the 93 objects with a date in their record (100% of the 93 in this virtual tour), the oldest is “Barrete de fumador/Masculino” (19663), from the XVIII century."))

    def test_nothing_dated(self) -> None:
        gateway = _AnalyticsGateway([], total_all=5, total_dated=0)
        service = self._service_with(gateway)
        response = asyncio.run(service.handle_message(self._request("qual a peça mais antiga?")))
        self.assertEqual(response.reply, "Não encontrei peças com data na ficha do museu para responder a isso.")

    def test_flag_off_goes_to_the_router(self) -> None:
        service = self._service_with(_AnalyticsGateway(DOCS), CHAT_CATALOGUE_ANALYTICS=False)
        with self.assertRaises(_RouterCalled):
            asyncio.run(service.handle_message(self._request("Qual é a peça mais antiga?")))

    def test_rooms_and_location_keep_precedence(self) -> None:
        service = self._service_with(_AnalyticsGateway(DOCS))
        response = asyncio.run(service.handle_message(self._request("Que salas há na visita?")))
        self.assertTrue(response.tour_rooms)


if __name__ == "__main__":
    unittest.main()
