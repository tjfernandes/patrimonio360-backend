"""Contagem por século pelo campo start_century do índice (CHAT_CATALOGUE_ANALYTICS_CENTURY_FIELD).

O histograma de start_year punha o ano -500 no século VI a.C. e o -100 no II a.C.

    python -m unittest tests.test_century_field_analytics
"""
from __future__ import annotations

import asyncio
import unittest

from app.core.config import Settings
from app.services.catalogue_analytics import century_of
from app.services.opensearch_client import OpenSearchGateway
from tests.test_catalogue_analytics import _AnalyticsGateway
from tests.test_tour_location import _Base


class _TermsClient:
    """Responde como o OpenSearch a uma agregação por termos de start_century."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def search(self, *, index, body):
        self.bodies.append(body)
        return {"hits": {"total": {"value": 33}, "hits": []},
                "aggregations": {"seculos": {"buckets": [{"key": 16, "doc_count": 30}, {"key": -6, "doc_count": 1},
                                                         {"key": -5, "doc_count": 1}, {"key": -1, "doc_count": 1}]}}}

    def count(self, *, index, body):
        return {"count": 40}


def _gateway(**flags) -> tuple[OpenSearchGateway, _TermsClient]:
    gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ARTIFACT="arts", **flags))
    fake = _TermsClient()
    gateway._client = fake
    return gateway, fake


class _GatewayByField(_AnalyticsGateway):
    async def aggregate_artifacts(self, **kwargs):
        return {**await super().aggregate_artifacts(**kwargs), "centuries_by_field": True}


class TestGateway(unittest.TestCase):
    def test_off_by_default(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_CATALOGUE_ANALYTICS_CENTURY_FIELD)
        gateway, fake = _gateway()
        result = asyncio.run(gateway.aggregate_artifacts(museum_slug="mnaa", museum_id="mnaa", centuries=True))
        self.assertIn("histogram", fake.bodies[0]["aggs"]["seculos"])
        self.assertFalse(result["centuries_by_field"])

    def test_on_counts_by_the_century_field(self) -> None:
        gateway, fake = _gateway(CHAT_CATALOGUE_ANALYTICS_CENTURY_FIELD=True)
        result = asyncio.run(gateway.aggregate_artifacts(museum_slug="mnaa", museum_id="mnaa", centuries=True))
        self.assertEqual(fake.bodies[0]["aggs"]["seculos"], {"terms": {"field": "start_century", "size": 200}})
        self.assertIn({"exists": {"field": "start_year"}}, fake.bodies[0]["query"]["bool"]["filter"])
        self.assertTrue(result["centuries_by_field"])
        self.assertEqual(sorted(result["centuries"]), [(-6, 1), (-5, 1), (-1, 1), (16, 30)])

    def test_on_other_questions_are_untouched(self) -> None:
        gateway, fake = _gateway(CHAT_CATALOGUE_ANALYTICS_CENTURY_FIELD=True)
        result = asyncio.run(gateway.aggregate_artifacts(museum_slug="mnaa", museum_id="mnaa", order="asc", size=24))
        self.assertNotIn("aggs", fake.bodies[0])
        self.assertFalse(result["centuries_by_field"])


class TestReply(_Base):
    def _service_with(self, gateway):
        settings = self._settings(CHAT_TOUR_LOCATION_ANSWERS=True, CHAT_CATALOGUE_ANALYTICS=True, CHAT_DETERMINISTIC_SMALLTALK=True)
        service = self._service(settings)      # _FailingLLM: o LLM nunca pode ser chamado
        service.opensearch_gateway = gateway
        return service

    def test_centuries_before_christ_by_field(self) -> None:
        gateway = _GatewayByField([], total_all=40, total_dated=33, centuries=[(16, 30), (-6, 1), (-5, 1), (-1, 1)])
        response = asyncio.run(self._service_with(gateway).handle_message(self._request("de que século é a maioria das peças?")))
        self.assertEqual(response.reply, "Das 33 peças com data na ficha do museu, a maioria é do século XVI (30). "
                                         "Por século: VI a.C. 1, V a.C. 1, I a.C. 1, XVI 30. Sem data na ficha: 7.")

    def test_the_histogram_keys_put_minus_500_in_the_sixth_century(self) -> None:
        # o erro que a flag corrige: -500 cai no intervalo que começa em -599
        self.assertEqual((century_of(-599), century_of(-500)), (-6, -5))
        gateway = _AnalyticsGateway([], total_all=40, total_dated=33, centuries=[(1501, 30), (-599, 2), (-199, 1)])
        response = asyncio.run(self._service_with(gateway).handle_message(self._request("de que século é a maioria das peças?")))
        self.assertIn("Por século: VI a.C. 2, II a.C. 1, XVI 30.", response.reply)


if __name__ == "__main__":
    unittest.main()
