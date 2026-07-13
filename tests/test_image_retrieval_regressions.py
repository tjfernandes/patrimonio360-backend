"""Regressões do diagnóstico i2i de 2026-07-13 (endpoint /chat/messages/image).

BUG 1 — o should ``{term: {in_tour: {boost}}}`` somava um score BM25/idf
(~+2.0 com boost=1, o dobro do máximo do cosseno), fazendo imagens in_tour
sequestrar o topo acima de matches perfeitos. A cláusula passou a
``constant_score`` (soma exatamente ``boost`` pontos — desempate previsível).

BUG 2 — a hidratação imagem→artefacto era inventory-first; a query de
inventários (terms + multi_match OR-concatenado, corte por ``size``) deixava
artefactos com inventários de tokens partilhados ("260 Cer MNSR" vs
"260 Cer CMP/ MNSR") fora dos slots, apagando resultados apesar de raw hit #1.
A hidratação passou a artifact_id-first (exata e ordenada), com inventário
apenas como fallback para hits sem artifact_id.

Testes offline (sem cluster):
    python -m unittest tests.test_image_retrieval_regressions
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chat_service import ChatService  # noqa: E402
from app.services.opensearch_client import OpenSearchGateway  # noqa: E402


class TestInTourBoostClause(unittest.TestCase):
    def _gateway(self) -> OpenSearchGateway:
        return OpenSearchGateway.__new__(OpenSearchGateway)

    def test_clause_is_constant_score_not_scored_term(self) -> None:
        clause = self._gateway()._build_in_tour_boost_clause(boost=0.05)
        self.assertIsNotNone(clause)
        self.assertIn("constant_score", clause)
        self.assertEqual(clause["constant_score"]["boost"], 0.05)
        self.assertEqual(
            clause["constant_score"]["filter"], {"term": {"in_tour": True}}
        )
        # A forma antiga ({"term": {"in_tour": {"boost": ...}}}) multiplicava o
        # idf do BM25 e não pode voltar.
        self.assertNotIn("term", clause)

    def test_zero_or_negative_boost_yields_no_clause(self) -> None:
        gateway = self._gateway()
        self.assertIsNone(gateway._build_in_tour_boost_clause(boost=0))
        self.assertIsNone(gateway._build_in_tour_boost_clause(boost=-1))


class _StubGateway:
    def __init__(self, by_ids_result=None, by_inventory_result=None) -> None:
        self.by_ids_result = by_ids_result if by_ids_result is not None else []
        self.by_inventory_result = (
            by_inventory_result if by_inventory_result is not None else []
        )
        self.by_ids_calls: list[dict] = []
        self.by_inventory_calls: list[dict] = []

    async def fetch_artifacts_by_ids(self, **kwargs):
        self.by_ids_calls.append(kwargs)
        return self.by_ids_result

    async def fetch_artifacts_by_inventory_numbers(self, **kwargs):
        self.by_inventory_calls.append(kwargs)
        return self.by_inventory_result


def _service_with(gateway: _StubGateway) -> ChatService:
    service = ChatService.__new__(ChatService)
    service.opensearch_gateway = gateway
    return service


def _hydrate(service: ChatService, image_hits, artifact_museum_id="mnsr"):
    return asyncio.run(
        ChatService._fetch_artifact_docs_for_image_hits(
            service,
            museum_slug="museu_nacional_soares_dos_reis",
            museum_id="mnsr",
            artifact_museum_id=artifact_museum_id,
            image_hits=image_hits,
            top_k=8,
        )
    )


class TestArtifactIdFirstHydration(unittest.TestCase):
    # Inventários com tokens partilhados — o cenário que fazia a via de
    # inventário descartar artefactos (S17 do diagnóstico).
    HITS = [
        {"artifact_id": "raiz:movel:1051026", "inventory_number": "260 Cer MNSR"},
        {"artifact_id": "raiz:movel:305093", "inventory_number": "617 Cer CMP/ MNSR"},
    ]

    def test_prefers_artifact_ids_and_skips_inventory_join(self) -> None:
        docs = [{"artifact_id": "raiz:movel:1051026"}, {"artifact_id": "raiz:movel:305093"}]
        gateway = _StubGateway(by_ids_result=docs)
        result = _hydrate(_service_with(gateway), self.HITS)
        self.assertEqual(result, docs)
        self.assertEqual(len(gateway.by_ids_calls), 1)
        self.assertEqual(
            gateway.by_ids_calls[0]["artifact_ids"],
            ["raiz:movel:1051026", "raiz:movel:305093"],
        )
        self.assertEqual(gateway.by_inventory_calls, [])

    def test_falls_back_to_inventory_when_hits_lack_artifact_ids(self) -> None:
        hits = [{"artifact_id": "", "inventory_number": "260 Cer MNSR"}]
        docs = [{"artifact_id": "raiz:movel:1051026"}]
        gateway = _StubGateway(by_inventory_result=docs)
        result = _hydrate(_service_with(gateway), hits)
        self.assertEqual(result, docs)
        self.assertEqual(gateway.by_ids_calls, [])
        self.assertEqual(len(gateway.by_inventory_calls), 1)
        self.assertEqual(
            gateway.by_inventory_calls[0]["inventory_numbers"], ["260 Cer MNSR"]
        )

    def test_falls_back_to_inventory_when_ids_return_nothing(self) -> None:
        docs = [{"artifact_id": "raiz:movel:1051026"}]
        gateway = _StubGateway(by_ids_result=[], by_inventory_result=docs)
        result = _hydrate(_service_with(gateway), self.HITS)
        self.assertEqual(result, docs)
        self.assertEqual(len(gateway.by_ids_calls), 1)
        self.assertEqual(len(gateway.by_inventory_calls), 1)

    def test_empty_hits_return_empty(self) -> None:
        gateway = _StubGateway()
        self.assertEqual(_hydrate(_service_with(gateway), []), [])
        self.assertEqual(gateway.by_ids_calls, [])
        self.assertEqual(gateway.by_inventory_calls, [])


if __name__ == "__main__":
    unittest.main()
