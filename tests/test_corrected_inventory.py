"""Número anunciado pela visita -> número do catálogo.

A visita anuncia o número que lá está escrito. Quando não é o do catálogo, a
correspondência está no mapa dos POIs (`inventoryIdsOriginal` -> `inventoryIds`)
e é aplicada pelo backend, porque os ficheiros da visita não se podem editar
(o player do 3DVista passa a mostrar a marca de água).
"""
from __future__ import annotations

import asyncio
import unittest

from tests.test_tour_location import _Base, _Gateway


class _GatewayWithCandidates(_Gateway):
    async def search_artifacts_by_inventory_candidates(self, *, museum_slug, museum_id, inventory_numbers, top_k):
        return []


class TestCorrectedInventory(_Base):
    POI_BY_MUSEUM = {
        "mnaz": [
            {"overlayId": "ov1", "panoramaKey": "pan1", "inventoryIds": ["MNAz 6991/2 Az"],
             "inventoryIdsOriginal": ["MNAz 6991 Az"], "location": "Exposição", "title": "Balaustrada com meninos"},
            # o original só difere na escrita: não é uma correspondência
            {"overlayId": "ov2", "panoramaKey": "pan1", "inventoryIds": ["MNAz 185 Az"],
             "inventoryIdsOriginal": ["MNAz_185_Az"], "location": "Exposição", "title": "Painel"},
            {"overlayId": "ov3", "panoramaKey": "pan2", "inventoryIds": ["MNAz 1690 Az"],
             "location": "Exposição", "title": "Fuga para o Egito"},
        ]
    }

    def _corrected(self, inventory: str) -> list[str]:
        return self.nav.corrected_inventories(museum_slug="mnaz", museum_id="mnaz", inventory=inventory)

    def test_visit_number_maps_to_catalogue_number(self) -> None:
        self.assertEqual(self._corrected("MNAz 6991 Az"), ["MNAz 6991/2 Az"])
        self.assertEqual(self._corrected("mnaz-6991-az"), ["MNAz 6991/2 Az"])

    def test_numbers_already_in_the_catalogue_are_left_alone(self) -> None:
        for inventory in ("MNAz 6991/2 Az", "MNAz 185 Az", "MNAz_185_Az", "MNAz 1690 Az", "MNAz 9999 Az", ""):
            self.assertEqual(self._corrected(inventory), [], inventory)

    def test_unknown_museum(self) -> None:
        self.assertEqual(self.nav.corrected_inventories(museum_slug="mnt", museum_id="mnt", inventory="MNAz 6991 Az"), [])

    def test_lookup_by_inventory_uses_the_catalogue_number(self) -> None:
        service = self._service()
        service.opensearch_gateway = _GatewayWithCandidates()
        for asked, searched in (("MNAz 6991 Az", "MNAz 6991/2 Az"), ("MNAz 1690 Az", "MNAz 1690 Az")):
            service.opensearch_gateway.inventory_calls.clear()
            asyncio.run(service.get_artifact_full_by_inventory(museum_slug="mnaz", museum_id="mnaz", inventory_number=asked))
            self.assertEqual(service.opensearch_gateway.inventory_calls, [[searched]], asked)


if __name__ == "__main__":
    unittest.main()
