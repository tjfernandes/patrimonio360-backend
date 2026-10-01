"""«Ver na tour» na lista das salas: as peças da lista levam o artifact_id do
catálogo, e o pedido describe_artifact serve só com o n.º de inventário
(traduzido pelo mapa dos POIs quando é o n.º escrito na visita).

    python -m unittest tests.test_tour_room_piece_describe
"""
from __future__ import annotations

import asyncio
import unittest

from tests.test_describe_artifact_intent import SAPATOS, _DescribeLLM, _IdGateway, _SearchCalled
from tests.test_tour_location import DOCS, _Base


class _Gateway(_IdGateway):
    def __init__(self) -> None:
        super().__init__(docs_by_id={})
        self.candidate_calls: list[list[str]] = []

    async def search_artifacts_by_inventory_candidates(self, *, museum_slug, museum_id, inventory_numbers, top_k):
        self.candidate_calls.append(list(inventory_numbers))
        return []


class TestRoomsListCarriesArtifactIds(_Base):
    def test_pieces_get_the_catalogue_id_in_one_fetch(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(self._request("Que salas há na visita?")))
        self.assertTrue(response.tour_rooms)
        pieces = {p.inventory_id: p.artifact_id for room in response.tour_rooms for p in room.pieces}
        self.assertEqual(pieces["19663"], "a-19663")
        self.assertEqual(pieces["5262"], "a-5262")
        self.assertEqual(pieces["38090"], "a-38090")
        self.assertIsNone(pieces["4703"])   # sem ficha no catálogo
        # uma só ida ao índice, com todos os inventários, sem repetidos
        self.assertEqual(len(service.opensearch_gateway.inventory_calls), 1)
        self.assertEqual(sorted(service.opensearch_gateway.inventory_calls[0]), ["19663", "38090", "4703", "4704", "5262"])

    def test_index_failure_leaves_pieces_without_id(self) -> None:
        service = self._service()

        async def _down(**_kwargs):
            raise RuntimeError("índice em baixo")

        service.opensearch_gateway.fetch_artifacts_by_inventory_numbers = _down
        response = asyncio.run(service.handle_message(self._request("Que salas há na visita?")))
        self.assertTrue(response.tour_rooms)
        self.assertTrue(all(p.artifact_id is None for room in response.tour_rooms for p in room.pieces))


class TestDescribeByInventoryOnly(_Base):
    POI_BY_MUSEUM = {"mnt": [
        {"overlayId": "ov1", "panoramaKey": "pan_musica", "inventoryIds": ["19663"], "inventoryIdsOriginal": ["19663 visita"],
         "location": "Sala da Música", "title": "Barrete de fumador"},
    ]}

    def _describe_service(self, **flags):
        settings = self._settings(**{"CHAT_DESCRIBE_ARTIFACT_INTENT": True, "CHAT_SELECTED_ARTIFACT_AUTO_RELEASE": True,
                                     "CHAT_COMPACT_ANSWER_CONTEXT": True, **flags})
        service = self._service(settings, llm=_DescribeLLM("Um barrete de fumador."))
        service.opensearch_gateway = _Gateway()

        async def _no_search(*_args, **_kwargs):
            raise _SearchCalled("a pesquisa não devia correr")

        service._handle_message_with_selection = _no_search
        return service

    def _request_for(self, selected: dict):
        request = self._request("Barrete de fumador/Masculino (19663)")
        request.metadata.update({"intent": "describe_artifact", "selected_artifact": selected})
        return request

    def test_inventory_only_is_enough(self) -> None:
        service = self._describe_service()
        response = asyncio.run(service.handle_message(self._request_for(
            {"inventory_number": "19663", "title": "Barrete de fumador", "source": "tour_room_piece"}
        )))
        self.assertEqual(response.reply, "Um barrete de fumador.")
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["a-19663"])
        self.assertEqual([t.inventory_id for t in response.navigation_targets], ["19663"])
        self.assertEqual(response.selected_artifact_status, "used")
        self.assertFalse(response.results_has_more)
        self.assertEqual(service.opensearch_gateway.id_calls, [])
        self.assertEqual(service.opensearch_gateway.inventory_calls, [["19663"]])
        event = service.query_logger.events[-1]
        self.assertEqual(event["retrieval_mode"], "describe_artifact")
        self.assertEqual(event["filters_applied"]["filters"], {"artifact_id": "a-19663"})

    def test_inventory_id_key_as_the_rooms_list_sends_it(self) -> None:
        service = self._describe_service()
        response = asyncio.run(service.handle_message(self._request_for({"inventory_id": "19663", "title": "Barrete"})))
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["a-19663"])

    def test_visit_number_is_translated_by_the_poi_map(self) -> None:
        service = self._describe_service()
        response = asyncio.run(service.handle_message(self._request_for({"inventory_number": "19663 visita"})))
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["a-19663"])
        self.assertEqual(service.opensearch_gateway.inventory_calls, [["19663"]])

    def test_unknown_number_tries_candidates_then_says_not_found(self) -> None:
        service = self._describe_service()
        response = asyncio.run(service.handle_message(self._request_for({"inventory_number": "99999"})))
        self.assertIn("Não encontrei a ficha", response.reply)
        self.assertEqual(response.artifact_results, [])
        self.assertEqual(service.opensearch_gateway.candidate_calls, [["99999"]])

    def test_without_intent_or_number_nothing_changes(self) -> None:
        service = self._describe_service()
        request = self._request_for({"title": "Barrete"})     # sem n.º nem id
        with self.assertRaises(_SearchCalled):
            asyncio.run(service.handle_message(request))
        request = self._request("Barrete de fumador/Masculino (19663)")
        request.metadata["selected_artifact"] = {"inventory_number": "19663"}     # sem intenção
        with self.assertRaises(_SearchCalled):
            asyncio.run(service.handle_message(request))

    def test_card_with_id_still_goes_by_id(self) -> None:
        service = self._describe_service()
        service.opensearch_gateway.docs_by_id = {"a-4270": SAPATOS}
        response = asyncio.run(service.handle_message(self._request_for(
            {"artifact_id": "a-4270", "inventory_number": "4270", "title": "Sapatos (par)/Feminino"}
        )))
        self.assertEqual([a.artifact_id for a in response.artifact_results], ["a-4270"])
        self.assertEqual(service.opensearch_gateway.id_calls, [["a-4270"]])
        self.assertEqual(service.opensearch_gateway.inventory_calls, [])


if __name__ == "__main__":
    unittest.main()
