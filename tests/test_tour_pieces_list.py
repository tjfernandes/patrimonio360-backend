"""«Que peças posso ver nesta visita?»: todas as peças assinaladas na visita,
pela ordem da visita, sem pesquisa nem LLM (CHAT_TOUR_LOCATION_ANSWERS).

    python -m unittest tests.test_tour_pieces_list
"""
from __future__ import annotations

import asyncio
import unittest

from app.schemas.chat import ChatResultsPageRequest
from tests.test_tour_location import _Base

POSITIVES = [
    "que peças posso ver nesta visita", "Que peças posso ver nesta visita?", "o que posso ver nesta visita?",
    "quais são as peças da visita?", "que peças há na visita virtual?", "lista as peças desta visita",
    "mostra-me todas as peças da visita", "que objetos existem neste tour?", "que obras posso ver aqui na visita?",
    "quais as peças assinaladas na visita?", "o que há para ver na visita?", "que peças fazem parte da visita",
    "What can I see in this tour?", "Which pieces are in this tour?", "list the objects of the visit",
]
NEGATIVES = [
    "que vestidos há nesta visita", "encontra esculturas na visita", "que peças azuis há na visita",
    "peças de 1850 na visita", "que peças tem o museu", "quero ver vestidos", "o que posso ver",
    "que peças há nesta sala", "onde estou", "mostra-me leques", "que salas há na visita",
    "peças parecidas com esta na visita", "find red dresses in this tour",
]


class TestDetection(_Base):
    def test_positives(self) -> None:
        service = self._service()
        wrong = [m for m in POSITIVES if not service._is_tour_pieces_list_query(m)]
        self.assertEqual(wrong, [])

    def test_negatives(self) -> None:
        service = self._service()
        wrong = [m for m in NEGATIVES if service._is_tour_pieces_list_query(m)]
        self.assertEqual(wrong, [])


class TestAnswer(_Base):
    def test_all_tour_pieces_in_tour_order_without_llm(self) -> None:
        service = self._service()   # _FailingLLM: qualquer chamada ao LLM rebenta
        response = asyncio.run(service.handle_message(self._request("Que peças posso ver nesta visita?")))
        # POI: Sala da Música (19663, 5262, 38090), Pátio (nada), Hall (4703, 4704 sem ficha), Sala Cantonal (5262, 38090 repetidos)
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["19663", "5262", "38090"])
        self.assertEqual(service.opensearch_gateway.inventory_calls, [["19663", "5262", "38090", "4703", "4704"]])
        self.assertEqual(response.reply, "A visita virtual tem 3 peças assinaladas, em 3 zonas. Os cartões estão abaixo, pela ordem da visita: abra um para ver a ficha ou pergunte-me por uma delas.")
        self.assertEqual(sorted(t.inventory_id for t in response.navigation_targets), ["19663", "38090", "38090", "5262", "5262"])
        self.assertFalse(response.results_has_more)
        self.assertEqual(response.results_total, 3)
        self.assertEqual(response.model_hint, "deterministic:tour_pieces")
        event = service.query_logger.events[-1]
        self.assertEqual(event["retrieval_mode"], "tour_pieces")
        self.assertEqual(event["tour_pieces"], {"inventories": 5, "rooms": 3})
        self.assertIsNone(event["participant_id"])

    def test_pages_come_from_the_session_cache(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(self._request("lista as peças desta visita", results_page_size=2)))
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["19663", "5262"])
        self.assertTrue(response.results_has_more)
        page2 = asyncio.run(service.get_results_page(ChatResultsPageRequest(
            museum_slug="mnt", museum_id="mnt", conversation_id=response.conversation_id, results_page=2,
            results_page_size=2, results_request_id=response.results_request_id,
        )))
        self.assertEqual([a.inventory_number for a in page2.artifact_results], ["38090"])
        self.assertFalse(page2.results_has_more)
        self.assertEqual(len(service.opensearch_gateway.inventory_calls), 1)   # sem nova ida ao índice

    def test_english(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(self._request("What can I see in this tour?", language="en")))
        self.assertTrue(response.reply.startswith("The virtual tour has 3 highlighted objects, in 3 areas."))

    def test_tour_without_pieces(self) -> None:
        service = self._service()

        async def _none(**_kwargs):
            return []

        service.opensearch_gateway.fetch_artifacts_by_inventory_numbers = _none
        response = asyncio.run(service.handle_message(self._request("que peças há na visita?")))
        self.assertEqual(response.artifact_results, [])
        self.assertIn("não tem peças assinaladas", response.reply)

    def test_flag_off_goes_to_the_router(self) -> None:
        from tests.test_tour_location import _RouterCalled

        service = self._service(self._settings(CHAT_TOUR_LOCATION_ANSWERS=False))
        with self.assertRaises(_RouterCalled):
            asyncio.run(service.handle_message(self._request("Que peças posso ver nesta visita?")))


if __name__ == "__main__":
    unittest.main()
