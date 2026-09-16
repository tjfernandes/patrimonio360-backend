"""«Onde estou?» respondido a partir da visita (CHAT_TOUR_LOCATION_ANSWERS).

    python -m unittest tests.test_tour_location
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ChatMessageRequest  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.chat_session_store import ChatSessionStore  # noqa: E402
from app.services.tour_navigation import TourNavigationService  # noqa: E402

POI = [
    {"overlayId": "ov1", "panoramaKey": "pan_musica", "inventoryIds": ["19663"], "location": "Sala da Música", "title": "Barrete de fumador"},
    {"overlayId": "ov2", "panoramaKey": "pan_musica", "inventoryIds": ["5262"], "location": "Sala da Música", "title": "Leque"},
    {"overlayId": "ov3", "panoramaKey": "pan_musica_2", "inventoryIds": ["38090"], "location": "Sala da Música", "title": "Chapéu"},
    {"overlayId": "ov4", "panoramaKey": "pan_patio", "inventoryIds": [], "location": "Pátio", "title": "Pátio"},
    {"overlayId": "ov5", "panoramaKey": "pan_hall", "inventoryIds": ["4703", "4704"], "location": "Hall", "title": "Botões (dois)"},
]

DOCS = {
    "19663": {"artifact_id": "a-19663", "inventory_number": "19663", "title": "Barrete de fumador/Masculino", "image_count": 0},
    "5262": {"artifact_id": "a-5262", "inventory_number": "5262", "title": "Leque/Feminino", "image_count": 0},
    "38090": {"artifact_id": "a-38090", "inventory_number": "38090", "title": "Chapéu/Feminino", "image_count": 0},
}


class _RouterCalled(RuntimeError):
    pass


class _FailingLLM:
    async def generate(self, **_kwargs):
        raise _RouterCalled("LLM chamado")


class _Gateway:
    def __init__(self) -> None:
        self.inventory_calls: list[list[str]] = []

    async def fetch_artifacts_by_inventory_numbers(self, *, museum_slug, museum_id, inventory_numbers, top_k):
        self.inventory_calls.append(list(inventory_numbers))
        # OpenSearch devolve por pontuação, não pela ordem pedida.
        found = [DOCS[i] for i in inventory_numbers if i in DOCS]
        return list(reversed(found))

    async def fetch_images_by_artifact_ids(self, **_kwargs):
        return []


class _Logger:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def log_event(self, event: dict) -> None:
        self.events.append(event)


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        poi_dir = Path(self._tmp.name)
        (poi_dir / "panorama-overlays-inventory-mnt.json").write_text(json.dumps(POI), encoding="utf-8")
        self.settings = Settings(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False,
                                 CHAT_TOUR_LOCATION_ANSWERS=True, CHAT_DETERMINISTIC_SMALLTALK=True,
                                 POI_TOURS_DIR=str(poi_dir))
        self.nav = TourNavigationService(self.settings)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _service(self, settings: Settings | None = None) -> ChatService:
        service = object.__new__(ChatService)
        service.settings = settings or self.settings
        service.session_store = ChatSessionStore(service.settings)
        service.llm_service = _FailingLLM()
        service.query_logger = _Logger()
        service.opensearch_gateway = _Gateway()
        service.embedding_provider = None
        service.model_retrieval_service = None
        service.tour_navigation_service = self.nav
        service._reranker_service = None
        service._vl_reranker_service = None
        return service

    @staticmethod
    def _request(message: str, tour_location: dict | None = None, **kwargs) -> ChatMessageRequest:
        metadata = {"session_id": "s1"}
        if tour_location is not None:
            metadata["tour_location"] = tour_location
        payload = dict(museum_slug="mnt", museum_id="mnt", museum_name="Museu Nacional do Traje",
                       language="pt", message=message, metadata=metadata)
        payload.update(kwargs)
        return ChatMessageRequest(**payload)


class TestDetection(_Base):
    def test_positives_pt_en(self) -> None:
        service = self._service()
        for text in ["Onde estou?", "onde é que estou", "Onde estou agora?", "onde me encontro", "onde estamos",
                     "Em que sala estou?", "em que sala é que estou", "que sala é esta?", "Qual é a sala em que estou",
                     "o que há nesta sala?", "O que posso ver aqui?", "que peças há nesta sala", "o que é que há por aqui",
                     "que sala é esta no museu", "Where am I?", "which room am I in", "what is in this room?",
                     # formulação real de produção (2026-09-16) que ia à pesquisa
                     "onde estou localizado?", "onde é que estou situada", "em que parte do museu estou?",
                     "em que ponto da visita me encontro", "onde fica isto?", "onde me situo",
                     "what's here", "What can I see here?"]:
            self.assertTrue(service._is_tour_location_query(text), text)

    def test_negatives_go_to_search(self) -> None:
        service = self._service()
        for text in ["onde estão os vestidos", "em que sala está o biombo", "que sala tem chapéus",
                     "onde posso encontrar um túmulo de pedra", "o que há de traje de noiva", "Olá",
                     "esta mesa 793 Mob MNSR, está em que lugar do museu?", "where is the sarcophagus",
                     "o que há nesta sala de vestidos"]:
            self.assertFalse(service._is_tour_location_query(text), text)

    def test_metadata_parsing(self) -> None:
        service = self._service()
        self.assertIsNone(service._tour_location_from_metadata(None))
        self.assertIsNone(service._tour_location_from_metadata({"tour_location": {"title": "x"}}))
        parsed = service._tour_location_from_metadata({"tour_location": {"panorama_key": " pan_musica ", "room": "Sala da Música"}})
        self.assertEqual(parsed, {"panorama_key": "pan_musica", "room": "Sala da Música", "title": None})


class TestDescribeLocation(_Base):
    def test_panorama_with_pieces(self) -> None:
        described = self.nav.describe_location(museum_slug="mnt", museum_id="mnt", panorama_key="pan_musica", room=None)
        self.assertTrue(described["known"])
        self.assertEqual(described["scope"], "panorama")
        self.assertEqual(described["room"], "Sala da Música")
        self.assertEqual(described["inventories"], ["19663", "5262"])
        self.assertEqual(described["room_inventory_count"], 3)

    def test_panorama_without_pieces_falls_back_to_room(self) -> None:
        described = self.nav.describe_location(museum_slug="mnt", museum_id="mnt", panorama_key="pan_desconhecido", room="sala da musica")
        self.assertEqual(described["scope"], "room")
        self.assertEqual(described["inventories"], ["19663", "5262", "38090"])
        empty = self.nav.describe_location(museum_slug="mnt", museum_id="mnt", panorama_key="pan_patio", room="Pátio")
        self.assertFalse(empty["known"])
        self.assertEqual(empty["inventories"], [])
        self.assertEqual(empty["room"], "Pátio")

    def test_unknown_museum(self) -> None:
        described = self.nav.describe_location(museum_slug="xx", museum_id="xx", panorama_key="pan_musica", room=None)
        self.assertFalse(described["known"])


class TestFastPath(_Base):
    def test_flag_off_goes_to_router(self) -> None:
        settings = Settings(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False,
                            CHAT_TOUR_LOCATION_ANSWERS=False, POI_TOURS_DIR=self.settings.POI_TOURS_DIR)
        service = self._service(settings)
        with self.assertRaises(_RouterCalled):
            asyncio.run(service.handle_message(self._request("Onde estou?", {"panorama_key": "pan_musica"})))

    def test_known_panorama_lists_pieces_in_tour_order(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(
            self._request("Onde estou?", {"panorama_key": "pan_musica", "room": "Sala da Música"}, conversation_id="c1")))
        self.assertEqual(response.model_hint, "deterministic:tour_location")
        self.assertIn("Está em «Sala da Música».", response.reply)
        self.assertIn("2 peças", response.reply)
        self.assertLess(response.reply.index("Barrete"), response.reply.index("Leque"))
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["19663", "5262"])
        self.assertEqual({t.panorama_key for t in response.navigation_targets}, {"pan_musica"})
        self.assertEqual(sorted(t.inventory_id for t in response.navigation_targets), ["19663", "5262"])
        self.assertEqual(service.opensearch_gateway.inventory_calls, [["19663", "5262"]])
        event = service.query_logger.events[-1]
        self.assertEqual(event["retrieval_mode"], "tour_location")
        self.assertEqual(event["fast_path"], "tour_location")
        self.assertEqual(event["latency_llm_ms"], 0.0)
        self.assertEqual(event["tour_location"]["scope"], "panorama")
        self.assertEqual([a["inventory_number"] for a in event["shown_artifacts"]], ["19663", "5262"])
        state = service.session_store.get("c1")
        self.assertEqual(state.last_result_ids, ["a-19663", "a-5262"])
        self.assertEqual([t.role for t in state.history], ["user", "assistant"])

    def test_room_fallback_and_empty_area(self) -> None:
        service = self._service()
        by_room = asyncio.run(service.handle_message(self._request("que sala é esta?", {"room": "Sala da Música"})))
        self.assertIn("Nesta sala a visita assinala 3 peças", by_room.reply)
        self.assertEqual(len(by_room.artifact_results), 3)
        patio = asyncio.run(service.handle_message(self._request("o que há aqui?", {"panorama_key": "pan_patio", "room": "Pátio"})))
        self.assertIn("Está em «Pátio».", patio.reply)
        self.assertIn("não tem peças assinaladas", patio.reply)
        self.assertEqual(patio.artifact_results, [])

    def test_without_position_says_so_without_search(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(self._request("Onde estou?")))
        self.assertIn("não me indicou a sua posição", response.reply)
        self.assertEqual(response.artifact_results, [])
        self.assertEqual(service.opensearch_gateway.inventory_calls, [])
        self.assertIsNone(service.query_logger.events[-1]["tour_location"]["sent"])

    def test_english_reply(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(
            self._request("Where am I?", {"panorama_key": "pan_hall", "room": "Hall"}, language="en")))
        self.assertIn('You are in "Hall".', response.reply)
        # 4703/4704 não existem no acervo de teste: a sala é conhecida mas sem fichas.
        self.assertIn("no highlighted objects", response.reply)


if __name__ == "__main__":
    unittest.main()
