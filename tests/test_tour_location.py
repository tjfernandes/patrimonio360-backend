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
    {"overlayId": "ov6", "panoramaKey": "pan_cantonal", "inventoryIds": ["5262"], "location": "Sala Cantonal", "title": "Leque"},
    {"overlayId": "ov7", "panoramaKey": "pan_cantonal_2", "inventoryIds": ["38090"], "location": "Sala Cantonal", "title": "Chapéu"},
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


class _LLMReply:
    def __init__(self, text: str) -> None:
        self.text, self.finish_reason, self.usage, self.endpoint = text, "stop", {"completion_tokens": 30}, "primary"


class _WordingLLM:
    """Redige a partir do prompt; guarda o prompt para inspeção."""

    def __init__(self, text: str | None) -> None:
        self.text, self.prompts = text, []

    async def generate(self, *, message, **_kwargs):
        self.prompts.append(message)
        if self.text is None:
            from app.services.llm_service import LLMServiceError
            raise LLMServiceError("LLM em baixo")
        return _LLMReply(self.text)


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
    POI_BY_MUSEUM: dict = {"mnt": POI}
    TOUR_LOCATIONS: dict = {}  # sem lugares da visita 3DVista: só os poi_tours

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        poi_dir = Path(self._tmp.name)
        for museum, poi in self.POI_BY_MUSEUM.items():
            (poi_dir / f"panorama-overlays-inventory-{museum}.json").write_text(json.dumps(poi), encoding="utf-8")
        self.tour_locations_path = str(poi_dir / "tour_locations.json")
        Path(self.tour_locations_path).write_text(json.dumps(self.TOUR_LOCATIONS, ensure_ascii=False), encoding="utf-8")
        self.settings = self._settings(CHAT_TOUR_LOCATION_ANSWERS=True, CHAT_DETERMINISTIC_SMALLTALK=True)
        self.nav = TourNavigationService(self.settings)

    def _settings(self, **overrides) -> Settings:
        base = dict(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False,
                    POI_TOURS_DIR=str(Path(self._tmp.name)), TOUR_LOCATIONS_PATH=self.tour_locations_path)
        base.update(overrides)
        return Settings(**base)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _service(self, settings: Settings | None = None, llm=None) -> ChatService:
        service = object.__new__(ChatService)
        service.settings = settings or self.settings
        service.session_store = ChatSessionStore(service.settings)
        service.llm_service = llm or _FailingLLM()
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
    def test_panorama_title_does_not_override_room_name(self) -> None:
        # O tour manda o título do panorama («Sala da Música 12»); a zona é a do ficheiro.
        described = self.nav.describe_location(museum_slug="mnt", museum_id="mnt", panorama_key="pan_musica", room="Sala da Música 12")
        self.assertEqual(described["room"], "Sala da Música")
        self.assertEqual(described["scope"], "room")

    def test_panorama_with_pieces(self) -> None:
        described = self.nav.describe_location(museum_slug="mnt", museum_id="mnt", panorama_key="pan_musica", room=None)
        self.assertTrue(described["known"])
        # a sala inteira, com as peças deste panorama à frente
        self.assertEqual(described["scope"], "room")
        self.assertEqual(described["room"], "Sala da Música")
        self.assertEqual(described["panorama_inventories"], ["19663", "5262"])
        self.assertEqual(described["inventories"], ["19663", "5262", "38090"])
        self.assertEqual(described["room_inventory_count"], 3)

    def test_rooms_and_find_room(self) -> None:
        rooms = self.nav.list_rooms(museum_slug="mnt", museum_id="mnt")
        self.assertEqual([r["room"] for r in rooms], ["Sala da Música", "Pátio", "Hall", "Sala Cantonal"])
        cantonal = rooms[-1]
        self.assertEqual(cantonal["panorama_keys"], ["pan_cantonal", "pan_cantonal_2"])
        self.assertEqual(cantonal["inventories"], ["5262", "38090"])
        find = lambda text: (self.nav.find_room(museum_slug="mnt", museum_id="mnt", text=text) or {}).get("room")
        self.assertEqual(find("E a sala cantonal?"), "Sala Cantonal")
        self.assertEqual(find("leva-me à Cantonal"), "Sala Cantonal")
        self.assertEqual(find("o que há na sala da música"), "Sala da Música")
        self.assertEqual(find("quero ver a musica"), "Sala da Música")
        self.assertEqual(find("vai para o hall"), "Hall")
        self.assertIsNone(find("encontra vestidos de noiva"))
        self.assertIsNone(find("sala das armas"))

    def test_panorama_without_pieces_falls_back_to_room(self) -> None:
        described = self.nav.describe_location(museum_slug="mnt", museum_id="mnt", panorama_key="pan_desconhecido", room="sala da musica")
        self.assertEqual(described["scope"], "room")
        self.assertEqual(described["inventories"], ["19663", "5262", "38090"])
        empty = self.nav.describe_location(museum_slug="mnt", museum_id="mnt", panorama_key="pan_patio", room="Pátio")
        self.assertTrue(empty["known"])  # o Pátio é um lugar da visita, só não tem peças
        self.assertEqual(empty["scope"], "none")
        self.assertEqual(empty["inventories"], [])
        self.assertEqual(empty["room"], "Pátio")

    def test_unknown_museum(self) -> None:
        described = self.nav.describe_location(museum_slug="xx", museum_id="xx", panorama_key="pan_musica", room=None)
        self.assertFalse(described["known"])


class TestFastPath(_Base):
    def test_flag_off_goes_to_router(self) -> None:
        settings = self._settings(CHAT_TOUR_LOCATION_ANSWERS=False)
        service = self._service(settings)
        with self.assertRaises(_RouterCalled):
            asyncio.run(service.handle_message(self._request("Onde estou?", {"panorama_key": "pan_musica"})))

    def test_known_panorama_lists_whole_room_current_panorama_first(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(
            self._request("Onde estou?", {"panorama_key": "pan_musica", "room": "Sala da Música"}, conversation_id="c1")))
        self.assertEqual(response.model_hint, "deterministic:tour_location")
        self.assertIn("Está em «Sala da Música».", response.reply)
        # a sala inteira (3 peças), não só os hotspots deste panorama (2);
        # as do panorama atual vêm primeiro
        self.assertIn("3 peças", response.reply)
        self.assertLess(response.reply.index("Barrete"), response.reply.index("Leque"))
        self.assertLess(response.reply.index("Leque"), response.reply.index("Chapéu"))
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["19663", "5262", "38090"])
        self.assertEqual(sorted(t.inventory_id for t in response.navigation_targets), ["19663", "38090", "5262"])
        self.assertEqual(service.opensearch_gateway.inventory_calls, [["19663", "5262", "38090"]])
        event = service.query_logger.events[-1]
        self.assertEqual(event["retrieval_mode"], "tour_location")
        self.assertEqual(event["fast_path"], "tour_location")
        self.assertEqual(event["latency_llm_ms"], 0.0)
        self.assertEqual(event["tour_location"]["scope"], "room")
        self.assertEqual([a["inventory_number"] for a in event["shown_artifacts"]], ["19663", "5262", "38090"])
        state = service.session_store.get("c1")
        self.assertEqual(state.last_result_ids, ["a-19663", "a-5262", "a-38090"])
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


class TestRoomQueries(_Base):
    def test_matching_kinds(self) -> None:
        service = self._service()
        match = lambda text: service._match_tour_room_query(text, museum_slug="mnt", museum_id="mnt")
        self.assertEqual(match("E a sala cantonal?")["kind"], "room")
        self.assertEqual(match("Leva-me à sala da música")["room"], "Sala da Música")
        self.assertEqual(match("o que há na sala cantonal")["room"], "Sala Cantonal")
        self.assertEqual(match("que salas há?")["kind"], "list")
        self.assertEqual(match("Quais são as zonas da visita?")["kind"], "list")
        self.assertEqual(match("e a sala das armas?"), {"kind": "unknown", "name": "armas"})
        # pesquisas com uma sala no meio continuam a ser pesquisas
        self.assertIsNone(match("vestidos de noiva na sala cantonal"))
        self.assertIsNone(match("Encontra chapéus de mulher"))
        self.assertIsNone(match("onde estão as capas"))

    def test_room_fast_path_with_go_button(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(self._request("E a sala cantonal?", conversation_id="c2")))
        self.assertEqual(response.model_hint, "deterministic:tour_room")
        self.assertIn("«Sala Cantonal» faz parte da visita virtual.", response.reply)
        self.assertIn("Nesta sala a visita assinala 2 peças", response.reply)
        self.assertIn("botão abaixo", response.reply)
        self.assertEqual(response.tour_room.room, "Sala Cantonal")
        self.assertEqual(response.tour_room.panorama_key, "pan_cantonal")
        self.assertEqual(response.tour_room.piece_count, 2)
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["5262", "38090"])
        self.assertEqual({t.panorama_key for t in response.navigation_targets}, {"pan_cantonal", "pan_cantonal_2"})
        event = service.query_logger.events[-1]
        self.assertEqual(event["retrieval_mode"], "tour_room")
        self.assertEqual(event["tour_room"]["panorama_key"], "pan_cantonal")
        self.assertEqual(event["wording"], "template")

    def test_rooms_list_and_unknown_room(self) -> None:
        service = self._service()
        listed = asyncio.run(service.handle_message(self._request("que salas há?")))
        self.assertIn("A visita virtual tem 4 zonas.", listed.reply)
        self.assertIsNone(listed.tour_room)
        self.assertEqual([r.room for r in listed.tour_rooms], ["Sala da Música", "Pátio", "Hall", "Sala Cantonal"])
        self.assertEqual([r.piece_count for r in listed.tour_rooms], [3, 0, 2, 2])
        self.assertEqual(listed.tour_rooms[0].panorama_key, "pan_musica")
        # descrição curada do ficheiro real (Sala da Música do MNT); as outras ficam vazias
        self.assertTrue((listed.tour_rooms[0].description or "").startswith("Sala do andar nobre"))
        self.assertIsNone(listed.tour_rooms[3].description)
        for text in ["Que outras salas existem?", "há mais salas?", "mostra-me as zonas da visita", "quais são as salas do museu"]:
            self.assertEqual(service._match_tour_room_query(text, museum_slug="mnt", museum_id="mnt"), {"kind": "list"}, text)
        unknown = asyncio.run(service.handle_message(self._request("e a sala das armas?")))
        self.assertIn("Não encontro nenhuma zona da visita com esse nome", unknown.reply)
        self.assertEqual(len(unknown.tour_rooms), 4)
        self.assertEqual(unknown.artifact_results, [])
        cantonal = asyncio.run(service.handle_message(self._request("E a sala cantonal?")))
        self.assertEqual(cantonal.tour_rooms, [])

    def test_llm_wording_uses_facts_and_falls_back_to_template(self) -> None:
        settings = self._settings(CHAT_TOUR_LOCATION_ANSWERS=True, CHAT_TOUR_CONTEXT_LLM_WORDING=True,
                                  CHAT_GROUNDING_CHECK="log")
        llm = _WordingLLM("Está na Sala da Música, onde a visita destaca o Barrete de fumador/Masculino (19663). O cartão está abaixo.")
        service = self._service(settings, llm=llm)
        response = asyncio.run(service.handle_message(
            self._request("Onde estou?", {"panorama_key": "pan_musica", "room": "Sala da Música"})))
        self.assertEqual(response.model_hint, "llm:tour_location")
        self.assertTrue(response.reply.startswith("Está na Sala da Música"))
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("FACTOS:", llm.prompts[0])
        self.assertIn("O visitante está agora em: Sala da Música", llm.prompts[0])
        self.assertIn("Barrete de fumador/Masculino (19663)", llm.prompts[0])
        self.assertIn("CARTÕES:", llm.prompts[0])
        event = service.query_logger.events[-1]
        self.assertEqual(event["wording"], "llm")
        self.assertEqual(event["grounding"]["violations"], 0)
        # sem posição: não há factos, não se chama o LLM
        no_position = asyncio.run(service.handle_message(self._request("Onde estou?")))
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("não me indicou a sua posição", no_position.reply)
        # LLM em baixo: template
        down = self._service(settings, llm=_WordingLLM(None))
        fallback = asyncio.run(down.handle_message(self._request("E a sala cantonal?")))
        self.assertEqual(fallback.model_hint, "deterministic:tour_room")
        self.assertIn("«Sala Cantonal» faz parte da visita virtual.", fallback.reply)


class TestTourPlaces(_Base):
    """Lugares da visita 3DVista fundidos com os poi_tours e pedidos «encontra …»."""

    POI_BY_MUSEUM = {
        "mnt": POI + [
            # nome com gralha nos poi_tours, mesmo panorama que «Sala Chinoiserie» na visita
            {"overlayId": "ov8", "panoramaKey": "pan_chino", "inventoryIds": ["5262"], "location": "Sala Chinioiserie"},
            # nome mais longo nos poi_tours para a «Escadaria» da visita
            {"overlayId": "ov9", "panoramaKey": "pan_escadaria_m", "inventoryIds": ["19663"], "location": "Escadaria Monumental"},
        ],
        "mj": [
            {"overlayId": "mj1", "panoramaKey": "pan_manuel", "inventoryIds": ["MJ_1"], "location": "Capela dos filhos de D. Manuel I"},
        ],
    }
    TOUR_LOCATIONS = {
        "mnt": [
            {"room": "Pátio", "panorama_keys": ["pan_patio_t", "pan_patio"]},
            {"room": "Hall", "panorama_keys": ["pan_hall"]},
            {"room": "Sala Cantonal", "panorama_keys": ["pan_cantonal_t", "pan_cantonal"]},
            {"room": "Sala Chinoiserie", "panorama_keys": ["pan_chino"]},
            {"room": "Jardim", "panorama_keys": ["pan_jardim"]},
            {"room": "Sala do Arco", "panorama_keys": ["pan_arco"]},
            {"room": "Sala do Arco 1930-1970", "panorama_keys": ["pan_arco_2"]},
            {"room": "Escadaria", "panorama_keys": ["pan_escadaria"]},
        ],
        "mj": [
            {"room": "Claustro Superior", "panorama_keys": ["pan_cs"]},
            {"room": "Claustro Térreo", "panorama_keys": ["pan_ct"]},
            {"room": "Túmulo de V. da Gama", "panorama_keys": ["pan_gama"]},
            {"room": "Túmulo de L. de Camões", "panorama_keys": ["pan_camoes"]},
        ],
    }

    def test_merge_order_aliases_and_pieces(self) -> None:
        rooms = {r["room"]: r for r in self.nav.list_rooms(museum_slug="mnt", museum_id="mnt")}
        self.assertEqual(list(rooms), ["Pátio", "Hall", "Sala Cantonal", "Sala Chinoiserie", "Jardim", "Sala do Arco",
                                       "Sala do Arco 1930-1970", "Escadaria", "Sala da Música"])
        self.assertIn("sala chinioiserie", rooms["Sala Chinoiserie"]["aliases"])
        self.assertEqual(rooms["Sala Chinoiserie"]["inventories"], ["5262"])
        # descrição curada guardada com o nome dos poi_tours chega ao lugar fundido
        self.assertTrue(rooms["Sala Chinoiserie"]["description"])
        self.assertIn("escadaria monumental", rooms["Escadaria"]["aliases"])
        self.assertEqual(rooms["Escadaria"]["inventories"], ["19663"])
        self.assertEqual(rooms["Sala Cantonal"]["panorama_keys"][:2], ["pan_cantonal_t", "pan_cantonal"])
        self.assertEqual(rooms["Jardim"]["inventories"], [])

    def test_encontra_phrasings(self) -> None:
        service = self._service()
        match = lambda text, museum="mnt": service._match_tour_room_query(text, museum_slug=museum, museum_id=museum)
        self.assertEqual(match("Encontra as salas deste museu"), {"kind": "list"})
        self.assertEqual(match("que outras salas existem?"), {"kind": "list"})
        self.assertEqual(match("encontra a sala cantonal")["room"], "Sala Cantonal")
        self.assertEqual(match("Encontra o pátio")["room"], "Pátio")
        self.assertEqual(match("leva-me ao jardim")["room"], "Jardim")
        self.assertEqual(match("procura a sala chinioiserie")["room"], "Sala Chinoiserie")
        self.assertEqual(match("encontra a escadaria monumental")["room"], "Escadaria")
        self.assertEqual(match("encontra a sala do arco 1930-1970")["room"], "Sala do Arco 1930-1970")
        arco = match("encontra o arco")
        self.assertEqual(arco["kind"], "choices")
        self.assertEqual([c["room"] for c in arco["candidates"]], ["Sala do Arco", "Sala do Arco 1930-1970"])
        self.assertEqual(match("encontra a sala das armas"), {"kind": "unknown", "name": "armas"})
        self.assertIsNone(match("vestidos de noiva na sala cantonal"))
        self.assertIsNone(match("encontra vestidos da sala"))
        self.assertIsNone(match("Encontra chapéus de mulher"))
        claustro = match("Encontra o claustro", "mj")
        self.assertEqual([c["room"] for c in claustro["candidates"]], ["Claustro Superior", "Claustro Térreo"])
        self.assertEqual(match("leva-me ao túmulo de vasco da gama", "mj")["room"], "Túmulo de V. da Gama")
        self.assertIsNone(match("encontra peças de D. Manuel", "mj"))
        self.assertEqual(match("encontra a capela dos filhos de D. Manuel I", "mj")["room"], "Capela dos filhos de D. Manuel I")

    def test_choices_answer_lists_candidates_with_buttons(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(
            self._request("Encontra o claustro", museum_slug="mj", museum_id="mj", museum_name="Mosteiro dos Jerónimos")))
        self.assertIn("Há 2 zonas da visita com esse nome", response.reply)
        self.assertEqual([r.room for r in response.tour_rooms], ["Claustro Superior", "Claustro Térreo"])
        self.assertEqual([r.panorama_key for r in response.tour_rooms], ["pan_cs", "pan_ct"])
        self.assertIsNone(response.tour_room)

    def test_place_without_pieces_gets_go_button(self) -> None:
        service = self._service()
        response = asyncio.run(service.handle_message(self._request("Encontra o pátio")))
        self.assertIn("«Pátio» faz parte da visita virtual.", response.reply)
        self.assertIn("não tem peças assinaladas", response.reply)
        self.assertEqual(response.tour_room.panorama_key, "pan_patio_t")

    def test_where_am_i_uses_tour_place_names(self) -> None:
        by_panorama = self.nav.describe_location(museum_slug="mnt", museum_id="mnt", panorama_key="pan_patio_t", room="Pátio 5")
        self.assertEqual((by_panorama["room"], by_panorama["known"]), ("Pátio", True))
        raw = self.nav.describe_location(museum_slug="mj", museum_id="mj", panorama_key="pan_raw",
                                         room="07_Claustro_Terreo_002_09112024_092340")
        self.assertEqual(raw["room"], "Claustro Térreo")
        unnamed = self.nav.describe_location(museum_slug="mj", museum_id="mj", panorama_key="pan_raw", room="E06_P1_A_AlaEste_043")
        self.assertIsNone(unnamed["room"])


if __name__ == "__main__":
    unittest.main()
