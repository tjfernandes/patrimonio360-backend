"""«Apontar e perguntar»: projeção da vista, recorte, pesquisa com prioridade
à visita, decisão e eventos SSE.

    python -m unittest tests.test_point_and_ask
"""
from __future__ import annotations

import asyncio
import math
import unittest
from types import SimpleNamespace

import numpy as np

from app.core.config import Settings
from app.schemas.chat import ResponseFormatObject
from app.services.chat_service import ChatService
from app.services.llm_service import LLMResponse
from app.services.point_ask import Candidate, PointAskService, PointAssetStore, PointOutcome, PointRequest, normalized_score
from app.services.point_segmenter import choose_mask, crop_and_cutout, douglas_peucker, mask_polygon, trace_contour
from app.services.point_view import FACES, Camera, ProjectedView, face_coordinates, point_in_view, project_view, ray_from_point
from tests.test_tour_location import _Base

PANO = "panorama_83D8666B_91ED_0BA2_41D9_9A5E8973B146"
FUGA_OVERLAY = "overlay_C29262E1_9F52_29DC_41CB_60D97463C7D0"
CAMERA = Camera(yaw=-0.07, pitch=0.48, hfov=100, width=999, height=759)


# ----- projeção -----

class _ColourFaces:
    """Faces do cubo de uma cor cada, para saber que face a projeção leu."""

    COLOURS = {"f": (255, 0, 0), "r": (0, 255, 0), "b": (0, 0, 255), "l": (255, 255, 0), "u": (0, 255, 255), "d": (255, 0, 255)}
    face_size = 64

    def face(self, museum, panorama_key, name):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[...] = self.COLOURS[name]
        image[:4, :, :] = 0  # faixa preta no topo da face, para ver a orientação
        return image


class TestProjection(unittest.TestCase):
    def test_ray_matches_the_frontend_example(self) -> None:
        yaw, pitch = ray_from_point(CAMERA, 0.65, 0.49)
        self.assertAlmostEqual(yaw, 19.6, delta=0.1)   # o frontend disse yaw 19.6
        self.assertAlmostEqual(pitch, 1.4, delta=0.4)
        self.assertEqual(ray_from_point(CAMERA, 0.5, 0.5), (CAMERA.yaw, CAMERA.pitch))

    def test_point_in_view_inverts_ray(self) -> None:
        for x, y in ((0.5, 0.5), (0.1, 0.2), (0.9, 0.8)):
            yaw, pitch = ray_from_point(CAMERA, x, y)
            back = point_in_view(CAMERA, yaw, pitch)
            self.assertIsNotNone(back)
            self.assertAlmostEqual(back[0], x, places=5)
            self.assertAlmostEqual(back[1], y, places=5)
        self.assertIsNone(point_in_view(CAMERA, 180.0, 0.0))  # atrás da câmara

    def test_faces_by_direction(self) -> None:
        directions = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64)
        face, s, t = face_coordinates(directions)
        self.assertEqual([FACES[i] for i in face], ["f", "r", "b", "l", "u", "d"])
        np.testing.assert_allclose(s, 0.5)
        np.testing.assert_allclose(t, 0.5)

    def test_project_view_reads_the_right_face(self) -> None:
        store = _ColourFaces()
        for yaw, pitch, expected in ((0, 0, "f"), (90, 0, "r"), (180, 0, "b"), (-90, 0, "l"), (0, 89, "u"), (0, -89, "d")):
            view = project_view(store, museum="m", panorama_key="p", camera=Camera(yaw, pitch, 60, 400, 300), width=128)
            centre = tuple(int(v) for v in view.image[view.image.shape[0] // 2, view.image.shape[1] // 2])
            self.assertEqual(centre, _ColourFaces.COLOURS[expected], (yaw, pitch))
            self.assertIn(expected, view.faces_used)
        self.assertEqual(view.image.shape, (96, 128, 3))

    def test_mask_in_camera(self) -> None:
        from app.services.point_view import mask_in_camera

        mask = np.zeros((300, 400), dtype=bool)
        mask[100:200, 150:250] = True
        same = mask_in_camera(mask, Camera(0, 0, 60, 400, 300), Camera(0, 0, 60, 400, 300), 400, 300)
        self.assertGreater((same & mask).sum() / mask.sum(), 0.98)
        # uma vista apertada à volta da caixa: a máscara enche-a quase toda
        zoom = PointAskService._zoom_camera(Camera(0, 0, 60, 400, 300), mask, margin=0.0)
        tight = mask_in_camera(mask, Camera(0, 0, 60, 400, 300), zoom, 200, 200)
        self.assertGreater(tight.mean(), 0.85)
        # fora da vista de origem fica False
        behind = mask_in_camera(mask, Camera(0, 0, 60, 400, 300), Camera(180, 0, 60, 400, 300), 50, 50)
        self.assertFalse(behind.any())

    def test_top_of_front_face_is_up(self) -> None:
        # a faixa preta do topo da face da frente aparece em cima quando se olha um pouco para cima
        view = project_view(_ColourFaces(), museum="m", panorama_key="p", camera=Camera(0, 40, 60, 400, 300), width=128)
        top = tuple(int(v) for v in view.image[2, 64])
        self.assertIn(top, {(0, 0, 0), _ColourFaces.COLOURS["u"]})


# ----- recorte -----

class TestSegmenterHelpers(unittest.TestCase):
    def test_choose_mask_prefers_the_whole_object(self) -> None:
        # as 3 propostas são encaixadas (todo, parte, pormenor): a maior dentro da janela
        self.assertEqual(choose_mask([0.9, 0.8, 0.7], [0.6, 0.2, 0.05]), 1)    # 0,6 passa dos 30 %
        self.assertEqual(choose_mask([0.7, 0.9, 0.8], [0.03, 0.01, 0.004]), 0)  # a maior, mesmo com score menor
        self.assertEqual(choose_mask([0.9, 0.8, 0.7], [0.6, 0.5, 0.45]), 0)    # nenhuma na janela: a de melhor score
        self.assertEqual(choose_mask([0.95, 0.3, 0.2], [0.001, 0.3, 0.02]), 0)  # as outras valem < metade: fica a melhor
        self.assertEqual(choose_mask([0.9, 0.2, 0.1], [0.05, 0.2, 0.3]), 0)    # score < metade do melhor não conta
        self.assertEqual(choose_mask([0.9, 0.8, 0.7], [0.9, 0.5, 0.1], min_area=0.15, max_area=0.98), 0)  # vista apertada

    def test_polygon_of_a_rectangle(self) -> None:
        mask = np.zeros((200, 300), dtype=bool)
        mask[50:150, 80:220] = True
        polygon = mask_polygon(mask)
        self.assertGreaterEqual(len(polygon), 12)
        self.assertLessEqual(len(polygon), 40)
        xs, ys = zip(*polygon)
        self.assertTrue(all(0 <= v <= 1 for v in xs + ys))
        self.assertAlmostEqual(min(xs), 80 / 300, delta=0.01)
        self.assertAlmostEqual(max(xs), 219 / 300, delta=0.01)
        self.assertAlmostEqual(min(ys), 50 / 200, delta=0.01)

    def test_polygon_of_a_disc_is_simplified(self) -> None:
        yy, xx = np.mgrid[:240, :320]
        mask = (xx - 160) ** 2 + (yy - 120) ** 2 < 70 ** 2
        polygon = mask_polygon(mask)
        self.assertGreaterEqual(len(polygon), 12)
        self.assertLessEqual(len(polygon), 40)
        contour = trace_contour(mask)
        self.assertGreater(len(contour), 300)   # o contorno cru tem centenas de píxeis
        self.assertEqual(mask_polygon(np.zeros((10, 10), dtype=bool)), [])

    def test_douglas_peucker_keeps_corners(self) -> None:
        square = [(0, 0), (5, 0), (10, 0), (10, 5), (10, 10), (5, 10), (0, 10), (0, 5), (0, 0)]
        self.assertEqual(douglas_peucker(square, 1.0), [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])

    def test_crop_and_cutout(self) -> None:
        image = np.full((100, 200, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 200), dtype=bool)
        mask[20:60, 50:150] = True
        cutout = crop_and_cutout(image, mask)
        x0, y0, x1, y1 = cutout.box
        self.assertEqual((x0, y0, x1, y1), (42, 16, 158, 64))   # 8 % de margem: 8 px em x, 4 px em y (arredondado para cima)
        self.assertTrue(cutout.crop_jpeg.startswith(b"\xff\xd8") and cutout.cutout_jpeg.startswith(b"\xff\xd8"))
        self.assertLess(len(cutout.cutout_jpeg), 60_000)
        self.assertEqual(crop_and_cutout(image, mask, whole=True).box, (0, 0, 200, 100))


# ----- pedido, scores e boosts -----

def _point_metadata(**overrides):
    base = {
        "request_id": "point-1", "panorama_key": PANO, "panorama_title": "Sala 52",
        "camera": {"yaw": -0.07, "pitch": 0.48, "hfov": 100, "vfov": 62.3},
        "viewport": {"width": 999, "height": 759}, "point": {"x": 0.65, "y": 0.49},
        "ray": {"yaw": 19.6, "pitch": 1.1}, "hotspot_overlay_id": None,
    }
    base.update(overrides)
    return base


class TestRequestAndScores(unittest.TestCase):
    def test_parse(self) -> None:
        request = PointRequest.parse(_point_metadata())
        self.assertEqual((request.request_id, request.panorama_key, request.camera.width, request.point_x), ("point-1", PANO, 999, 0.65))
        self.assertEqual(request.ray_yaw, 19.6)
        self.assertIsNone(PointRequest.parse(_point_metadata(point={"x": 1.5, "y": 0})))
        self.assertIsNone(PointRequest.parse(_point_metadata(panorama_key="")))
        self.assertIsNone(PointRequest.parse(_point_metadata(camera={"yaw": 0, "pitch": 0, "hfov": 0})))
        self.assertIsNone(PointRequest.parse("x"))
        self.assertTrue(PointRequest.parse(_point_metadata(request_id="../x")).request_id.startswith("point-"))

    def test_normalized_score_is_the_cosine(self) -> None:
        self.assertAlmostEqual(normalized_score(0.9), 0.8)
        self.assertAlmostEqual(normalized_score(0.5), 0.0)
        self.assertEqual(normalized_score(None), 0.0)
        self.assertEqual(normalized_score(0.2), 0.0)

    def test_rank_groups_by_piece_and_boosts_tour_and_panorama(self) -> None:
        service = PointAskService.__new__(PointAskService)
        service.settings = Settings(_env_file=None, POINT_PANORAMA_BOOST=1.3, POINT_IN_TOUR_BOOST=1.15, POINT_CANDIDATES=3)
        hits = [
            {"artifact_id": "a", "inventory_number": "1 Pint", "score": 0.86, "local_path": "Images/a1.jpg", "artifact_title": "A"},   # cos 0,72
            {"artifact_id": "a", "inventory_number": "1 Pint", "score": 0.80, "local_path": "Images/a2.jpg", "artifact_title": "A"},
            {"artifact_id": "b", "inventory_number": "2 Pint", "score": 0.88, "local_path": "Images/b.jpg", "artifact_title": "B"},    # cos 0,76 (visita)
            {"artifact_id": "c", "inventory_number": "3 Pint", "score": 0.85, "local_path": "Images/c.jpg", "artifact_title": "C"},    # cos 0,70 (panorama)
            {"artifact_id": "d", "inventory_number": "4 Pint", "score": 0.84, "local_path": "Images/d.jpg", "artifact_title": "D"},
        ]
        ranked = service.rank(hits, in_tour_keys={"2PINT", "3PINT"}, panorama_keys={"3PINT"}, normalize_key=lambda s: s.replace(" ", "").upper())
        self.assertEqual([c.artifact_id for c in ranked], ["c", "b", "a"])
        self.assertAlmostEqual(ranked[0].score, 0.91, places=2)   # 0,70 × 1,3
        self.assertAlmostEqual(ranked[1].score, 0.874, places=3)  # 0,76 × 1,15
        self.assertAlmostEqual(ranked[2].score, 0.72, places=2)
        self.assertEqual(ranked[2].image_name, "Images/a1.jpg")   # a melhor imagem da peça
        self.assertTrue(ranked[0].in_panorama and ranked[1].in_tour and not ranked[2].in_tour)

    def test_asset_store_ttl(self) -> None:
        store = PointAssetStore(ttl_seconds=0.05)
        store.put("r1", b"crop", b"cut", museum_slug="mnaa", panorama_key=PANO)
        self.assertEqual(store.get("r1").crop_jpeg, b"crop")
        import time

        time.sleep(0.08)
        self.assertIsNone(store.get("r1"))


# ----- de ponta a ponta -----

DOCS = {
    "raiz:1874": {"artifact_id": "raiz:1874", "inventory_number": "1874 Pint", "title": "Fuga para o Egipto", "museum_id": "mnaa",
                  "category": "Pintura", "creators": ["Tiepolo"], "date_or_period": "1765", "in_tour": True, "image_count": 1},
    "raiz:14": {"artifact_id": "raiz:14", "inventory_number": "14 Pint", "title": "Fuga para o Egipto", "museum_id": "mnaa",
                "category": "Pintura", "in_tour": False, "image_count": 1},
    "raiz:9": {"artifact_id": "raiz:9", "inventory_number": "9 Pint", "title": "Paisagem", "museum_id": "mnaa",
               "category": "Pintura", "in_tour": False, "image_count": 1},
}


class _Gateway:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def search_similar_images_page(self, **kwargs):
        self.calls.append(("knn", kwargs["page_size"]))
        return SimpleNamespace(results=list(self.hits), total=len(self.hits), query_body={})

    async def fetch_artifacts_by_ids(self, *, artifact_ids, **_k):
        return [dict(DOCS[i]) for i in artifact_ids if i in DOCS]

    async def fetch_artifacts_by_inventory_numbers(self, *, inventory_numbers, **_k):
        return [dict(d) for d in DOCS.values() if d["inventory_number"] in inventory_numbers]

    async def fetch_images_by_artifact_ids(self, *, artifact_ids, **_k):
        return [{"artifact_id": i, "image_id": f"img-{i}", "local_path": f"Images/{i.split(':')[1]}.jpg",
                 "inventory_number": DOCS[i]["inventory_number"], "museum_id": "mnaa", "score": 1.0} for i in artifact_ids if i in DOCS]

    async def fetch_set_object_counts(self, *, set_ids):
        return {}


class _Embedder:
    def __init__(self):
        self.calls = 0

    async def embed_multimodal_image_bytes(self, *, image_bytes, text=None):
        self.calls += 1
        return [0.1] * 8


class _LLM:
    def __init__(self, text):
        self.text, self.prompts = text, []

    async def generate(self, *, message, response_format=ResponseFormatObject(type="text"), **_k):
        self.prompts.append(message)
        return LLMResponse(text=self.text, model="fake", response_format=response_format, finish_reason="stop")

    async def generate_stream(self, *, message, response_format, on_delta=None, **_k):
        self.prompts.append(message)
        for start in range(0, len(self.text), 9):
            await on_delta(self.text[start:start + 9])
        return LLMResponse(text=self.text, model="fake", response_format=response_format, finish_reason="stop")


class _FakePoint(PointAskService):
    """Sem GPU: a vista é sintética e o recorte é um retângulo à volta do ponto."""

    def __init__(self, settings, *, fail_render=False):
        self.settings = settings
        self.assets = PointAssetStore(ttl_seconds=60)
        self.fail_render = fail_render
        self.describe_calls = 0
        self.segmenter = SimpleNamespace(ready=True)

    async def render(self, request, *, museum_slug):
        if self.fail_render:
            raise FileNotFoundError("sem tiles")
        image = np.full((778, 1024, 3), 180, dtype=np.uint8)
        return ProjectedView(image=image, camera=request.camera, faces_used=("f",), elapsed_ms=50.0)

    async def cut(self, outcome, *, museum_slug):
        image = outcome.view.image
        h, w = image.shape[:2]
        x, y = int(outcome.request.point_x * w), int(outcome.request.point_y * h)
        mask = np.zeros((h, w), dtype=bool)
        mask[max(0, y - 60): y + 60, max(0, x - 50): x + 50] = True
        outcome.point_px = (x, y)
        outcome.timings_ms["sam"] = 30.0
        outcome.segment_score, outcome.segment_area = 0.95, float(mask.mean())
        outcome.mask_polygon = mask_polygon(mask)
        cut = crop_and_cutout(image, mask)
        outcome.crop_jpeg = cut.crop_jpeg
        self.assets.put(outcome.request.request_id, cut.crop_jpeg, cut.cutout_jpeg, museum_slug=museum_slug, panorama_key=outcome.request.panorama_key)
        prefix = f"/api/v1/chat/point/{outcome.request.request_id}"
        outcome.crop_url, outcome.cutout_url = f"{prefix}/crop.jpg", f"{prefix}/cutout.jpg"

    async def describe_crop(self, crop_jpeg, *, prompt, system_prompt, on_delta=None):
        self.describe_calls += 1
        text = "Vejo uma moldura dourada com uma pintura clara."
        for start in range(0, len(text), 10):
            await on_delta(text[start:start + 10])
        return text


class TestPointAndAsk(_Base):
    POI_BY_MUSEUM = {"mnaa": [
        {"overlayId": FUGA_OVERLAY, "panoramaKey": PANO, "inventoryIds": ["1874 Pint"], "location": "Sala 52", "title": "Fuga para o Egipto"},
        {"overlayId": "overlay_x", "panoramaKey": "panorama_outro", "inventoryIds": ["9 Pint"], "location": "Sala 1", "title": "Paisagem"},
    ]}

    def _service(self, hits, *, llm_text="Esta pintura de Tiepolo mostra a Fuga para o Egipto.", fail_render=False, enabled=True):
        settings = self._settings(POINT_AND_ASK_ENABLED=enabled, CHAT_DESCRIBE_ARTIFACT_INTENT=True, CHAT_COMPACT_ANSWER_CONTEXT=True,
                                  CHAT_STREAM_FINAL_ANSWER=True, POINT_MATCH_THRESHOLD=0.8, POINT_LOW_CONFIDENCE_FLOOR=0.45)
        service = super()._service(settings, llm=_LLM(llm_text))
        service.opensearch_gateway = _Gateway(hits)
        service.embedding_provider = _Embedder()
        service.point_ask_service = _FakePoint(settings, fail_render=fail_render)
        return service

    def _ask(self, service, *, intent="point_and_ask", message="O que é isto?", **point_overrides):
        events: list[dict] = []

        async def status_cb(event):
            events.append(event)

        request = self._request(message, museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga")
        request = request.model_copy(update={"metadata": {"session_id": "s1", "intent": intent, "point": _point_metadata(**point_overrides)}})
        return asyncio.run(service.handle_message(request, status_cb=status_cb)), events

    @staticmethod
    def _stages(events):
        return [e["payload"]["stage"] for e in events if e["type"] == "point"]

    HITS = [
        {"artifact_id": "raiz:1874", "inventory_number": "1874 Pint", "score": 0.92, "local_path": "Images/1874.jpg", "artifact_title": "Fuga para o Egipto"},
        {"artifact_id": "raiz:14", "inventory_number": "14 Pint", "score": 0.84, "local_path": "Images/14.jpg", "artifact_title": "Fuga para o Egipto"},
        {"artifact_id": "raiz:9", "inventory_number": "9 Pint", "score": 0.80, "local_path": "Images/9.jpg", "artifact_title": "Paisagem"},
    ]

    def test_identified_piece_streams_the_description(self) -> None:
        service = self._service(self.HITS)
        response, events = self._ask(service)
        self.assertEqual(self._stages(events), ["captured", "segmented", "candidates", "decided"])
        decided = next(e["payload"] for e in events if e["type"] == "point" and e["payload"]["stage"] == "decided")
        # sem boosts: 1874 Pint cos 0,84 ≥ 0,8; 14 Pint 0,68; 9 Pint 0,60 (o hotspot no panorama já não pesa)
        self.assertEqual(decided["chosen"], {"artifact_id": "raiz:1874", "score": 0.84})
        self.assertFalse(decided["hotspot_match"])
        self.assertEqual([c["inventory_number"] for c in decided["candidates"]], ["1874 Pint", "14 Pint", "9 Pint"])
        self.assertEqual(decided["candidates"][0]["image_name"], "Images/1874.jpg")
        segmented = next(e["payload"] for e in events if e["type"] == "point" and e["payload"]["stage"] == "segmented")
        self.assertGreaterEqual(len(segmented["mask"]), 12)
        self.assertTrue(all(0 <= x <= 1 and 0 <= y <= 1 for x, y in segmented["mask"]))
        self.assertEqual(segmented["cutout_url"], "/api/v1/chat/point/point-1/cutout.jpg")
        statuses = [e["key"] for e in events if e["type"] == "status"]
        self.assertEqual(statuses[:2], ["status.point_segmenting", "status.point_searching"])
        self.assertIn("status.generating_final_answer", statuses)
        self.assertTrue(any(e["type"] == "token" for e in events))
        self.assertEqual(response.reply, "Esta pintura de Tiepolo mostra a Fuga para o Egipto.")
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["1874 Pint", "14 Pint", "9 Pint"])
        self.assertEqual([t.inventory_id for t in response.navigation_targets], ["1874 Pint", "9 Pint"])
        self.assertEqual(response.point["stage"], "decided")
        self.assertEqual(response.point["chosen"]["artifact_id"], "raiz:1874")
        self.assertIn("PEÇA: Fuga para o Egipto", service.llm_service.prompts[0])
        event = service.query_logger.events[-1]
        self.assertEqual(event["retrieval_mode"], "point_and_ask")
        point_log = event["point"]
        self.assertEqual((point_log["panorama_key"], point_log["chosen"], point_log["threshold"]), (PANO, "raiz:1874", 0.8))
        self.assertAlmostEqual(point_log["ray"]["yaw"], 19.6, delta=0.1)
        self.assertEqual(len(point_log["candidates"]), 3)
        for key in ("projection", "sam", "search", "decided", "llm_first_token"):
            self.assertIn(key, point_log["timings_ms"], key)
        self.assertNotIn("crop_jpeg", str(point_log))
        self.assertEqual(service.embedding_provider.calls, 1)

    def test_between_floor_and_threshold_presents_without_certainty(self) -> None:
        hits = [dict(h, score=0.82) for h in self.HITS]   # cos 0,64: abaixo do limiar 0,8, acima do chão 0,45
        service = self._service(hits, llm_text="Não tenho a certeza, mas parece ser a Fuga para o Egipto.")
        response, events = self._ask(service)
        decided = next(e["payload"] for e in events if e["type"] == "point" and e["payload"]["stage"] == "decided")
        self.assertEqual(decided["chosen"], {"artifact_id": "raiz:1874", "score": 0.64})
        self.assertEqual(decided["confidence"], "low")
        self.assertEqual(response.reply, "Não tenho a certeza, mas parece ser a Fuga para o Egipto.")
        self.assertEqual(len(response.artifact_results), 3)
        self.assertEqual(response.point["confidence"], "low")
        prompt = service.llm_service.prompts[0]
        self.assertIn("a identificação não é segura (correspondência de 64 %", prompt)
        self.assertIn("Não tenho a certeza, mas parece ser", prompt)
        self.assertEqual(service.query_logger.events[-1]["point"]["confidence"], "low")

    def test_above_threshold_has_no_caution(self) -> None:
        service = self._service(self.HITS)
        self._ask(service)
        self.assertNotIn("não é segura", service.llm_service.prompts[0])

    def test_defaults_never_say_not_found_with_a_candidate(self) -> None:
        # por omissão: limiar 0,70 e chão 0: um candidato a 0,40 apresenta-se sem certeza
        hits = [dict(h, score=0.70) for h in self.HITS]   # cos 0,40
        service = self._service(hits, llm_text="Não tenho a certeza, mas parece ser a Fuga para o Egipto.")
        service.settings = self._settings(POINT_AND_ASK_ENABLED=True, CHAT_DESCRIBE_ARTIFACT_INTENT=True, CHAT_COMPACT_ANSWER_CONTEXT=True,
                                          CHAT_STREAM_FINAL_ANSWER=True)
        service.point_ask_service.settings = service.settings
        self.assertEqual((service.settings.POINT_MATCH_THRESHOLD, service.settings.POINT_LOW_CONFIDENCE_FLOOR), (0.7, 0.0))
        response, events = self._ask(service)
        decided = next(e["payload"] for e in events if e["type"] == "point" and e["payload"]["stage"] == "decided")
        self.assertEqual((decided["chosen"]["artifact_id"], decided["confidence"]), ("raiz:1874", "low"))
        self.assertIn("correspondência de 40 %", service.llm_service.prompts[0])
        # sem candidatos nenhuns, aí sim «não encontrei»
        service = self._service([])
        response, events = self._ask(service)
        self.assertTrue(response.reply.startswith("Não encontrei esta peça no acervo"))
        self.assertIsNone(response.point["chosen"])

    def test_below_floor_says_not_found_and_keeps_candidates(self) -> None:
        hits = [dict(h, score=0.70) for h in self.HITS]   # cos 0,40 < chão 0,45 (chão explícito no teste)
        service = self._service(hits)
        response, events = self._ask(service)
        decided = next(e["payload"] for e in events if e["type"] == "point" and e["payload"]["stage"] == "decided")
        self.assertIsNone(decided["chosen"])
        self.assertIsNone(decided["confidence"])
        self.assertEqual(len(decided["candidates"]), 3)
        self.assertTrue(response.reply.startswith("Não encontrei esta peça no acervo"))
        self.assertEqual(len(response.artifact_results), 3)
        self.assertIsNone(response.point["chosen"])
        self.assertFalse(any(e["type"] == "token" for e in events))
        self.assertEqual(len(service.llm_service.prompts), 0)
        self.assertEqual(service.query_logger.events[-1]["point"]["chosen"], None)

    def test_hotspot_shortcut_skips_the_search(self) -> None:
        service = self._service(self.HITS)
        response, events = self._ask(service, hotspot_overlay_id=FUGA_OVERLAY)
        decided = next(e["payload"] for e in events if e["type"] == "point" and e["payload"]["stage"] == "decided")
        self.assertEqual(decided["chosen"], {"artifact_id": "raiz:1874", "score": 0.99})
        self.assertTrue(decided["hotspot_match"])
        self.assertEqual(decided["candidates"][0]["image_name"], "Images/1874.jpg")
        self.assertEqual(self._stages(events), ["captured", "segmented", "candidates", "decided"])
        self.assertEqual(service.embedding_provider.calls, 0)
        self.assertEqual(service.opensearch_gateway.calls, [])
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["1874 Pint"])

    def test_render_failure_is_reported(self) -> None:
        service = self._service(self.HITS, fail_render=True)
        response, events = self._ask(service)
        self.assertTrue(response.reply.startswith("Não consegui reconstruir a vista"))
        self.assertEqual(self._stages(events), ["decided"])
        self.assertEqual(service.query_logger.events[-1]["status"], "error")

    def test_describe_point_crop_reuses_the_crop(self) -> None:
        service = self._service(self.HITS)
        self._ask(service)   # guarda o crop do point-1
        response, events = self._ask(service, intent="describe_point_crop", message="Descreve-me o que vês")
        self.assertEqual(response.reply, "Vejo uma moldura dourada com uma pintura clara.")
        self.assertEqual(response.artifact_results, [])
        self.assertEqual(response.point["stage"], "described")
        self.assertTrue(any(e["type"] == "token" for e in events))
        self.assertIn("status.point_describing", [e["key"] for e in events if e["type"] == "status"])
        log = service.query_logger.events[-1]
        self.assertEqual((log["retrieval_mode"], log["point"]["reused_crop"]), ("describe_point_crop", True))
        # sem recorte guardado, reconstrói
        response, _ = self._ask(service, intent="describe_point_crop", request_id="point-2")
        self.assertEqual(response.reply, "Vejo uma moldura dourada com uma pintura clara.")
        self.assertFalse(service.query_logger.events[-1]["point"]["reused_crop"])

    def test_flag_off_or_bad_metadata_goes_the_normal_way(self) -> None:
        service = self._service(self.HITS, enabled=False)
        service._handle_message_with_selection = lambda payload, status_cb=None: _normal()
        response, events = self._ask(service)
        self.assertEqual(response.reply, "normal")
        service = self._service(self.HITS)
        service._handle_message_with_selection = lambda payload, status_cb=None: _normal()
        response, _ = self._ask(service, panorama_key="")
        self.assertEqual(response.reply, "normal")


async def _normal():
    from app.schemas.chat import ChatMessageResponse

    return ChatMessageResponse(reply="normal")


if __name__ == "__main__":
    unittest.main()
