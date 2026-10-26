"""Objetos pré-indexados por panorama (GET /api/v1/chat/tour/objects).

    python -m unittest tests.test_panorama_objects
"""
from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from app.api.routes import chat as chat_routes
from app.services.panorama_objects import PanoramaObjectStore, hotspot_overlays_by_inventory, public_object
from app.services.point_view import Camera, point_in_view, ray_from_point
from app.services.tour_pois import TourPoiStore

PANO = "panorama_AAA"
OBJECT = {
    "object_id": f"{PANO}:1", "artifact_id": "raiz:1874", "inventory_number": "1874 Pint", "title": "Fuga para o Egipto",
    "score": 0.9123, "runner_up_score": 0.61, "image_name": "Images/1874.jpg", "yaw": 2.4, "pitch": -2.5,
    "bbox": {"yaw_min": -8.0, "yaw_max": 12.5, "pitch_min": -14.0, "pitch_max": 9.0},
    "polygon": [[-8.0, 9.0], [12.5, 9.0], [12.5, -14.0], [-8.0, -14.0]], "prompt": "artwork",
}


class _Files(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "panorama-objects-mnaa.json").write_text(json.dumps({
            "museum_slug": "mnaa", "generated_at": "2026-10-26T10:00:00+00:00",
            "panoramas": {PANO: {"views": 16, "segmented": 7, "identified": 1, "objects": [OBJECT]}},
        }), encoding="utf-8")
        (self.dir / "panorama-overlays-inventory-mnaa.json").write_text(json.dumps([
            {"overlayId": "ov-fuga", "panoramaKey": PANO, "inventoryIds": ["1874 Pint"], "title": "Fuga"},
            {"overlayId": "ov-outro", "panoramaKey": "panorama_BBB", "inventoryIds": ["1874 Pint"], "title": "Fuga"},
        ]), encoding="utf-8")
        self.store = PanoramaObjectStore(self.dir, check_interval=0.0)
        self.pois = TourPoiStore(self.dir, check_interval=0.0)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestStore(_Files):
    def test_reads_and_reloads(self) -> None:
        self.assertEqual(self.store.museums(), ["mnaa"])
        self.assertEqual(self.store.panorama("mnaa", PANO)["identified"], 1)
        self.assertIsNone(self.store.panorama("mnaa", "panorama_ZZZ"))
        self.assertIsNone(self.store.panorama("mnt", PANO))
        (self.dir / "panorama-objects-mnaa.json").write_text(json.dumps({"museum_slug": "mnaa", "panoramas": {}}), encoding="utf-8")
        self.assertIsNone(self.store.panorama("mnaa", PANO))

    def test_public_object_and_hotspot_link(self) -> None:
        hotspots = hotspot_overlays_by_inventory(self.pois.entries("mnaa"), PANO)
        self.assertEqual(hotspots, {"1874pint": "ov-fuga"})
        public = public_object(OBJECT, hotspot_overlay_id="ov-fuga")
        self.assertEqual((public["inventoryNumber"], public["score"], public["yaw"], public["pitch"]), ("1874 Pint", 0.912, 2.4, -2.5))
        self.assertEqual(public["polygon"][0], [-8.0, 9.0])
        self.assertEqual(public["bbox"]["yaw_max"], 12.5)
        self.assertEqual(public["hotspotOverlayId"], "ov-fuga")
        self.assertNotIn("runner_up_score", public)


class TestRoute(_Files):
    def _get(self, museum: str, panorama: str) -> dict:
        with mock.patch.object(chat_routes, "get_panorama_object_store", lambda: self.store), \
             mock.patch.object(chat_routes, "get_tour_poi_store", lambda: self.pois):
            response = asyncio.run(chat_routes.get_tour_objects(museum_slug=museum, panorama_key=panorama))
        self.assertEqual(response.headers["cache-control"], "public, max-age=300")
        return json.loads(response.body)

    def test_indexed_panorama(self) -> None:
        body = self._get("mnaa", PANO)
        self.assertTrue(body["indexed"])
        self.assertEqual(body["segmented"], 7)
        self.assertEqual([o["inventoryNumber"] for o in body["objects"]], ["1874 Pint"])
        self.assertEqual(body["objects"][0]["hotspotOverlayId"], "ov-fuga")

    def test_unindexed_panorama_is_empty_not_an_error(self) -> None:
        body = self._get("mnaa", "panorama_ZZZ")
        self.assertFalse(body["indexed"])
        self.assertEqual(body["objects"], [])

    def test_museum_without_file_is_404(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self._get("mnt", PANO)
        self.assertEqual(ctx.exception.status_code, 404)


class TestAngles(unittest.TestCase):
    def test_polygon_in_degrees_round_trips_to_any_camera(self) -> None:
        # o indexador guarda (yaw, pitch) por vértice; o frontend projeta-os com a câmara atual
        index_camera = Camera(yaw=-45.0, pitch=0.0, hfov=90.0, width=1024, height=768)
        yaw, pitch = ray_from_point(index_camera, 0.7, 0.4)
        view_camera = Camera(yaw=-30.0, pitch=5.0, hfov=70.0, width=999, height=759)
        xy = point_in_view(view_camera, yaw, pitch)
        self.assertIsNotNone(xy)
        back = ray_from_point(view_camera, *xy)
        self.assertTrue(math.isclose(back[0], yaw, abs_tol=1e-6) and math.isclose(back[1], pitch, abs_tol=1e-6))
        # atrás da câmara: fora da vista
        self.assertIsNone(point_in_view(view_camera, yaw + 180.0, pitch))


if __name__ == "__main__":
    unittest.main()
