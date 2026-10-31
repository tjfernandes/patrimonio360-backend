"""«Ver na tour» para peças sem hotspot, localizadas offline num panorama
(TOUR_OBJECT_TARGETS_ENABLED; poi_tours/panorama-objects-<museu>.json).

    python -m unittest tests.test_tour_object_targets
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.chat import TourNavigationTarget
from app.services.panorama_objects import PanoramaObjectStore, object_navigation_target
from tests.test_tour_location import _Base


def _obj(pano: str, inv: str, score: float, *, accepted_by: str | None = None, area: float = 0.05, yaw: float = 10.0, pitch: float = -2.0, idx: int = 0, title: str = "Peça") -> dict:
    obj = {
        "object_id": f"{pano}:{idx}", "panorama_key": pano, "artifact_id": f"a-{inv}", "inventory_number": inv, "title": title,
        "score": score, "runner_up_score": 0.5, "image_name": f"Images/{inv}.jpg", "yaw": yaw, "pitch": pitch, "area_fraction": area,
        "bbox": {"yaw_min": yaw - 5, "yaw_max": yaw + 5, "pitch_min": pitch - 5, "pitch_max": pitch + 5}, "polygon": [[yaw - 5, pitch + 5], [yaw + 5, pitch + 5], [yaw + 5, pitch - 5]],
        "prompt": "artwork",
    }
    if accepted_by:
        obj["accepted_by"] = accepted_by
    return obj


# A visita dos testes (test_tour_location.POI) tem hotspots para 19663, 5262, 38090, 4703/4704.
OBJECTS = {
    # peça SEM hotspot confirmada pelo VLM, vista em dois panoramas: o melhor é o confirmado
    "pan_musica": {"objects": [
        _obj("pan_musica", "777", 0.71, accepted_by="vlm", yaw=33.3, pitch=-4.4, idx=0, title="Casaca bordada"),
        _obj("pan_musica", "5262", 0.95, accepted_by="vlm", idx=1, title="Leque"),   # tem hotspot: não duplica
        _obj("pan_musica", "888", 0.79, idx=2, title="Chapéu de plumas"),            # só cosseno, abaixo de 0,80
    ]},
    "pan_cantonal": {"objects": [
        _obj("pan_cantonal", "777", 0.93, yaw=-120.0, pitch=1.0, idx=0, title="Casaca bordada"),  # cosseno maior mas não confirmado
        _obj("pan_cantonal", "999", 0.86, yaw=70.5, pitch=12.25, idx=1, title="Botas"),           # só cosseno, ≥ 0,80
    ]},
    "pan_desconhecido": {"objects": [
        _obj("pan_desconhecido", "1010", 0.99, accepted_by="hotspot", idx=0, title="Sem sala"),
    ]},
}
TOUR_LOCATIONS = {"mnt": [{"room": "Sala da Música", "panorama_keys": ["pan_musica", "pan_musica_2"]}]}


class _WithObjects(_Base):
    TOUR_LOCATIONS = TOUR_LOCATIONS

    def setUp(self) -> None:
        super().setUp()
        (Path(self._tmp.name) / "panorama-objects-mnt.json").write_text(json.dumps({
            "museum_slug": "mnt", "generated_at": "2026-10-30T09:00:00+00:00", "panoramas": OBJECTS,
        }), encoding="utf-8")
        self.store = PanoramaObjectStore(self._tmp.name, check_interval=0.0)

    def _enabled(self, **overrides):
        settings = self._settings(CHAT_TOUR_LOCATION_ANSWERS=True, TOUR_OBJECT_TARGETS_ENABLED=True, **overrides)
        service = self._service(settings)
        service._panorama_object_store = self.store
        return service


class TestLocate(_WithObjects):
    def test_prefers_confirmed_over_higher_cosine(self) -> None:
        located = self.store.locate("mnt", "777")
        self.assertEqual((located["panorama_key"], located["accepted_by"], located["yaw"]), ("pan_musica", "vlm", 33.3))

    def test_cosine_only_needs_min_score(self) -> None:
        self.assertIsNone(self.store.locate("mnt", "888"))
        self.assertEqual(self.store.locate("mnt", "888", min_score=0.75)["panorama_key"], "pan_musica")
        self.assertEqual(self.store.locate("mnt", "999")["panorama_key"], "pan_cantonal")

    def test_unknown_inputs(self) -> None:
        self.assertIsNone(self.store.locate("mnt", "0000"))
        self.assertIsNone(self.store.locate("mnaa", "777"))
        self.assertIsNone(self.store.locate("mnt", ""))
        # a normalização do inventário é a dos POIs («777 » = «777»)
        self.assertIsNotNone(self.store.locate("mnt", " 777 "))

    def test_reload_clears_index(self) -> None:
        self.assertIsNotNone(self.store.locate("mnt", "777"))
        (Path(self._tmp.name) / "panorama-objects-mnt.json").write_text(json.dumps({"museum_slug": "mnt", "panoramas": {}}), encoding="utf-8")
        self.assertIsNone(self.store.locate("mnt", "777"))

    def test_object_navigation_target_shape(self) -> None:
        target = object_navigation_target(self.store.locate("mnt", "999"), inventory_id="999", location="Sala Cantonal")
        self.assertEqual(target, {
            "overlay_id": "object:pan_cantonal:1", "panorama_key": "pan_cantonal", "inventory_id": "999", "location": "Sala Cantonal",
            "title": "Botas", "source": "object", "yaw": 70.5, "pitch": 12.25, "score": 0.86,
        })
        model = TourNavigationTarget(**target)
        self.assertEqual((model.source, model.yaw, model.pitch), ("object", 70.5, 12.25))
        # os alvos dos hotspots continuam iguais, com a origem por omissão
        hotspot = TourNavigationTarget(overlay_id="ov1", panorama_key="pan_musica", inventory_id="19663")
        self.assertEqual((hotspot.source, hotspot.yaw, hotspot.pitch, hotspot.score), ("hotspot", None, None, None))


class TestPanoramaRoom(_WithObjects):
    def test_room_from_tour_place_then_hotspots_then_none(self) -> None:
        self.assertEqual(self.nav.panorama_room(museum_slug="mnt", museum_id="mnt", panorama_key="pan_musica"), "Sala da Música")
        self.assertEqual(self.nav.panorama_room(museum_slug="mnt", museum_id="mnt", panorama_key="pan_cantonal"), "Sala Cantonal")
        self.assertIsNone(self.nav.panorama_room(museum_slug="mnt", museum_id="mnt", panorama_key="pan_desconhecido"))


class TestResolveTargets(_WithObjects):
    DOCS = [
        {"inventory_number": "19663"},  # hotspot
        {"inventory_number": "777"},    # sem hotspot, VLM
        {"inventory_number": "888"},    # sem hotspot, cosseno baixo
        {"inventory_number": "999"},    # sem hotspot, cosseno alto
        {"inventory_number": "1010"},   # sem hotspot, panorama sem sala
        {"inventory_number": "4242"},   # não localizada
    ]

    def test_disabled_keeps_hotspots_only(self) -> None:
        service = self._service()
        service._panorama_object_store = self.store
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=self.DOCS)
        self.assertEqual([(t.inventory_id, t.source) for t in targets], [("19663", "hotspot")])

    def test_enabled_adds_objects_for_pieces_without_hotspot(self) -> None:
        service = self._enabled()
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=self.DOCS)
        self.assertEqual(
            [(t.inventory_id, t.source, t.panorama_key) for t in targets],
            [("19663", "hotspot", "pan_musica"), ("777", "object", "pan_musica"), ("999", "object", "pan_cantonal"), ("1010", "object", "pan_desconhecido")],
        )
        by_inv = {t.inventory_id: t for t in targets}
        self.assertEqual(by_inv["19663"].overlay_id, "ov1")
        self.assertEqual((by_inv["777"].overlay_id, by_inv["777"].location, by_inv["777"].title), ("object:pan_musica:0", "Sala da Música", "Casaca bordada"))
        self.assertEqual((by_inv["777"].yaw, by_inv["777"].pitch, by_inv["777"].score), (33.3, -4.4, 0.71))
        self.assertEqual(by_inv["999"].location, "Sala Cantonal")
        self.assertIsNone(by_inv["1010"].location)
        # o payload da API leva a origem e a direção; o frontend antigo ignora-as sem partir
        dumped = by_inv["777"].model_dump()
        self.assertEqual(dumped["source"], "object")
        self.assertEqual({"overlay_id", "panorama_key", "inventory_id", "location", "title", "source", "yaw", "pitch", "score"}, set(dumped))

    def test_min_score_is_configurable(self) -> None:
        service = self._enabled(TOUR_OBJECT_TARGETS_MIN_SCORE=0.75)
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=[{"inventory_number": "888"}])
        self.assertEqual([(t.inventory_id, t.score) for t in targets], [("888", 0.79)])

    def test_piece_with_hotspot_is_not_duplicated(self) -> None:
        service = self._enabled()
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=[{"inventory_number": "5262"}])
        # resolve_targets devolve no máximo len(inventories) entradas: aqui só a 1.ª dos dois hotspots da peça
        self.assertEqual([(t.source, t.panorama_key) for t in targets], [("hotspot", "pan_musica")])

    def test_museum_without_objects_file(self) -> None:
        service = self._enabled()
        targets = service._resolve_navigation_targets(museum_slug="mnaa", museum_id="mnaa", docs=self.DOCS)
        self.assertEqual(targets, [])

    def test_related_lookup_and_log_entries(self) -> None:
        service = self._enabled()
        lookup = service._build_navigation_lookup_for_related(museum_slug="mnt", museum_id="mnt", related_docs=[{"inventory_number": "777"}, {"inventory_number": "19663"}])
        self.assertEqual({k: v.get("source", "hotspot") for k, v in lookup.items()}, {"777": "object", "19663": "hotspot"})
        item = service._build_related_item({"artifact_id": "a-777", "inventory_number": "777", "title": "Casaca"}, navigation_lookup=lookup, images_by_artifact={})
        self.assertEqual((item.navigation_target.source, item.navigation_target.yaw), ("object", 33.3))
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=[{"inventory_number": "777"}])
        entries = service._log_navigation_target_entries(navigation_targets=targets, artifact_results=[])
        self.assertEqual((entries[0]["source"], entries[0]["overlay_id"]), ("object", "object:pan_musica:0"))
