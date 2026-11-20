"""«Ver na tour» para peças sem hotspot, localizadas offline num panorama
(TOUR_OBJECT_TARGETS_ENABLED; poi_tours/panorama-objects-<museu>.json).

    python -m unittest tests.test_tour_object_targets
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.chat import TourNavigationTarget
from app.services.panorama_objects import PanoramaObjectStore, angular_size, object_navigation_target
from app.services.route_planning.graph import TourGraph
from tests.test_tour_location import _Base


def _obj(pano: str, inv: str, score: float, *, accepted_by: str | None = None, area: float = 0.05, yaw: float = 10.0, pitch: float = -2.0, idx: int = 0, title: str = "Peça", half: float = 5.0) -> dict:
    obj = {
        "object_id": f"{pano}:{idx}", "panorama_key": pano, "artifact_id": f"a-{inv}", "inventory_number": inv, "title": title,
        "score": score, "runner_up_score": 0.5, "image_name": f"Images/{inv}.jpg", "yaw": yaw, "pitch": pitch, "area_fraction": area,
        "bbox": {"yaw_min": yaw - half, "yaw_max": yaw + half, "pitch_min": pitch - half, "pitch_max": pitch + half}, "polygon": [[yaw - 5, pitch + 5], [yaw + 5, pitch + 5], [yaw + 5, pitch - 5]],
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
        # um alvo por peça: dos dois hotspots da 5262 fica o do panorama em que a indexação a recortou
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


class _RelatedGateway:
    """Um artefacto num conjunto com três peças relacionadas: uma com hotspot,
    uma localizada offline num panorama e uma que a visita não mostra."""

    RELATED = [
        {"artifact_id": "a-19663", "inventory_number": "19663", "title": "Com hotspot", "in_tour": True},
        {"artifact_id": "a-777", "inventory_number": "777", "title": "Casaca bordada", "in_tour": False},
        {"artifact_id": "a-4242", "inventory_number": "4242", "title": "Fora da visita", "in_tour": False},
    ]

    async def fetch_artifacts_by_ids(self, **_kwargs):
        return [{"artifact_id": "a-1", "inventory_number": "1", "set_ids": ["s1"]}]

    async def fetch_entities_by_ids(self, *, tipo, entity_ids):
        return [{"entity_id": "conjunto:s1", "name": "Conjunto"}] if tipo == "conjunto" else []

    async def fetch_artifacts_by_entity(self, *, tipo, entity_id, top_k, from_offset=0, **_kwargs):
        if tipo != "conjunto":
            return [], 0
        return [dict(d) for d in self.RELATED[from_offset: from_offset + top_k]], len(self.RELATED)


class TestDetailContextTargets(_WithObjects):
    """A ficha do artefacto (detail-context) e /artifacts/related dão o mesmo
    alvo à mesma peça relacionada (2026-11-20: a ficha só dava hotspots)."""

    def _service_with_related(self, **overrides):
        settings = self._settings(**{"CHAT_TOUR_LOCATION_ANSWERS": True, "TOUR_OBJECT_TARGETS_ENABLED": True, **overrides})
        service = self._service(settings)
        service._panorama_object_store = self.store
        service.opensearch_gateway = _RelatedGateway()

        async def _no_images(**_kwargs):
            return {}

        service._build_related_images_lookup = _no_images
        return service

    @staticmethod
    def _targets(items) -> dict:
        return {i.inventory_number: (i.navigation_target.model_dump() if i.navigation_target else None) for i in items}

    def _both(self, service):
        import asyncio

        detail = asyncio.run(service.get_artifact_detail_context(museum_slug="mnt", museum_id="mnt", artifact_id="a-1"))
        page = asyncio.run(service.get_related_artifacts_page(
            museum_slug="mnt", museum_id="mnt", artifact_id="a-1", tipo="conjunto", entity_id="conjunto:s1", offset=0, limit=10))
        return detail.sets[0].artifacts, page.artifacts

    def test_detail_context_and_related_page_give_the_same_targets(self) -> None:
        detail, page = self._both(self._service_with_related())
        self.assertEqual(self._targets(detail), self._targets(page))
        targets = self._targets(detail)
        self.assertEqual(targets["19663"]["source"], "hotspot")
        self.assertEqual((targets["777"]["source"], targets["777"]["panorama_key"], targets["777"]["yaw"], targets["777"]["pitch"]),
                         ("object", "pan_musica", 33.3, -4.4))
        self.assertIsNone(targets["4242"])

    def test_they_agree_with_object_targets_off_too(self) -> None:
        detail, page = self._both(self._service_with_related(TOUR_OBJECT_TARGETS_ENABLED=False))
        self.assertEqual(self._targets(detail), self._targets(page))
        self.assertIsNone(self._targets(detail)["777"])

    def test_in_tour_follows_the_target(self) -> None:
        for items in self._both(self._service_with_related(CHAT_IN_TOUR_FROM_NAVIGATION=True)):
            self.assertEqual({i.inventory_number: i.in_tour for i in items}, {"19663": True, "777": True, "4242": False})
        # sem a flag fica o campo do índice
        for items in self._both(self._service_with_related()):
            self.assertEqual({i.inventory_number: i.in_tour for i in items}, {"19663": True, "777": False, "4242": False})

    def test_in_tour_never_goes_from_yes_to_no(self) -> None:
        service = self._service_with_related(CHAT_IN_TOUR_FROM_NAVIGATION=True)
        item = service._build_related_item({"artifact_id": "a-4242", "inventory_number": "4242", "in_tour": True}, navigation_lookup={})
        self.assertTrue(item.in_tour)
        self.assertIsNone(item.navigation_target)


# ----- A vista mais próxima da peça (TOUR_TARGETS_CLOSEST_VIEW, 2026-11-20) ----- #

# Uma visita em fila: a — b — c — d — e — f, mais as salas dos hotspots do
# test_tour_location (Música ao lado da Música 2; Cantonal ao lado da Cantonal 2).
_LINE = ["pan_a", "pan_b", "pan_c", "pan_d", "pan_e", "pan_f"]
_PAIRS = [*zip(_LINE, _LINE[1:]), ("pan_musica", "pan_musica_2"), ("pan_cantonal", "pan_cantonal_2")]
GRAPH = {
    "museum_slug": "mnt",
    "nodes": [{"panorama_key": k, "playlist_index": i, "room": "Sala"} for i, k in enumerate(
        [*_LINE, "pan_musica", "pan_musica_2", "pan_cantonal", "pan_cantonal_2"])],
    # as ligações só num sentido também contam como vizinhança
    "edges": [{"source": a, "target": b, "kinds": ["adjacent"]} for a, b in _PAIRS],
}
NEAR_OBJECTS = {
    "pan_a": {"objects": [
        _obj("pan_a", "2001", 0.95, half=2, idx=0),                  # o maior cosseno, mas de longe
        _obj("pan_a", "2002", 0.93, half=2, idx=1),
        _obj("pan_a", "2003", 0.95, half=2, idx=2),
        _obj("pan_a", "2004", 0.70, accepted_by="vlm", half=3, idx=3),
    ]},
    "pan_b": {"objects": [
        _obj("pan_b", "2001", 0.85, half=10, yaw=-40.0, pitch=-8.0, idx=0),  # a vista mais próxima
        _obj("pan_b", "2004", 0.99, half=12, idx=1),                 # maior e com mais cosseno, mas não confirmada
        _obj("pan_b", "2005", 0.60, half=12, idx=2),                 # abaixo do mínimo: não conta
    ]},
    "pan_c": {"objects": [
        _obj("pan_c", "2001", 0.82, half=6, idx=0),
        _obj("pan_c", "2003", 0.84, half=4, idx=1),
        _obj("pan_c", "2005", 0.90, half=2, idx=2),
    ]},
    "pan_e": {"objects": [_obj("pan_e", "2003", 0.83, half=9, idx=0)]},      # a dois de c, que está a dois de a
    "pan_f": {"objects": [_obj("pan_f", "2002", 0.85, half=15, idx=0)]},     # enorme, mas noutra zona: peça parecida
    # hotspots: a 38090 vê-se maior do 2.º hotspot do ficheiro; a 5262 só foi recortada no 2.º
    "pan_musica_2": {"objects": [_obj("pan_musica_2", "38090", 0.70, half=3, idx=0)]},
    "pan_cantonal_2": {"objects": [_obj("pan_cantonal_2", "38090", 0.66, half=11, idx=0)]},
    "pan_cantonal": {"objects": [_obj("pan_cantonal", "5262", 0.55, half=4, idx=0)]},
}


class _WithGraph(_Base):
    def setUp(self) -> None:
        super().setUp()
        (Path(self._tmp.name) / "panorama-objects-mnt.json").write_text(json.dumps({"museum_slug": "mnt", "panoramas": NEAR_OBJECTS}), encoding="utf-8")
        (Path(self._tmp.name) / "tour-graph-mnt.json").write_text(json.dumps(GRAPH), encoding="utf-8")
        self.store = PanoramaObjectStore(self._tmp.name, check_interval=0.0)
        self.graph = TourGraph(GRAPH)

    def _enabled(self, **overrides):
        service = self._service(self._settings(CHAT_TOUR_LOCATION_ANSWERS=True, TOUR_OBJECT_TARGETS_ENABLED=True, **overrides))
        service._panorama_object_store = self.store
        return service

    def _panorama(self, service, inventory: str) -> str | None:
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=[{"inventory_number": inventory}])
        return targets[0].panorama_key if targets else None


class TestAngularSize(_WithGraph):
    def test_size_comes_from_the_angular_box(self) -> None:
        self.assertAlmostEqual(angular_size(_obj("p", "1", 0.9, pitch=0.0, half=5)), 100.0)
        # a mesma largura em yaw ocupa menos perto dos polos
        self.assertAlmostEqual(angular_size(_obj("p", "1", 0.9, pitch=-60.0, half=5)), 50.0)
        # a caixa que atravessa os ±180° não fica com 350° de largura
        wrapped = {"bbox": {"yaw_min": 175.0, "yaw_max": -175.0, "pitch_min": -5.0, "pitch_max": 5.0}}
        self.assertAlmostEqual(angular_size(wrapped), 100.0)
        for broken in ({}, {"bbox": None}, {"bbox": {"yaw_min": 1}}, {"bbox": {"yaw_min": "x", "yaw_max": 2, "pitch_min": 0, "pitch_max": 1}}):
            self.assertEqual(angular_size(broken), 0.0)

    def test_graph_nearby_ignores_direction(self) -> None:
        self.assertEqual(self.graph.nearby("pan_c", 1), {"pan_b", "pan_c", "pan_d"})
        self.assertEqual(self.graph.nearby("pan_a", 2), {"pan_a", "pan_b", "pan_c"})
        self.assertEqual(self.graph.nearby("pan_a", 0), {"pan_a"})
        self.assertEqual(self.graph.nearby("pan_nenhum", 2), set())


class TestClosestView(_WithGraph):
    def _locate(self, inventory: str, hops: int = 2) -> str | None:
        located = self.store.locate("mnt", inventory, nearby=lambda key: self.graph.nearby(key, hops))
        return located["panorama_key"] if located else None

    def test_closest_view_of_the_same_place(self) -> None:
        self.assertEqual(self.store.locate("mnt", "2001")["panorama_key"], "pan_a")   # sem grafo: a mais segura
        located = self.store.locate("mnt", "2001", nearby=lambda key: self.graph.nearby(key, 2))
        self.assertEqual((located["panorama_key"], located["yaw"], located["pitch"]), ("pan_b", -40.0, -8.0))

    def test_lookalike_in_another_zone_is_ignored(self) -> None:
        self.assertEqual(self._locate("2002"), "pan_a")

    def test_place_follows_neighbours_that_also_see_the_piece(self) -> None:
        self.assertEqual(self._locate("2003"), "pan_e")
        # com um só passo, c já não é vizinho de a
        self.assertEqual(self._locate("2003", hops=1), "pan_a")

    def test_only_definitive_locations_count(self) -> None:
        # confirmada pelo VLM em a; em b só cosseno (≥ 0,80 → definitiva) e maior
        self.assertEqual(self._locate("2004"), "pan_b")
        # a de b está abaixo do mínimo: fica a de c
        self.assertEqual(self._locate("2005"), "pan_c")

    def test_broken_graph_keeps_the_safest(self) -> None:
        def broken(_key: str):
            raise RuntimeError("grafo estragado")
        self.assertEqual(self.store.locate("mnt", "2001", nearby=broken)["panorama_key"], "pan_a")

    def test_view_size(self) -> None:
        self.assertGreater(self.store.view_size("mnt", "38090", "pan_cantonal_2"), self.store.view_size("mnt", " 38090 ", "pan_musica_2"))
        for args in (("mnt", "38090", "pan_a"), ("mnt", "0000", "pan_a"), ("mnaa", "38090", "pan_musica_2"), ("mnt", "", "pan_a"), ("mnt", "38090", None)):
            self.assertEqual(self.store.view_size(*args), 0.0)


class TestClosestTargets(_WithGraph):
    def test_object_target_is_the_closest_view(self) -> None:
        service = self._enabled()
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=[{"inventory_number": "2001"}])
        self.assertEqual([(t.source, t.panorama_key, t.overlay_id, t.yaw, t.pitch) for t in targets], [("object", "pan_b", "object:pan_b:0", -40.0, -8.0)])
        self.assertEqual([self._panorama(service, i) for i in ("2002", "2003")], ["pan_a", "pan_e"])

    def test_hops_are_configurable(self) -> None:
        self.assertEqual(self._panorama(self._enabled(TOUR_TARGETS_CLOSEST_VIEW_HOPS=1), "2003"), "pan_a")

    def test_disabled_keeps_the_safest_identification(self) -> None:
        service = self._enabled(TOUR_TARGETS_CLOSEST_VIEW=False)
        self.assertEqual([self._panorama(service, i) for i in ("2001", "2003")], ["pan_a", "pan_a"])

    def test_without_graph_file_keeps_the_safest_identification(self) -> None:
        (Path(self._tmp.name) / "tour-graph-mnt.json").unlink()
        self.assertEqual(self._panorama(self._enabled(), "2001"), "pan_a")

    def test_hotspot_of_the_closest_panorama(self) -> None:
        service = self._enabled()
        docs = [{"inventory_number": i} for i in ("38090", "5262", "19663", "4703")]
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=docs)
        # um alvo por peça, pela ordem pedida; nenhuma fica de fora por outra ter dois hotspots
        self.assertEqual(
            [(t.inventory_id, t.source, t.overlay_id, t.panorama_key) for t in targets],
            [("38090", "hotspot", "ov7", "pan_cantonal_2"), ("5262", "hotspot", "ov6", "pan_cantonal"),
             ("19663", "hotspot", "ov1", "pan_musica"), ("4703", "hotspot", "ov5", "pan_hall")],
        )

    def test_hotspots_never_cut_out_keep_file_order(self) -> None:
        (Path(self._tmp.name) / "panorama-objects-mnt.json").write_text(json.dumps({"museum_slug": "mnt", "panoramas": {}}), encoding="utf-8")
        service = self._enabled()
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=[{"inventory_number": "38090"}, {"inventory_number": "5262"}])
        self.assertEqual([(t.overlay_id, t.panorama_key) for t in targets], [("ov3", "pan_musica_2"), ("ov2", "pan_musica")])

    def test_hotspots_do_not_need_object_targets(self) -> None:
        service = self._service(self._settings(CHAT_TOUR_LOCATION_ANSWERS=True))
        service._panorama_object_store = self.store
        self.assertEqual(self._panorama(service, "38090"), "pan_cantonal_2")
        self.assertIsNone(self._panorama(service, "2001"))   # sem TOUR_OBJECT_TARGETS_ENABLED não há alvos de objetos

    def test_disabled_keeps_the_first_hotspots(self) -> None:
        service = self._enabled(TOUR_TARGETS_CLOSEST_VIEW=False)
        targets = service._resolve_navigation_targets(museum_slug="mnt", museum_id="mnt", docs=[{"inventory_number": "38090"}])
        self.assertEqual([(t.overlay_id, t.panorama_key) for t in targets], [("ov3", "pan_musica_2")])

    def test_hotspots_only_lookup(self) -> None:
        service = self._enabled()
        targets = service._resolve_target_dicts(museum_slug="mnt", museum_id="mnt", inventories=["38090", "2001"], objects=False)
        self.assertEqual([(t["inventory_id"], t["overlay_id"]) for t in targets], [("38090", "ov7")])
