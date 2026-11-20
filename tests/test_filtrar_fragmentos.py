"""Filtro dos recortes que são um bocado de outra peça (tools/apontar/filtrar_fragmentos.py).

    python -m unittest tests.test_filtrar_fragmentos
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "apontar"))

import filtrar_fragmentos as F  # noqa: E402

PANO = "panorama_AAA"


def _obj(object_id: str, inventory: str, score: float, box: tuple[float, float, float, float], **extra) -> dict:
    yaw_min, yaw_max, pitch_min, pitch_max = box
    return {
        "object_id": f"{PANO}:{object_id}", "inventory_number": inventory, "title": inventory, "score": score,
        "bbox": {"yaw_min": yaw_min, "yaw_max": yaw_max, "pitch_min": pitch_min, "pitch_max": pitch_max},
        "polygon": [[yaw_min, pitch_max], [yaw_max, pitch_max], [yaw_max, pitch_min], [yaw_min, pitch_min]],
        "accepted_by": "vlm", **extra,
    }


# a escultura inteira e a sua base, que ficou com o n.º de outra peça (o caso de 2026-11-20)
WHOLE = _obj("3", "1206 Esc", 0.874, (-99.4, -82.7, -17.1, 17.2))
BASE = _obj("7", "63 Esc", 0.655, (-99.5, -82.8, -19.7, -7.7))


def _payload(*objects: dict) -> dict:
    return {"museum_slug": "mnaa", "panoramas": {PANO: {"objects": [copy.deepcopy(o) for o in objects]}}}


class FindFragmentsTests(unittest.TestCase):
    def test_base_of_an_identified_piece_is_a_fragment(self) -> None:
        found = F.find_fragments([WHOLE, BASE])
        self.assertEqual([(a["inventory_number"], b["inventory_number"]) for a, b in found], [("63 Esc", "1206 Esc")])

    def test_higher_score_inside_a_lower_one_stays(self) -> None:
        base = dict(BASE, score=0.9)
        self.assertEqual(F.find_fragments([WHOLE, base]), [])

    def test_same_piece_twice_is_not_a_fragment(self) -> None:
        base = dict(BASE, inventory_number="1206 esc")
        self.assertEqual(F.find_fragments([WHOLE, base]), [])

    def test_hotspot_piece_stays(self) -> None:
        base = dict(BASE, accepted_by="hotspot")
        self.assertEqual(F.find_fragments([WHOLE, base]), [])

    def test_container_must_be_clearly_larger(self) -> None:
        twin = _obj("8", "5 Esc", 0.7, (-99.0, -83.0, -17.0, 17.0))
        self.assertEqual(F.find_fragments([WHOLE, twin]), [])

    def test_piece_beside_another_stays(self) -> None:
        beside = _obj("9", "9 Our", 0.7, (-82.0, -70.0, -10.0, 0.0))
        self.assertEqual(F.find_fragments([WHOLE, beside]), [])

    def test_inside_the_box_but_outside_the_outline_stays(self) -> None:
        # contorno em L: a caixa cobre o canto vazio, o contorno não
        outer = _obj("1", "1 Tap", 0.9, (0.0, 40.0, 0.0, 40.0))
        outer["polygon"] = [[0, 40], [40, 40], [40, 20], [20, 20], [20, 0], [0, 0]]
        chair = _obj("2", "2 Mov", 0.7, (24.0, 38.0, 2.0, 18.0))
        self.assertEqual(F.find_fragments([outer, chair]), [])
        inside = _obj("3", "3 Mov", 0.7, (2.0, 18.0, 2.0, 18.0))
        self.assertEqual(len(F.find_fragments([outer, inside])), 1)

    def test_polygon_inside_is_an_area_fraction(self) -> None:
        self.assertGreater(F.polygon_inside(BASE, WHOLE), 0.75)
        self.assertEqual(F.polygon_inside(_obj("9", "9 Our", 0.7, (-60.0, -50.0, 0.0, 5.0)), WHOLE), 0.0)
        self.assertEqual(F.polygon_inside(dict(BASE, polygon=[]), WHOLE), 1.0)


class FilterPayloadTests(unittest.TestCase):
    def test_fragment_leaves_and_report_says_where_it_was(self) -> None:
        payload = _payload(WHOLE, BASE)
        report = F.filter_payload(payload)
        self.assertEqual([o["inventory_number"] for o in payload["panoramas"][PANO]["objects"]], ["1206 Esc"])
        self.assertEqual((report["mantido"], report["apagado_fragmento"], report["apagado_a_mao"]), (1, 1, 0))
        self.assertEqual(report["fragmentos"][0]["inside_object_id"], f"{PANO}:3")

    def test_remove_takes_every_location_of_the_piece(self) -> None:
        far = _obj("12", "63 Esc", 0.70, (138.5, 140.1, -1.6, 2.4))
        payload = _payload(WHOLE, BASE, far)
        report = F.filter_payload(payload, remove={"63 Esc"})
        self.assertEqual([o["inventory_number"] for o in payload["panoramas"][PANO]["objects"]], ["1206 Esc"])
        self.assertEqual((report["apagado_fragmento"], report["apagado_a_mao"]), (1, 1))
        self.assertEqual(report["a_mao"][0]["object_id"], f"{PANO}:12")

    def test_keep_spares_a_fragment(self) -> None:
        payload = _payload(WHOLE, BASE)
        report = F.filter_payload(payload, keep={f"{PANO}:7"})
        self.assertEqual(len(payload["panoramas"][PANO]["objects"]), 2)
        self.assertEqual((report["apagado_fragmento"], report["poupado"]), (0, [f"{PANO}:7"]))

    def test_nothing_to_remove_leaves_the_payload_alone(self) -> None:
        payload = _payload(WHOLE)
        before = copy.deepcopy(payload)
        report = F.filter_payload(payload)
        self.assertEqual(payload, before)
        self.assertEqual((report["mantido"], report["apagado_fragmento"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
