"""Filtro pela vantagem sobre a segunda candidata (tools/apontar/filtrar_margem.py).

    python -m unittest tests.test_filtrar_margem
"""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "apontar"))

import filtrar_margem as F  # noqa: E402


def _obj(object_id: str, score: float, runner_up: float | None, accepted_by: str = "vlm") -> dict:
    return {"object_id": object_id, "inventory_number": object_id, "score": score, "runner_up_score": runner_up, "accepted_by": accepted_by}


CLEAR = _obj("p:0", 0.78, 0.66)          # vantagem 0,12
HIGH = _obj("p:1", 0.87, 0.866)          # sem vantagem, cosseno alto
CLOSE = _obj("p:2", 0.655, 0.6546)       # a «Caridade» de 2026-11-20
MID = _obj("p:3", 0.79, 0.75)            # vantagem 0,04, cosseno abaixo de 0,85
HOTSPOT = _obj("p:4", 0.60, 0.59, accepted_by="hotspot")
ALONE = _obj("p:5", 0.72, None)          # sem segunda candidata


class KeepsTests(unittest.TestCase):
    def test_rule(self) -> None:
        self.assertTrue(F.keeps(CLEAR))
        self.assertTrue(F.keeps(HIGH))
        self.assertFalse(F.keeps(CLOSE))
        self.assertFalse(F.keeps(MID))

    def test_hotspot_always_stays(self) -> None:
        self.assertTrue(F.keeps(HOTSPOT))

    def test_no_runner_up_counts_the_score(self) -> None:
        self.assertTrue(F.keeps(ALONE))
        self.assertFalse(F.keeps(_obj("p:6", 0.07, None)))

    def test_exact_threshold_stays(self) -> None:
        self.assertTrue(F.keeps(_obj("p:7", 0.73, 0.65)))   # 0,08 em vírgula flutuante dá 0,0799…
        self.assertTrue(F.keeps(_obj("p:8", 0.85, 0.84)))


class RestoreTests(unittest.TestCase):
    def test_manual_confirmation_survives_the_filter(self) -> None:
        self.assertTrue(F.keeps(dict(CLOSE, confirmed_by="manual")))

    def test_restore_moves_it_back_marked_as_manual(self) -> None:
        payload = {"museum_slug": "mnaa", "panoramas": {"a": {"objects": [copy.deepcopy(CLEAR)]}}}
        queue = {"museum_slug": "mnaa", "panoramas": {"a": {"objects": [copy.deepcopy(CLOSE)]}, "c": {"objects": [copy.deepcopy(MID)]}},
                 "margin_filter": {"na_fila": 2}}
        restored = F.restore(payload, queue, {"p:2", "p:3"}, when="2026-11-20T17:00:00")
        self.assertEqual([o["object_id"] for o in restored], ["p:2", "p:3"])
        self.assertEqual([o["object_id"] for o in payload["panoramas"]["a"]["objects"]], ["p:0", "p:2"])
        back = payload["panoramas"]["c"]["objects"][0]
        self.assertEqual((back["accepted_by"], back["confirmed_by"], back["confirmed_at"]), ("vlm", "manual", "2026-11-20T17:00:00"))
        self.assertEqual(queue["panoramas"]["a"]["objects"], [])
        self.assertEqual(queue["margin_filter"]["na_fila"], 0)
        self.assertEqual(payload["manual_confirmations"], [{"when": "2026-11-20T17:00:00", "object_ids": ["p:2", "p:3"]}])
        # o filtro, corrido outra vez, não as volta a tirar
        queued, report = F.split_payload(payload)
        self.assertEqual((queued, report["mantido"]), ({}, 3))

    def test_unknown_id_restores_nothing(self) -> None:
        payload, queue = {"panoramas": {}}, {"panoramas": {"a": {"objects": [copy.deepcopy(CLOSE)]}}}
        self.assertEqual(F.restore(payload, queue, {"p:99"}), [])
        self.assertEqual((payload, len(queue["panoramas"]["a"]["objects"])), ({"panoramas": {}}, 1))


class SplitTests(unittest.TestCase):
    def _payload(self) -> dict:
        return {"museum_slug": "mnaa", "panoramas": {
            "a": {"objects": [copy.deepcopy(o) for o in (CLEAR, CLOSE, HOTSPOT)]},
            "b": {"objects": [copy.deepcopy(o) for o in (HIGH, ALONE)]},
            "c": {"objects": [copy.deepcopy(MID)]},
        }}

    def test_split_moves_the_close_ones(self) -> None:
        payload = self._payload()
        queued, report = F.split_payload(payload)
        self.assertEqual([o["object_id"] for o in payload["panoramas"]["a"]["objects"]], ["p:0", "p:4"])
        self.assertEqual(len(payload["panoramas"]["b"]["objects"]), 2)
        self.assertEqual(payload["panoramas"]["c"]["objects"], [])
        self.assertEqual({k: [o["object_id"] for o in v] for k, v in queued.items()}, {"a": ["p:2"], "c": ["p:3"]})
        self.assertEqual((report["mantido"], report["a_auditar"]), (4, 2))

    def test_queue_keeps_the_whole_record_and_does_not_repeat(self) -> None:
        queued, report = F.split_payload(self._payload())
        queue = F.merge_queue({}, "mnaa", queued, report)
        self.assertEqual(queue["panoramas"]["a"]["objects"][0], CLOSE)
        again = F.merge_queue(queue, "mnaa", queued, report)
        self.assertEqual(again["margin_filter"]["na_fila"], 2)
        more = F.merge_queue(again, "mnaa", {"a": [_obj("p:9", 0.7, 0.69)]}, report)
        self.assertEqual(more["margin_filter"]["na_fila"], 3)


if __name__ == "__main__":
    unittest.main()
