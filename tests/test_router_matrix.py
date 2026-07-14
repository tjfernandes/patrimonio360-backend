"""E12.7 — router visual-intent matrix (PT/EN), precision/recall gates.

Offline (deterministic router). Also printable for the report via
    python -m tests.test_router_matrix   (prints the confusion + per-group)
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.retrieval.visual_intent import decide_visual_intent  # noqa: E402

MATRIX = json.loads(
    (BACKEND / "benchmarks/multimodal/router_matrix.json").read_text(encoding="utf-8")
)["cases"]


def evaluate():
    tp = fp = tn = fn = 0
    rows = []
    for case in MATRIX:
        decision = decide_visual_intent(case["query"], mode="intent")
        got = decision.use_visual
        expected = case["expect_visual"]
        if expected and got:
            tp += 1
        elif expected and not got:
            fn += 1
        elif not expected and got:
            fp += 1
        else:
            tn += 1
        rows.append((case, got, decision.rule))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "fp_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "fn_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "rows": rows,
    }


class TestRouterMatrix(unittest.TestCase):
    def test_precision_recall_gates(self) -> None:
        m = evaluate()
        self.assertGreaterEqual(m["precision"], 0.85, f"precision {m['precision']:.3f}")
        self.assertGreaterEqual(m["recall"], 0.80, f"recall {m['recall']:.3f}")

    def test_greetings_and_admin_never_visual(self) -> None:
        for case in MATRIX:
            if case["group"] in {"greeting", "admin"}:
                self.assertFalse(
                    decide_visual_intent(case["query"], mode="intent").use_visual,
                    case["query"],
                )


if __name__ == "__main__":
    m = evaluate()
    print(f"precision={m['precision']:.3f} recall={m['recall']:.3f} "
          f"fp_rate={m['fp_rate']:.3f} fn_rate={m['fn_rate']:.3f} "
          f"(tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']})")
    for case, got, rule in m["rows"]:
        flag = "" if got == case["expect_visual"] else "  <-- MISS"
        print(f"  [{case['id']:12s}] exp={case['expect_visual']!s:5s} got={got!s:5s} rule={rule}{flag}")
