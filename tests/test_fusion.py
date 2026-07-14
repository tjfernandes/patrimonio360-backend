"""Testes unitários das primitivas de fusão (Fase 3, Etapa 3).

Puro/offline: sem OpenSearch, sem settings, sem modelos.
    python -m unittest tests.test_fusion
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.retrieval.fusion import (  # noqa: E402
    BranchHit,
    FusedResult,
    apply_score_floor,
    group_image_hits_by_artifact,
    promote_in_tour_within_margin,
    weighted_rrf,
)


def hit(aid, rank, score=None, img=None):
    return BranchHit(artifact_id=aid, rank=rank, score=score, matched_image_id=img)


class TestWeightedRRFFormula(unittest.TestCase):
    def test_formula_matches_hand_computation(self) -> None:
        results = weighted_rrf(
            {
                "artifact_search": [hit("A", 1, 12.3), hit("B", 2, 11.0)],
                "text_to_image": [hit("B", 1, 0.41, img="imgB"), hit("C", 2, 0.35, img="imgC")],
            },
            weights={"artifact_search": 1.0, "text_to_image": 0.7},
            rrf_k=60,
        )
        by_id = {r.artifact_id: r for r in results}
        self.assertAlmostEqual(by_id["A"].fusion_score, 1.0 / 61)
        self.assertAlmostEqual(by_id["B"].fusion_score, 1.0 / 62 + 0.7 / 61)
        self.assertAlmostEqual(by_id["C"].fusion_score, 0.7 / 62)
        # B aparece nos dois ramos -> uma única entrada, score somado, 1º lugar.
        self.assertEqual([r.artifact_id for r in results], ["B", "A", "C"])

    def test_weights_change_ordering(self) -> None:
        branches = {
            "artifact_search": [hit("A", 1)],
            "text_to_image": [hit("B", 1, img="imgB")],
        }
        heavier_text = weighted_rrf(
            branches, weights={"artifact_search": 1.0, "text_to_image": 2.0}, rrf_k=60
        )
        self.assertEqual(heavier_text[0].artifact_id, "B")
        heavier_artifacts = weighted_rrf(
            branches, weights={"artifact_search": 2.0, "text_to_image": 1.0}, rrf_k=60
        )
        self.assertEqual(heavier_artifacts[0].artifact_id, "A")

    def test_rrf_k_softens_rank_gaps(self) -> None:
        branches = {"a": [hit("X", 1), hit("Y", 10)]}
        small_k = weighted_rrf(branches, weights={"a": 1.0}, rrf_k=1)
        large_k = weighted_rrf(branches, weights={"a": 1.0}, rrf_k=1000)
        gap_small = small_k[0].fusion_score - small_k[1].fusion_score
        gap_large = large_k[0].fusion_score - large_k[1].fusion_score
        self.assertGreater(gap_small, gap_large)

    def test_unknown_branch_weight_defaults_to_zero_but_keeps_provenance(self) -> None:
        results = weighted_rrf(
            {"artifact_search": [hit("A", 1)], "mystery": [hit("A", 1)]},
            weights={"artifact_search": 1.0},
            rrf_k=60,
        )
        self.assertAlmostEqual(results[0].fusion_score, 1.0 / 61)
        self.assertIn("mystery", results[0].sources)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            weighted_rrf({"a": [hit("A", 1)]}, weights={"a": 1.0}, rrf_k=0)
        with self.assertRaises(ValueError):
            weighted_rrf({"a": [hit("A", 1)]}, weights={"a": -0.1}, rrf_k=60)
        with self.assertRaises(ValueError):
            weighted_rrf({"a": [hit("A", 0)]}, weights={"a": 1.0}, rrf_k=60)


class TestDeduplicationAndTies(unittest.TestCase):
    def test_duplicate_artifact_within_branch_keeps_best_rank(self) -> None:
        results = weighted_rrf(
            {"i2i": [hit("A", 1, 0.9, img="img1"), hit("A", 3, 0.7, img="img2")]},
            weights={"i2i": 1.0},
            rrf_k=60,
        )
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].fusion_score, 1.0 / 61)
        self.assertEqual(results[0].matched_image_id, "img1")

    def test_single_branch_artifacts_are_supported(self) -> None:
        results = weighted_rrf(
            {"artifact_search": [hit("A", 1)], "text_to_image": []},
            weights={"artifact_search": 1.0, "text_to_image": 0.7},
            rrf_k=60,
        )
        self.assertEqual([r.artifact_id for r in results], ["A"])

    def test_tie_break_is_deterministic_rank_then_id(self) -> None:
        # B e A com fusion_score igual; melhor rank individual decide; depois id.
        results = weighted_rrf(
            {
                "x": [hit("B", 1), hit("A", 2)],
                "y": [hit("A", 1), hit("B", 2)],
            },
            weights={"x": 1.0, "y": 1.0},
            rrf_k=60,
        )
        self.assertAlmostEqual(results[0].fusion_score, results[1].fusion_score)
        # Empate total (mesmo score, mesmo melhor rank=1) -> artifact_id asc.
        self.assertEqual([r.artifact_id for r in results], ["A", "B"])

    def test_output_is_deterministic_across_input_dict_order(self) -> None:
        b1 = {
            "artifact_search": [hit("A", 1), hit("B", 2)],
            "text_to_image": [hit("C", 1, img="i")],
        }
        b2 = dict(reversed(list(b1.items())))
        w = {"artifact_search": 1.0, "text_to_image": 0.7}
        r1 = [(r.artifact_id, r.fusion_score) for r in weighted_rrf(b1, weights=w)]
        r2 = [(r.artifact_id, r.fusion_score) for r in weighted_rrf(b2, weights=w)]
        self.assertEqual(r1, r2)


class TestMatchedImagePreservation(unittest.TestCase):
    def test_best_visual_rank_defines_matched_image(self) -> None:
        results = weighted_rrf(
            {
                "image_to_image": [hit("A", 2, 0.8, img="i2i-img")],
                "text_to_image": [hit("A", 1, 0.4, img="t2i-img")],
            },
            weights={"image_to_image": 1.0, "text_to_image": 0.7},
            rrf_k=60,
        )
        self.assertEqual(results[0].matched_image_id, "t2i-img")
        self.assertEqual(results[0].sources["image_to_image"].matched_image_id, "i2i-img")

    def test_textual_branch_never_overwrites_matched_image(self) -> None:
        results = weighted_rrf(
            {
                "artifact_search": [hit("A", 1)],
                "text_to_image": [hit("A", 5, 0.4, img="t2i-img")],
            },
            weights={"artifact_search": 1.0, "text_to_image": 0.7},
            rrf_k=60,
        )
        self.assertEqual(results[0].matched_image_id, "t2i-img")

    def test_provenance_payload_shape(self) -> None:
        result = weighted_rrf(
            {"text_to_image": [hit("A", 1, 0.42, img="img")]},
            weights={"text_to_image": 0.7},
            rrf_k=60,
        )[0]
        payload = result.provenance()
        self.assertEqual(payload["artifact_id"], "A")
        self.assertEqual(payload["sources"]["text_to_image"]["rank"], 1)
        self.assertEqual(payload["sources"]["text_to_image"]["score"], 0.42)
        self.assertEqual(payload["sources"]["text_to_image"]["matched_image_id"], "img")


class TestGrouping(unittest.TestCase):
    def test_multiple_images_of_same_artifact_group_to_best(self) -> None:
        hits = [
            {"artifact_id": "A", "image_id": "a1", "score": 0.95, "local_path": "p1"},
            {"artifact_id": "B", "image_id": "b1", "score": 0.90, "local_path": "p2"},
            {"artifact_id": "A", "image_id": "a2", "score": 0.88, "local_path": "p3"},
        ]
        grouped = group_image_hits_by_artifact(hits)
        self.assertEqual([g.artifact_id for g in grouped], ["A", "B"])
        self.assertEqual([g.rank for g in grouped], [1, 2])
        self.assertEqual(grouped[0].matched_image_id, "a1")
        self.assertEqual(grouped[0].matched_image_local_path, "p1")

    def test_hits_without_artifact_id_are_skipped(self) -> None:
        grouped = group_image_hits_by_artifact(
            [{"artifact_id": "", "image_id": "x"}, {"artifact_id": "A", "image_id": "a"}]
        )
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].rank, 1)


class TestScoreFloor(unittest.TestCase):
    def test_floor_drops_weak_hits_and_recompacts_ranks(self) -> None:
        hits = [hit("A", 1, 0.9, img="a"), hit("B", 2, 0.3, img="b"), hit("C", 3, 0.8, img="c")]
        kept, dropped = apply_score_floor(hits, min_score=0.5)
        self.assertEqual([k.artifact_id for k in kept], ["A", "C"])
        self.assertEqual([k.rank for k in kept], [1, 2])
        self.assertEqual([d.artifact_id for d in dropped], ["B"])

    def test_zero_floor_keeps_everything(self) -> None:
        hits = [hit("A", 1, 0.01)]
        kept, dropped = apply_score_floor(hits, min_score=0.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])

    def test_hits_without_score_survive_the_floor(self) -> None:
        kept, dropped = apply_score_floor([hit("A", 1, None)], min_score=0.5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])


def fused(aid, score):
    return FusedResult(artifact_id=aid, fusion_score=score)


class TestInTourPostFusionPolicy(unittest.TestCase):
    """E10 — preferência in_tour pós-fusão, conservadora e desligável."""

    def test_promotes_one_position_within_margin(self) -> None:
        results = [fused("A", 0.500), fused("B", 0.498)]
        ordered, n = promote_in_tour_within_margin(
            results, in_tour_by_artifact={"B": True}, margin=0.01
        )
        self.assertEqual([r.artifact_id for r in ordered], ["B", "A"])
        self.assertEqual(n, 1)
        # scores originais intactos
        self.assertEqual(ordered[0].fusion_score, 0.498)

    def test_never_overtakes_a_clear_match_beyond_margin(self) -> None:
        # Cenário do bug histórico: match perfeito no topo, in_tour abaixo com
        # gap grande — NUNCA ultrapassa.
        results = [fused("perfect", 1.0), fused("tour", 0.62)]
        ordered, n = promote_in_tour_within_margin(
            results, in_tour_by_artifact={"tour": True}, margin=0.05
        )
        self.assertEqual([r.artifact_id for r in ordered], ["perfect", "tour"])
        self.assertEqual(n, 0)

    def test_single_step_only_no_chained_climbing(self) -> None:
        results = [fused("A", 0.503), fused("B", 0.502), fused("T", 0.501)]
        ordered, n = promote_in_tour_within_margin(
            results, in_tour_by_artifact={"T": True}, margin=0.01
        )
        # T sobe UMA posição (acima de B), não em cadeia até ao topo.
        self.assertEqual([r.artifact_id for r in ordered], ["A", "T", "B"])
        self.assertEqual(n, 1)

    def test_disabled_at_zero_margin(self) -> None:
        results = [fused("A", 0.500), fused("B", 0.4999)]
        ordered, n = promote_in_tour_within_margin(
            results, in_tour_by_artifact={"B": True}, margin=0.0
        )
        self.assertEqual([r.artifact_id for r in ordered], ["A", "B"])
        self.assertEqual(n, 0)

    def test_in_tour_above_stays_put(self) -> None:
        results = [fused("T", 0.5), fused("A", 0.5)]
        ordered, n = promote_in_tour_within_margin(
            results, in_tour_by_artifact={"T": True}, margin=0.01
        )
        self.assertEqual([r.artifact_id for r in ordered], ["T", "A"])
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
