from __future__ import annotations

import math


def _relevant_set(relevant_artifacts: list[str]) -> set[str]:
    return {artifact_id for artifact_id in relevant_artifacts if artifact_id}


def recall_at_k(ranking: list[str], relevant_artifacts: list[str], k: int) -> float | None:
    relevant = _relevant_set(relevant_artifacts)
    if not relevant:
        return None
    if k <= 0:
        return 0.0
    hits = sum(1 for artifact_id in ranking[:k] if artifact_id in relevant)
    return float(hits) / float(len(relevant))


def hit_at_k(ranking: list[str], relevant_artifacts: list[str], k: int) -> float | None:
    relevant = _relevant_set(relevant_artifacts)
    if not relevant:
        return None
    if k <= 0:
        return 0.0
    window = ranking[:k]
    return 1.0 if any(artifact_id in relevant for artifact_id in window) else 0.0


def precision_at_k(ranking: list[str], relevant_artifacts: list[str], k: int) -> float | None:
    relevant = _relevant_set(relevant_artifacts)
    if not relevant:
        return None
    if k <= 0:
        return 0.0
    hits = sum(1 for artifact_id in ranking[:k] if artifact_id in relevant)
    return float(hits) / float(k)


def reciprocal_rank(ranking: list[str], relevant_artifacts: list[str]) -> float | None:
    relevant = _relevant_set(relevant_artifacts)
    if not relevant:
        return None
    for index, artifact_id in enumerate(ranking, start=1):
        if artifact_id in relevant:
            return 1.0 / float(index)
    return 0.0


def ndcg_at_k(ranking: list[str], relevant_artifacts: list[str], k: int) -> float | None:
    relevant = _relevant_set(relevant_artifacts)
    if not relevant:
        return None
    if k <= 0:
        return 0.0

    dcg = 0.0
    for index, artifact_id in enumerate(ranking[:k], start=1):
        if artifact_id in relevant:
            dcg += 1.0 / math.log2(index + 1.0)

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(index + 1.0) for index in range(1, ideal_hits + 1))
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg / ideal_dcg


def selected_artifact_hit(
    selected_artifact_id: str | None,
    relevant_artifacts: list[str],
) -> float | None:
    relevant = _relevant_set(relevant_artifacts)
    if not relevant:
        return None
    selected = (selected_artifact_id or "").strip()
    if not selected:
        return None
    return 1.0 if selected in relevant else 0.0


def score_ranking(
    ranking: list[str],
    *,
    relevant_artifacts: list[str],
    mode: str,
) -> dict[str, float | None]:
    if mode == "text_multi":
        return {
            "recall_at_1": None,
            "recall_at_5": None,
            "recall_at_10": recall_at_k(ranking, relevant_artifacts, 10),
            "hit_at_5": hit_at_k(ranking, relevant_artifacts, 5),
            "hit_at_10": hit_at_k(ranking, relevant_artifacts, 10),
            "precision_at_5": precision_at_k(ranking, relevant_artifacts, 5),
            "mrr": None,
            "ndcg_at_5": ndcg_at_k(ranking, relevant_artifacts, 5),
            "ndcg_at_10": ndcg_at_k(ranking, relevant_artifacts, 10),
        }

    return {
        "recall_at_1": recall_at_k(ranking, relevant_artifacts, 1),
        "recall_at_5": recall_at_k(ranking, relevant_artifacts, 5),
        "recall_at_10": recall_at_k(ranking, relevant_artifacts, 10),
        "hit_at_5": None,
        "hit_at_10": None,
        "precision_at_5": None,
        "mrr": reciprocal_rank(ranking, relevant_artifacts),
        "ndcg_at_5": None,
        "ndcg_at_10": ndcg_at_k(ranking, relevant_artifacts, 10),
    }
