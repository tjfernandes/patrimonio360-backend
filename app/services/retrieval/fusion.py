"""Weighted Reciprocal Rank Fusion ao nível de artifact_id (Fase 3).

Módulo puro: sem OpenSearch, sem settings, sem I/O. Os ramos entregam listas
ordenadas de hits e a fusão combina RANKS (nunca somar scores brutos BM25 /
híbridos / cosseno / HNSW — escalas incomparáveis):

    fusion_score(artifact) = Σ_ramo  weight_ramo / (rrf_k + rank_ramo)

Determinismo: para inputs iguais a saída é idêntica, incluindo desempates
(fusion_score desc → melhor rank individual asc → artifact_id asc).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BranchHit:
    """Um resultado de um ramo de pesquisa, já ao nível do artefacto.

    ``rank`` é 1-based dentro do ramo. ``score`` é o score ORIGINAL do ramo
    (apenas provenance/diagnóstico — nunca entra na fórmula da fusão).
    ``matched_image_id`` identifica a imagem que fez match nos ramos visuais.
    """

    artifact_id: str
    rank: int
    score: float | None = None
    matched_image_id: str | None = None
    matched_image_local_path: str | None = None


@dataclass(frozen=True)
class BranchProvenance:
    rank: int
    score: float | None = None
    matched_image_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"rank": self.rank}
        if self.score is not None:
            payload["score"] = self.score
        if self.matched_image_id:
            payload["matched_image_id"] = self.matched_image_id
        return payload


@dataclass
class FusedResult:
    artifact_id: str
    fusion_score: float
    matched_image_id: str | None = None
    matched_image_local_path: str | None = None
    sources: dict[str, BranchProvenance] = field(default_factory=dict)

    def provenance(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "fusion_score": self.fusion_score,
            "matched_image_id": self.matched_image_id,
            "sources": {name: prov.as_dict() for name, prov in self.sources.items()},
        }


def group_image_hits_by_artifact(
    image_hits: list[dict[str, object]],
) -> list[BranchHit]:
    """Agrupa hits de imagem (ordenados por score do ramo) por artifact_id.

    O rank do artefacto é a posição do seu MELHOR hit (primeira ocorrência na
    lista, que vem ordenada); esse hit define matched_image_id/score. Ranks
    saem compactados (1..n artefactos únicos), preservando a ordem relativa.
    """

    grouped: list[BranchHit] = []
    seen: set[str] = set()
    for hit in image_hits:
        artifact_id = str(hit.get("artifact_id") or "").strip()
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        raw_score = hit.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        grouped.append(
            BranchHit(
                artifact_id=artifact_id,
                rank=len(grouped) + 1,
                score=score,
                matched_image_id=str(hit.get("image_id") or "").strip() or None,
                matched_image_local_path=str(hit.get("local_path") or "").strip() or None,
            )
        )
    return grouped


def apply_score_floor(
    hits: list[BranchHit],
    *,
    min_score: float,
) -> tuple[list[BranchHit], list[BranchHit]]:
    """Separa (mantidos, excluídos) por floor de score, recompactando ranks.

    Hits sem score conhecido são mantidos (o floor só corta evidência
    comprovadamente fraca). Com ``min_score <= 0`` devolve tudo intacto.
    """

    if min_score <= 0:
        return list(hits), []
    kept: list[BranchHit] = []
    dropped: list[BranchHit] = []
    for hit in hits:
        if hit.score is not None and hit.score < min_score:
            dropped.append(hit)
            continue
        kept.append(
            BranchHit(
                artifact_id=hit.artifact_id,
                rank=len(kept) + 1,
                score=hit.score,
                matched_image_id=hit.matched_image_id,
                matched_image_local_path=hit.matched_image_local_path,
            )
        )
    return kept, dropped


def weighted_rrf(
    branches: dict[str, list[BranchHit]],
    *,
    weights: dict[str, float],
    rrf_k: int = 60,
) -> list[FusedResult]:
    """Funde os ramos por artifact_id com Weighted RRF.

    - Deduplicação: dentro de um ramo, conta apenas o melhor rank por artefacto.
    - Artefactos presentes num único ramo entram normalmente.
    - Ramos sem peso declarado usam peso 0 (não contribuem, mas ficam na
      provenance) — pesos explícitos são responsabilidade do orquestrador.
    - matched_image_id do resultado = o do ramo visual com melhor rank que o
      tenha (empate entre ramos resolvido por nome de ramo, determinístico).
    """

    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1; got {rrf_k}")
    for name, weight in weights.items():
        if weight < 0:
            raise ValueError(f"branch weight must be >= 0; got {name}={weight}")

    fused: dict[str, FusedResult] = {}

    for branch_name in sorted(branches):
        weight = float(weights.get(branch_name, 0.0))
        best_in_branch: dict[str, BranchHit] = {}
        for hit in branches[branch_name]:
            artifact_id = (hit.artifact_id or "").strip()
            if not artifact_id:
                continue
            if hit.rank < 1:
                raise ValueError(
                    f"ranks are 1-based; got rank={hit.rank} in branch {branch_name!r}"
                )
            current = best_in_branch.get(artifact_id)
            if current is None or hit.rank < current.rank:
                best_in_branch[artifact_id] = hit

        for artifact_id, hit in best_in_branch.items():
            result = fused.get(artifact_id)
            if result is None:
                result = FusedResult(artifact_id=artifact_id, fusion_score=0.0)
                fused[artifact_id] = result
            result.fusion_score += weight / (rrf_k + hit.rank)
            result.sources[branch_name] = BranchProvenance(
                rank=hit.rank,
                score=hit.score,
                matched_image_id=hit.matched_image_id,
            )
            if hit.matched_image_id:
                # Melhor imagem: rank mais baixo entre ramos visuais; empate
                # resolvido pela ordem alfabética de ramo (loop ordenado).
                current_rank = _matched_image_rank(result)
                if current_rank is None or hit.rank < current_rank:
                    result.matched_image_id = hit.matched_image_id
                    result.matched_image_local_path = hit.matched_image_local_path
                    result.sources[branch_name] = BranchProvenance(
                        rank=hit.rank,
                        score=hit.score,
                        matched_image_id=hit.matched_image_id,
                    )

    def _best_rank(result: FusedResult) -> int:
        return min(prov.rank for prov in result.sources.values())

    return sorted(
        fused.values(),
        key=lambda r: (-r.fusion_score, _best_rank(r), r.artifact_id),
    )


def promote_in_tour_within_margin(
    results: list[FusedResult],
    *,
    in_tour_by_artifact: dict[str, bool],
    margin: float,
) -> tuple[list[FusedResult], int]:
    """Etapa 10 — preferência in_tour APÓS a fusão, conservadora.

    Uma única passagem estável: um resultado in_tour sobe no máximo UMA posição,
    e apenas quando a diferença de fusion_score para o vizinho acima é <= margin.
    Nunca altera scores, nunca introduz candidatos, e um match destacado (gap >
    margin) nunca é ultrapassado. margin <= 0 desliga a política (ordem intacta).

    Devolve (nova_ordem, nº de promoções).
    """

    if margin <= 0 or len(results) < 2:
        return list(results), 0
    ordered = list(results)
    promotions = 0
    index = 1
    while index < len(ordered):
        above = ordered[index - 1]
        current = ordered[index]
        above_in_tour = bool(in_tour_by_artifact.get(above.artifact_id, False))
        current_in_tour = bool(in_tour_by_artifact.get(current.artifact_id, False))
        if (
            current_in_tour
            and not above_in_tour
            and (above.fusion_score - current.fusion_score) <= margin
        ):
            ordered[index - 1], ordered[index] = current, above
            promotions += 1
            index += 2  # o promovido não continua a subir em cadeia
        else:
            index += 1
    return ordered, promotions


def _matched_image_rank(result: FusedResult) -> int | None:
    ranks = [
        prov.rank
        for prov in result.sources.values()
        if prov.matched_image_id and prov.matched_image_id == result.matched_image_id
    ]
    return min(ranks) if ranks else None
