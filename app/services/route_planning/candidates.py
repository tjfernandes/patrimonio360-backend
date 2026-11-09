"""Candidatos ao percurso: os quatro ramos fundidos por Weighted RRF.

Reutiliza `app.services.retrieval.fusion.weighted_rrf` com pesos próprios
(ROUTE_WEIGHT_*). Cada candidato traz a paragem `best_placement` da peça, o
fusion_score, a proveniência por ramo, `why[]` (as facetas do tema que a
peça satisfaz, em português leigo) e `look_for[]` (as observações do VLM).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.core.logging import log_event
from app.services.retrieval.fusion import BranchHit, weighted_rrf
from app.services.route_planning.search import ATTRIBUTE_FIELDS, RouteStopSearch
from app.services.route_planning.theme import ThemeFacets, fold

logger = logging.getLogger(__name__)

_ROMAN_BY_NUMBER = {12: "XII", 13: "XIII", 14: "XIV", 15: "XV", 16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX", 21: "XXI"}


@dataclass
class Candidate:
    artifact_id: str
    stop: dict[str, Any]
    fusion_score: float
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    why: list[str] = field(default_factory=list)
    look_for: list[str] = field(default_factory=list)

    def as_planner_dict(self) -> dict[str, Any]:
        catalogue = self.stop.get("catalogue") or {}
        return {
            "artifact_id": self.artifact_id,
            "stop_id": self.stop.get("stop_id"),
            "panorama_key": self.stop.get("panorama_key"),
            "room": self.stop.get("room"),
            "playlist_index": self.stop.get("playlist_index"),
            "fusion_score": self.fusion_score,
            "category": catalogue.get("category"),
            "start_century": catalogue.get("start_century"),
            "why": list(self.why),
            "look_for": list(self.look_for),
            "sources": dict(self.sources),
            "stop": self.stop,
        }


def _doc_values(stop: dict[str, Any], dotted: str) -> list[str]:
    node: Any = stop
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return []
        node = node.get(part)
    if node is None:
        return []
    if isinstance(node, list):
        return [fold(str(v)) for v in node if v is not None]
    return [fold(str(node))]


def explain(stop: dict[str, Any], facets: ThemeFacets, *, year_tolerance: int = 0) -> list[str]:
    """Facetas do tema que esta paragem satisfaz, para o «porquê» do cartão.

    Época: dentro do século pedido sai «séc. XVII»; só dentro da folga dos
    filtros (`year_tolerance`) sai a data da ficha, para o visitante ver porque entrou."""
    why: list[str] = []
    for kind, fields in ATTRIBUTE_FIELDS.items():
        wanted = [v for v in getattr(facets, kind, []) if v]
        if not wanted:
            continue
        present: set[str] = set()
        for field_name, _boost in fields:
            present.update(_doc_values(stop, field_name))
        for value in wanted:
            if fold(value) in present and value not in why:
                why.append(value)
    if getattr(facets, "national_treasure", False) and stop.get("national_treasure"):
        why.append("Tesouro Nacional")
    catalogue = stop.get("catalogue") or {}
    window = facets.year_window(0)
    if window is not None:
        start, end = catalogue.get("start_year"), catalogue.get("end_year")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            strict = start <= window[1] and end >= window[0]
            loose = facets.year_window(year_tolerance)
            within_tolerance = loose is not None and start <= loose[1] and end >= loose[0]
            if strict and facets.centuries:
                centuries = [c for c in facets.centuries if (c - 1) * 100 <= end and c * 100 - 1 >= start]
                for century_number in centuries[:2]:
                    label = f"séc. {_ROMAN_BY_NUMBER.get(century_number, century_number)}"
                    if label not in why:
                        why.append(label)
            elif strict or within_tolerance:
                raw = str(catalogue.get("date_or_period") or "").strip()
                if raw.isdigit() and 1 <= int(raw) <= 21:
                    label = f"séc. {_ROMAN_BY_NUMBER.get(int(raw), raw)}"   # a ficha diz só o século («18»)
                else:
                    label = raw or (f"{int(start)}–{int(end)}" if start != end else str(int(start)))
                if label not in why:
                    why.append(label)
    return why


def _look_for(stop: dict[str, Any]) -> list[str]:
    visual = stop.get("visual") or {}
    values = visual.get("look_for") or []
    if isinstance(values, str):
        values = [values]
    return [str(v).strip() for v in values if str(v).strip()][:3]


async def find_candidates(
    *,
    search: RouteStopSearch,
    facets: ThemeFacets,
    museum_slug: str,
    museum_id: str,
    theme_text: str,
    embed_text: Callable[[str], Awaitable[list[float]]] | None,
    embed_image_text: Callable[[str], Awaitable[list[float]]] | None,
    weights: dict[str, float],
    rrf_k: int,
    branch_size: int,
    year_tolerance: int,
    limit: int,
    room: str | None = None,
) -> tuple[list[Candidate], dict[str, Any]]:
    """Os quatro ramos (os que puderem correr) fundidos; devolve (candidatos, diagnóstico)."""
    filters = RouteStopSearch.hard_filters(museum_slug=museum_slug, facets=facets, year_tolerance=year_tolerance, room=room)
    query_text = " ".join(part for part in [theme_text, *facets.colors, *facets.pigments, *facets.motifs, *facets.iconography] if part).strip() or theme_text
    branches: dict[str, list[BranchHit]] = {}
    docs: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {"branches": {}, "filters": filters, "query_text": query_text}

    async def run(name: str, fn: Callable[[], list[tuple[str, float, dict[str, Any]]]]) -> None:
        try:
            hits = await asyncio.to_thread(fn)
        except Exception as exc:  # noqa: BLE001 — um ramo em baixo não impede o percurso
            log_event(logger, logging.WARNING, "route.candidates.branch_error", branch=name, error=exc)
            diagnostics["branches"][name] = {"error": str(exc)[:200]}
            return
        branches[name] = [BranchHit(artifact_id=a, rank=i + 1, score=s) for i, (a, s, _d) in enumerate(hits)]
        for artifact_id, _score, doc in hits:
            if doc.get("stop_id"):
                docs.setdefault(artifact_id, doc)
        diagnostics["branches"][name] = {"hits": len(hits)}

    text_embedding: list[float] | None = None
    image_embedding: list[float] | None = None
    if embed_text is not None and weights.get("semantic", 0) > 0:
        try:
            text_embedding = await embed_text(query_text)
        except Exception as exc:  # noqa: BLE001
            log_event(logger, logging.WARNING, "route.candidates.embed_error", kind="text", error=exc)
    if embed_image_text is not None and weights.get("image", 0) > 0:
        try:
            image_embedding = await embed_image_text(query_text)
        except Exception as exc:  # noqa: BLE001
            log_event(logger, logging.WARNING, "route.candidates.embed_error", kind="image", error=exc)

    tasks = [run("lexical", lambda: search.lexical(query_text=query_text, filters=filters, size=branch_size))]
    if getattr(facets, "national_treasure", False):
        # «tesouros nacionais» sem mais facetas: o filtro duro já escolhe; a ordem
        # vem da visita (todas as peças com selo entram, pela playlist)
        tasks.append(run("treasures", lambda: search.all_filtered(filters=filters, size=max(branch_size, 60))))
    if text_embedding:
        tasks.append(run("semantic", lambda: search.semantic(embedding=text_embedding, filters=filters, size=branch_size)))
    if facets.has_visual() and weights.get("attributes", 0) > 0:
        tasks.append(run("attributes", lambda: search.attributes(facets=facets, filters=filters, size=branch_size)))
    await asyncio.gather(*tasks)

    if image_embedding:
        try:
            eligible = await asyncio.to_thread(lambda: search.artifact_ids(filters=filters))
        except Exception as exc:  # noqa: BLE001
            log_event(logger, logging.WARNING, "route.candidates.branch_error", branch="image", error=exc)
            eligible = []
        if eligible:
            await run("image", lambda: search.image(embedding=image_embedding, museum_id=museum_id, artifact_ids=eligible, size=branch_size))

    fused = weighted_rrf(branches, weights=weights, rrf_k=rrf_k)
    fused.sort(key=lambda r: (-r.fusion_score, r.artifact_id))
    top = fused[:limit]
    missing = [r.artifact_id for r in top if r.artifact_id not in docs]
    if missing:
        try:
            docs.update(await asyncio.to_thread(lambda: search.stops_by_artifact(museum_slug=museum_slug, artifact_ids=missing)))
        except Exception as exc:  # noqa: BLE001
            log_event(logger, logging.WARNING, "route.candidates.fetch_error", error=exc)
    candidates: list[Candidate] = []
    for result in top:
        stop = docs.get(result.artifact_id)
        if not stop:
            continue
        candidates.append(Candidate(
            artifact_id=result.artifact_id,
            stop=stop,
            fusion_score=float(result.fusion_score),
            sources={name: {"rank": prov.rank, "score": prov.score} for name, prov in result.sources.items()},
            why=explain(stop, facets, year_tolerance=year_tolerance),
            look_for=_look_for(stop),
        ))
    diagnostics["fused"] = len(fused)
    return candidates, diagnostics
