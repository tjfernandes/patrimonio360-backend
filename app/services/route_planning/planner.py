"""Planeador determinístico do percurso (sem LLM).

Entrada: candidatos já fundidos (um por peça, com a paragem `best_placement`,
fusion_score e provenance), o grafo da visita e o panorama onde o visitante
está. Saída: k paragens ordenadas, cada uma com os focos no mesmo panorama
(`also_here`) e a transição vinda do grafo.

  k = stops_requested | ⌊time_budget_min / ROUTE_MINUTES_PER_STOP⌋ | ROUTE_DEFAULT_STOPS, em [ROUTE_MIN_STOPS, ROUTE_MAX_STOPS]
  escolha gulosa: relevância normalizada + diversidade (sala nova, categoria
    nova, século novo quando o tema é temporal) − custo de caminho (saltos no
    grafo a partir da última paragem escolhida, ou do visitante)
  peças no mesmo panorama de uma paragem já escolhida entram como also_here
  ordem: caminho hamiltoniano mínimo em saltos (exato por programação
    dinâmica para k ≤ 10, o máximo), a começar no visitante
  com menos de ROUTE_MIN_CANDIDATES candidatos relevantes -> None
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from app.services.route_planning.graph import TourGraph


@dataclass
class PlannerSettings:
    default_stops: int = 6
    min_stops: int = 3
    max_stops: int = 10
    minutes_per_stop: float = 2.5
    min_fusion_score: float = 0.012
    min_candidates: int = 3
    diversity_weight: float = 0.35
    path_cost_weight: float = 0.02
    max_also_here: int = 3


@dataclass
class PlannedStop:
    candidate: dict[str, Any]
    also_here: list[dict[str, Any]] = field(default_factory=list)
    transition: str | None = None
    hops_from_previous: int | None = None

    @property
    def panorama_key(self) -> str:
        return str(self.candidate.get("panorama_key") or "")


def choose_k(*, stops_requested: int | None, time_budget_min: int | None, settings: PlannerSettings) -> int:
    if stops_requested:
        k = int(stops_requested)
    elif time_budget_min:
        k = int(math.floor(float(time_budget_min) / max(settings.minutes_per_stop, 0.5)))
    else:
        k = settings.default_stops
    return max(settings.min_stops, min(settings.max_stops, k))


def relevant(candidates: list[dict[str, Any]], *, min_fusion_score: float) -> list[dict[str, Any]]:
    return [c for c in candidates if float(c.get("fusion_score") or 0.0) >= min_fusion_score]


def _century(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("start_century")
    return int(value) if isinstance(value, (int, float)) else None


def select_stops(
    candidates: list[dict[str, Any]],
    *,
    k: int,
    graph: TourGraph | None,
    start_panorama: str | None,
    settings: PlannerSettings,
    temporal: bool = False,
) -> list[PlannedStop]:
    """Escolha gulosa de k paragens (panoramas) entre os candidatos ordenados por relevância."""
    pool = [c for c in candidates if c.get("panorama_key")]
    if not pool:
        return []
    top = max(float(c.get("fusion_score") or 0.0) for c in pool) or 1.0
    chosen: list[PlannedStop] = []
    used_rooms: set[str] = set()
    used_categories: set[str] = set()
    used_centuries: set[int] = set()
    remaining = list(pool)
    current = start_panorama if graph is not None and graph.has(start_panorama) else None

    while remaining and len(chosen) < k:
        best_score, best = -math.inf, None
        for candidate in remaining:
            relevance = float(candidate.get("fusion_score") or 0.0) / top
            bonus = 0.0
            room = candidate.get("room")
            if room and room not in used_rooms:
                bonus += settings.diversity_weight
            category = candidate.get("category")
            if category and category not in used_categories:
                bonus += settings.diversity_weight / 2
            century = _century(candidate)
            if temporal and century is not None and century not in used_centuries:
                bonus += settings.diversity_weight / 2
            cost = 0.0
            if graph is not None and current is not None:
                hops = graph.hops(current, candidate["panorama_key"])
                cost = settings.path_cost_weight * (hops if hops is not None else 30)
            score = relevance + bonus - cost
            if score > best_score:
                best_score, best = score, candidate
        assert best is not None
        remaining.remove(best)
        stop = PlannedStop(candidate=best)
        chosen.append(stop)
        if best.get("room"):
            used_rooms.add(str(best["room"]))
        if best.get("category"):
            used_categories.add(str(best["category"]))
        if _century(best) is not None:
            used_centuries.add(_century(best))  # type: ignore[arg-type]
        current = best["panorama_key"]
        # as outras peças relevantes no mesmo panorama viram focos desta paragem
        same = [c for c in remaining if c.get("panorama_key") == best["panorama_key"]]
        for other in same[: settings.max_also_here]:
            stop.also_here.append(other)
            remaining.remove(other)
        for other in same[settings.max_also_here:]:
            remaining.remove(other)
    return chosen


def _distance(graph: TourGraph | None, a: str | None, b: str) -> float:
    if graph is None or not a:
        return 1.0
    hops = graph.hops(a, b)
    return float(hops) if hops is not None else 30.0


def order_stops(stops: list[PlannedStop], *, graph: TourGraph | None, start_panorama: str | None) -> list[PlannedStop]:
    """Caminho hamiltoniano mínimo (em saltos) a partir do visitante; exato até 10 paragens."""
    n = len(stops)
    if n <= 1:
        return list(stops)
    start = start_panorama if graph is not None and graph.has(start_panorama) else None
    keys = [s.panorama_key for s in stops]
    dist = [[_distance(graph, keys[i], keys[j]) for j in range(n)] for i in range(n)]
    from_start = [_distance(graph, start, keys[j]) if start else 0.0 for j in range(n)]
    if graph is None:
        # sem grafo: pela ordem da visita (playlist)
        return sorted(stops, key=lambda s: (s.candidate.get("playlist_index") is None, s.candidate.get("playlist_index") or 0))
    if n > 10:
        order = _nearest_neighbour(dist, from_start)
    else:
        order = _held_karp(dist, from_start)
    return [stops[i] for i in order]


def _nearest_neighbour(dist: list[list[float]], from_start: list[float]) -> list[int]:
    n = len(dist)
    current = min(range(n), key=lambda j: (from_start[j], j))
    order = [current]
    left = set(range(n)) - {current}
    while left:
        current = min(left, key=lambda j: (dist[current][j], j))
        order.append(current)
        left.remove(current)
    return order


def _held_karp(dist: list[list[float]], from_start: list[float]) -> list[int]:
    n = len(dist)
    full = (1 << n) - 1
    # best[mask][last] = (custo, anterior)
    best: dict[tuple[int, int], tuple[float, int | None]] = {}
    for j in range(n):
        best[(1 << j, j)] = (from_start[j], None)
    for size in range(2, n + 1):
        for subset in combinations(range(n), size):
            mask = 0
            for j in subset:
                mask |= 1 << j
            for last in subset:
                previous_mask = mask & ~(1 << last)
                candidates = []
                for prev in subset:
                    if prev == last:
                        continue
                    entry = best.get((previous_mask, prev))
                    if entry is None:
                        continue
                    candidates.append((entry[0] + dist[prev][last], prev))
                if candidates:
                    best[(mask, last)] = min(candidates, key=lambda t: (t[0], t[1]))
    last = min(range(n), key=lambda j: (best[(full, j)][0], j))
    order: list[int] = []
    mask = full
    while last is not None:
        order.append(last)
        _cost, prev = best[(mask, last)]
        mask &= ~(1 << last)
        last = prev  # type: ignore[assignment]
    return order[::-1]


def annotate_transitions(stops: list[PlannedStop], *, graph: TourGraph | None, start_panorama: str | None, language: str | None) -> None:
    previous = start_panorama if graph is not None and graph.has(start_panorama) else None
    for stop in stops:
        if graph is not None:
            stop.transition = graph.transition_text(previous, stop.panorama_key, language=language)
            stop.hops_from_previous = graph.hops(previous, stop.panorama_key) if previous else None
        previous = stop.panorama_key


def estimate_minutes(stops: list[PlannedStop], *, minutes_per_stop: float) -> int:
    hops = sum(s.hops_from_previous or 0 for s in stops)
    focos = sum(len(s.also_here) for s in stops)
    total = len(stops) * minutes_per_stop + focos * (minutes_per_stop / 3) + hops * 0.25
    return max(1, int(round(total)))


def plan(
    candidates: list[dict[str, Any]],
    *,
    graph: TourGraph | None,
    start_panorama: str | None,
    settings: PlannerSettings,
    stops_requested: int | None,
    time_budget_min: int | None,
    temporal: bool,
    language: str | None,
) -> tuple[list[PlannedStop] | None, dict[str, Any]]:
    """Devolve (paragens ordenadas | None, info). None = poucos candidatos relevantes."""
    pool = relevant(candidates, min_fusion_score=settings.min_fusion_score)
    info: dict[str, Any] = {"candidate_count": len(candidates), "relevant_count": len(pool)}
    if len(pool) < settings.min_candidates:
        return None, info
    k = choose_k(stops_requested=stops_requested, time_budget_min=time_budget_min, settings=settings)
    info["k"] = k
    chosen = select_stops(pool, k=k, graph=graph, start_panorama=start_panorama, settings=settings, temporal=temporal)
    ordered = order_stops(chosen, graph=graph, start_panorama=start_panorama)
    annotate_transitions(ordered, graph=graph, start_panorama=start_panorama, language=language)
    info["estimated_minutes"] = estimate_minutes(ordered, minutes_per_stop=settings.minutes_per_stop)
    return ordered, info
