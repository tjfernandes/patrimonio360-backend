"""Grafo de navegação da visita (poi_tours/tour-graph-<museu>.json) e distâncias.

Gerado por tools/percursos/extrair_grafo_visita.py: nós = panoramas (com
sala e posição na playlist), arestas = ligações clicáveis (adjacentPanoramas e
hotspots de navegação; `inferred` só para nós sem ligações). As seis visitas
são uma só componente ligada, pelo que `hops` quase nunca é None.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_PREFIX = "tour-graph-"
_SLUG_RE = re.compile(r"^[a-z0-9_-]{1,40}$")


class TourGraph:
    """Um grafo carregado: nós por panorama, adjacência dirigida (só arestas
    ativas) e, para o caso de uma ligação só existir num sentido, a versão
    sem sentido como recurso."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.museum_slug = str(payload.get("museum_slug") or "")
        self.nodes: dict[str, dict[str, Any]] = {}
        for node in payload.get("nodes") or []:
            key = str(node.get("panorama_key") or "").strip()
            if key:
                self.nodes[key] = node
        self.order: list[str] = sorted(
            self.nodes, key=lambda k: (self.nodes[k].get("playlist_index") is None, self.nodes[k].get("playlist_index") or 0)
        )
        self._directed: dict[str, list[str]] = {k: [] for k in self.nodes}
        self._undirected: dict[str, set[str]] = {k: set() for k in self.nodes}
        for edge in payload.get("edges") or []:
            a, b = str(edge.get("source") or ""), str(edge.get("target") or "")
            if a not in self.nodes or b not in self.nodes or a == b:
                continue
            if edge.get("enabled", True):
                self._directed[a].append(b)
            self._undirected[a].add(b)
            self._undirected[b].add(a)
        self._paths: dict[str, dict[str, list[str]]] = {}
        self._undirected_done: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ nós
    def has(self, panorama_key: str | None) -> bool:
        return bool(panorama_key) and panorama_key in self.nodes

    def room(self, panorama_key: str | None) -> str | None:
        node = self.nodes.get(panorama_key or "")
        return (node or {}).get("room") or None

    def playlist_index(self, panorama_key: str | None) -> int | None:
        node = self.nodes.get(panorama_key or "")
        return (node or {}).get("playlist_index")

    def panoramas_in_room(self, room: str) -> list[str]:
        folded = _fold(room)
        return [k for k in self.order if _fold(self.room(k) or "") == folded]

    # ------------------------------------------------------------- caminhos
    def _bfs(self, start: str, adjacency: dict[str, Any]) -> dict[str, list[str]]:
        parent: dict[str, str | None] = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for other in adjacency.get(current, ()):
                if other not in parent:
                    parent[other] = current
                    queue.append(other)
        paths: dict[str, list[str]] = {}
        for target in parent:
            path: list[str] = []
            node: str | None = target
            while node is not None:
                path.append(node)
                node = parent[node]
            paths[target] = path[::-1]
        return paths

    def path(self, start: str | None, target: str | None) -> list[str] | None:
        """Caminho mais curto (em saltos) de `start` a `target`; None se não houver."""
        if not self.has(start) or not self.has(target):
            return None
        assert start is not None and target is not None
        if start == target:
            return [start]
        with self._lock:
            paths = self._paths.get(start)
            if paths is None:
                paths = self._bfs(start, self._directed)
                self._paths[start] = paths
            if target not in paths and start not in self._undirected_done:
                # ligação só num sentido (ou desligada): vale o grafo sem sentido
                for key, value in self._bfs(start, self._undirected).items():
                    paths.setdefault(key, value)
                self._undirected_done.add(start)
        return paths.get(target)

    def hops(self, start: str | None, target: str | None) -> int | None:
        path = self.path(start, target)
        return None if path is None else len(path) - 1

    def rooms_crossed(self, start: str | None, target: str | None) -> int | None:
        """Quantas vezes se muda de sala ao longo do caminho mais curto."""
        path = self.path(start, target)
        if path is None:
            return None
        changes = 0
        previous = self.room(path[0])
        for key in path[1:]:
            room = self.room(key)
            if room and room != previous:
                changes += 1
                previous = room
        return changes

    def transition_text(self, start: str | None, target: str | None, *, language: str | None = None) -> str | None:
        """Frase curta da transição («Duas salas adiante»), a partir do grafo."""
        en = (language or "pt").lower().startswith("en")
        if not self.has(target):
            return None
        if not self.has(start):
            room = self.room(target)
            if not room:
                return None
            return f"In {room}" if en else f"Em {room}"
        if start == target:
            return "Right here" if en else "Aqui mesmo"
        crossed = self.rooms_crossed(start, target)
        hops = self.hops(start, target)
        if crossed is None or hops is None:
            room = self.room(target)
            return (f"Elsewhere in the tour ({room})" if en else f"Noutra zona da visita ({room})") if room else None
        if crossed == 0:
            return ("A few steps ahead" if hops > 1 else "Right next to you") if en else ("Uns passos adiante" if hops > 1 else "Mesmo ao lado")
        if crossed == 1:
            return "In the next room" if en else "Na sala seguinte"
        words_pt = {2: "Duas", 3: "Três", 4: "Quatro", 5: "Cinco", 6: "Seis"}
        words_en = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
        if en:
            return f"{words_en.get(crossed, str(crossed))} rooms ahead"
        return f"{words_pt.get(crossed, str(crossed))} salas adiante"


def _fold(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", (value or "").casefold())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in normalized if not unicodedata.combining(c))).strip()


class TourGraphStore:
    """Os ficheiros tour-graph-<museu>.json de uma pasta, relidos quando mudam."""

    def __init__(self, directory: Path | str, *, check_interval: float = 5.0) -> None:
        self.directory = Path(directory)
        self.check_interval = check_interval
        self._lock = threading.Lock()
        self._graphs: dict[str, tuple[tuple[int, int], TourGraph]] = {}
        self._checked: dict[str, float] = {}

    def get(self, museum_slug: str | None) -> TourGraph | None:
        slug = (museum_slug or "").strip().lower()
        if not _SLUG_RE.match(slug):
            return None
        path = self.directory / f"{_PREFIX}{slug}.json"
        now = time.monotonic()
        with self._lock:
            cached = self._graphs.get(slug)
            if cached is not None and now - self._checked.get(slug, float("-inf")) < self.check_interval:
                return cached[1]
            self._checked[slug] = now
            try:
                stat = path.stat()
            except OSError:
                self._graphs.pop(slug, None)
                return None
            signature = (stat.st_mtime_ns, stat.st_size)
            if cached is not None and cached[0] == signature:
                return cached[1]
            try:
                graph = TourGraph(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001 — um ficheiro estragado não derruba o chat
                logger.warning("tour_graph: %s ilegível (%s)", path.name, exc)
                self._graphs.pop(slug, None)
                return None
            self._graphs[slug] = (signature, graph)
            return graph


@lru_cache(maxsize=1)
def get_tour_graph_store() -> TourGraphStore:
    return TourGraphStore(get_settings().poi_tours_dir_resolved)
