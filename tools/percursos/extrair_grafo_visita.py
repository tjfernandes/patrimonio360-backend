# -*- coding: utf-8 -*-
"""Grafo de navegação de uma visita 3DVista (percursos temáticos, Fase 0).

    python tools/percursos/extrair_grafo_visita.py mnaa
    python tools/percursos/extrair_grafo_visita.py mnaa --visita tmp/visitas/mnaa -o poi_tours/tour-graph-mnaa.json

Lê o script_general.js da visita (a cópia em tmp/visitas/<museu>/, descarregada
do VPS se faltar) e escreve poi_tours/tour-graph-<museu>.json:

  nós     cada panorama: panorama_key, playlist_index (posição na mainPlayList),
          label e subtitle (locale/*.txt), room (app/data/tour_locations.json,
          senão a zona dos hotspots do mapa dos POIs, senão o subtitle).
  arestas as ligações entre panoramas, de três fontes:
          - `adjacent`: a lista adjacentPanoramas de cada panorama no export
            (com yaw, distância e o overlay que a dispara; `enabled:false`
            quando o 3DVista a tem desligada);
          - `hotspot`: os hotspots de navegação, cujo click chama
            setPlayListSelectedIndex(this.mainPlayList, N) (só ou depois de
            setPanoramaCameraWithSpot) ou setMediaBehaviour(…, this.panorama_X);
            origem = panorama que contém o overlay, destino = N-ésimo item da
            mainPlayList (a mesma cadeia area → overlay → panorama do
            tools/poi_mapping/extract_poi_mapping.py);
          - `inferred`: só para nós sem nenhuma ligação, os vizinhos na
            mainPlayList (marcados inferred: true).
  overlays todos os hotspots do export com a sua direção (yaw, pitch) no
          panorama, para dar direção também aos POIs das peças.

Relatório no fim: nós, arestas por fonte, componentes ligadas (sem sentido),
órfãos e nós sem sala. Nada disto toca na visita publicada nem nos índices.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import datetime as dt
import json
import re
import ssl
import sys
import urllib.request
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "poi_mapping"))
from extract_poi_mapping import innermost, is_human_name, load_locale, object_spans  # noqa: E402

PANORAMA_ID = r"panorama_[0-9A-F]{8}_[0-9A-F]{4}_[0-9A-F]{4}_[0-9A-F]{4}_[0-9A-F]{12}"
PANORAMA_ID_RE = re.compile(r'"id":"(' + PANORAMA_ID + r')"')
ANY_ID_RE = re.compile(r'"id":"([A-Za-z0-9_]+)"')
OVERLAY_REF_RE = re.compile(r'"this\.((?:overlay|sprite)_[0-9A-F_]+)"')
ADJ_TARGET_RE = re.compile(r'"panorama":"this\.(' + PANORAMA_ID + r')"')
NUM = r"(-?[0-9]+(?:\.[0-9]+)?)"
CLICK_PLAYLIST_RE = re.compile(r"setPlayListSelectedIndex\(this\.mainPlayList, ?(\d+)\)")
CLICK_MEDIA_RE = re.compile(r"setMediaBehaviour\(this\.[A-Za-z0-9_]+, ?\d+, ?this\.(" + PANORAMA_ID + r")")
LABEL_RE = re.compile(r'"label":"([^"]*)"')
DEFAULT_BASE = "https://patrimonio360.webredirect.org:5500"


def _num(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def fetch(url: str, dest: Path) -> None:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE  # o servidor das visitas usa certificado próprio
    with urllib.request.urlopen(url, context=context, timeout=180) as response:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.read())


def ensure_tour_files(tour_dir: Path, slug: str, base: str) -> Path:
    """script_general.js e locale/*.txt em tour_dir (descarrega do VPS o que faltar)."""
    script = tour_dir / "script_general.js"
    if not script.exists():
        print(f"a descarregar {base}/{slug}/script_general.js …")
        fetch(f"{base}/{slug}/script_general.js", script)
    if not (tour_dir / "locale").is_dir() or not any((tour_dir / "locale").glob("*.txt")):
        index = tour_dir / "index.html"
        try:
            fetch(f"{base}/{slug}/", index)
            for name in sorted(set(re.findall(r"locale/([\w-]+)\.txt", index.read_text(encoding="utf-8", errors="replace")))):
                fetch(f"{base}/{slug}/locale/{name}.txt", tour_dir / "locale" / f"{name}.txt")
        except Exception as exc:  # noqa: BLE001
            print(f"aviso: não descarreguei os locale/*.txt ({exc}); nós sem label")
    return script


def _balanced_array(text: str, open_index: int) -> str:
    depth, in_string, escaped = 0, False, False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index]
    return ""


class TourScript:
    """O export, com os spans dos objetos e os índices que o grafo precisa."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.spans = object_spans(text)
        self.starts = [s[0] for s in self.spans]

    def object_at(self, pos: int) -> tuple[int, int] | None:
        return innermost(self.spans, self.starts, pos)

    def object_text(self, span: tuple[int, int]) -> str:
        return self.text[span[0]:span[1]]

    def parent_span(self, span: tuple[int, int]) -> tuple[int, int] | None:
        """O objeto que contém `span` (o mais interior dos que começam antes e acabam depois)."""
        start = span[0]
        i = bisect_right(self.starts, start - 1) - 1
        while i >= 0:
            candidate = self.spans[i]
            if candidate[0] < start and candidate[1] >= span[1]:
                return candidate
            i -= 1
        return None

    def referrers(self, object_id: str) -> list[tuple[int, int]]:
        """Objetos que referem "this.<object_id>"."""
        found: list[tuple[int, int]] = []
        for match in re.finditer(r'"this\.' + re.escape(object_id) + r'"', self.text):
            span = self.object_at(match.start())
            if span and span not in found:
                found.append(span)
        return found


def playlist_items(script: TourScript) -> list[str | None]:
    """Panorama de cada item da mainPlayList, pela ordem (None para itens que não são panoramas)."""
    text = script.text
    item_media: dict[str, str | None] = {}
    for match in re.finditer(r'"id":"((?:Panorama|Model3D|Video|Photo)?PlayListItem_[0-9A-F_]+)"', text):
        span = script.object_at(match.start())
        if not span:
            continue
        media = re.search(r'"media":"this\.(' + PANORAMA_ID + r')"', script.object_text(span))
        item_media[match.group(1)] = media.group(1) if media else None
    for match in re.finditer(r'"id":"mainPlayList"', text):
        start = text.rfind("{", 0, match.start())
        items_at = text.find('"items":[', start)
        if items_at < 0:
            continue
        array = _balanced_array(text, items_at + len('"items":'))
        order: list[str | None] = []
        # itens como referências («this.PanoramaPlayListItem_…») ou objetos em linha com "media"
        for token in re.finditer(r'this\.([A-Za-z0-9]*PlayListItem_[0-9A-F_]+)"|"media":"this\.(' + PANORAMA_ID + r')"', array):
            if token.group(1):
                order.append(item_media.get(token.group(1)))
            elif token.group(2):
                order.append(token.group(2))
        if order:
            return order
    return []


def panorama_definitions(script: TourScript) -> dict[str, dict]:
    """Por panorama: label literal (se houver), overlays, adjacentPanoramas."""
    text = script.text
    panoramas: dict[str, dict] = {}
    for match in re.finditer(r'"class":"Panorama"', text):
        span = script.object_at(match.start())
        if not span:
            continue
        obj = script.object_text(span)
        pid = PANORAMA_ID_RE.search(obj)
        if not pid:
            continue
        key = pid.group(1)
        if key in panoramas:
            continue
        overlays_at = obj.find('"overlays":[')
        overlays = re.findall(r'"this\.((?:overlay|sprite)_[0-9A-F_]+)"', _balanced_array(obj, overlays_at + len('"overlays":'))) if overlays_at >= 0 else []
        label_match = re.search(r'"data":\{[^{}]*?"label":"([^"]*)"', obj)
        adjacents: list[dict] = []
        inner_spans = object_spans(obj)
        inner_starts = [s[0] for s in inner_spans]
        for adj in re.finditer(r'"class":"AdjacentPanorama"', obj):
            inner = innermost(inner_spans, inner_starts, adj.start())
            if not inner:
                continue
            adj_text = obj[inner[0]:inner[1]]
            target = ADJ_TARGET_RE.search(adj_text)
            if not target or target.group(1) == key:
                continue
            overlay = re.search(r'"overlayID":"((?:overlay|sprite)_[0-9A-F_]+)"', adj_text)
            adjacents.append({
                "target": target.group(1),
                "yaw": _num(r'"yaw":' + NUM, adj_text),
                "distance": _num(r'"distance":' + NUM, adj_text),
                "backward_yaw": _num(r'"backwardYaw":' + NUM, adj_text),
                "overlay_id": overlay.group(1) if overlay else None,
                "enabled": '"enabled":false' not in adj_text,
            })
        panoramas[key] = {"label": label_match.group(1) if label_match else None, "overlays": overlays, "adjacents": adjacents}
    return panoramas


def overlay_directions(script: TourScript, overlay_to_panorama: dict[str, str]) -> dict[str, dict]:
    """overlay_id -> {panorama_key, yaw, pitch, label} de todos os hotspots do export."""
    text = script.text
    result: dict[str, dict] = {}
    for match in re.finditer(r'"id":"((?:overlay|sprite)_[0-9A-F_]+)"', text):
        overlay_id = match.group(1)
        if overlay_id in result:
            continue
        span = script.object_at(match.start())
        if not span:
            continue
        obj = script.object_text(span)
        # HotspotPanoramaOverlay (setas de navegação), FlatHotspotPanoramaOverlay (os POIs das peças) e sprites
        if not re.search(r'"class":"(?:Flat)?HotspotPanorama|"class":"Sprite', obj):
            continue
        label = LABEL_RE.search(obj)
        result[overlay_id] = {
            "panorama_key": overlay_to_panorama.get(overlay_id),
            "yaw": _num(r'"yaw":' + NUM, obj),
            "pitch": _num(r'"pitch":' + NUM, obj),
            "label": label.group(1) if label else None,
        }
    return result


def click_edges(script: TourScript, items: list[str | None], overlay_to_panorama: dict[str, str], directions: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Arestas `hotspot` a partir dos clicks de navegação; devolve (arestas, não resolvidos)."""
    text = script.text
    edges: list[dict] = []
    unresolved: list[dict] = []
    seen_areas: set[str] = set()
    for kind, pattern in (("playlist", CLICK_PLAYLIST_RE), ("media", CLICK_MEDIA_RE)):
        for match in pattern.finditer(text):
            span = script.object_at(match.start())
            if not span:
                continue
            obj = script.object_text(span)
            if '"click"' not in obj:
                continue
            area = ANY_ID_RE.search(obj)
            area_id = area.group(1) if area else f"pos{span[0]}"
            if area_id in seen_areas:
                continue
            seen_areas.add(area_id)
            if kind == "playlist":
                index = int(match.group(1))
                target = items[index] if 0 <= index < len(items) else None
                if target is None:
                    unresolved.append({"area": area_id, "motivo": f"item {index} da mainPlayList não é um panorama"})
                    continue
            else:
                target = match.group(1)
            # área → overlay pai (ou a própria área já é um overlay) → panorama
            overlay_id = area_id if area_id.startswith(("overlay_", "sprite_")) else None
            if overlay_id is None:
                for ref_span in script.referrers(area_id):
                    candidate = ANY_ID_RE.search(script.object_text(ref_span))
                    if candidate and candidate.group(1).startswith(("overlay_", "sprite_")):
                        overlay_id = candidate.group(1)
                        break
            if overlay_id is None:
                unresolved.append({"area": area_id, "motivo": "sem overlay pai (botão fora dos panoramas?)"})
                continue
            source = overlay_to_panorama.get(overlay_id)
            if source is None:
                for ref_span in script.referrers(overlay_id):
                    candidate = PANORAMA_ID_RE.search(script.object_text(ref_span))
                    if candidate:
                        source = candidate.group(1)
                        break
            if source is None:
                unresolved.append({"area": area_id, "overlay": overlay_id, "motivo": "overlay sem panorama (mapa/menu)"})
                continue
            if source == target:
                continue
            direction = directions.get(overlay_id) or {}
            edges.append({
                "source": source, "target": target, "overlay_id": overlay_id,
                "yaw": direction.get("yaw"), "pitch": direction.get("pitch"),
                "click": kind,
            })
    return edges, unresolved


def load_rooms(slug: str) -> tuple[dict[str, str], dict[str, str]]:
    """panorama -> sala: dos lugares da visita (tour_locations.json) e dos hotspots do mapa dos POIs."""
    from_tour: dict[str, str] = {}
    try:
        payload = json.loads((ROOT / "app" / "data" / "tour_locations.json").read_text(encoding="utf-8"))
        for place in payload.get(slug) or []:
            for key in place.get("panorama_keys") or []:
                from_tour.setdefault(key, place["room"])
    except Exception:  # noqa: BLE001
        pass
    from_pois: dict[str, str] = {}
    poi_path = ROOT / "poi_tours" / f"panorama-overlays-inventory-{slug}.json"
    if poi_path.exists():
        for entry in json.loads(poi_path.read_text(encoding="utf-8")):
            if entry.get("location") and entry.get("panoramaKey"):
                from_pois.setdefault(entry["panoramaKey"], entry["location"])
    return from_tour, from_pois


def build_graph(slug: str, tour_dir: Path) -> dict:
    script = TourScript((tour_dir / "script_general.js").read_text(encoding="utf-8", errors="replace"))
    names, locale_name = load_locale(tour_dir)
    items = playlist_items(script)
    definitions = panorama_definitions(script)
    overlay_to_panorama = {o: key for key, meta in definitions.items() for o in meta["overlays"]}
    directions = overlay_directions(script, overlay_to_panorama)
    rooms_tour, rooms_pois = load_rooms(slug)

    order: list[str] = []
    first_index: dict[str, int] = {}
    for index, key in enumerate(items):
        if key and key not in first_index:
            first_index[key] = index
            order.append(key)
    for key in definitions:
        if key not in first_index:
            order.append(key)

    nodes: list[dict] = []
    for key in order:
        locale = names.get(key, {})
        label = locale.get("label") or (definitions.get(key) or {}).get("label")
        subtitle = locale.get("subtitle")
        room = rooms_tour.get(key) or rooms_pois.get(key) or (subtitle if is_human_name(subtitle or "") else None)
        nodes.append({
            "panorama_key": key,
            "playlist_index": first_index.get(key),
            "label": label,
            "subtitle": subtitle,
            "room": room,
            "room_source": "tour_locations" if key in rooms_tour else "poi_tours" if key in rooms_pois else "subtitle" if room else None,
        })
    node_keys = {n["panorama_key"] for n in nodes}

    edges: dict[tuple[str, str], dict] = {}

    def add_edge(source: str, target: str, kind: str, **fields) -> None:
        if source not in node_keys or target not in node_keys or source == target:
            return
        edge = edges.setdefault((source, target), {"source": source, "target": target, "kinds": [], "inferred": False})
        if kind not in edge["kinds"]:
            edge["kinds"].append(kind)
        for name, value in fields.items():
            if value is not None and edge.get(name) is None:
                edge[name] = value

    for key, meta in definitions.items():
        for adj in meta["adjacents"]:
            add_edge(key, adj["target"], "adjacent", yaw=adj["yaw"], distance=adj["distance"], overlay_id=adj["overlay_id"], enabled=adj["enabled"])
    hotspot_edges, unresolved = click_edges(script, items, overlay_to_panorama, directions)
    for edge in hotspot_edges:
        add_edge(edge["source"], edge["target"], "hotspot", yaw=edge["yaw"], pitch=edge["pitch"], overlay_id=edge["overlay_id"])
    for edge in edges.values():
        edge.setdefault("enabled", True)

    # nós sem nenhuma ligação: vizinhos na playlist, marcados inferred
    linked = {e["source"] for e in edges.values()} | {e["target"] for e in edges.values()}
    inferred_for: list[str] = []
    for position, key in enumerate(order):
        if key in linked or first_index.get(key) is None:
            continue
        neighbours = [order[i] for i in (position - 1, position + 1) if 0 <= i < len(order) and first_index.get(order[i]) is not None]
        for other in neighbours:
            for a, b in ((key, other), (other, key)):
                if (a, b) not in edges:
                    edges[(a, b)] = {"source": a, "target": b, "kinds": ["inferred"], "inferred": True, "enabled": True}
        if neighbours:
            inferred_for.append(key)

    # componentes ligadas (sem sentido) e órfãos
    adjacency: dict[str, set[str]] = {k: set() for k in node_keys}
    for (a, b) in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    component_of: dict[str, int] = {}
    components: list[list[str]] = []
    for key in order:
        if key in component_of:
            continue
        queue, members = deque([key]), []
        component_of[key] = len(components)
        while queue:
            current = queue.popleft()
            members.append(current)
            for other in adjacency[current]:
                if other not in component_of:
                    component_of[other] = len(components)
                    queue.append(other)
        components.append(members)
    for node in nodes:
        node["component"] = component_of.get(node["panorama_key"])
    orphans = [k for k in order if k not in linked]

    edge_list = sorted(edges.values(), key=lambda e: (first_index.get(e["source"], 10**6), first_index.get(e["target"], 10**6)))
    kinds_count = {"adjacent": 0, "hotspot": 0, "inferred": 0, "both": 0}
    for edge in edge_list:
        for kind in edge["kinds"]:
            kinds_count[kind] += 1
        if "adjacent" in edge["kinds"] and "hotspot" in edge["kinds"]:
            kinds_count["both"] += 1

    return {
        "museum_slug": slug,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {"script": str(tour_dir / "script_general.js"), "locale": locale_name, "playlist_items": len(items)},
        "stats": {
            "nodes": len(nodes),
            "nodes_in_playlist": sum(1 for n in nodes if n["playlist_index"] is not None),
            "nodes_with_room": sum(1 for n in nodes if n["room"]),
            "edges": len(edge_list),
            "edges_by_kind": kinds_count,
            "components": len(components),
            "component_sizes": sorted((len(c) for c in components), reverse=True)[:10],
            "orphans": len(orphans),
            "inferred_for": inferred_for,
            "unresolved_clicks": len(unresolved),
        },
        "nodes": nodes,
        "edges": edge_list,
        "overlays": {k: v for k, v in directions.items() if v.get("panorama_key")},
        "_unresolved": unresolved[:50],
    }


def report(graph: dict) -> None:
    stats = graph["stats"]
    print(f"{graph['museum_slug']}: {stats['nodes']} nós ({stats['nodes_in_playlist']} na playlist, {stats['nodes_with_room']} com sala), "
          f"{stats['edges']} arestas (adjacent {stats['edges_by_kind']['adjacent']}, hotspot {stats['edges_by_kind']['hotspot']}, "
          f"ambas {stats['edges_by_kind']['both']}, inferidas {stats['edges_by_kind']['inferred']})")
    print(f"  componentes: {stats['components']} (tamanhos {stats['component_sizes']}); órfãos: {stats['orphans']}; "
          f"clicks não resolvidos: {stats['unresolved_clicks']}; overlays com direção: {len(graph['overlays'])}")
    sem_sala = [n for n in graph["nodes"] if not n["room"]]
    if sem_sala:
        print(f"  nós sem sala ({len(sem_sala)}): " + ", ".join((n["label"] or n["panorama_key"][:20]) for n in sem_sala[:8]) + (" …" if len(sem_sala) > 8 else ""))
    if stats["inferred_for"]:
        print(f"  ligações inferidas para: {len(stats['inferred_for'])} nós")
    motivos: dict[str, int] = {}
    for item in graph["_unresolved"]:
        motivos[item["motivo"]] = motivos.get(item["motivo"], 0) + 1
    for motivo, count in motivos.items():
        print(f"  não resolvido: {motivo} ×{count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museum", nargs="+", help="slug(s) da visita: mnt mnsr mnaz mj cmag mnaa")
    parser.add_argument("--visita", help="pasta do export (default tmp/visitas/<museu>)")
    parser.add_argument("-o", "--out", help="ficheiro de saída (default poi_tours/tour-graph-<museu>.json)")
    parser.add_argument("--base", default=DEFAULT_BASE, help="URL das visitas publicadas, para descarregar o que faltar")
    args = parser.parse_args()
    if len(args.museum) > 1 and (args.visita or args.out):
        parser.error("--visita/-o só com um museu")
    for slug in args.museum:
        tour_dir = Path(args.visita) if args.visita else ROOT / "tmp" / "visitas" / slug
        ensure_tour_files(tour_dir, slug, args.base.rstrip("/"))
        graph = build_graph(slug, tour_dir)
        out = Path(args.out) if args.out else ROOT / "poi_tours" / f"tour-graph-{slug}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        public = {k: v for k, v in graph.items() if not k.startswith("_")}
        out.write_text(json.dumps(public, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        report(graph)
        print(f"  escrito em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
