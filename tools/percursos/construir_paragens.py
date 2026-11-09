# -*- coding: utf-8 -*-
"""Paragens de uma visita: cada colocação de uma peça num panorama (Fase 0 dos
percursos temáticos).

    python tools/percursos/construir_paragens.py mnaz
    python tools/percursos/construir_paragens.py mnt mnsr mnaz mj cmag mnaa --objetos-min-score 0.70

Junta, por museu:
  - os hotspots do mapa dos POIs (poi_tours/panorama-overlays-inventory-<museu>.json),
    com a direção (yaw, pitch) que o grafo da visita guardou para cada overlay;
  - os objetos localizados offline (poi_tours/panorama-objects-<museu>.json):
    entram os aceites pelo VLM ou por hotspot com score ≥ --objetos-min-score;
    num ficheiro sem verificação VLM (ainda só kNN, sem `accepted_by`) entram
    os de score ≥ --objetos-min-score-sem-vlm (0,80, o mesmo limiar do «Ver na
    tour» por objetos);
  - a ficha do índice de peças (leitura, só os campos do cartão), pelo n.º de
    inventário normalizado (sem maiúsculas, espaços nem pontuação).

Um registo por (peça, panorama), em poi_tours/tour-stops-<museu>.ndjson:
  stop_id, museum_slug, artifact_id, inventory_number, panorama_key, room,
  playlist_index, placements[] ({kind: hotspot|object, overlay_id|object_id,
  yaw, pitch, bbox, polygon, area_fraction, placement_score}), best_placement
  (a melhor vista da peça na visita: hotspot primeiro, depois a maior
  area_fraction), catalogue{…} (cópia da ficha) e catalogue_found.

Relatório de cobertura no fim (e em poi_tours/tour-stops-<museu>.report.json):
peças visíveis com/sem fotografia, por sala, hotspots vs objetos, inventários
sem ficha. Só lê o OpenSearch; não escreve em nenhum índice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.national_treasures import inventory_key  # noqa: E402
from app.services.tour_navigation import TourNavigationService  # noqa: E402

CATALOGUE_FIELDS = [
    "artifact_id", "inventory_number", "inventory_type", "title", "description", "technique", "support_or_material",
    "date_or_period", "start_year", "end_year", "creators", "creator_ids", "production_center", "category",
    "super_category", "origin_history", "dimensions_cm", "in_tour", "image_count", "image_paths", "image_ids", "detail_url",
]
VLM_ACCEPTED = frozenset({"vlm", "hotspot"})


def century(year: Any) -> int | None:
    try:
        value = int(year)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return (value - 1) // 100 + 1


def load_catalogue(museum_id: str) -> dict[str, list[dict[str, Any]]]:
    """inventário normalizado -> fichas do museu (só leitura; só os campos do cartão)."""
    from opensearchpy.helpers import scan

    from app.services.opensearch_client import OpenSearchGateway

    settings = get_settings()
    client = OpenSearchGateway(settings)._ensure_client()
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in scan(client, index=settings.OPENSEARCH_INDEX_ARTIFACT, size=1000,
                    query={"_source": CATALOGUE_FIELDS, "query": {"term": {"museum_id": museum_id}}}):
        source = hit.get("_source") or {}
        key = inventory_key(str(source.get("inventory_number") or ""))
        if key:
            source.setdefault("artifact_id", hit.get("_id"))
            index[key].append(source)
    return index


def catalogue_copy(doc: dict[str, Any]) -> dict[str, Any]:
    start_year = doc.get("start_year")
    end_year = doc.get("end_year")
    return {
        "title": doc.get("title"),
        "description": doc.get("description"),
        "technique": doc.get("technique"),
        "support_or_material": doc.get("support_or_material"),
        "date_or_period": doc.get("date_or_period"),
        "start_year": start_year,
        "end_year": end_year,
        "start_century": century(start_year),
        "end_century": century(end_year),
        "creators": list(doc.get("creators") or []),
        "production_center": doc.get("production_center"),
        "category": doc.get("category"),
        "super_category": doc.get("super_category"),
        "inventory_type": doc.get("inventory_type"),
        "dimensions_cm": doc.get("dimensions_cm"),
        "in_tour": bool(doc.get("in_tour")),
        "image_count": int(doc.get("image_count") or 0),
        "image_paths": list(doc.get("image_paths") or []),
        "image_ids": list(doc.get("image_ids") or []),
        "detail_url": doc.get("detail_url"),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stops(slug: str, *, objects_min_score: float, objects_min_score_no_vlm: float) -> tuple[list[dict], dict]:
    nav = TourNavigationService(get_settings())
    graph = load_json(ROOT / "poi_tours" / f"tour-graph-{slug}.json")
    node_by_key = {n["panorama_key"]: n for n in graph["nodes"]}
    overlay_dir = graph.get("overlays") or {}
    poi_path = ROOT / "poi_tours" / f"panorama-overlays-inventory-{slug}.json"
    pois = load_json(poi_path) if poi_path.exists() else []
    objects_path = ROOT / "poi_tours" / f"panorama-objects-{slug}.json"
    objects_payload = load_json(objects_path) if objects_path.exists() else {"panoramas": {}}
    catalogue = load_catalogue(slug)

    # tesouro nacional: por peça, união das entradas do mapa dos POIs
    treasures = {inventory_key(i) for e in pois if e.get("nationalTreasure") for i in (e.get("inventoryIds") or [])}

    stops: dict[tuple[str, str], dict] = {}
    unresolved: Counter[str] = Counter()
    collisions: dict[str, int] = {}

    def resolve(inventory_text: str) -> tuple[str, dict[str, Any] | None]:
        """(chave normalizada, ficha) para um n.º tal como a visita o anuncia."""
        for candidate in nav._extract_candidates(inventory_text):
            key = inventory_key(candidate)
            docs = catalogue.get(key)
            if docs:
                if len(docs) > 1:
                    collisions[key] = len(docs)
                return key, docs[0]
        return inventory_key(inventory_text), None

    def stop_for(key: str, doc: dict[str, Any] | None, panorama_key: str, *, inventory_number: str, title: str | None) -> dict:
        artifact_id = str((doc or {}).get("artifact_id") or f"{slug}:inv:{key}")
        stop = stops.get((artifact_id, panorama_key))
        if stop is None:
            node = node_by_key.get(panorama_key) or {}
            stop = {
                "stop_id": f"{slug}:{artifact_id}@{panorama_key}",
                "museum_slug": slug,
                "artifact_id": artifact_id,
                "inventory_number": str((doc or {}).get("inventory_number") or inventory_number),
                "inventory_key": key,
                "title": (doc or {}).get("title") or title,
                "panorama_key": panorama_key,
                "room": node.get("room"),
                "playlist_index": node.get("playlist_index"),
                "panorama_known": panorama_key in node_by_key,
                "placements": [],
                "best_placement": False,
                "national_treasure": key in treasures,
                "catalogue_found": doc is not None,
                "catalogue": catalogue_copy(doc) if doc else None,
            }
            stops[(artifact_id, panorama_key)] = stop
        return stop

    # 1. hotspots
    for entry in pois:
        panorama_key = str(entry.get("panoramaKey") or "").strip()
        overlay_id = str(entry.get("overlayId") or "").strip()
        inventory_ids = entry.get("inventoryIds") or []
        if isinstance(inventory_ids, str):
            inventory_ids = [inventory_ids]
        for inventory_text in inventory_ids:
            # «47, 48, 51 Mob MNSR»: cada número é uma peça
            parts = [c for c in nav._extract_candidates(inventory_text) if c != inventory_text] or [inventory_text]
            if len(parts) > 1 and any(inventory_key(p) in catalogue for p in parts):
                pieces = parts
            else:
                pieces = [inventory_text]
            for piece in pieces:
                key, doc = resolve(piece)
                if doc is None:
                    unresolved[f"hotspot {piece}"] += 1
                direction = overlay_dir.get(overlay_id) or {}
                stop = stop_for(key, doc, panorama_key, inventory_number=piece, title=entry.get("title"))
                if not any(p["kind"] == "hotspot" and p["overlay_id"] == overlay_id for p in stop["placements"]):
                    stop["placements"].append({
                        "kind": "hotspot", "overlay_id": overlay_id,
                        "yaw": direction.get("yaw"), "pitch": direction.get("pitch"),
                        "bbox": None, "polygon": None, "area_fraction": None, "placement_score": 1.0,
                    })

    # 2. objetos localizados offline
    all_objects = [o for p in (objects_payload.get("panoramas") or {}).values() if isinstance(p, dict) for o in p.get("objects") or []]
    has_vlm = any(o.get("accepted_by") for o in all_objects)
    objects_kept = objects_seen = 0
    for panorama_key, panorama in (objects_payload.get("panoramas") or {}).items():
        for obj in (panorama.get("objects") or []) if isinstance(panorama, dict) else []:
            objects_seen += 1
            score = float(obj.get("score") or 0.0)
            accepted_by = str(obj.get("accepted_by") or "")
            if has_vlm:
                ok = accepted_by in VLM_ACCEPTED and score >= objects_min_score
            else:
                ok = score >= objects_min_score_no_vlm
            if not ok:
                continue
            objects_kept += 1
            key, doc = resolve(str(obj.get("inventory_number") or ""))
            if doc is None:
                unresolved[f"objeto {obj.get('inventory_number')}"] += 1
            stop = stop_for(key, doc, str(panorama_key), inventory_number=str(obj.get("inventory_number") or ""), title=obj.get("title"))
            stop["placements"].append({
                "kind": "object", "object_id": obj.get("object_id"),
                "yaw": obj.get("yaw"), "pitch": obj.get("pitch"),
                "bbox": obj.get("bbox"), "polygon": obj.get("polygon"),
                "area_fraction": obj.get("area_fraction"), "placement_score": round(score, 4),
                "accepted_by": accepted_by or None,
            })

    # 3. melhor vista de cada peça: hotspot primeiro (o mais cedo na visita), depois a maior área
    by_artifact: dict[str, list[dict]] = defaultdict(list)
    for stop in stops.values():
        stop["placements"].sort(key=lambda p: (p["kind"] != "hotspot", -(p.get("area_fraction") or 0.0)))
        stop["kinds"] = sorted({p["kind"] for p in stop["placements"]})
        by_artifact[stop["artifact_id"]].append(stop)
    for artifact_stops in by_artifact.values():
        def rank(stop: dict) -> tuple:
            has_hotspot = "hotspot" in stop["kinds"]
            area = max((p.get("area_fraction") or 0.0) for p in stop["placements"])
            index = stop["playlist_index"] if stop["playlist_index"] is not None else 10**6
            return (not has_hotspot, -area if not has_hotspot else index, index)
        best = min(artifact_stops, key=rank)
        best["best_placement"] = True
        for stop in artifact_stops:
            stop["placement_count"] = len(artifact_stops)

    ordered = sorted(stops.values(), key=lambda s: (s["playlist_index"] if s["playlist_index"] is not None else 10**6, s["inventory_number"]))

    # relatório
    pieces = {s["artifact_id"]: s for s in ordered if s["best_placement"]}
    with_photo = sum(1 for s in pieces.values() if (s.get("catalogue") or {}).get("image_count", 0) > 0)
    by_room: Counter[str] = Counter(s["room"] or "(sem sala)" for s in pieces.values())
    report = {
        "museum_slug": slug,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "thresholds": {"objects_min_score": objects_min_score, "objects_min_score_no_vlm": objects_min_score_no_vlm, "objects_file_has_vlm": has_vlm},
        "records": len(ordered),
        "pieces": len(pieces),
        "pieces_with_catalogue": sum(1 for s in pieces.values() if s["catalogue_found"]),
        "pieces_with_photo": with_photo,
        "pieces_without_photo": len(pieces) - with_photo,
        "pieces_from_hotspots": sum(1 for s in pieces.values() if "hotspot" in s["kinds"]),
        "pieces_only_objects": sum(1 for s in pieces.values() if s["kinds"] == ["object"]),
        "pieces_with_dates": sum(1 for s in pieces.values() if (s.get("catalogue") or {}).get("start_century")),
        "national_treasures": sum(1 for s in pieces.values() if s["national_treasure"]),
        "objects_seen": objects_seen,
        "objects_kept": objects_kept,
        "panoramas_with_stops": len({s["panorama_key"] for s in ordered}),
        "panoramas_unknown": sorted({s["panorama_key"] for s in ordered if not s["panorama_known"]}),
        "rooms": dict(by_room.most_common()),
        "unresolved_inventories": dict(unresolved.most_common(60)),
        "inventory_collisions": collisions,
    }
    return ordered, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museum", nargs="+", help="slug(s): mnt mnsr mnaz mj cmag mnaa")
    parser.add_argument("--objetos-min-score", type=float, default=0.70, help="objetos aceites pelo VLM/hotspot com score ≥ (default 0,70)")
    parser.add_argument("--objetos-min-score-sem-vlm", type=float, default=0.80, help="ficheiros ainda sem verificação VLM: só cosseno ≥ (default 0,80)")
    parser.add_argument("--out-dir", default=str(ROOT / "poi_tours"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for slug in args.museum:
        stops, report = build_stops(slug, objects_min_score=args.objetos_min_score, objects_min_score_no_vlm=args.objetos_min_score_sem_vlm)
        out = out_dir / f"tour-stops-{slug}.ndjson"
        with out.open("w", encoding="utf-8") as handle:
            for stop in stops:
                handle.write(json.dumps(stop, ensure_ascii=False) + "\n")
        (out_dir / f"tour-stops-{slug}.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{slug}: {report['records']} registos, {report['pieces']} peças ({report['pieces_with_catalogue']} com ficha, "
              f"{report['pieces_with_photo']} com foto, {report['pieces_with_dates']} com século, {report['national_treasures']} tesouros); "
              f"hotspots {report['pieces_from_hotspots']}, só objetos {report['pieces_only_objects']} "
              f"(objetos {report['objects_kept']}/{report['objects_seen']}, VLM={report['thresholds']['objects_file_has_vlm']}); "
              f"{report['panoramas_with_stops']} panoramas")
        rooms = list(report["rooms"].items())
        print("  salas: " + ", ".join(f"{r} {n}" for r, n in rooms[:10]) + (" …" if len(rooms) > 10 else ""))
        if report["unresolved_inventories"]:
            items = list(report["unresolved_inventories"].items())
            print(f"  sem ficha ({len(items)}): " + ", ".join(k for k, _ in items[:8]) + (" …" if len(items) > 8 else ""))
        if report["panoramas_unknown"]:
            print(f"  panoramas fora do grafo: {len(report['panoramas_unknown'])}")
        if report["inventory_collisions"]:
            print(f"  n.ºs com mais de uma ficha: {len(report['inventory_collisions'])}")
        print(f"  escrito em {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
