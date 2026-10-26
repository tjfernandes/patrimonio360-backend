"""Pré-indexa os objetos de cada panorama: onde estão e que peça do acervo são.

    python tools/apontar/indexar_objetos_panorama.py mnaa --panorama panorama_83D8… --debug /tmp/apontar/objetos
    python tools/apontar/indexar_objetos_panorama.py mnt --hotspots          # os panoramas com hotspots
    python tools/apontar/indexar_objetos_panorama.py mnaa --all              # todos os panoramas em cache

Para cada panorama percorre vistas retilíneas sobrepostas (yaw de 45° em 45°,
hfov 90°, nos pitches pedidos), recorta todas as instâncias do SAM 3 por
conceito, projeta uma vista apertada no nível fino para cada uma, passa-a pelo
Qwen3-VL-Embedding e vai ao kNN do índice de imagens (só leitura). Guarda as
que se identificam com confiança (score ≥ --threshold e à frente da segunda
peça por ≥ --margin), com o centro (yaw, pitch), a caixa e o contorno em
graus, em poi_tours/panorama-objects-<museu>.json (junta ao que lá estiver).

Com --debug grava cada vista com os contornos (verde: identificado, com o n.º
e o score; vermelho: recortado mas sem peça) e uma faixa com as vistas do
horizonte. No fim imprime, por panorama, quantos objetos recortou, quantos
identificou e quantas das peças com hotspot nesse panorama encontrou.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import get_embedding_provider  # noqa: E402
from app.services.opensearch_client import get_opensearch_gateway  # noqa: E402
from app.services.point_ask import PointAskService  # noqa: E402
from app.services.point_segmenter import crop_and_cutout, mask_polygon  # noqa: E402
from app.services.point_view import Camera, mask_in_camera, point_in_view, project_view, ray_from_point  # noqa: E402
from app.services.tour_navigation import TourNavigationService  # noqa: E402
from app.services.tour_pois import get_tour_poi_store  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("indexar_objetos")


def unwrap(yaw: float, around: float) -> float:
    """O yaw mais próximo de `around` (evita saltos em ±180 nos contornos)."""
    return around + ((yaw - around + 180.0) % 360.0 - 180.0)


def normalize_yaw(yaw: float) -> float:
    return (yaw + 180.0) % 360.0 - 180.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a, b).sum())


def bbox_iou(a: dict, b: dict) -> float:
    """IoU de duas caixas angulares (yaws já desenrolados perto um do outro)."""
    b_yaw_min = unwrap(b["yaw_min"], a["yaw_min"])
    b_yaw_max = b_yaw_min + (b["yaw_max"] - b["yaw_min"])
    w = min(a["yaw_max"], b_yaw_max) - max(a["yaw_min"], b_yaw_min)
    h = min(a["pitch_max"], b["pitch_max"]) - max(a["pitch_min"], b["pitch_min"])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area_a = (a["yaw_max"] - a["yaw_min"]) * (a["pitch_max"] - a["pitch_min"])
    area_b = (b["yaw_max"] - b["yaw_min"]) * (b["pitch_max"] - b["pitch_min"])
    return float(inter / max(area_a + area_b - inter, 1e-9))


class Indexer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.settings = get_settings()
        self.museum = args.museum
        self.point = PointAskService(self.settings)
        self.point.concepts.prompts = tuple(p.strip() for p in args.prompts.split(",") if p.strip())
        self.point.concepts.threshold = float(args.concept_threshold)
        if not self.point.concepts.load():
            raise SystemExit(f"SAM 3 por conceito não carregou: {self.point.concepts.error}")
        self.embedder = get_embedding_provider()._get_multimodal_embedder()
        self.gateway = get_opensearch_gateway()
        self.nav = TourNavigationService(self.settings)
        self.height = int(round(args.width * 0.75))

    # --- uma vista ---------------------------------------------------------

    def cameras(self) -> list[Camera]:
        pitches = [float(p) for p in self.args.pitches.split(",") if p.strip()]
        return [
            Camera(yaw=float(yaw), pitch=pitch, hfov=float(self.args.hfov), width=self.args.width, height=self.height)
            for pitch in pitches
            for yaw in range(-180, 180, int(self.args.yaw_step))
        ]

    def segment_view(self, image: np.ndarray) -> list:
        hits = self.point.concepts.instances_sync(image, min_area=self.args.min_area, max_area=self.args.max_area)
        hits.sort(key=lambda h: h.score, reverse=True)
        kept: list = []
        for hit in hits:
            # a mesma coisa vista por dois prompts («artwork» e «painting»): fica o melhor score
            if any(mask_iou(hit.mask, other.mask) > 0.7 for other in kept):
                continue
            # cortado pela borda lateral da vista: a vista vizinha (45° ao lado) vê-o inteiro
            if hit.mask[:, :3].any() or hit.mask[:, -3:].any():
                continue
            kept.append(hit)
        return kept

    def identify(self, panorama_key: str, camera: Camera, mask: np.ndarray) -> dict | None:
        try:
            zoom_camera = PointAskService._zoom_camera(camera, mask, margin=float(self.settings.POINT_CROP_MARGIN))
            zoom = project_view(self.point.zoom_faces, museum=self.museum, panorama_key=panorama_key,
                                camera=zoom_camera, width=int(self.settings.POINT_CROP_WIDTH))
            zh, zw = zoom.image.shape[:2]
            crop = crop_and_cutout(zoom.image, mask_in_camera(mask, camera, zoom_camera, zw, zh), margin=0.0, whole=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("vista apertada falhou (%s); recorte na vista geral", exc)
            crop = crop_and_cutout(np.asarray(self._current_view), mask)
        embedding = self.embedder.embed_image_bytes(crop.crop_jpeg)
        page = asyncio.run(self.gateway.search_similar_images_page(
            museum_slug=self.museum, museum_id=self.museum, image_embedding=embedding,
            from_offset=0, page_size=max(int(self.settings.POINT_SEARCH_K), 3), retrieval_window_size=None,
        ))
        candidates = self.point.rank(list(page.results), in_tour_keys=set(), panorama_keys=set(),
                                     normalize_key=self.nav.normalize_inventory_key)
        if not candidates:
            return None
        top = candidates[0]
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        return {"artifact_id": top.artifact_id, "inventory_number": top.inventory_number, "title": top.title,
                "score": round(top.score, 4), "runner_up_score": round(runner_up, 4), "image_name": top.image_name}

    # --- um panorama -------------------------------------------------------

    def index_panorama(self, panorama_key: str) -> tuple[dict, list[dict]]:
        started = time.perf_counter()
        raw: list[dict] = []
        views: list[tuple[Camera, np.ndarray]] = []
        for camera in self.cameras():
            view = project_view(self.point.faces, museum=self.museum, panorama_key=panorama_key, camera=camera, width=self.args.width)
            self._current_view = view.image
            views.append((camera, view.image))
            for hit in self.segment_view(view.image):
                ys, xs = np.nonzero(hit.mask)
                cx, cy = float(xs.mean() + 0.5) / camera.width, float(ys.mean() + 0.5) / camera.height
                yaw, pitch = ray_from_point(camera, cx, cy)
                polygon = []
                for x, y in mask_polygon(hit.mask):
                    vy, vp = ray_from_point(camera, x, y)
                    polygon.append([round(unwrap(vy, yaw), 3), round(vp, 3)])
                yaws, pitches = [p[0] for p in polygon], [p[1] for p in polygon]
                match = self.identify(panorama_key, camera, hit.mask)
                raw.append({
                    "view": {"yaw": camera.yaw, "pitch": camera.pitch}, "prompt": hit.prompt, "segment_score": round(hit.score, 3),
                    "area_fraction": round(hit.area_fraction, 4), "yaw": round(yaw, 3), "pitch": round(pitch, 3),
                    "bbox": {"yaw_min": min(yaws), "yaw_max": max(yaws), "pitch_min": min(pitches), "pitch_max": max(pitches)},
                    "polygon": polygon, **(match or {"artifact_id": None, "score": 0.0, "runner_up_score": 0.0}),
                })
        identified = [o for o in raw if o.get("artifact_id") and o["score"] >= self.args.threshold
                      and (o["score"] - o["runner_up_score"]) >= self.args.margin]
        identified.sort(key=lambda o: o["score"], reverse=True)
        final: list[dict] = []
        for obj in identified:
            if any(other["artifact_id"] == obj["artifact_id"] for other in final):
                continue   # a mesma peça vista de duas vistas: fica a de melhor score
            if any(bbox_iou(other["bbox"], obj["bbox"]) > 0.5 for other in final):
                continue   # duas peças no mesmo sítio: fica a de melhor score
            final.append(obj)
        for index, obj in enumerate(final, start=1):
            obj["object_id"] = f"{panorama_key}:{index}"
        hotspot_keys = self.nav.panorama_inventory_keys(museum_slug=self.museum, museum_id=self.museum, panorama_key=panorama_key)
        found_keys = {self.nav.normalize_inventory_key(o.get("inventory_number")) for o in final}
        # «MNAA, Inv, 73 Tap» no mapa dos POIs vs «73 Tap» no índice: conta como encontrado se o n.º termina igual
        hotspot_found = {k for k in hotspot_keys if any(k == f or (len(f) >= 3 and k.endswith(f)) for f in found_keys)}
        entry = {
            "views": len(views), "segmented": len(raw), "identified": len(final),
            "hotspot_pieces": sorted(hotspot_keys), "hotspot_pieces_found": sorted(hotspot_found),
            "elapsed_s": round(time.perf_counter() - started, 1),
            "objects": [{k: v for k, v in o.items() if k not in {"view"}} for o in final],
        }
        if self.args.debug:
            self.dump_debug(panorama_key, views, raw, final)
        return entry, raw

    # --- imagens de depuração ----------------------------------------------

    def dump_debug(self, panorama_key: str, views, raw: list[dict], final: list[dict]) -> None:
        from PIL import Image, ImageDraw

        folder = Path(self.args.debug) / self.museum / panorama_key
        folder.mkdir(parents=True, exist_ok=True)
        final_ids = {id(o) for o in final}
        strip: list[Image.Image] = []
        for camera, image in views:
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            w, h = pil.size
            for obj in raw:
                pts = []
                for vy, vp in obj["polygon"]:
                    xy = point_in_view(camera, vy, vp)
                    if xy is None or not (-0.2 <= xy[0] <= 1.2 and -0.2 <= xy[1] <= 1.2):
                        pts = []
                        break
                    pts.append((xy[0] * w, xy[1] * h))
                if len(pts) < 3:
                    continue
                if id(obj) in final_ids:
                    draw.polygon(pts, outline=(60, 220, 90), width=4)
                    label = f"{obj.get('inventory_number') or obj.get('artifact_id')} · {obj['score']:.2f}"
                    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                    draw.rectangle([x0, max(y0 - 18, 0), x0 + 8 * len(label) + 8, max(y0 - 18, 0) + 18], fill=(60, 220, 90))
                    draw.text((x0 + 4, max(y0 - 17, 0)), label, fill=(0, 0, 0))
                elif obj.get("view") == {"yaw": camera.yaw, "pitch": camera.pitch}:
                    draw.polygon(pts, outline=(230, 70, 70), width=2)
                    draw.text((pts[0][0], pts[0][1]), f"{obj.get('inventory_number') or '—'} {obj['score']:.2f}", fill=(230, 70, 70))
            name = f"yaw{int(camera.yaw):+04d}_pitch{int(camera.pitch):+03d}.jpg"
            pil.save(folder / name, quality=82)
            if camera.pitch == 0.0:
                strip.append(pil.resize((w // 2, h // 2)))
        if strip:
            band = Image.new("RGB", (sum(i.width for i in strip), strip[0].height))
            x = 0
            for i in strip:
                band.paste(i, (x, 0))
                x += i.width
            band.save(folder / "faixa_horizonte.jpg", quality=80)


def panorama_keys_for(args: argparse.Namespace, settings) -> list[str]:
    keys = list(args.panorama or [])
    if args.hotspots:
        entries = get_tour_poi_store().entries(args.museum) or []
        for entry in entries:
            key = str(entry.get("panoramaKey") or "").strip()
            if key and key not in keys:
                keys.append(key)
    if args.all:
        media = Path(settings.point_tiles_cache_dir_resolved) / args.museum / "media"
        for path in sorted(media.glob("panorama_*_0")):
            key = path.name[:-2]
            if key not in keys:
                keys.append(key)
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museum")
    parser.add_argument("--panorama", action="append", help="panorama_key (repetível)")
    parser.add_argument("--hotspots", action="store_true", help="todos os panoramas com hotspots no mapa dos POIs")
    parser.add_argument("--all", action="store_true", help="todos os panoramas com faces em cache")
    parser.add_argument("--threshold", type=float, default=0.75, help="cosseno mínimo da melhor peça")
    parser.add_argument("--margin", type=float, default=0.05, help="avanço mínimo sobre a segunda peça")
    parser.add_argument("--yaw-step", type=int, default=45)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--pitches", default="0,-30", help="pitches das fiadas de vistas, em graus")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--prompts", default="artwork,painting,sculpture,furniture,tapestry,garment,vase")
    parser.add_argument("--concept-threshold", type=float, default=0.3)
    parser.add_argument("--min-area", type=float, default=0.003)
    parser.add_argument("--max-area", type=float, default=0.40)
    parser.add_argument("--out", default=None, help="pasta dos poi_tours (por omissão a do backend)")
    parser.add_argument("--debug", default=None, help="pasta para as vistas com os contornos")
    parser.add_argument("--raw", default=None, help="ficheiro JSONL com todos os recortes, identificados ou não")
    args = parser.parse_args()

    indexer = Indexer(args)
    keys = panorama_keys_for(args, indexer.settings)
    if not keys:
        parser.error("indica --panorama, --hotspots ou --all")
    out_dir = Path(args.out) if args.out else indexer.settings.poi_tours_dir_resolved
    out_path = out_dir / f"panorama-objects-{args.museum}.json"
    payload = {"museum_slug": args.museum, "panoramas": {}}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    payload.update({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": args.threshold, "margin": args.margin,
        "models": {"segmenter": indexer.settings.POINT_SAM_MODEL, "prompts": args.prompts,
                   "embedder": indexer.settings.multimodal_embedding_model_resolved},
        "views": {"yaw_step": args.yaw_step, "hfov": args.hfov, "pitches": args.pitches, "width": args.width},
    })
    raw_file = open(args.raw, "a", encoding="utf-8") if args.raw else None
    print(f"{len(keys)} panorama(s) de {args.museum} · limiar {args.threshold} · margem {args.margin}")
    for key in keys:
        try:
            entry, raw = indexer.index_panorama(key)
        except Exception as exc:  # noqa: BLE001
            print(f"  {key}: FALHOU ({type(exc).__name__}: {exc})")
            continue
        payload["panoramas"][key] = entry
        if raw_file:
            for obj in raw:
                raw_file.write(json.dumps({"panorama_key": key, **obj}, ensure_ascii=False) + "\n")
        hp, hf = entry["hotspot_pieces"], entry["hotspot_pieces_found"]
        print(f"  {key}: {entry['views']} vistas · {entry['segmented']} recortes · {entry['identified']} identificados"
              f" · hotspots {len(hf)}/{len(hp)}" + (f" (faltam {sorted(set(hp) - set(hf))})" if len(hf) < len(hp) else "")
              + f" · {entry['elapsed_s']} s")
        for obj in entry["objects"]:
            print(f"      {obj['inventory_number']!s:14} {obj['score']:.3f} (2.º {obj['runner_up_score']:.2f})  yaw {obj['yaw']:7.1f}  pitch {obj['pitch']:6.1f}  {obj['prompt']:10} {(obj.get('title') or '')[:40]}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if raw_file:
        raw_file.close()
    print(f"escrito em {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
