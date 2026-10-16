# -*- coding: utf-8 -*-
"""Aquece a cache das faces do cubo para o «Apontar e perguntar».

O primeiro toque num panorama descarrega as faces da visita publicada (nível 1
para a vista, nível 0 para o recorte): 10 a 20 s. Isto descarrega-as todas de
antemão, para tmp/visitas/<museu>/media/ (a cache do backend).

    ~/miniconda3/envs/visitas_virtuais/bin/python tools/apontar/descarregar_faces.py mnaa
    ~/miniconda3/envs/visitas_virtuais/bin/python tools/apontar/descarregar_faces.py              # todos os museus com visita
    ~/miniconda3/envs/visitas_virtuais/bin/python tools/apontar/descarregar_faces.py mnaa --completo

Os panoramas vêm de app/data/tour_locations.json (todos os lugares da visita)
e de poi_tours/ (os que têm hotspots). Por omissão só o nível da vista (~2 MB
por panorama; o MNAA tem 471 panoramas); com --completo também o nível fino do
recorte (~8 MB por panorama), que de outro modo se descarrega no primeiro
toque em cada panorama (1 a 3 s).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from app.core.config import get_settings  # noqa: E402
from app.services.point_view import CubeFaceStore  # noqa: E402


def panoramas(museu: str) -> list[str]:
    keys: list[str] = []
    try:
        for place in json.loads((RAIZ / "app" / "data" / "tour_locations.json").read_text(encoding="utf-8")).get(museu, []):
            for key in place.get("panorama_keys") or []:
                if key not in keys:
                    keys.append(key)
    except FileNotFoundError:
        pass
    poi = RAIZ / "poi_tours" / f"panorama-overlays-inventory-{museu}.json"
    if poi.is_file():
        for entry in json.loads(poi.read_text(encoding="utf-8")):
            key = entry.get("panoramaKey")
            if key and key not in keys:
                keys.append(key)
    return [k for k in keys if k.startswith("panorama_")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museus", nargs="*", help="slugs (por omissão, os de poi_tours/)")
    parser.add_argument("--completo", action="store_true", help="também o nível fino do recorte (POINT_CROP_FACE_LEVEL)")
    args = parser.parse_args()
    args.so_vista = not args.completo
    settings = get_settings()
    museus = args.museus or sorted(p.stem.removeprefix("panorama-overlays-inventory-") for p in (RAIZ / "poi_tours").glob("panorama-overlays-inventory-*.json"))
    stores = [CubeFaceStore(base_url=settings.POINT_TOURS_BASE_URL, cache_dir=settings.point_tiles_cache_dir_resolved,
                            tiles_per_side=int(settings.POINT_FACE_TILES), level=int(settings.POINT_FACE_LEVEL))]
    if not args.so_vista:
        stores.append(CubeFaceStore(base_url=settings.POINT_TOURS_BASE_URL, cache_dir=settings.point_tiles_cache_dir_resolved,
                                    tiles_per_side=int(settings.POINT_CROP_FACE_TILES), level=int(settings.POINT_CROP_FACE_LEVEL)))
    for museu in museus:
        keys = panoramas(museu)
        started = time.time()
        total = 0
        falhas = []
        for index, key in enumerate(keys, start=1):
            for store in stores:
                try:
                    total += store.prefetch(museu, key)
                except Exception as exc:
                    falhas.append((key, store.level, str(exc)[:80]))
            print(f"\r{museu}: {index}/{len(keys)} panoramas, {total} tiles, {time.time() - started:.0f} s", end="", flush=True)
        print()
        for key, level, erro in falhas[:10]:
            print(f"  falhou {key} nível {level}: {erro}")
    print(f"cache em {settings.point_tiles_cache_dir_resolved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
