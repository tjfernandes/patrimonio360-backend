# -*- coding: utf-8 -*-
"""
Verifica se cada numero de inventario de um mapeamento de POIs existe mesmo no
catalogo indexado. Tem de correr na maquina do backend (precisa do OpenSearch).

    (a partir da raiz do repositório)
    python tools/poi_mapping/validate_poi_mapping.py poi_tours/panorama-overlays-inventory-mnsr.json --museum mnsr

Sai com codigo 1 se houver POIs que nao resolvem, para poder ser usado em CI.
Nao escreve nada: so le.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raiz do repositório do backend

from app.services.opensearch_client import get_opensearch_gateway  # noqa: E402
from app.services.tour_navigation import TourNavigationService  # noqa: E402
from app.core.config import get_settings  # noqa: E402


async def run(mapping_path: Path, museum: str, verbose: bool) -> int:
    entries = json.loads(mapping_path.read_text(encoding="utf-8"))
    gateway = get_opensearch_gateway()
    navigator = TourNavigationService(get_settings())

    inventories: dict[str, list[dict]] = {}
    for entry in entries:
        for inv in entry.get("inventoryIds") or []:
            inventories.setdefault(inv, []).append(entry)

    print("mapeamento : " + mapping_path.name)
    print("museu      : " + museum)
    print("POIs       : " + str(len(entries)))
    print("inventarios: " + str(len(inventories)))
    print()

    falhas = []
    parciais = []
    for inv in sorted(inventories):
        # os mesmos candidatos que o TourNavigationService usaria
        candidatos = navigator._extract_candidates(inv)
        encontrados = []
        for candidato in candidatos:
            docs = await gateway.fetch_artifacts_by_inventory_numbers(
                museum_slug=museum,
                museum_id=museum,
                inventory_numbers=[candidato],
                top_k=1,
            )
            if docs:
                encontrados.append((candidato, docs[0].get("inventory_number")))

        titulo = inventories[inv][0].get("title")
        if not encontrados:
            falhas.append((inv, titulo))
            print("  FALHA    {:40} {}".format(inv, titulo))
        elif len(encontrados) < len(candidatos):
            parciais.append((inv, titulo, encontrados))
            if verbose:
                print("  parcial  {:40} {} -> {}".format(inv, titulo, encontrados))
        elif verbose:
            print("  ok       {:40} {}".format(inv, titulo))

    print()
    print("resolvem totalmente: " + str(len(inventories) - len(falhas) - len(parciais)))
    print("resolvem em parte  : " + str(len(parciais)))
    print("nao resolvem       : " + str(len(falhas)))

    if falhas:
        print()
        print("POIs sem correspondencia no catalogo (o 'Ver no tour' nao vai funcionar):")
        for inv, titulo in falhas:
            print("    {:40} {}".format(inv, titulo))

    return 1 if falhas else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mapping", help="panorama-overlays-inventory-<slug>.json")
    ap.add_argument("--museum", required=True, help="slug do museu (mnt, mj, mnaz, mnsr)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    code = asyncio.run(run(Path(args.mapping), args.museum, args.verbose))
    sys.exit(code)


if __name__ == "__main__":
    main()
