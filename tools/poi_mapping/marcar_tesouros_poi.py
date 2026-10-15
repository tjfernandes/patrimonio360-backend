# -*- coding: utf-8 -*-
"""
Escreve o `nationalTreasure` (true/false) em poi_tours/panorama-overlays-inventory-<museu>.json
a partir da visita publicada, sem regenerar o mapa: mantem as correccoes feitas a mao
(inventoryIdsOriginal, listas de numeros separadas, zonas).

    tools/poi_mapping/obter_poi_mapping.sh mnaa --so-comparar   # descarrega a visita e as paginas dos POIs
    python tools/poi_mapping/marcar_tesouros_poi.py mnaa [--simular] [--acrescentar-novos]

Uma entrada e Tesouro Nacional quando a pagina do POI que o hotspot abre tem o selo
<b style="color: goldenrod"> Tesouro Nacional </b> (o "treasure" do JSON embutido nao
conta). Cada entrada fica com o valor do hotspot com o mesmo overlayId na visita; sem
ele, pelo numero de inventario. A mesma peca em varios hotspots fica com o mesmo valor.

--acrescentar-novos junta ao mapa os POIs da visita que ele ainda nao tem, com os
numeros tal como a visita os anuncia: confirmam-se depois com validar_inventarios_poi.py
e, quando a visita anuncia um numero que nao e o do catalogo, escreve-se a
correspondencia com inventoryIdsOriginal (ver README).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from extract_poi_mapping import (
    apply_locations,
    extract,
    inventory_key,
    load_locale,
    public,
    unify_treasures,
    with_treasure_field,
)

BACKEND = Path(__file__).resolve().parents[2]


def indent_of(text: str) -> int:
    for line in text.splitlines()[1:]:
        stripped = line.lstrip(" ")
        if stripped:
            return len(line) - len(stripped)
    return 2


def marcar(museu: str, visitas: Path, simular: bool, acrescentar: bool) -> int:
    ficheiro = BACKEND / "poi_tours" / f"panorama-overlays-inventory-{museu}.json"
    tour_dir = visitas / museu
    if not ficheiro.is_file():
        print(f"{museu}: nao ha {ficheiro}")
        return 1
    if not tour_dir.is_dir():
        print(f"{museu}: nao ha {tour_dir}; correr antes tools/poi_mapping/obter_poi_mapping.sh {museu} --so-comparar")
        return 1
    texto = ficheiro.read_text(encoding="utf-8")
    mapa = json.loads(texto)
    visita, _, _ = extract(tour_dir)
    sem_pagina = [e for e in visita if e.get("_page") and not e.get("_pageFound")]
    if sem_pagina and all(not e.get("_pageFound") for e in visita if e.get("_page")):
        print(f"{museu}: as paginas dos POIs nao estao em {tour_dir}; correr o obter_poi_mapping.sh")
        return 1

    por_overlay: dict[str, bool] = {}
    por_numero: dict[str, bool] = {}
    for entrada in visita:
        por_overlay[entrada["overlayId"]] = por_overlay.get(entrada["overlayId"], False) or entrada["nationalTreasure"]
        for numero in entrada["inventoryIds"]:
            por_numero[inventory_key(numero)] = por_numero.get(inventory_key(numero), False) or entrada["nationalTreasure"]

    antes = {e["overlayId"]: e.get("nationalTreasure") for e in mapa}
    sem_par = []
    for entrada in mapa:
        if entrada["overlayId"] in por_overlay:
            entrada["nationalTreasure"] = por_overlay[entrada["overlayId"]]
            continue
        numeros = (entrada.get("inventoryIds") or []) + (entrada.get("inventoryIdsOriginal") or [])
        conhecidos = [por_numero[inventory_key(n)] for n in numeros if inventory_key(n) in por_numero]
        entrada["nationalTreasure"] = any(conhecidos)
        if not conhecidos:
            sem_par.append(entrada)

    novos = []
    if acrescentar:
        conhecidos_ids = {e["overlayId"] for e in mapa}
        novos = [public(e) for e in visita if e["overlayId"] not in conhecidos_ids]
        nomes, _ = load_locale(tour_dir)
        if nomes:
            apply_locations(novos, nomes, "subtitle")
        mapa.extend(novos)

    unify_treasures(mapa)
    mapa = [with_treasure_field(e, e["nationalTreasure"]) for e in mapa]

    tesouros = [e for e in mapa if e["nationalTreasure"]]
    numeros = {inventory_key(n): n for e in tesouros for n in e["inventoryIds"]}
    mudados = [e for e in mapa if e["overlayId"] in antes and antes[e["overlayId"]] != e["nationalTreasure"]]
    print(f"== {museu}: {len(mapa)} entradas · Tesouro Nacional em {len(tesouros)} entradas, "
          f"{len(numeros)} numeros · {len(mudados)} entradas mudaram · {len(novos)} POIs novos")
    for e in tesouros:
        print(f"    * {', '.join(e['inventoryIds']):40} {e.get('title') or ''}")
    for e in novos:
        print(f"    + {', '.join(e['inventoryIds']):40} {e.get('title') or ''} (novo; confirmar o numero)")
    for e in sem_par:
        print(f"    ? {', '.join(e.get('inventoryIds') or []):40} {e.get('title') or ''}: sem hotspot na visita, fica false")
    if simular:
        print("    (--simular: nao escrevi nada)")
        return 0
    saida = json.dumps(mapa, ensure_ascii=False, indent=indent_of(texto))
    ficheiro.write_text(saida + ("\n" if texto.endswith("\n") else ""), encoding="utf-8")
    print(f"    escrito: {ficheiro}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museus", nargs="*", help="slugs (por omissao, todos os de poi_tours/)")
    parser.add_argument("--visitas", default=str(BACKEND / "tmp" / "visitas"), help="pasta com as visitas descarregadas")
    parser.add_argument("--acrescentar-novos", action="store_true", help="junta os POIs da visita que o mapa nao tem")
    parser.add_argument("--simular", action="store_true", help="so mostra, nao escreve")
    args = parser.parse_args()
    museus = args.museus or sorted(
        p.stem.removeprefix("panorama-overlays-inventory-") for p in (BACKEND / "poi_tours").glob("panorama-overlays-inventory-*.json")
    )
    codigo = 0
    for museu in museus:
        codigo |= marcar(museu, Path(args.visitas), args.simular, args.acrescentar_novos)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
