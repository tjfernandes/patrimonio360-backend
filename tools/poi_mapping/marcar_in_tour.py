"""Sincroniza o `in_tour` do índice com os POIs da visita.

O `in_tour` das peças e das imagens é calculado na ingestão a partir dos
ficheiros `poi_tours/panorama-overlays-inventory-<museu>.json`. Quando esses
ficheiros mudam — por exemplo depois de `validar_inventarios_poi.py` corrigir os
números do MNAz — o índice fica desatualizado e as peças da visita deixam de ser
preferidas nas respostas («existem painéis religiosos nesta visita?» devolvia
peças sem hotspot). Medido a 2026-09-18: MNAz com 10 peças marcadas para 42
números nos POIs.

Reingerir tudo era caro; isto compara o índice com os POIs e escreve só os
documentos cujo `in_tour` está errado. A regra de comparação é a mesma da
ingestão e do assistente: maiúsculas, só letras e algarismos.

    python tools/poi_mapping/marcar_in_tour.py --simular
    python tools/poi_mapping/marcar_in_tour.py mnaz
    python tools/poi_mapping/marcar_in_tour.py            # os quatro museus

Corre com o Python do backend (lê o `backend/.env`):
/home/hedge6000pro/miniconda3/envs/visitas_virtuais/bin/python
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.core.config import get_settings  # noqa: E402
from opensearchpy import OpenSearch, helpers  # noqa: E402

MUSEUS = ("mnt", "mnsr", "mnaz", "mj")


def normalizar(valor: object) -> str:
    """Igual à ingestão e ao TourNavigationService: maiúsculas alfanuméricas."""
    return re.sub(r"[^A-Z0-9]+", "", str(valor or "").upper())


def numeros_dos_pois(museu: str) -> set[str]:
    ficheiro = BACKEND / "poi_tours" / f"panorama-overlays-inventory-{museu}.json"
    if not ficheiro.exists():
        return set()
    numeros = set()
    for poi in json.loads(ficheiro.read_text(encoding="utf-8")):
        for inventario in poi.get("inventoryIds") or []:
            if normalizar(inventario):
                numeros.add(normalizar(inventario))
    return numeros


def ligar() -> OpenSearch:
    s = get_settings()
    anfitriao = s.OPENSEARCH_HOST.replace("https://", "").replace("http://", "")
    return OpenSearch(
        hosts=[{"host": anfitriao, "port": int(s.OPENSEARCH_PORT)}],
        http_auth=(s.OPENSEARCH_USERNAME, s.OPENSEARCH_PASSWORD),
        use_ssl=str(s.OPENSEARCH_SCHEME or "https").startswith("https"),
        verify_certs=bool(getattr(s, "OPENSEARCH_VERIFY_CERTS", False)),
        ssl_show_warn=False,
        timeout=60,
    )


def sincronizar(cliente: OpenSearch, indice: str, museu: str, numeros: set[str], simular: bool) -> tuple[int, int, list[str]]:
    """Devolve (a ligar, a desligar, exemplos das que passam a estar na visita)."""
    consulta = {"query": {"bool": {"filter": [{"term": {"museum_id": museu}}]}}}
    ligar_docs, desligar_docs, exemplos = [], [], []
    for doc in helpers.scan(
        cliente, index=indice, query=consulta, _source=["inventory_number", "in_tour", "title"], size=1000, request_timeout=120
    ):
        fonte = doc.get("_source") or {}
        deve = normalizar(fonte.get("inventory_number")) in numeros
        esta = bool(fonte.get("in_tour"))
        if deve == esta:
            continue
        (ligar_docs if deve else desligar_docs).append(
            {"_op_type": "update", "_index": indice, "_id": doc["_id"], "doc": {"in_tour": deve}}
        )
        if deve and len(exemplos) < 5:
            exemplos.append(f"{fonte.get('inventory_number')} · {str(fonte.get('title') or '')[:40]}")
    if not simular and (ligar_docs or desligar_docs):
        helpers.bulk(cliente, ligar_docs + desligar_docs, request_timeout=180)
        cliente.indices.refresh(index=indice)
    return len(ligar_docs), len(desligar_docs), exemplos


def principal(museus: list[str], simular: bool) -> int:
    cliente = ligar()
    s = get_settings()
    for museu in museus:
        numeros = numeros_dos_pois(museu)
        print(f"\n=== {museu}: {len(numeros)} números nos POIs")
        if not numeros:
            print("   sem ficheiro de POIs, não mexo")
            continue
        for indice in (s.OPENSEARCH_INDEX_ARTIFACT, s.OPENSEARCH_INDEX_IMAGE):
            ligados, desligados, exemplos = sincronizar(cliente, indice, museu, numeros, simular)
            print(f"   {indice}: +{ligados} a marcar | -{desligados} a desmarcar")
            for exemplo in exemplos:
                print(f"      {exemplo}")
    if simular:
        print("\n(--simular: não escrevi nada)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museus", nargs="*", default=None, help=f"museus a sincronizar (por omissão {', '.join(MUSEUS)})")
    parser.add_argument("--simular", action="store_true", help="só mostra o que mudaria")
    args = parser.parse_args()
    sys.exit(principal(args.museus or list(MUSEUS), args.simular))
