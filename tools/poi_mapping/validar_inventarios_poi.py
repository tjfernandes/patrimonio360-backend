"""Valida e corrige os números de inventário de um ficheiro de POIs da visita.

O `overlay-inventory-mapper.js` tira da visita 3DVista o que lá está escrito, e
o que lá está nem sempre é o número real: no MNAz os números foram escritos com
`_` onde o catálogo tem espaços (`MNAz_185_Az` -> `MNAz 185 Az`), há etiquetas
com dois números juntos (`MNAz_3097_Az_1000_Az`) e sufixos de intervalo que o
catálogo não usa (`MNAz_7266_a-z_Az` -> `MNAz 7266 Az`). No MJ, pelo contrário,
os `_` fazem parte do número real (`MJ_SN_3`) e não se tocam.

Em vez de adivinhar, cada hipótese é confirmada no índice: só entra no ficheiro
um número que exista mesmo no catálogo daquele museu. Os que não resolvem ficam
como estavam (o hotspot não se perde) e saem no relatório para verificação.

    python tools/poi_mapping/validar_inventarios_poi.py mnaz
    python tools/poi_mapping/validar_inventarios_poi.py mnaz --simular
    python tools/poi_mapping/validar_inventarios_poi.py mnaz --ficheiro /caminho/x.json

Corre com o Python do ambiente do backend (lê o `.env` da raiz do repositório
para chegar ao índice): /home/hedge6000pro/miniconda3/envs/visitas_virtuais/bin/python
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]  # raiz do repositório do backend
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.core.config import get_settings  # noqa: E402
from app.services.opensearch_client import OpenSearchGateway  # noqa: E402

PEDIDOS_EM_PARALELO = 8


def candidatos(bruto: str) -> list[tuple[str, list[str]]]:
    """Hipóteses para um número, da mais conservadora à mais arriscada."""
    bruto = (bruto or "").strip()
    if not bruto:
        return []
    hipoteses: list[tuple[str, list[str]]] = [("tal como está", [bruto])]
    if "_" in bruto:
        hipoteses.append(("_ por espaço", [re.sub(r"_+", " ", bruto).strip()]))
    # «MNAz_3097_Az_1000_Az»: duas peças na mesma etiqueta
    partes = re.findall(r"([A-Za-z]*\s*\d+(?:[a-z]|-[a-z])*)[_ ]+([A-Za-z]{2,4})(?=[_ ]|$)", bruto.replace("_", " "))
    if len(partes) > 1:
        prefixo = re.match(r"^([A-Za-z]+)[_ ]", bruto)
        pref = f"{prefixo.group(1)} " if prefixo else ""
        so_numero = lambda n: re.sub(r"^[A-Za-z]+\s*", "", n).strip()  # noqa: E731
        numeros = [f"{pref}{so_numero(n)} {suf}" for n, suf in partes]
        hipoteses.append(("vários números", numeros))
    # «MNAz 7266 a-z Az»: intervalo de letras que o catálogo não usa
    sem_intervalo = re.sub(r"\s+[a-z]-[a-z](?=\s|$)", "", re.sub(r"_+", " ", bruto)).strip()
    if sem_intervalo != re.sub(r"_+", " ", bruto).strip():
        hipoteses.append(("sem sufixo de intervalo", [sem_intervalo]))
    return hipoteses


def mesmo_numero(a: str, b: str) -> bool:
    limpa = lambda v: re.sub(r"[^a-z0-9]", "", (v or "").lower())  # noqa: E731
    return limpa(a) == limpa(b)


async def existe_no_catalogo(
    gateway: OpenSearchGateway, museu: str, museu_id: str, valor: str, sem: asyncio.Semaphore
) -> str | None:
    """Título da peça se o número existir tal e qual no catálogo, senão None."""
    async with sem:
        try:
            achados = await gateway.fetch_artifacts_by_inventory_numbers(
                museum_slug=museu, museum_id=museu_id, inventory_numbers=[valor], top_k=1
            )
        except Exception as exc:  # rede/índice: não inventa, marca como não confirmado
            print(f"    aviso: consulta a {valor!r} falhou ({exc})")
            return None
    for artefacto in achados or []:
        if mesmo_numero(str(artefacto.get("inventory_number") or ""), valor):
            return str(artefacto.get("title") or "")
    return None


async def principal(museu: str, museu_id: str, ficheiro: Path, simular: bool) -> int:
    pois: list[dict[str, Any]] = json.loads(ficheiro.read_text(encoding="utf-8"))
    gateway = OpenSearchGateway(get_settings())
    sem = asyncio.Semaphore(PEDIDOS_EM_PARALELO)

    # uma consulta por candidato distinto (o top_k limita o total, não por número)
    distintos = sorted({
        valor
        for poi in pois
        for bruto in (poi.get("inventoryIds") or [])
        for _, valores in candidatos(bruto)
        for valor in valores
    })
    print(f"{len(pois)} POIs, {len(distintos)} candidatos distintos a confirmar no catálogo de {museu}…")
    titulos = dict(zip(
        distintos,
        await asyncio.gather(*(existe_no_catalogo(gateway, museu, museu_id, v, sem) for v in distintos)),
    ))
    confirmados = {v: t for v, t in titulos.items() if t is not None}
    print(f"{len(confirmados)} candidatos existem no catálogo\n")

    corrigidos = por_resolver = 0
    relatorio: dict[str, Any] = {
        "museu": museu, "ficheiro": str(ficheiro), "data": datetime.date.today().isoformat(),
        "alterados": [], "por_resolver": [],
    }
    for poi in pois:
        finais: list[str] = []
        for bruto in poi.get("inventoryIds") or []:
            for nome, valores in candidatos(bruto):
                if valores and all(v in confirmados for v in valores):
                    if valores != [bruto]:
                        corrigidos += 1
                        relatorio["alterados"].append({
                            "de": bruto, "para": valores, "regra": nome,
                            "titulos": [confirmados[v] for v in valores],
                            "overlayId": poi.get("overlayId"),
                        })
                    finais.extend(valores)
                    break
            else:
                por_resolver += 1
                relatorio["por_resolver"].append({
                    "numero": bruto, "overlayId": poi.get("overlayId"),
                    "title": poi.get("title"), "location": poi.get("location"),
                })
                finais.append(bruto)  # fica como estava: o hotspot não se perde
        if finais != (poi.get("inventoryIds") or []):
            poi["inventoryIdsOriginal"] = poi.get("inventoryIds")
            poi["inventoryIds"] = finais

    for alterado in relatorio["alterados"][:10]:
        print(f"  {alterado['de']:30} -> {alterado['para']}  [{alterado['regra']}]  {alterado['titulos'][0][:38]!r}")
    if len(relatorio["alterados"]) > 10:
        print(f"  … mais {len(relatorio['alterados']) - 10}")
    if relatorio["por_resolver"]:
        print("\nsem correspondência no catálogo (ficam como estavam, verificar à mão):")
        for falhado in relatorio["por_resolver"]:
            print(f"  {falhado['numero']:30} {str(falhado['title'] or '')[:46]}")

    print(f"\ncorrigidos {corrigidos} | por resolver {por_resolver} | POIs {len(pois)}")
    # o relatório NÃO pode ficar em poi_tours: o indexador lê tudo o que lá está
    # como inventário de uma visita (apareceria um museu «<slug>.validacao»)
    pasta_relatorios = Path(__file__).resolve().parent / "relatorios"
    pasta_relatorios.mkdir(exist_ok=True)
    relatorio_path = pasta_relatorios / f"{ficheiro.stem}.validacao.json"
    if simular:
        print("(--simular: não escrevi nada)")
        return 0
    copia = ficheiro.with_suffix(f".json.bak-{datetime.date.today().isoformat()}")
    if not copia.exists():
        shutil.copy2(ficheiro, copia)
    ficheiro.write_text(json.dumps(pois, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    relatorio_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"escrito: {ficheiro}\ncópia:   {copia}\nrelatório: {relatorio_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museu", help="slug do museu (mnt, mnsr, mnaz, mj)")
    parser.add_argument("--museu-id", default=None, help="id no índice (por omissão, igual ao slug)")
    parser.add_argument("--ficheiro", default=None, help="POIs a validar (por omissão, os do backend)")
    parser.add_argument("--simular", action="store_true", help="só mostra, não escreve")
    args = parser.parse_args()
    caminho = Path(args.ficheiro) if args.ficheiro else BACKEND / "poi_tours" / f"panorama-overlays-inventory-{args.museu}.json"
    if not caminho.exists():
        sys.exit(f"não encontrei {caminho}")
    sys.exit(asyncio.run(principal(args.museu, args.museu_id or args.museu, caminho, args.simular)))
