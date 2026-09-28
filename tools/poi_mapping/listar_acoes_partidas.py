"""Lista os hotspots cuja ação não compila, para corrigir no editor do 3DVista.

Cada hotspot tem uma ação «Execute JavaScript» que avisa o assistente:

    eval('window.postMessage({ type: "openedInfoWindow", inventoryIds: ["..."],
          title: "...", location: "..." }, "*")')

No MNAz o `location` foi escrito com aspas curvas (“…”), o eval rebenta com
«Invalid or unexpected token» e o try/catch da visita engole o erro — o aviso
nunca sai. O tour-focus-artifact.js repara isto em tempo de execução (dá ele o
aviso), mas o erro continua a aparecer na consola e a correção definitiva é no
projeto: trocar as aspas curvas por aspas normais.

Não se pode corrigir no ficheiro publicado: o player valida um hash do projeto
e passa a desenhar a marca de água do 3DVista.

    python tools/poi_mapping/listar_acoes_partidas.py mnaz
    python tools/poi_mapping/listar_acoes_partidas.py mnt mnsr mnaz mj --csv /tmp/acoes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import sys
import urllib.request

BASE = "https://patrimonio360.webredirect.org:5500"
ACAO = re.compile(
    r"showWindowBase\(this\.(window_[A-Za-z0-9_]+)[\s\S]{0,600}?openedInfoWindow"
    r"[\s\S]{0,400}?inventoryIds:\s*\[([^\]]*)\]([\s\S]{0,400}?)\}"
)
CURVAS = re.compile(r"[\u201c\u201d\u2018\u2019]")
TEXTO = re.compile(r"\\?[\"'\u201c\u2018]([^\"'\u201d\u2019\\]+)\\?[\"'\u201d\u2019]")


def ler_visita(museu: str) -> str:
    contexto = ssl.create_default_context()
    contexto.check_hostname = False
    contexto.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(f"{BASE}/{museu}/script_general.js", context=contexto, timeout=120) as resposta:
        return resposta.read().decode("utf-8", "replace")


def acoes(texto: str) -> list[dict[str, str]]:
    encontradas = []
    for janela, lista, resto in ACAO.findall(texto):
        numeros = TEXTO.findall(lista)
        if not numeros:
            continue
        titulo = re.search(r"title:\s*\\?[\"'\u201c]([^\"'\u201d\\]*)", resto)
        local = re.search(r"location:\s*(\\?[\"'\u201c])([^\"'\u201d\\]*)", resto)
        encontradas.append({
            "janela": janela,
            "inventario": " | ".join(numeros),
            "titulo": titulo.group(1) if titulo else "",
            "location": local.group(2) if local else "",
            "aspas_do_location": "curvas" if local and CURVAS.search(local.group(1)) else "normais",
            "partida": bool(CURVAS.search(resto)),
        })
    return encontradas


def principal(museus: list[str], destino_csv: str | None) -> int:
    linhas = []
    for museu in museus:
        try:
            lista = acoes(ler_visita(museu))
        except Exception as exc:
            print(f"{museu}: não consegui ler a visita ({exc})")
            continue
        partidas = [a for a in lista if a["partida"]]
        print(f"\n=== {museu}: {len(lista)} ações, {len(partidas)} por corrigir")
        for acao in partidas:
            print(f"  {acao['inventario']:24} {acao['titulo'][:46]:48} location: {acao['location']}")
            linhas.append({"museu": museu, **acao})
    if destino_csv and linhas:
        with open(destino_csv, "w", newline="", encoding="utf-8") as ficheiro:
            escritor = csv.DictWriter(ficheiro, fieldnames=list(linhas[0].keys()))
            escritor.writeheader()
            escritor.writerows(linhas)
        print(f"\nescrito: {destino_csv} ({len(linhas)} ações)")
    if linhas:
        print("\nNo editor do 3DVista, em cada hotspot acima: abrir a ação «Execute "
              "JavaScript» e trocar as aspas curvas do `location` por aspas normais.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museus", nargs="*", default=None, help="mnt, mnsr, mnaz, mj (por omissão, os quatro)")
    parser.add_argument("--csv", dest="csv", default=None, help="guardar a lista em CSV")
    args = parser.parse_args()
    sys.exit(principal(args.museus or ["mnt", "mnsr", "mnaz", "mj"], args.csv))
