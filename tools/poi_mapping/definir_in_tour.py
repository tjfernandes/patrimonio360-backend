"""Liga ou desliga o `in_tour` de uma lista de números de inventário.

O `marcar_in_tour.py` sincroniza o índice com os POIs da visita e é essa a via
normal. Isto é a via manual, para quando a resposta certa não está nos POIs:
uma peça que está na sala mas cujo hotspot ainda não foi criado, uma que saiu de
exposição, ou uma correção pontual sem esperar por uma reingestão.

    python tools/poi_mapping/definir_in_tour.py mnaz --valor true --numeros "MNAz 185 Az, MNAz 1000 Az"
    python tools/poi_mapping/definir_in_tour.py mnaz --valor false --ficheiro lista.txt
    python tools/poi_mapping/definir_in_tour.py mnaz --valor true --ficheiro lista.txt --simular

O ficheiro tem um número por linha (`#` comenta). A comparação é a mesma da
ingestão e do assistente — maiúsculas, só letras e algarismos — por isso
`MNAz 185 Az`, `mnaz-185-az` e `MNAZ185AZ` são o mesmo número.

Escreve nos dois índices (peças e imagens) e só nos documentos cujo `in_tour`
está diferente do pedido. Os números que não existirem no catálogo daquele
museu são listados no fim: não são inventados nem ignorados em silêncio.

Atenção: isto é uma marca manual e o índice não guarda de onde ela veio. A
próxima corrida do `marcar_in_tour.py` (ou uma reingestão) volta a pôr o
`in_tour` a valer o que estiver nos POIs e desfaz o que aqui se fizer. Para
uma peça ficar mesmo na visita, o sítio certo é o ficheiro de POIs.

Corre com o Python do backend (lê o `.env` da raiz do repositório):
/home/hedge6000pro/miniconda3/envs/visitas_virtuais/bin/python
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]  # raiz do repositório do backend
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.core.config import get_settings  # noqa: E402
from opensearchpy import OpenSearch, helpers  # noqa: E402


def normalizar(valor: object) -> str:
    """Igual à ingestão e ao TourNavigationService: maiúsculas alfanuméricas."""
    return re.sub(r"[^A-Z0-9]+", "", str(valor or "").upper())


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


def ler_numeros(args: argparse.Namespace) -> dict[str, str]:
    """Números pedidos: {normalizado: como foi escrito}."""
    brutos: list[str] = []
    if args.numeros:
        brutos += re.split(r"[,;\n]", args.numeros)
    if args.ficheiro:
        caminho = Path(args.ficheiro)
        if not caminho.exists():
            sys.exit(f"não encontrei {caminho}")
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            brutos.append(linha.split("#", 1)[0])
    pedidos: dict[str, str] = {}
    for bruto in brutos:
        bruto = bruto.strip()
        if bruto and normalizar(bruto):
            pedidos.setdefault(normalizar(bruto), bruto)
    return pedidos


def aplicar(
    cliente: OpenSearch, indice: str, museu: str, pedidos: dict[str, str], valor: bool, simular: bool
) -> tuple[int, int, set[str]]:
    """Devolve (documentos alterados, já certos, números encontrados)."""
    consulta = {"query": {"bool": {"filter": [{"term": {"museum_id": museu}}]}}}
    alteracoes, ja_certos, encontrados = [], 0, set()
    for doc in helpers.scan(
        cliente, index=indice, query=consulta, _source=["inventory_number", "in_tour", "title"],
        size=1000, request_timeout=120,
    ):
        fonte = doc.get("_source") or {}
        chave = normalizar(fonte.get("inventory_number"))
        if chave not in pedidos:
            continue
        encontrados.add(chave)
        if bool(fonte.get("in_tour")) == valor:
            ja_certos += 1
            continue
        alteracoes.append({"_op_type": "update", "_index": indice, "_id": doc["_id"], "doc": {"in_tour": valor}})
    if not simular and alteracoes:
        helpers.bulk(cliente, alteracoes, request_timeout=180)
        cliente.indices.refresh(index=indice)
    return len(alteracoes), ja_certos, encontrados


def principal(museu: str, pedidos: dict[str, str], valor: bool, indices: str, simular: bool) -> int:
    cliente = ligar()
    s = get_settings()
    escolhidos = {
        "artefactos": [s.OPENSEARCH_INDEX_ARTIFACT],
        "imagens": [s.OPENSEARCH_INDEX_IMAGE],
        "ambos": [s.OPENSEARCH_INDEX_ARTIFACT, s.OPENSEARCH_INDEX_IMAGE],
    }[indices]

    print(f"=== {museu}: pôr in_tour={str(valor).lower()} em {len(pedidos)} número(s)")
    encontrados: set[str] = set()
    for indice in escolhidos:
        alterados, ja_certos, achados = aplicar(cliente, indice, museu, pedidos, valor, simular)
        encontrados |= achados
        print(f"   {indice}: {alterados} alterado(s) | {ja_certos} já estava(m) assim")

    em_falta = [pedidos[chave] for chave in pedidos if chave not in encontrados]
    if em_falta:
        print(f"\n{len(em_falta)} número(s) sem correspondência no catálogo de {museu} (nada foi escrito para estes):")
        for numero in em_falta:
            print(f"   {numero}")
        print("   verificar o número, ou o museu: a comparação ignora espaços, traços e maiúsculas")

    if simular:
        print("\n(--simular: não escrevi nada)")
    elif encontrados:
        print("\nfeito. Nota: a próxima corrida do marcar_in_tour.py repõe o in_tour a partir dos POIs.")
    return 1 if em_falta else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museu", help="museum_id no índice (mnt, mnsr, mnaz, mj)")
    parser.add_argument("--valor", required=True, choices=("true", "false"), help="pôr in_tour a true ou false")
    parser.add_argument("--numeros", default="", help='números separados por vírgula: "MNAz 185 Az, MNAz 1000 Az"')
    parser.add_argument("--ficheiro", default=None, help="ficheiro com um número por linha (# comenta)")
    parser.add_argument("--indices", choices=("ambos", "artefactos", "imagens"), default="ambos",
                        help="onde escrever (por omissão nos dois)")
    parser.add_argument("--simular", action="store_true", help="só mostra o que mudaria")
    args = parser.parse_args()

    numeros = ler_numeros(args)
    if not numeros:
        sys.exit("sem números: usar --numeros \"A, B\" ou --ficheiro lista.txt")
    sys.exit(principal(args.museu, numeros, args.valor == "true", args.indices, args.simular))
