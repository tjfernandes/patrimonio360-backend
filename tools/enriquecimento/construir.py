"""Lê as fichas de um museu no índice das peças (só leitura), aplica as regras e guarda o enriquecimento.

    python -m tools.enriquecimento.construir --museu mnaa

Escreve tools/enriquecimento/.cache/<museu>.jsonl (um documento por peça, pronto para o indexar.py) e mostra
quanto ficou preenchido e que valores dos campos da ficha ficaram sem correspondência nas listas.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from opensearchpy import OpenSearch
from opensearchpy.helpers import scan

from app.core.config import get_settings
from tools.enriquecimento import regras, temas, vocab

CACHE = Path(__file__).resolve().parent / ".cache"
CAMPOS = ["artifact_id", "museum_id", "inventory_number", "title", "category", "technique", "support_or_material", "production_center",
          "description", "origin_history", "dimensions_cm", "in_tour", "start_year", "end_year"]


def cliente() -> OpenSearch:
    s = get_settings()
    host = s.OPENSEARCH_HOST.replace("https://", "").replace("http://", "")
    return OpenSearch(hosts=[{"host": host, "port": int(s.OPENSEARCH_PORT)}], http_auth=(s.OPENSEARCH_USERNAME, s.OPENSEARCH_PASSWORD),
                      use_ssl=True, verify_certs=False, ssl_show_warn=False, timeout=120)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--museu", required=True, help="museum_id (ex.: mnaa)")
    args = p.parse_args()
    s = get_settings()
    fichas = [h["_source"] for h in scan(cliente(), index=s.OPENSEARCH_INDEX_ARTIFACT, size=500,
                                         query={"query": {"term": {"museum_id": args.museu}}, "_source": CAMPOS})]
    fichas.sort(key=lambda f: str(f.get("artifact_id")))
    if not fichas:
        raise SystemExit(f"nenhuma ficha com museum_id={args.museu} em {s.OPENSEARCH_INDEX_ARTIFACT}")
    respostas = temas.ler_cache(args.museu)   # vazio enquanto o temas.py não tiver corrido para este museu
    docs = []
    for f in fichas:
        enr = regras.enriquecer(f)
        if f["artifact_id"] in respostas:
            enr["temas"] = temas.aceitar(f, respostas[f["artifact_id"]], enr["tipo_objeto"]["valor"] if enr["tipo_objeto"] else None)
        doc = regras.documento(f, enr)
        if "temas" in enr:
            doc["enr_temas"] = [t["valor"] for t in enr["temas"]]
            doc["enr_temas_palavras"] = temas.palavras(enr["temas"])
        docs.append(doc)
    CACHE.mkdir(exist_ok=True)
    saida = CACHE / f"{args.museu}.jsonl"
    with open(saida, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    n = len(docs)
    print(f"{n} peças de «{args.museu}» · listas {vocab.VERSAO} · temas respondidos pelo modelo: {len(respostas)} · {saida}\n")
    print(f"{'campo':<20} {'com valor':>10} {'valores distintos':>18}")
    for campo in ("enr_tipo", "enr_materiais", "enr_materiais_ficha", "enr_tecnicas", "enr_tecnicas_ficha", "enr_funcao", "enr_cores",
                  "enr_origem_pais", "enr_estilo", "enr_tamanho", "enr_temas"):
        valores = collections.Counter(v for d in docs for v in (d.get(campo) if isinstance(d.get(campo), list) else [d.get(campo)] if d.get(campo) else []))
        print(f"{campo:<20} {100 * sum(bool(d.get(campo)) for d in docs) / n:9.1f}% {len(valores):>18}")
    # o que as listas ainda não sabem ler, para as fazer crescer
    for campo, onde, enr in (("support_or_material", "material", "materiais"), ("technique", "técnica", "tecnicas"), ("production_center", "centro de produção", "origem")):
        fora: collections.Counter[str] = collections.Counter()
        for f, d in zip(fichas, docs):
            if vocab.n(f.get(campo)) and not any(v.get("onde") == onde for v in d["enr_detalhe"][enr]):
                fora[vocab.n(f.get(campo))[:60]] += 1
        print(f"\nsem correspondência no campo «{onde}»: {sum(fora.values())} peças — " + "; ".join(f"{k} ({v})" for k, v in fora.most_common(10)))


if __name__ == "__main__":
    main()
