"""Escreve o enriquecimento de um museu no índice próprio (nunca no índice das peças).

    python -m tools.enriquecimento.indexar --museu mnaa             # simula: diz o que faria
    python -m tools.enriquecimento.indexar --museu mnaa --executar  # cria o índice se faltar e escreve

O índice (OPENSEARCH_INDEX_ENRICHMENT) tem um documento por peça, com o mesmo artifact_id da ficha, e é pequeno:
não leva vetores. Voltar a correr substitui os documentos do museu pelos novos; não há forma de apagar o índice aqui.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from opensearchpy.helpers import bulk

from app.core.config import get_settings
from tools.enriquecimento import vocab
from tools.enriquecimento.construir import CACHE, cliente

PREFIXO = "cultural_heritage_enrichment_"   # o guião recusa escrever noutro índice
K = {"type": "keyword", "normalizer": "lowercase_ascii"}
MAPEAMENTO = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0},
                 "analysis": {"normalizer": {"lowercase_ascii": {"type": "custom", "filter": ["lowercase", "asciifolding"]}}}},
    "mappings": {
        "dynamic": "false",
        "_meta": {"tipo": "enriquecimento por regras", "listas": vocab.VERSAO},
        "properties": {
            "artifact_id": {"type": "keyword"}, "museum_id": {"type": "keyword"}, "inventory_number": K, "title": {"type": "text"},
            "category": K, "in_tour": {"type": "boolean"}, "start_year": {"type": "integer"}, "end_year": {"type": "integer"},
            "enr_versao": {"type": "keyword"}, "enr_tipo": K, "enr_tipo_caminho": K,
            "enr_materiais": K, "enr_materiais_ficha": K, "enr_tecnicas": K, "enr_tecnicas_ficha": K, "enr_estilo": K, "enr_cores": K,
            "enr_funcao": K, "enr_origem_pais": K, "enr_origem_regiao": K, "enr_tamanho": K,
            "enr_temas": K, "enr_temas_palavras": K, "enr_foto": K, "enr_foto_palavras": K,
            "enr_altura_cm": {"type": "float"}, "enr_largura_cm": {"type": "float"}, "enr_comprimento_cm": {"type": "float"},
            "enr_profundidade_cm": {"type": "float"}, "enr_diametro_cm": {"type": "float"},
            "enr_detalhe": {"type": "object", "enabled": False},   # a fonte e o excerto de cada valor: guardados, não pesquisáveis
        },
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--museu", required=True)
    p.add_argument("--executar", action="store_true", help="sem isto só simula")
    args = p.parse_args()
    indice = get_settings().OPENSEARCH_INDEX_ENRICHMENT
    if not indice.startswith(PREFIXO):
        raise SystemExit(f"recuso escrever em «{indice}»: o índice de enriquecimento tem de começar por {PREFIXO}")
    ficheiro = Path(CACHE) / f"{args.museu}.jsonl"
    if not ficheiro.is_file():
        raise SystemExit(f"falta {ficheiro}: correr primeiro o construir.py")
    docs = [json.loads(linha) for linha in open(ficheiro, encoding="utf-8")]
    c = cliente()
    existe = c.indices.exists(index=indice)
    ja = c.count(index=indice, body={"query": {"term": {"museum_id": args.museu}}})["count"] if existe else 0
    print(f"índice {indice}: {'existe' if existe else 'não existe'} · documentos de «{args.museu}» lá: {ja} · a escrever: {len(docs)}")
    if not args.executar:
        print("simulação: nada foi escrito (--executar para escrever)")
        return
    if not existe:
        c.indices.create(index=indice, body=MAPEAMENTO)
    else:
        # campos acrescentados depois de o índice existir (os temas): só se acrescenta, nunca se muda o que lá está
        c.indices.put_mapping(index=indice, body={"properties": MAPEAMENTO["mappings"]["properties"]})
    acoes = ({"_op_type": "index", "_index": indice, "_id": d["artifact_id"], "_source": d} for d in docs)
    certos, erros = bulk(c, acoes, chunk_size=100, max_chunk_bytes=700_000, raise_on_error=False)   # o nginx recusa corpos acima de 1 MB
    # peças que deixaram de existir no museu saem do índice de enriquecimento
    atuais = {d["artifact_id"] for d in docs}
    velhos = [h["_id"] for h in c.search(index=indice, body={"size": 10000, "_source": False, "query": {"term": {"museum_id": args.museu}}})["hits"]["hits"]
              if h["_id"] not in atuais]
    for _id in velhos:
        c.delete(index=indice, id=_id)
    c.indices.refresh(index=indice)
    total = c.count(index=indice, body={"query": {"term": {"museum_id": args.museu}}})["count"]
    print(f"escritos {certos} · erros {len(erros) if isinstance(erros, list) else erros} · retirados {len(velhos)} · no índice agora: {total}")
    if erros:
        print("primeiro erro:", json.dumps(erros[0], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
