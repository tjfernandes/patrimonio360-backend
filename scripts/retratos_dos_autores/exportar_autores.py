"""Exporta os autores do catálogo (seis museus) com o contexto das peças. Só leitura.
Uso (na pasta de produção): PYTHONPATH=. python exportar_autores.py <saida.json>"""
import json
import sys

from app.core.config import Settings
from app.services.author_cards import build_author_directory, clean_life_date, life_year, merged_author_fields
from app.services.opensearch_client import OpenSearchGateway

MUSEUS = {"mnaa": "Museu Nacional de Arte Antiga", "mnsr": "Museu Nacional Soares dos Reis", "mnaz": "Museu Nacional do Azulejo",
          "cmag": "Casa-Museu Dr. Anastácio Gonçalves", "mj": "Mosteiro dos Jerónimos", "mnt": "Museu Nacional do Traje"}
settings = Settings(_env_file=".env")
gateway = OpenSearchGateway(settings)
client = gateway._ensure_client()


def lista(valor):
    return [valor] if isinstance(valor, str) else [str(v) for v in valor or []]


por_id: dict[str, dict] = {}
for slug, nome_museu in MUSEUS.items():
    filtros = gateway._build_filter_clauses(museum_slug=slug, museum_id=slug, filters=None)
    corpo = {"size": 0, "query": {"bool": {"filter": filtros}}, "aggs": {"autores": {
        "terms": {"field": "creator_ids", "size": 5000},
        "aggs": {
            "ficha": {"top_hits": {"size": 5, "_source": ["creators", "creator_ids", "title", "date_or_period", "category"]}},
            "categorias": {"terms": {"field": "category", "size": 3}},
            "super": {"terms": {"field": "super_category", "size": 2}},
            "seculo_min": {"min": {"field": "start_century"}},
            "seculo_max": {"max": {"field": "end_century"}},
        }}}}
    resposta = client.search(index=settings.OPENSEARCH_INDEX_ARTIFACT, body=corpo)
    baldes = resposta["aggregations"]["autores"]["buckets"]
    n = 0
    for balde in baldes:
        chave = str(balde["key"]).strip()
        hits = balde["ficha"]["hits"]["hits"]
        nome = ""
        for hit in hits:
            ids, nomes = lista(hit["_source"].get("creator_ids")), lista(hit["_source"].get("creators"))
            if chave in ids and len(ids) == len(nomes):
                nome = nomes[ids.index(chave)].strip()
                break
        if not nome:
            continue
        n += 1
        reg = por_id.setdefault(chave, {"id": chave, "name": nome, "count": 0, "museums": {}, "categories": {}, "super": {},
                                         "titles": [], "start_century": None, "end_century": None})
        reg["count"] += balde["doc_count"]
        reg["museums"][nome_museu] = balde["doc_count"]
        for b in balde["categorias"]["buckets"]:
            reg["categories"][b["key"]] = reg["categories"].get(b["key"], 0) + b["doc_count"]
        for b in balde["super"]["buckets"]:
            reg["super"][b["key"]] = reg["super"].get(b["key"], 0) + b["doc_count"]
        for hit in hits[:4]:
            titulo = str(hit["_source"].get("title") or "").strip()
            data = str(hit["_source"].get("date_or_period") or "").strip()
            if titulo:
                reg["titles"].append(f"{titulo} ({data})" if data else titulo)
        lo, hi = balde["seculo_min"].get("value"), balde["seculo_max"].get("value")
        if lo is not None:
            reg["start_century"] = int(lo) if reg["start_century"] is None else min(reg["start_century"], int(lo))
        if hi is not None:
            reg["end_century"] = int(hi) if reg["end_century"] is None else max(reg["end_century"], int(hi))
    print(slug, "autores:", n, file=sys.stderr)

linhas = sorted(por_id.values(), key=lambda r: -r["count"])
diretorio = build_author_directory(linhas)

# fichas do índice de autores
ids = list(por_id)
docs: dict[str, dict] = {}
indice = gateway._entity_index("autor")
for i in range(0, len(ids), 500):
    r = client.search(index=indice, body={"size": 500, "query": {"ids": {"values": ids[i:i + 500]}},
                                          "_source": ["entity_id", "name", "activity", "birth_date", "death_date", "birth_place",
                                                      "death_place", "biography", "url", "start_year", "end_year"]})
    for hit in r["hits"]["hits"]:
        docs[hit["_id"]] = hit["_source"]
        docs[str(hit["_source"].get("entity_id") or hit["_id"])] = hit["_source"]

saida = []
for grupo in diretorio.groups:
    campos = merged_author_fields([docs[i] for i in grupo.ids if i in docs])
    museus, categorias, superc, titulos = {}, {}, {}, []
    lo = hi = None
    for i in grupo.ids:
        reg = por_id[i]
        for k, v in reg["museums"].items():
            museus[k] = museus.get(k, 0) + v
        for k, v in reg["categories"].items():
            categorias[k] = categorias.get(k, 0) + v
        for k, v in reg["super"].items():
            superc[k] = superc.get(k, 0) + v
        titulos += [t for t in reg["titles"] if t not in titulos]
        if reg["start_century"] is not None:
            lo = reg["start_century"] if lo is None else min(lo, reg["start_century"])
        if reg["end_century"] is not None:
            hi = reg["end_century"] if hi is None else max(hi, reg["end_century"])
    saida.append({
        "key": grupo.key, "name": grupo.name, "names": grupo.names, "ids": grupo.ids, "count": grupo.count, "museums": museus,
        "activity": campos["activity"], "birth_date": campos["birth_date"], "death_date": campos["death_date"],
        "birth_year": life_year(campos["birth_date"]), "death_year": life_year(campos["death_date"]),
        "birth_place": campos["birth_place"], "death_place": campos["death_place"],
        "biography": (campos["biography"] or "")[:700] or None, "url": campos["url"],
        "categories": sorted(categorias, key=lambda k: -categorias[k])[:3], "super_categories": sorted(superc, key=lambda k: -superc[k])[:2],
        "titles": titulos[:5], "start_century": lo, "end_century": hi,
    })
json.dump(saida, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
com_ano = sum(1 for a in saida if a["birth_year"] or a["death_year"])
print(f"grupos: {len(saida)} | com algum ano: {com_ano} | com biografia: {sum(1 for a in saida if a['biography'])}", file=sys.stderr)
for a in saida[:6]:
    print(json.dumps({k: a[k] for k in ("name", "count", "activity", "birth_year", "death_year", "categories", "start_century", "end_century", "titles")}, ensure_ascii=False)[:300], file=sys.stderr)
