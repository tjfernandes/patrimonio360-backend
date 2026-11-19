"""Passo 2: candidatos na Wikidata para cada autor (pessoas, com ou sem retrato), pelo serviço
de consultas, 20 nomes por pedido. Retomável."""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from comum import AGENTE, gravar, ler

autores = {a["key"]: a for a in ler("autores.json")}
nomes = ler("nomes.json")
procura = ler("procura.json", {})      # nome procurado -> [qid]
entidades = ler("entidades.json", {})  # qid -> dados


def sparql(consulta):
    dados = urllib.parse.urlencode({"query": consulta, "format": "json"}).encode()
    for tentativa in range(5):
        try:
            pedido = urllib.request.Request("https://query.wikidata.org/sparql", data=dados,
                                            headers={"User-Agent": AGENTE, "Accept": "application/sparql-results+json"})
            return json.loads(urllib.request.urlopen(pedido, timeout=120).read())["results"]["bindings"]
        except urllib.error.HTTPError as erro:
            espera = int(erro.headers.get("Retry-After") or 0) or 15 * (tentativa + 1)
            print("  http", erro.code, "espera", espera, flush=True)
            time.sleep(espera)
        except Exception as erro:  # noqa: BLE001
            print("  erro", str(erro)[:80], flush=True)
            time.sleep(10 * (tentativa + 1))
    return None


def variantes(chave):
    """Nomes a procurar: o que o Qwen escreveu e o do catálogo sem parênteses, com «Apelido, Nome» trocado."""
    v = []
    q = nomes[chave]["nome"]
    if q:
        v.append(q)
    base = re.sub(r"[(\[][^)\]]*[)\]]?", " ", autores[chave]["name"])
    base = re.sub(r"\s+", " ", base).strip(" ,;?")
    if base.count(",") == 1:
        apelido, proprio = [p.strip() for p in base.split(",")]
        base = f"{proprio} {apelido}".strip()
    if base and base.casefold() not in {x.casefold() for x in v}:
        v.append(base)
    return [x.replace('"', "").replace("\\", "") for x in v if len(x) >= 4][:2]


por_autor = {k: variantes(k) for k in autores if nomes.get(k, {}).get("nome")}
todos = sorted({n for v in por_autor.values() for n in v} - set(procura))
if len(sys.argv) > 1:
    todos = todos[: int(sys.argv[1])]
print("nomes a procurar:", len(todos), flush=True)
LOTE = 20
for i in range(0, len(todos), LOTE):
    lote = todos[i:i + LOTE]
    valores = " ".join(json.dumps(n, ensure_ascii=False) for n in lote)
    linhas = sparql(f"""SELECT DISTINCT ?nome ?item WHERE {{
      VALUES ?nome {{ {valores} }}
      SERVICE wikibase:mwapi {{ bd:serviceParam wikibase:endpoint "www.wikidata.org"; wikibase:api "EntitySearch";
        mwapi:search ?nome; mwapi:language "pt". ?item wikibase:apiOutputItem mwapi:item. }}
      ?item wdt:P31 wd:Q5.
    }}""")
    if linhas is None:
        print("  lote falhado, fica para a próxima execução", flush=True)
        continue
    for n in lote:
        procura[n] = []
    for b in linhas:
        procura[b["nome"]["value"]].append(b["item"]["value"].rsplit("/", 1)[-1])
    gravar("procura.json", procura)
    if (i // LOTE) % 10 == 0:
        print(f"  {i + len(lote)}/{len(todos)}", flush=True)
    time.sleep(1.0)

falta = sorted({q for ids in procura.values() for q in ids} - set(entidades))
print("entidades a ler:", len(falta), flush=True)
for i in range(0, len(falta), 80):
    lote = falta[i:i + 80]
    valores = " ".join(f"wd:{q}" for q in lote)
    linhas = sparql(f"""SELECT ?item ?itemLabel ?itemDescription (SAMPLE(?img) AS ?imagem) (MIN(?nasc) AS ?n) (MIN(?morte) AS ?m)
        (SAMPLE(?pn) AS ?precN) (SAMPLE(?pm) AS ?precM) (SAMPLE(?lnLabel) AS ?ln) (SAMPLE(?lmLabel) AS ?lm)
        (GROUP_CONCAT(DISTINCT ?ocLabel; separator="; ") AS ?oc) (SAMPLE(?paisLabel) AS ?pais) (SAMPLE(?pt) AS ?ptwiki) WHERE {{
      VALUES ?item {{ {valores} }}
      OPTIONAL {{ ?item wdt:P18 ?img }}
      OPTIONAL {{ ?item p:P569/psv:P569 [ wikibase:timeValue ?nasc; wikibase:timePrecision ?pn ] }}
      OPTIONAL {{ ?item p:P570/psv:P570 [ wikibase:timeValue ?morte; wikibase:timePrecision ?pm ] }}
      OPTIONAL {{ ?item wdt:P19 ?lnasc }} OPTIONAL {{ ?item wdt:P20 ?lmorte }} OPTIONAL {{ ?item wdt:P106 ?ocup }} OPTIONAL {{ ?item wdt:P27 ?pais_ }}
      OPTIONAL {{ ?pt schema:about ?item; schema:isPartOf <https://pt.wikipedia.org/> }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt,en,es,fr".
        ?item rdfs:label ?itemLabel. ?item schema:description ?itemDescription. ?lnasc rdfs:label ?lnLabel. ?lmorte rdfs:label ?lmLabel.
        ?ocup rdfs:label ?ocLabel. ?pais_ rdfs:label ?paisLabel. }}
    }} GROUP BY ?item ?itemLabel ?itemDescription""")
    if linhas is None:
        print("  lote de entidades falhado", flush=True)
        continue

    def v(b, campo):
        return b.get(campo, {}).get("value") or None

    def ano(texto):
        m = re.match(r"(-?)(\d{4})", texto or "")
        return (int(m.group(2)) * (-1 if m.group(1) else 1)) if m else None

    for b in linhas:
        qid = b["item"]["value"].rsplit("/", 1)[-1]
        imagem = v(b, "imagem")
        entidades[qid] = {
            "qid": qid, "label": v(b, "itemLabel"), "description": v(b, "itemDescription"),
            "image": urllib.parse.unquote(imagem.rsplit("/", 1)[-1]) if imagem else None,
            "birth_year": ano(v(b, "n")), "death_year": ano(v(b, "m")),
            "birth_precision": int(v(b, "precN")) if v(b, "precN") else None, "death_precision": int(v(b, "precM")) if v(b, "precM") else None,
            "birth_place": v(b, "ln"), "death_place": v(b, "lm"), "occupations": v(b, "oc"), "country": v(b, "pais"),
            "ptwiki": urllib.parse.unquote(v(b, "ptwiki").rsplit("/", 1)[-1]).replace("_", " ") if v(b, "ptwiki") else None,
        }
    gravar("entidades.json", entidades)
    print(f"  entidades {i + len(lote)}/{len(falta)}", flush=True)
    time.sleep(1.0)

candidatos = {}
for chave, nomes_procurados in por_autor.items():
    ids = []
    for n in nomes_procurados:
        ids += [q for q in procura.get(n, []) if q not in ids]
    candidatos[chave] = [entidades[q] for q in ids if q in entidades]
gravar("candidatos.json", candidatos)
com = sum(1 for c in candidatos.values() if c)
com_img = sum(1 for c in candidatos.values() if any(x["image"] for x in c))
print(f"autores procurados: {len(candidatos)} | com alguma pessoa: {com} | com alguma pessoa com retrato: {com_img} | "
      f"pares autor–pessoa com retrato: {sum(1 for c in candidatos.values() for x in c if x['image'])}")
