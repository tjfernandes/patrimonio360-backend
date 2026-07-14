"""E12 — testes formais de integração via API real (8001). Um modo por execução.
Uso: python e12_api.py <off|intent|always> <out.json>
"""
import json, ssl, base64, time, sys, urllib.request, urllib.parse
from pathlib import Path

MODE = sys.argv[1]
OUT = sys.argv[2]
BASE = "http://127.0.0.1:8001"
API = "/api/v1/chat"
SLUG = {"mnt": "museu_nacional_do_traje", "mnaz": "museu_nacional_do_azulejo",
        "mnsr": "museu_nacional_soares_dos_reis", "mj": "mosteiro_dos_jeronimos"}

def read_env(p):
    e = {}
    for l in open(p, encoding="utf-8-sig"):
        l = l.strip()
        if not l or l.startswith("#") or "=" not in l: continue
        k, v = l.split("=", 1); e[k.strip()] = v.strip().strip("\"'")
    return e
IDX = read_env("/home/hedge6000pro/Projects/AMALIA/patrimonio360_indexer/.env")
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

def os_search(index, body):
    r = urllib.request.Request(IDX["OPENSEARCH_URL"] + f"/{index}/_search", method="POST")
    r.add_header("Authorization", "Basic " + base64.b64encode((IDX["OPENSEARCH_USER"] + ":" + IDX["OPENSEARCH_PASSWORD"]).encode()).decode())
    r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r, data=json.dumps(body).encode(), context=CTX, timeout=30) as resp:
        return json.loads(resp.read())

CODES = {}
def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + API + path, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read()); code = r.status
    except urllib.error.HTTPError as e:
        out = {}; code = e.code
    CODES[str(code)] = CODES.get(str(code), 0) + 1
    return out, code, round((time.perf_counter() - t0) * 1000)

def get(path, timeout=120):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(BASE + API + path, timeout=timeout) as r:
            out = json.loads(r.read()); code = r.status
    except urllib.error.HTTPError as e:
        out = {}; code = e.code
    CODES[str(code)] = CODES.get(str(code), 0) + 1
    return out, code, round((time.perf_counter() - t0) * 1000)

def post_image(museum, message=None, timeout=300):
    art = "raiz:movel:1051026" if museum == "mnsr" else "raiz:movel:229394"
    img = os_search("cultural_heritage_images_v4", {"size": 1, "query": {"term": {"artifact_id": art}}, "_source": ["local_path"]})["hits"]["hits"][0]["_source"]
    path = Path("/home/hedge6000pro/Projects/AMALIA/raiz_scraper/output") / img["local_path"]
    boundary = "----e12b"
    parts = []
    fields = [("museum_slug", SLUG[museum]), ("museum_id", museum), ("language", "pt")]
    if message: fields.append(("message", message))
    for k, v in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode() + path.read_bytes() + b"\r\n")
    payload = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + API + "/messages/image", data=payload, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read()); code = r.status
    except urllib.error.HTTPError as e:
        out = {}; code = e.code
    CODES[str(code)] = CODES.get(str(code), 0) + 1
    return out, code, round((time.perf_counter() - t0) * 1000)

def museums_pure(out, expected):
    ms = {a.get("museum_id") for a in out.get("artifact_results", [])}
    return not ms or ms == {expected}

R = {"mode": MODE, "sections": {}}
h, _, _ = get("/health")
R["models"] = (h.get("text_embedding_model"), h.get("multimodal_embedding_model"))

doc_q = [
    ("mnaz", "painel com cabeca de perfil e coracao"),
    ("mnt", "vestido de cerimonia em seda"),
    ("mnsr", "quem foi o autor mais representado na colecao?"),
    ("mnt", "pecas do seculo XIX"),
    ("mnaz", "qual o horario de abertura do museu?"),
    ("mnsr", "quantas pinturas ha na colecao?"),
]
doc_rows = []
for m, q in doc_q:
    out, code, ms = post("/messages", {"museum_slug": SLUG[m], "museum_id": m, "language": "pt", "message": q, "results_page_size": 5})
    ids = [a.get("artifact_id") for a in out.get("artifact_results", [])][:5]
    doc_rows.append({"museum": m, "q": q, "code": code, "ms": ms, "ids": ids,
                     "museum_pure": museums_pure(out, m), "n": len(out.get("artifact_results", []))})
R["sections"]["12.1_documental"] = doc_rows

vis_q = [
    ("mnaz", "pecas azuis"), ("mnt", "objetos com flores"), ("mnsr", "imagens de santos"),
    ("mnaz", "azulejos com padroes geometricos"), ("mnsr", "esculturas douradas"),
    ("mnt", "objetos com animais"),
]
vis_rows = []
for m, q in vis_q:
    out, code, ms = post("/messages", {"museum_slug": SLUG[m], "museum_id": m, "language": "pt", "message": q, "results_page_size": 5})
    matches = out.get("image_matches", [])
    ids = [a.get("artifact_id") for a in out.get("artifact_results", [])][:5]
    vis_rows.append({"museum": m, "q": q, "code": code, "ms": ms, "ids": ids,
                     "n": len(out.get("artifact_results", [])), "museum_pure": museums_pure(out, m),
                     "thumb1": (matches[0].get("image_id") if matches else None),
                     "thumb_matches_top": bool(matches and ids and matches[0].get("artifact_id") == ids[0])})
R["sections"]["12.2_visual"] = vis_rows

it_rows = []
for msg in [None, "com decoracao floral", "apenas do seculo XVIII"]:
    out, code, ms = post_image("mnsr", msg)
    arts = [{"id": a.get("artifact_id"), "sy": a.get("start_year"), "ey": a.get("end_year")} for a in out.get("artifact_results", [])][:5]
    matches = out.get("image_matches", [])
    it_rows.append({"message": msg, "code": code, "ms": ms, "ids": [a["id"] for a in arts],
                    "years": [(a["sy"], a["ey"]) for a in arts], "museum_pure": museums_pure(out, "mnsr"),
                    "thumb1": (matches[0].get("image_id") if matches else None)})
R["sections"]["12.4_image_text"] = it_rows

out, code, ms = post("/messages", {"museum_slug": SLUG["mnaz"], "museum_id": "mnaz", "language": "pt", "message": "pecas azuis", "results_page_size": 4})
conv = out.get("conversation_id"); rid = out.get("results_request_id")
p1 = [a.get("artifact_id") for a in out.get("artifact_results", [])]
out2, c2, _ = post("/messages/results", {"museum_slug": SLUG["mnaz"], "museum_id": "mnaz", "conversation_id": conv, "results_page": 2, "results_request_id": rid})
p2 = [a.get("artifact_id") for a in out2.get("artifact_results", [])]
out1b, _, _ = post("/messages/results", {"museum_slug": SLUG["mnaz"], "museum_id": "mnaz", "conversation_id": conv, "results_page": 1, "results_request_id": rid})
p1b = [a.get("artifact_id") for a in out1b.get("artifact_results", [])]
out_oob, c_oob, _ = post("/messages/results", {"museum_slug": SLUG["mnaz"], "museum_id": "mnaz", "conversation_id": conv, "results_page": 99, "results_request_id": rid})
R["sections"]["12.5_pagination"] = {
    "page1": p1, "page2": p2, "page1_repeat": p1b, "page1_stable": p1 == p1b,
    "no_overlap": not (set(p1) & set(p2)), "has_more": out.get("results_has_more"),
    "oob_code": c_oob, "oob_n": len(out_oob.get("artifact_results", [])), "codes_ok": code == 200 and c2 == 200,
}

R["codes"] = CODES
R["fivexx"] = sum(v for k, v in CODES.items() if k.startswith("5"))
Path(OUT).write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[{MODE}] escrito {OUT} | 5xx={R['fivexx']} codes={CODES}")
