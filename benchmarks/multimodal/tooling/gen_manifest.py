"""E13.1 — gera o manifesto de avaliação congelado (versionável, sem pesados).

Fontes de ground-truth:
- documental/visual-com-alvo: benchmark_cases_live_ids.json + benchmark_cases_multimodal.json
  (target_artifact real, remapeado na Fase 0A).
- image_only: amostra determinística (seed) por museu; alvo = próprio artefacto.
- image_text: image_only + refinamento (visual e temporal); alvo = próprio artefacto.

Escreve backend/benchmarks/multimodal/manifest.json. Read-only no cluster.
"""
import json, ssl, base64, urllib.request, random
from pathlib import Path

BK = Path("/home/hedge6000pro/Projects/AMALIA/patrimonio360_assistant/backend")
OUT = BK / "benchmarks/multimodal/manifest.json"
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

def load_cases(name):
    d = json.loads((BK / "benchmarks/cases" / name).read_text(encoding="utf-8"))
    return d["cases"] if isinstance(d, dict) and "cases" in d else d

cases = []
cid = 0
def add(**kw):
    global cid
    cid += 1
    kw["query_id"] = f"Q{cid:03d}"
    cases.append(kw)

# --- A. documental (target conhecido) ---
for c in load_cases("benchmark_cases_live_ids.json"):
    if c.get("mode") == "text_single" and c.get("enabled", True) and c.get("target_artifact"):
        add(museum_id=c["museum_id"], category="documental", subtype="text_single",
            type="text_documental", text=c["query"], image_ref=None,
            relevant=[c["target_artifact"]], filters={}, notes="target remapeado Fase 0A",
            source=f"live_ids:{c.get('case_id')}")

# --- B. visual com target conhecido (t2i) ---
for c in load_cases("benchmark_cases_multimodal.json"):
    if c.get("mode") == "text_to_image" and c.get("enabled", True) and c.get("target_artifact"):
        add(museum_id=c["museum_id"], category="visual", subtype="text_to_image",
            type="text_visual", text=c["query"], image_ref=None,
            relevant=[c["target_artifact"]], filters={},
            notes="alvo visual conhecido (proxy: 1 alvo)", source=f"multimodal:{c.get('case_id')}")

# --- C. image_only: amostra por museu (seed), alvo = próprio artefacto ---
random.seed(20260714)
for museum, n in [("mnaz", 4), ("mnt", 4), ("mnsr", 4), ("mj", 2)]:
    hits = os_search("cultural_heritage_images_v4", {
        "size": 40, "query": {"function_score": {"query": {"term": {"museum_id": museum}},
                                                   "random_score": {"seed": 20260714, "field": "_seq_no"}}},
        "_source": ["image_id", "artifact_id", "local_path", "inventory_number"]})["hits"]["hits"]
    picked = 0
    for h in hits:
        s = h["_source"]
        p = Path("/home/hedge6000pro/Projects/AMALIA/raiz_scraper/output") / s["local_path"]
        if not p.exists():
            continue
        add(museum_id=museum, category="image_only", subtype="self_retrieval",
            type="image_only", text=None, image_ref=s["local_path"], image_id=s["image_id"],
            relevant=[s["artifact_id"]], filters={}, exclude_image_ids=[s["image_id"]],
            notes="self-retrieval leave-self-out", source="sampled")
        picked += 1
        if picked >= n:
            break

# --- D. image_text: reutiliza algumas image_only + refinamento ---
img_cases = [c for c in cases if c["type"] == "image_only"]
refinements = [
    ("com decoracao floral", "visual", {}),
    ("mais azul", "visual", {}),
    ("apenas do seculo XVIII", "documental", {"_temporal_interval": {"start_year": 1700, "end_year": 1799, "include_unknown": False}}),
]
for i, base in enumerate(img_cases[:6]):
    text, rtype, filt = refinements[i % len(refinements)]
    add(museum_id=base["museum_id"], category="image_text", subtype=f"refine_{rtype}",
        type="image_text", text=text, image_ref=base["image_ref"], image_id=base["image_id"],
        relevant=base["relevant"], filters=filt, exclude_image_ids=[base["image_id"]],
        notes=f"imagem + refinamento {rtype}; alvo=proprio artefacto", source="derived")

manifest = {
    "version": "phase3-mm-benchmark-v1",
    "frozen_at": "2026-07-14",
    "seed": 20260714,
    "n_cases": len(cases),
    "by_type": {t: sum(1 for c in cases if c["type"] == t) for t in
                ["text_documental", "text_visual", "image_only", "image_text"]},
    "by_museum": {m: sum(1 for c in cases if c["museum_id"] == m) for m in SLUG},
    "cases": cases,
}
OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"manifesto: {len(cases)} casos -> {OUT}")
print("por tipo:", manifest["by_type"])
print("por museu:", manifest["by_museum"])
