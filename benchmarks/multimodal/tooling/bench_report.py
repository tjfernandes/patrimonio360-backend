"""E13 — métricas + go/no-go. Lê bench_{off,intent,always}.json, e12_*.json,
manifest, split, router matrix. Escreve summary.json + report.md."""
import json, statistics, sys
from pathlib import Path

BK = Path("/home/hedge6000pro/Projects/AMALIA/patrimonio360_assistant/backend")
EV = Path("/home/hedge6000pro/p3_eval")
MAN = json.loads((BK / "benchmarks/multimodal/manifest.json").read_text(encoding="utf-8"))
CASE_BY_ID = {c["query_id"]: c for c in MAN["cases"]}

def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

bench = {m: load(EV / f"bench_{m}.json") for m in ("off", "intent", "always")}
e12 = {m: load(EV / f"e12_{m}.json") for m in ("off", "intent", "always")}

def recall_at(rows, k, types):
    sel = [r for r in rows if r["type"] in types and "target_rank" in r]
    if not sel: return None, 0
    hit = sum(1 for r in sel if r["target_rank"] and r["target_rank"] <= k)
    return hit / len(sel), len(sel)

def mrr(rows, types):
    sel = [r for r in rows if r["type"] in types and "target_rank" in r]
    if not sel: return None
    return sum((1.0 / r["target_rank"]) if r["target_rank"] else 0.0 for r in sel) / len(sel)

def zero_rate(rows, types):
    sel = [r for r in rows if r["type"] in types]
    if not sel: return None
    return sum(1 for r in sel if r.get("n", 0) == 0) / len(sel)

DOC = {"text_documental"}
VIS = {"text_visual"}
IMG = {"image_only"}
IT = {"image_text"}

report = {"modes": {}, "gates": {}, "deltas": {}}

for m in ("off", "intent", "always"):
    b = bench[m]
    if not b: continue
    rows = b["results"]
    lat = [r["latency_ms"] for r in rows if "latency_ms" in r]
    contam = [r for r in rows if r.get("contamination") is True]
    dups = [r for r in rows if r.get("dup")]
    errs = [r for r in rows if "error" in r]
    report["modes"][m] = {
        "n": len(rows), "errors": len(errs),
        "retrieval_p50_ms": round(statistics.median(lat), 1) if lat else None,
        "retrieval_p95_ms": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 1) if lat else None,
        "contamination_cases": len(contam),
        "duplicate_cases": len(dups),
        "documental": {"R@1": recall_at(rows, 1, DOC)[0], "R@5": recall_at(rows, 5, DOC)[0],
                       "R@10": recall_at(rows, 10, DOC)[0], "MRR": mrr(rows, DOC), "n": recall_at(rows, 5, DOC)[1]},
        "visual": {"R@1": recall_at(rows, 1, VIS)[0], "R@5": recall_at(rows, 5, VIS)[0],
                   "R@10": recall_at(rows, 10, VIS)[0], "MRR": mrr(rows, VIS), "n": recall_at(rows, 5, VIS)[1],
                   "zero_rate": zero_rate(rows, VIS)},
        "image_only": {"R@1": recall_at(rows, 1, IMG)[0], "R@5": recall_at(rows, 5, IMG)[0],
                       "MRR": mrr(rows, IMG), "n": recall_at(rows, 5, IMG)[1]},
        "image_text": {"R@1": recall_at(rows, 1, IT)[0], "R@5": recall_at(rows, 5, IT)[0],
                       "MRR": mrr(rows, IT), "n": recall_at(rows, 5, IT)[1]},
    }

# off vs intent deltas (pp)
def pp(a, b):
    if a is None or b is None: return None
    return round((b - a) * 100, 1)
if bench["off"] and bench["intent"]:
    o, i = report["modes"]["off"], report["modes"]["intent"]
    report["deltas"]["intent_minus_off"] = {
        "documental_R@5_pp": pp(o["documental"]["R@5"], i["documental"]["R@5"]),
        "documental_MRR_pp": pp(o["documental"]["MRR"], i["documental"]["MRR"]),
        "visual_R@5_pp": pp(o["visual"]["R@5"], i["visual"]["R@5"]),
        "visual_zero_pp": pp(o["visual"]["zero_rate"], i["visual"]["zero_rate"]),
        "retrieval_p95_delta_ms": (i["retrieval_p95_ms"] - o["retrieval_p95_ms"])
        if (i["retrieval_p95_ms"] and o["retrieval_p95_ms"]) else None,
    }

# off==intent nos casos TEXT_ONLY (router não dispara)
exact_equiv = None
if bench["off"] and bench["intent"]:
    off_by = {r["query_id"]: r for r in bench["off"]["results"]}
    diffs = []
    for r in bench["intent"]["results"]:
        if r["type"] in DOC and not r.get("used_t2i", False):
            o = off_by.get(r["query_id"], {})
            if o.get("ranked_top10") != r.get("ranked_top10"):
                diffs.append(r["query_id"])
    exact_equiv = {"text_only_cases_identical_off_vs_intent": not diffs, "diffs": diffs}
report["gates"]["off_intent_text_only_identical"] = exact_equiv

# image+text: ambos os inputs alteram (intent vs off por caso image_text)
if bench["off"] and bench["intent"]:
    off_by = {r["query_id"]: r for r in bench["off"]["results"]}
    changed = 0; total = 0
    per = []
    for r in bench["intent"]["results"]:
        if r["type"] == IT.copy().pop():
            total += 1
            o = off_by.get(r["query_id"], {})
            ch = o.get("ranked_top10") != r.get("ranked_top10")
            changed += int(ch)
            per.append({"query_id": r["query_id"], "changed_by_text": ch})
    report["gates"]["image_text_both_inputs_change"] = {
        "changed": changed, "total": total,
        "rate": round(changed / total, 3) if total else None, "per_case": per}

# temporal compliance (image_text com filtro temporal)
temporal = []
if bench["intent"]:
    for r in bench["intent"]["results"]:
        c = CASE_BY_ID.get(r["query_id"], {})
        interval = (c.get("filters") or {}).get("_temporal_interval")
        if interval and r.get("ranked_top10"):
            temporal.append({"query_id": r["query_id"], "n": r.get("n"), "interval": interval})
report["gates"]["temporal_cases"] = temporal

# 5xx across E12
fivexx = {m: (e12[m]["fivexx"] if e12[m] else None) for m in ("off", "intent", "always")}
report["gates"]["e12_5xx"] = fivexx
report["gates"]["e12_museum_pure"] = {
    m: all(row.get("museum_pure", True)
           for sec in (e12[m]["sections"].values() if e12[m] else [])
           for row in (sec if isinstance(sec, list) else [sec]))
    for m in ("off", "intent", "always") if e12[m]}

(EV / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=1))
