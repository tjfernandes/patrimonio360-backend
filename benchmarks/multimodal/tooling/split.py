"""E13.2 — split calibration/holdout estratificado por tipo+museu (seed)."""
import json, random
from pathlib import Path

BK = Path("/home/hedge6000pro/Projects/AMALIA/patrimonio360_assistant/backend")
man = json.loads((BK / "benchmarks/multimodal/manifest.json").read_text(encoding="utf-8"))
rng = random.Random(20260714)
by_stratum = {}
for c in man["cases"]:
    by_stratum.setdefault((c["type"], c["museum_id"]), []).append(c["query_id"])
calib, holdout = [], []
for stratum, ids in sorted(by_stratum.items()):
    ids = sorted(ids); rng.shuffle(ids)
    cut = max(1, round(len(ids) * 0.6)) if len(ids) > 1 else 1
    calib += ids[:cut]; holdout += ids[cut:]
split = {"seed": 20260714, "method": "stratified by (type, museum), 60/40, min 1 in calib",
         "calibration": sorted(calib), "holdout": sorted(holdout),
         "n_calib": len(calib), "n_holdout": len(holdout)}
out = BK / "benchmarks/multimodal/split.json"
out.write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"calib={len(calib)} holdout={len(holdout)} -> {out}")
