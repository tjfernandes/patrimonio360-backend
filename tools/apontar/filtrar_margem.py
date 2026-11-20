"""Põe à parte as localizações do panorama-objects-<museu>.json em que a peça
ganhou por pouco à segunda candidata.

    python tools/apontar/filtrar_margem.py mnaa            # só mostra o que sairia
    python tools/apontar/filtrar_margem.py mnaa --apply    # reescreve os dois ficheiros

Medido a 2026-11-20 em 150 localizações do MNAA etiquetadas à mão (50 por
faixa de cosseno): a peça estava errada em 22/50 abaixo de 0,70, 21/50 entre
0,70 e 0,80 e 4/50 acima de 0,80, cerca de 27 % do ficheiro, todas elas
confirmadas pelo VLM par a par. O que separa é a vantagem do cosseno sobre a
segunda candidata (`score - runner_up_score`): com 0,08 ou mais, 0 erradas em
40; abaixo, 47 em 110.

Regra: uma localização aceite pelo VLM fica se a vantagem for ≥ --min-margin
(0,08) ou o cosseno ≥ --min-score (0,85). Na amostra ficam 55, 2 erradas; os
limiares saíram dessa mesma amostra, por isso o erro real será um pouco maior.
As aceites por hotspot ficam sempre; sem segunda candidata a vantagem é o
próprio cosseno.

As que saem não se apagam: vão para panorama-audit-queue-<museu>.json, com a
mesma forma (panoramas → objects), à espera de uma verificação melhor que as
recupere. O backend só lê panorama-objects-*.json. Correr outra vez junta à
fila o que sair de novo. O resumo fica em `margin_filter` nos dois ficheiros.

    python tools/apontar/filtrar_margem.py mnaa --apply --restore panorama_…:7

--restore repõe uma localização da fila que alguém viu e confirmou à mão (o
recorte ao lado da fotografia da ficha): volta ao ficheiro com
`confirmed_by: "manual"` e a data, e nem este filtro nem o dos bocados a
voltam a tirar. Fica com `accepted_by` como estava, que é o que o backend lê.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
QUEUE_PREFIX = "panorama-audit-queue-"


def margin(obj: dict[str, Any]) -> float:
    """Vantagem do cosseno da peça sobre a segunda candidata."""
    try:
        return float(obj.get("score") or 0.0) - float(obj.get("runner_up_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def keeps(obj: dict[str, Any], *, min_margin: float = 0.08, min_score: float = 0.85) -> bool:
    if str(obj.get("accepted_by") or "") != "vlm" or str(obj.get("confirmed_by") or "") == "manual":
        return True
    try:
        score = float(obj.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    # arredondado: os scores têm 4 casas e 0,7300 - 0,6500 não dá 0,08 em vírgula flutuante
    return round(margin(obj), 6) >= min_margin or score >= min_score


def split_payload(
    payload: dict[str, Any], *, min_margin: float = 0.08, min_score: float = 0.85,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Tira de `payload`, no sítio, as localizações que não passam e devolve-as
    por panorama, com o resumo."""
    queued: dict[str, list[dict[str, Any]]] = {}
    kept = 0
    for panorama_key, entry in (payload.get("panoramas") or {}).items():
        if not isinstance(entry, dict):
            continue
        stay, go = [], []
        for obj in entry.get("objects") or []:
            ok = not isinstance(obj, dict) or keeps(obj, min_margin=min_margin, min_score=min_score)
            (stay if ok else go).append(obj)
        if go:
            entry["objects"] = stay
            queued[str(panorama_key)] = go
        kept += len(stay)
    report = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "rule": {"min_margin": min_margin, "min_score": min_score, "only_accepted_by": "vlm"},
        "mantido": kept,
        "a_auditar": sum(len(v) for v in queued.values()),
    }
    return queued, report


def merge_queue(queue: dict[str, Any], museum: str, queued: dict[str, list[dict[str, Any]]], report: dict[str, Any]) -> dict[str, Any]:
    """Junta `queued` à fila já guardada (sem repetir object_id)."""
    queue = queue if isinstance(queue, dict) and isinstance(queue.get("panoramas"), dict) else {"museum_slug": museum, "panoramas": {}}
    for panorama_key, objects in queued.items():
        entry = queue["panoramas"].setdefault(panorama_key, {"objects": []})
        seen = {str(o.get("object_id") or "") for o in entry["objects"] if isinstance(o, dict)}
        entry["objects"].extend(o for o in objects if str(o.get("object_id") or "") not in seen)
    queue["margin_filter"] = dict(report, na_fila=sum(len(e.get("objects") or []) for e in queue["panoramas"].values()))
    return queue


def restore(payload: dict[str, Any], queue: dict[str, Any], object_ids: set[str], *, when: str | None = None) -> list[dict[str, Any]]:
    """Passa da fila para `payload`, no sítio, as localizações `object_ids`,
    marcadas como confirmadas à mão. Devolve as que repôs."""
    when = when or datetime.now().isoformat(timespec="seconds")
    restored: list[dict[str, Any]] = []
    for panorama_key, entry in (queue.get("panoramas") or {}).items():
        objects = entry.get("objects") or []
        back = [o for o in objects if isinstance(o, dict) and str(o.get("object_id") or "") in object_ids]
        if not back:
            continue
        entry["objects"] = [o for o in objects if o not in back]
        target = (payload.setdefault("panoramas", {})).setdefault(panorama_key, {"objects": []})
        present = {str(o.get("object_id") or "") for o in target.get("objects") or [] if isinstance(o, dict)}
        for obj in back:
            obj = dict(obj, confirmed_by="manual", confirmed_at=when)
            if obj["object_id"] not in present:
                target.setdefault("objects", []).append(obj)
            restored.append(obj)
    if restored:
        log = payload.setdefault("manual_confirmations", [])
        log.append({"when": when, "object_ids": [o["object_id"] for o in restored]})
        if isinstance(queue.get("margin_filter"), dict):
            queue["margin_filter"]["na_fila"] = sum(len(e.get("objects") or []) for e in queue["panoramas"].values())
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("museum")
    parser.add_argument("--dir", default=str(ROOT / "poi_tours"))
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--min-score", type=float, default=0.85)
    parser.add_argument("--restore", action="append", default=[], help="object_id da fila a repor como confirmado à mão")
    parser.add_argument("--apply", action="store_true", help="reescreve os ficheiros (sem isto só mostra)")
    args = parser.parse_args()

    path = Path(args.dir) / f"panorama-objects-{args.museum}.json"
    queue_path = Path(args.dir) / f"{QUEUE_PREFIX}{args.museum}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if args.restore:
        queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.exists() else {}
        restored = restore(payload, queue, set(args.restore))
        missing = set(args.restore) - {o["object_id"] for o in restored}
        for obj in restored:
            print(f"reposto: {obj['object_id']} · {obj.get('inventory_number')} «{obj.get('title')}»")
        if missing:
            print(f"não estão na fila: {sorted(missing)}")
            return 1
        if args.apply:
            # o ficheiro primeiro: se a segunda escrita falhar, a localização fica nos dois e não em nenhum
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=1), encoding="utf-8")
            print("ficheiros reescritos")
        return 0
    queued, report = split_payload(payload, min_margin=args.min_margin, min_score=args.min_score)
    print(f"{path.name}: ficam {report['mantido']}, vão para a fila {report['a_auditar']}")
    if args.apply and queued:
        old = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.exists() else {}
        queue = merge_queue(old, args.museum, queued, report)
        # a fila primeiro: se a segunda escrita falhar, nada se perdeu
        queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=1), encoding="utf-8")
        payload["margin_filter"] = report
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"ficheiros reescritos; fila em {queue_path.name} ({queue['margin_filter']['na_fila']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
