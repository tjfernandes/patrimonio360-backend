"""Tira do panorama-objects-<museu>.json os recortes que são um bocado de outra
peça já identificada no mesmo sítio.

    python tools/apontar/filtrar_fragmentos.py mnaa                # só mostra o que sairia
    python tools/apontar/filtrar_fragmentos.py mnaa --apply        # reescreve o ficheiro
    python tools/apontar/filtrar_fragmentos.py mnaa --apply --remove 63esc --keep panorama_…:12

O SAM recorta às vezes só uma parte de uma peça (a base de uma escultura, o
painel de baixo de um retábulo) e o kNN dá a esse bocado outra peça parecida,
que o VLM par a par confirma (2026-11-20: a base da «Santíssima Trindade»,
1206 Esc, ficou como «Caridade», 63 Esc, com cosseno 0,655 e «2 sim»). A peça
inteira está identificada no mesmo panorama com cosseno maior: o bocado sai.

Regra: o recorte A sai se houver no mesmo panorama um recorte B de outra peça
em que
  - pelo menos --inside (0,75) da caixa de A fica dentro da caixa de B;
  - pelo menos --polygon-inside (0,75) da área do contorno de A fica dentro
    do contorno de B (uma cadeira à frente de uma tapeçaria está na caixa
    dela mas não no contorno: fica);
  - a caixa de B é pelo menos --ratio (1,5) vezes maior;
  - o cosseno de A é menor do que o de B, e A não tem hotspot.

Os limiares são 0,75 e não 0,8 porque o bocado costuma levar um pouco do
plinto ou da moldura (o da «Caridade» tinha 0,78 da caixa dentro da peça).

--remove tira também todas as localizações das peças indicadas (n.º de
inventário, sem espaços nem maiúsculas), para as que se viu à mão que estão
erradas. --keep poupa um recorte (o seu object_id) que a regra apanha mas é
uma peça inteira ao lado de outra (uma custódia à frente de um sacrário, um
cofre apanhado pelo contorno de um grupo de custódias). O que sai e o que foi
poupado ficam registados em `fragment_filter` no ficheiro.

MNAA, 2026-11-20: vistos os candidatos em folhas com os dois contornos; dos
129 saem 127 e 2 ficam poupados; saem também as outras 2 localizações da
«Caridade» (nenhuma das 3 era a peça). De 3313 objetos ficam 3184.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def normalize_inventory(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _box(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox") or {}
    try:
        return float(bbox["yaw_min"]), float(bbox["yaw_max"]), float(bbox["pitch_min"]), float(bbox["pitch_max"])
    except (KeyError, TypeError, ValueError):
        return None


def _area(box: tuple[float, float, float, float]) -> float:
    return max(box[1] - box[0], 0.0) * max(box[3] - box[2], 0.0)


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[1], b[1]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[2], b[2])
    return max(width, 0.0) * max(height, 0.0)


def _point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def polygon_inside(inner: dict[str, Any], outer: dict[str, Any], *, grid: int = 48) -> float:
    """Fração da área do contorno de `inner` que fica dentro do contorno de
    `outer`, medida numa grelha de `grid`×`grid` pontos sobre a caixa de
    `inner` (1.0 se algum deles não tem contorno: fica a decidir a caixa).
    Pela área e não pelos vértices: o bocado partilha quase sempre a borda
    com a peça, e os vértices da borda caem de um lado ou do outro."""
    a, b = inner.get("polygon") or [], outer.get("polygon") or []
    box = _box(inner)
    if len(a) < 3 or len(b) < 3 or box is None:
        return 1.0
    yaw_min, yaw_max, pitch_min, pitch_max = box
    own = shared = 0
    for row in range(grid):
        y = pitch_min + (row + 0.5) * (pitch_max - pitch_min) / grid
        for col in range(grid):
            x = yaw_min + (col + 0.5) * (yaw_max - yaw_min) / grid
            if _point_in_polygon(x, y, a):
                own += 1
                shared += _point_in_polygon(x, y, b)
    return shared / own if own else 1.0


def find_fragments(
    objects: list[dict[str, Any]],
    *,
    inside: float = 0.75,
    polygon_min: float = 0.75,
    ratio: float = 1.5,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Os pares (bocado, peça que o contém) entre os objetos de um panorama."""
    found: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for a in objects:
        box_a = _box(a)
        if box_a is None or _area(box_a) <= 0 or str(a.get("accepted_by") or "") == "hotspot":
            continue
        if str(a.get("confirmed_by") or "") == "manual":
            continue  # alguém viu este recorte e confirmou a peça
        best: dict[str, Any] | None = None
        for b in objects:
            if b is a or normalize_inventory(b.get("inventory_number")) == normalize_inventory(a.get("inventory_number")):
                continue
            box_b = _box(b)
            if box_b is None or _area(box_b) < ratio * _area(box_a):
                continue
            if float(a.get("score") or 0.0) >= float(b.get("score") or 0.0):
                continue
            if _intersection(box_a, box_b) / _area(box_a) < inside:
                continue
            if polygon_inside(a, b) < polygon_min:
                continue
            if best is None or float(b.get("score") or 0.0) > float(best.get("score") or 0.0):
                best = b
        if best is not None:
            found.append((a, best))
    return found


def filter_payload(
    payload: dict[str, Any],
    *,
    inside: float = 0.75,
    polygon_min: float = 0.75,
    ratio: float = 1.5,
    remove: set[str] | None = None,
    keep: set[str] | None = None,
) -> dict[str, Any]:
    """Tira os bocados (e as peças de `remove`) de `payload`, no sítio, e
    devolve o relatório do que saiu. Os recortes de `keep` (object_id) ficam
    mesmo que a regra os apanhe."""
    remove = {normalize_inventory(r) for r in (remove or set())} - {""}
    keep = {str(k).strip() for k in (keep or set())} - {""}
    spared: list[str] = []
    fragments: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    kept_total = 0
    for panorama_key, entry in (payload.get("panoramas") or {}).items():
        if not isinstance(entry, dict):
            continue
        objects = [o for o in entry.get("objects") or [] if isinstance(o, dict)]
        gone: dict[int, None] = {}
        for a, b in find_fragments(objects, inside=inside, polygon_min=polygon_min, ratio=ratio):
            if str(a.get("object_id") or "") in keep:
                spared.append(str(a.get("object_id")))
                continue
            gone[id(a)] = None
            fragments.append({
                "object_id": a.get("object_id"), "panorama_key": panorama_key,
                "inventory_number": a.get("inventory_number"), "title": a.get("title"), "score": a.get("score"),
                "inside_object_id": b.get("object_id"), "inside_inventory_number": b.get("inventory_number"),
                "inside_score": b.get("score"),
            })
        for obj in objects:
            if id(obj) not in gone and normalize_inventory(obj.get("inventory_number")) in remove:
                gone[id(obj)] = None
                manual.append({
                    "object_id": obj.get("object_id"), "panorama_key": panorama_key,
                    "inventory_number": obj.get("inventory_number"), "title": obj.get("title"), "score": obj.get("score"),
                })
        if gone:
            entry["objects"] = [o for o in entry.get("objects") or [] if id(o) not in gone]
        kept_total += len(entry.get("objects") or [])
    return {
        "when": datetime.now().isoformat(timespec="seconds"),
        "rule": {"inside": inside, "polygon_inside": polygon_min, "ratio": ratio, "lower_score": True},
        "mantido": kept_total,
        "apagado_fragmento": len(fragments),
        "apagado_a_mao": len(manual),
        "poupado": sorted(spared),
        "fragmentos": fragments,
        "a_mao": manual,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("museum")
    parser.add_argument("--dir", default=str(ROOT / "poi_tours"))
    parser.add_argument("--inside", type=float, default=0.75)
    parser.add_argument("--polygon-inside", type=float, default=0.75)
    parser.add_argument("--ratio", type=float, default=1.5)
    parser.add_argument("--remove", action="append", default=[], help="n.º de inventário a tirar de todos os panoramas")
    parser.add_argument("--keep", action="append", default=[], help="object_id a poupar mesmo que a regra o apanhe")
    parser.add_argument("--apply", action="store_true", help="reescreve o ficheiro (sem isto só mostra)")
    parser.add_argument("--report", help="guarda o relatório completo neste JSON")
    args = parser.parse_args()

    path = Path(args.dir) / f"panorama-objects-{args.museum}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = filter_payload(
        payload, inside=args.inside, polygon_min=args.polygon_inside, ratio=args.ratio, remove=set(args.remove),
        keep=set(args.keep),
    )
    print(f"{path.name}: ficam {report['mantido']}, saem {report['apagado_fragmento']} bocados e {report['apagado_a_mao']} à mão; poupados {len(report['poupado'])}")
    for item in report["fragmentos"][:15]:
        print(f"  {item['inventory_number']} «{item['title']}» {item['score']} dentro de {item['inside_inventory_number']} {item['inside_score']}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.apply:
        payload["fragment_filter"] = report
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print("ficheiro reescrito")
    return 0


if __name__ == "__main__":
    sys.exit(main())
