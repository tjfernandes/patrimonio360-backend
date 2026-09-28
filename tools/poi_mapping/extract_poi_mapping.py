# -*- coding: utf-8 -*-
"""
Extrai o mapeamento POI -> inventario directamente do export 3DVista,
sem abrir o browser e sem disparar POIs um a um.

    python extract_poi_mapping.py tmp/visitas/mnsr -o poi_tours/panorama-overlays-inventory-mnsr.json
    python extract_poi_mapping.py tmp/visitas/mnsr --compare poi_tours/panorama-overlays-inventory-mnsr.json

Le script_general.js (fallback script_mobile.js), encontra os hotspots cujo click
emite postMessage({type:"openedInfoWindow", inventoryIds:[...]}) e resolve a
cadeia  area -> overlay -> panorama/model,  produzindo o mesmo formato que o
overlay-inventory-mapper.js gerava a partir do browser.

O nome da zona («location») de cada POI vem, por ordem, do subtitle do panorama
onde esta (locale/<lingua>.txt do export, com pt antes de en), e so na falta
dele do `location` escrito no bloco JS do hotspot. No MNAA os blocos dizem
«Sala 37», «Sala 64», … (25 zonas), e os subtitles dizem a coleccao («Pintura
Europeia», «Ourivesaria», …), que e o que o visitante procura. O valor do bloco
fica em `locationOriginal` quando e diferente. --location-from js repoe o
comportamento antigo; --location-from label usa o label do panorama.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

CLICK_MARKER = "openedInfoWindow"

ID_RE = re.compile(r'"id"\s*:\s*"([^"]+)"')
INV_RE = re.compile(r"inventoryIds\s*:\s*\[(.*?)\]", re.S)
STR_RE = re.compile(r'\\"(.*?)\\"|\u201c(.*?)\u201d', re.S)
TITLE_RE = re.compile(r'title\s*:\s*(?:\\"(.*?)\\"|\u201c(.*?)\u201d)', re.S)
LOC_RE = re.compile(r'location\s*:\s*(?:\\"(.*?)\\"|\u201c(.*?)\u201d)', re.S)

# \' \" \/ (e runs de barras) sao escapes do 3DVista, nao fazem parte do texto
UNESC_RE = re.compile(r"\\+(['\"/])")


def unescape(value: str) -> str:
    return re.sub(r"\s+", " ", UNESC_RE.sub(r"\1", value).replace("\\n", " ")).strip()


def first_group(match) -> str:
    return next((g for g in match.groups() if g is not None), "")

# varios numeros de inventario colados numa so string ("47, 48, 51 Mob MNSR")
MULTI_RE = re.compile(r"\d\s*(?:,|;|\be\b|\ba\b)\s*\d")


def object_spans(text: str) -> list[tuple[int, int]]:
    """Spans (inicio, fim) de todos os objectos, ignorando chavetas dentro de strings."""
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    in_str = False
    quote = ""
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            spans.append((stack.pop(), i + 1))
    spans.sort(key=lambda s: s[0])
    return spans


def innermost(spans, starts, pos):
    """Objecto mais interior que contem pos: o de maior inicio que ainda o cobre."""
    i = bisect.bisect_right(starts, pos) - 1
    while i >= 0:
        start, end = spans[i]
        if end > pos:
            return start, end
        i -= 1
    return None


def own_id(obj_text: str) -> str | None:
    m = ID_RE.search(obj_text)
    return m.group(1) if m else None


LOCALE_LINE_RE = re.compile(r"^((?:panorama|model)_[0-9A-F_]+)\.(label|subtitle) = (.*?)\s*$", re.M)


def load_locale(tour_dir: Path) -> tuple[dict[str, dict[str, str]], str | None]:
    """{chave do panorama: {"label": …, "subtitle": …}} do locale/<lingua>.txt; pt antes de en."""
    folder = tour_dir / "locale"
    if not folder.is_dir():
        return {}, None
    candidates = sorted(folder.glob("*.txt"), key=lambda p: {"pt": 0, "en": 1}.get(p.stem, 2))
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="replace")
        names: dict[str, dict[str, str]] = {}
        for key, field, value in LOCALE_LINE_RE.findall(text):
            if value.strip():
                names.setdefault(key, {})[field] = value.strip()
        if names:
            return names, path.name
    return {}, None


def is_human_name(value: str) -> bool:
    """«Pintura Europeia» sim; nomes de ficheiro de captura («E01_Terreo_32») nao."""
    return bool(value) and "_" not in value and not re.search(r"\d{2}\.\d{2}\.\d{4}", value)


def apply_locations(results: list, names: dict[str, dict[str, str]], source: str) -> int:
    """location <- subtitle (ou label) do panorama; o valor do bloco JS fica em locationOriginal."""
    changed = 0
    for entry in results:
        wanted = (names.get(entry["panoramaKey"], {}).get(source) or "").strip()
        if not is_human_name(wanted) or wanted == entry["location"]:
            continue
        if entry["location"]:
            entry["locationOriginal"] = entry["location"]
        entry["location"] = wanted
        changed += 1
    return changed


def extract(tour_dir: Path):
    src = None
    for name in ("script_general.js", "script_mobile.js"):
        if (tour_dir / name).exists():
            src = tour_dir / name
            break
    if src is None:
        sys.exit("Nao encontrei script_general.js nem script_mobile.js em " + str(tour_dir))

    text = src.read_text(encoding="utf-8", errors="replace")
    spans = object_spans(text)
    starts = [s[0] for s in spans]

    hotspots = []
    seen_area: set[str] = set()
    for m in re.finditer(CLICK_MARKER, text):
        span = innermost(spans, starts, m.start())
        if not span:
            continue
        obj = text[span[0]:span[1]]
        if "inventoryIds" not in obj:
            continue
        area_id = own_id(obj)
        if not area_id or area_id in seen_area:
            continue
        seen_area.add(area_id)

        inv_m = INV_RE.search(obj)
        ids = [unescape(first_group(m)) for m in STR_RE.finditer(inv_m.group(1))] if inv_m else []
        ids = [i for i in ids if i]
        if not ids:
            continue
        t_m = TITLE_RE.search(obj)
        l_m = LOC_RE.search(obj)
        hotspots.append({
            "areaId": area_id,
            "inventoryIds": ids,
            "title": unescape(first_group(t_m)) or None if t_m else None,
            "location": unescape(first_group(l_m)) or None if l_m else None,
        })

    def parent_of(child_id: str, prefixes: tuple[str, ...]) -> str | None:
        for m in re.finditer(r'"this\.' + re.escape(child_id) + r'"', text):
            span = innermost(spans, starts, m.start())
            if not span:
                continue
            pid = own_id(text[span[0]:span[1]])
            if pid and pid.startswith(prefixes):
                return pid
        return None

    results = []
    unresolved = []
    for h in hotspots:
        area_id = h["areaId"]
        if area_id.startswith(("overlay_", "sprite_")):
            overlay_id = area_id
        else:
            overlay_id = parent_of(area_id, ("overlay_", "sprite_"))
        if not overlay_id:
            unresolved.append(dict(h, motivo="sem overlay pai"))
            continue
        pano = parent_of(overlay_id, ("panorama_", "model_"))
        if not pano:
            unresolved.append(dict(h, overlayId=overlay_id, motivo="sem panorama"))
            continue
        results.append({
            "overlayId": overlay_id,
            "panoramaKey": pano,
            "inventoryIds": h["inventoryIds"],
            "location": h["location"],
            "title": h["title"],
        })

    return results, unresolved, src.name


def compare(new: list, old_path: Path) -> None:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old_by_id = {e["overlayId"]: e for e in old}
    new_by_id = {e["overlayId"]: e for e in new}
    novos = sorted(set(new_by_id) - set(old_by_id))
    mortos = sorted(set(old_by_id) - set(new_by_id))
    diverg = [
        k for k in set(old_by_id) & set(new_by_id)
        if old_by_id[k]["inventoryIds"] != new_by_id[k]["inventoryIds"]
        or old_by_id[k]["panoramaKey"] != new_by_id[k]["panoramaKey"]
    ]

    print()
    print("comparacao com " + old_path.name + ": " + str(len(old_by_id)) + " -> " + str(len(new_by_id)))
    print("  POIs novos            : " + str(len(novos)))
    print("  POIs que desapareceram: " + str(len(mortos)))
    print("  POIs alterados        : " + str(len(diverg)))
    for k in novos:
        e = new_by_id[k]
        print("    + {:30} {} / {}".format(str(e["inventoryIds"]), e["location"], e["title"]))
    for k in mortos:
        e = old_by_id[k]
        print("    - {:30} {} / {}  (so no ficheiro antigo)".format(str(e["inventoryIds"]), e["location"], e["title"]))
    for k in diverg:
        print("    ~ {} -> {}   {}".format(old_by_id[k]["inventoryIds"], new_by_id[k]["inventoryIds"], new_by_id[k]["title"]))


def keep_better(results: list, old_path: Path) -> list:
    """Onde o novo export colapsou varios inventarios numa string e o mapeamento
    antigo os tinha separados, mantem a versao antiga (melhor)."""
    old_by_id = {e["overlayId"]: e for e in json.loads(old_path.read_text(encoding="utf-8"))}
    mantidos = []
    for entry in results:
        antigo = old_by_id.get(entry["overlayId"])
        if not antigo:
            continue
        novo_ids = entry["inventoryIds"]
        antigo_ids = antigo.get("inventoryIds") or []
        if any(MULTI_RE.search(i) for i in novo_ids) and len(antigo_ids) > len(novo_ids):
            entry["inventoryIds"] = antigo_ids
            mantidos.append((entry["title"], novo_ids, antigo_ids))
    if mantidos:
        print()
        print("MANTIDOS do mapeamento antigo (" + str(len(mantidos)) + "): o novo export colapsou-os numa string")
        for titulo, novo_ids, antigo_ids in mantidos:
            print("    {} : {} -> {} numeros".format(titulo, novo_ids, len(antigo_ids)))
    return results


def report_suspicious(results: list) -> None:
    suspeitos = [e for e in results if any(MULTI_RE.search(i) for i in e["inventoryIds"])]
    if not suspeitos:
        return
    print()
    print("AVISO: " + str(len(suspeitos)) + " POIs com varios numeros de inventario numa so string.")
    print("O backend so separa por ; , | e quebra de linha, por isso partes destes nao resolvem.")
    for e in suspeitos:
        print("    {:62} {}".format(str(e["inventoryIds"]), e["title"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tour_dir", help="pasta do tour exportado (ex.: tmp/visitas/mnsr)")
    ap.add_argument("-o", "--out", help="ficheiro de saida (default: <tour_dir>/panorama-overlays-inventory.json)")
    ap.add_argument("--compare", help="mapeamento actual, para mostrar o que mudou")
    ap.add_argument("--keep-better", help="mapeamento actual: preserva listas de inventario que o novo export colapsou")
    ap.add_argument("--location-from", choices=["subtitle", "label", "js"], default="subtitle",
                    help="nome da zona: subtitle (default) ou label do panorama, em locale/*.txt; js = o location do bloco JS")
    args = ap.parse_args()

    tour_dir = Path(args.tour_dir)
    results, unresolved, src_name = extract(tour_dir)

    if args.keep_better:
        results = keep_better(results, Path(args.keep_better))

    if args.location_from != "js":
        names, locale_name = load_locale(tour_dir)
        if names:
            changed = apply_locations(results, names, args.location_from)
            print("zonas             : {} do {} de {} ({} POIs; os restantes ficam com o location do bloco JS)".format(
                len({e["location"] for e in results if e.get("location")}), args.location_from, locale_name, changed))
        else:
            print("zonas             : sem locale/*.txt em " + str(tour_dir) + "; fica o location do bloco JS")

    out = Path(args.out) if args.out else tour_dir / "panorama-overlays-inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("fonte             : " + str(tour_dir / src_name))
    print("POIs              : " + str(len(results)))
    print("panoramas         : " + str(len({r["panoramaKey"] for r in results})))
    print("inventarios unicos: " + str(len({i for r in results for i in r["inventoryIds"]})))
    print("escrito em        : " + str(out))

    if unresolved:
        print()
        print("NAO RESOLVIDOS: " + str(len(unresolved)) + " (POIs sem overlay ou panorama pai)")
        for u in unresolved:
            print("    {}: {} {} {}".format(u["motivo"], u["areaId"], u["inventoryIds"], u["title"]))

    report_suspicious(results)

    if args.compare:
        compare(results, Path(args.compare))


if __name__ == "__main__":
    main()
