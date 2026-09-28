# -*- coding: utf-8 -*-
"""
Audita um projecto 3DVista (.vtp) ANTES de exportar: que hotspots abrem uma
ficha de peca, quais ja tem o bloco JavaScript do assistente, quais o tem
partido, e que fichas em Conteudos/ ainda nao tem hotspot nenhum.

    python audit_vtp.py "caminho/Projecto.vtp" [--conteudos caminho/Conteudos] [--json saida.json]

O .vtp e um ZIP; o script.js la dentro e o modelo do projecto. Nao escreve nada
no projecto.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import re
import sys
import zipfile
from pathlib import Path

OPEN_Q = '"\'“‘'
CLOSE_OF = {'"': '"', "'": "'", "“": "”", "‘": "’"}
SMART = re.compile("[“”‘’]")


# ----------------------------------------------------------------- estrutura
def object_spans(text):
    spans, stack, in_str, quote, esc = [], [], False, "", False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            spans.append((stack.pop(), i + 1))
    spans.sort()
    return spans, [s[0] for s in spans]


def innermost(spans, starts, pos):
    i = bisect.bisect_right(starts, pos) - 1
    while i >= 0:
        if spans[i][1] > pos:
            return spans[i]
        i -= 1
    return None


# ----------------------------------------------------------- codigo do autor
def read_quoted(text, i):
    close = CLOSE_OF[text[i]]
    j, buf = i + 1, []
    while j < len(text) and text[j] != close:
        if text[j] == "\\" and j + 1 < len(text):
            buf.append(text[j + 1]); j += 2; continue
        buf.append(text[j]); j += 1
    return "".join(buf).strip(), j + 1


def read_field(code, name):
    m = re.search(name + r"\s*:\s*([" + OPEN_Q + "])", code)
    return read_quoted(code, m.start(1))[0] if m else None


def read_list(code):
    m = re.search(r"inventoryIds\s*:\s*\[(.*?)\]", code, re.S)
    if not m:
        return []
    seg, out, i = m.group(1), [], 0
    while i < len(seg):
        if seg[i] in OPEN_Q:
            v, i = read_quoted(seg, i)
            if v:
                out.append(v)
        else:
            i += 1
    return out


def parse_code(code):
    if "openedInfoWindow" not in code:
        return None
    title = read_field(code, "title")
    # aspas curvas dentro do titulo («trompe l’oeil», “A Fonte da Vida”) sao texto legitimo;
    # so contam as que estao a delimitar valores
    return {
        "inventoryIds": read_list(code),
        "title": title,
        "location": read_field(code, "location"),
        "smartQuotes": bool(SMART.search(code.replace(title or "", ""))),
        "htmlInTitle": bool(re.search(r"<[^>]+>", title or "")),
    }


# ------------------------------------------------------------------ projecto
def load_project(vtp: Path):
    with zipfile.ZipFile(vtp) as z:
        return z.read("script.js").decode("utf-8", "replace")


def audit(text, conteudos: Path | None):
    spans, starts = object_spans(text)

    def obj_at(pos):
        s = innermost(spans, starts, pos)
        return text[s[0]:s[1]] if s else ""

    def obj_by_id(oid):
        m = re.search(r'"id":"' + re.escape(oid) + '"', text)
        return obj_at(m.start()) if m else ""

    def own_id(o):
        m = re.search(r'"id":"([^"]+)"', o)
        return m.group(1) if m else None

    def parent_of(child, prefixes):
        for m in re.finditer(r'"this\.' + re.escape(child) + '"', text):
            pid = own_id(obj_at(m.start()))
            if pid and pid.startswith(prefixes):
                return pid
        return None

    locales = {k: json.loads('"' + v + '"') for k, v in
               re.findall(r'"([A-Za-z]+_[0-9A-F_]+\.\w+)":\{"value":"((?:[^"\\]|\\.)*)","class":"LocaleValue"\}', text)}

    hotspots = []
    for m in re.finditer(r'"id":"((?:Flat)?HotspotPanoramaOverlayArea_[0-9A-F_]+)"', text):
        area_id, area = m.group(1), obj_at(m.start())
        popups = re.findall(r'"this\.(PopupWebFrameBehaviour_[0-9A-F_]+)"', area)
        codes = [json.loads('"' + c + '"') for c in re.findall(r'"code":"((?:[^"\\]|\\.)*)"', area)]
        js = next((parse_code(c) for c in codes if parse_code(c)), None)
        if js:
            # o 3DVista exporta «sentences» (o codigo compilado), nao «code»: se o autor editou o
            # code na interface sem o 3DVista recompilar, o export sai com a versao antiga
            js["code"] = next(c for c in codes if parse_code(c))
            sents = [json.loads('"' + s + '"') for s in re.findall(r'"sentences":\["((?:[^"\\]|\\.)*)"\]', area)]
            js["sentences"] = next((s for s in sents if "openedInfoWindow" in s), None)
        if not popups and not js:
            continue
        overlay = parent_of(area_id, ("overlay_",))
        pano = parent_of(overlay, ("panorama_", "model_")) if overlay else None
        ov_obj = obj_by_id(overlay) if overlay else ""
        ov_label = (re.search(r'"label":"([^"]+)"', ov_obj) or [None, None])[1]
        page = locales.get(popups[0] + ".url") if popups else None
        hotspots.append({
            "areaId": area_id, "overlayId": overlay, "panoramaKey": pano,
            "panorama": locales.get((pano or "") + ".label"), "hotspotLabel": ov_label,
            "page": page, "pageRoom": page.split("/")[-2] if page and "/" in page else None,
            "hasPopup": bool(popups), "js": js,
        })

    # fichas em Conteudos
    pages = {}
    if conteudos:
        for f in glob.glob(str(conteudos / "HTML" / "iFrames" / "data" / "*" / "*.json")):
            d = json.load(open(f, encoding="utf-8"))
            key = os.path.relpath(f, conteudos / "HTML").replace(os.sep, "/").replace("/data/", "/generated-html/")[:-5] + ".html"
            pages[key] = {"room": d.get("room"), "inventory_number": d.get("inventory_number"), "header": d.get("header")}
    return hotspots, pages, locales


def report(hotspots, pages):
    ok, broken, missing_js, no_popup = [], [], [], []
    for h in hotspots:
        if not h["js"]:
            missing_js.append(h); continue
        if not h["hasPopup"]:
            no_popup.append(h)
        j = h["js"]
        page = pages.get(h["page"] or "", {})
        problems = []
        if j["smartQuotes"]:
            problems.append("aspas curvas")
        if j["htmlInTitle"]:
            problems.append("HTML no titulo")
        if not j["inventoryIds"]:
            problems.append("sem inventoryIds")
        norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
        ficha_inv = page.get("inventory_number") or ""
        if ficha_inv and not any(norm(i) and norm(i) in norm(ficha_inv) for i in j["inventoryIds"]):
            problems.append(f'inventario != ficha ({ficha_inv})')
        if h["pageRoom"] and j["location"] and j["location"] != h["pageRoom"]:
            problems.append(f'location != sala da ficha ({h["pageRoom"]})')
        if any(re.search(r"SN$", i) for i in j["inventoryIds"]):
            problems.append("SN sem numero unico")
        (broken if problems else ok).append((h, problems))

    used = {h["page"] for h in hotspots if h["page"]}
    orphan_pages = sorted(p for p in pages if p not in used)

    print(f"hotspots com ficha ou bloco JS : {len(hotspots)}")
    print(f"  bloco JS correcto            : {len(ok)}")
    print(f"  bloco JS com problemas       : {len(broken)}")
    print(f"  abrem ficha SEM bloco JS     : {len(missing_js)}")
    print(f"  bloco JS mas nao abrem ficha : {len(no_popup)}")
    print(f"fichas em Conteudos            : {len(pages)}   sem hotspot: {len(orphan_pages)}")

    def line(h, extra=""):
        j = h["js"] or {}
        return (f'  {h["panorama"] or "?":18} {h["hotspotLabel"] or "":11} '
                f'{(j.get("title") or (pages.get(h["page"] or "", {}).get("header")) or "")[:48]:48} '
                f'{" | ".join(j.get("inventoryIds") or []):26} {extra}')

    if missing_js:
        print("\n== ABREM FICHA SEM BLOCO JS (acrescentar o bloco) ==")
        for h in missing_js:
            print(line(h, "-> " + (h["page"] or "")))
    if broken:
        print("\n== BLOCO JS COM PROBLEMAS ==")
        for h, p in broken:
            print(line(h, "; ".join(p)))
    if no_popup:
        print("\n== BLOCO JS MAS SEM ACCAO DE ABRIR FICHA ==")
        for h in no_popup:
            print(line(h))
    if orphan_pages:
        print("\n== FICHAS SEM HOTSPOT (conteudo preparado, hotspot em falta) ==")
        for p in orphan_pages:
            d = pages[p]
            print(f'  {d["room"]:24} {d["inventory_number"]:14} {d["header"]}')
    return {"ok": ok, "broken": broken, "missing_js": missing_js, "no_popup": no_popup, "orphan_pages": orphan_pages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtp")
    ap.add_argument("--conteudos", help="pasta Conteudos do projecto (default: ao lado do .vtp)")
    ap.add_argument("--json", help="guardar o resultado completo em JSON")
    a = ap.parse_args()
    vtp = Path(a.vtp)
    conteudos = Path(a.conteudos) if a.conteudos else vtp.parent / "Conteudos"
    text = load_project(vtp)
    hotspots, pages, _ = audit(text, conteudos if conteudos.exists() else None)
    r = report(hotspots, pages)
    if a.json:
        Path(a.json).write_text(json.dumps({"hotspots": hotspots, "pages": pages, "orphan_pages": r["orphan_pages"]},
                                           ensure_ascii=False, indent=1), encoding="utf-8")
        print("\njson:", a.json)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
