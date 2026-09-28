"""Gera app/data/tour_locations.json a partir das visitas 3DVista publicadas.

Cada visita expõe os panoramas com o seu nome em locale/<língua>.txt
(`panorama_<id>.label = 03 Sala Cantonal 45`) e a ordem da visita na
mainPlayList de script_general.js. Os nomes «humanos» são os lugares da visita
(Pátio, Salão Nobre, Túmulo de L. de Camões…); os nomes de ficheiro de captura
(`E06_P1_A_AlaEste_043`, `02_Igreja_003_09.09.2024_…`) ficam de fora.

No CMAG e no MNAA os labels são nomes de ficheiro e o nome do lugar vem do
`subtitle` do panorama, que nessas visitas diz a sala ou a coleção («Sala
Wucai», «Pintura Europeia», «Ourivesaria»). A fonte é fixa por visita
(`--pelo-subtitle`): nas quatro visitas do estudo mantém-se o label, porque
os nomes dos lugares foram afinados com ele e as descrições curadas em
tour_rooms.json dependem deles.

    python scripts/build_tour_locations.py [--base https://…:5500] [--museus mnt mnsr mnaz mj cmag mnaa] [--pelo-subtitle cmag mnaa]

Correr outra vez sempre que uma visita for republicada.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import ssl
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "app" / "data" / "tour_locations.json"
PANORAMA_ID = r"panorama_[0-9A-F]{8}_[0-9A-F]{4}_[0-9A-F]{4}_[0-9A-F]{4}_[0-9A-F]{12}"


def fetch(url: str) -> str:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE  # o servidor das visitas usa certificado próprio
    with urllib.request.urlopen(url, context=context, timeout=180) as response:
        return response.read().decode("utf-8", "ignore")


def is_human_label(label: str) -> bool:
    """«03 Sala Cantonal 45», «Pátio», «Escadaria (Piso 1)» sim; nomes de ficheiro não."""
    return "_" not in label and not re.search(r"\d{2}\.\d{2}\.\d{4}", label)


def place_name(label: str) -> str:
    name = re.sub(r"^\d+\s+", "", label.strip())  # «03 Sala Cantonal 45» -> «Sala Cantonal 45»
    # número final = n.º do panorama («Sala Cantonal 45»), salvo quando faz
    # parte do nome («Exposição Temporária - Piso 1»)
    if not re.search(r"\b(?:piso|floor|nivel|nível|sala)\s+\d+$", name, re.IGNORECASE):
        name = re.sub(r"\s+\d+$", "", name)  # -> «Sala Cantonal»
    return re.sub(r"\s{2,}", " ", name).strip()


def _balanced_array(text: str, open_index: int) -> str:
    """Conteúdo de um array JSON-like a partir do '[' em open_index (com objetos lá dentro)."""
    depth, in_string, escaped = 0, False, False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index]
    return ""


def playlist_order(general: str) -> list[str]:
    """Panoramas pela ordem da mainPlayList.

    Os itens aparecem como referências («this.PanoramaPlayListItem_…») ou,
    nalgumas visitas, como objetos em linha com "media"; ambos contam.
    """
    item_media: dict[str, str] = {}
    for match in re.finditer(r'"id":"(PanoramaPlayListItem_[0-9A-F_]+)"', general):
        start = general.rfind("{", 0, match.start())
        end = general.find("}", match.end())
        media = re.search(r'"media":"this\.(' + PANORAMA_ID + r')"', general[start:end + 1])
        if media:
            item_media[match.group(1)] = media.group(1)
    for match in re.finditer(r'"id":"mainPlayList"', general):
        start = general.rfind("{", 0, match.start())
        items_at = general.find('"items":[', start)
        if items_at < 0:
            continue
        array = _balanced_array(general, items_at + len('"items":'))
        order: list[str] = []
        pattern = r'this\.(PanoramaPlayListItem_[0-9A-F_]+)"|"media":"this\.(' + PANORAMA_ID + r')"'
        for item in re.finditer(pattern, array):
            panorama = item_media.get(item.group(1)) if item.group(1) else item.group(2)
            if panorama and panorama not in order:
                order.append(panorama)
        if order:
            return order
    return []


def build_museum(base: str, slug: str, source: str = "label") -> list[dict]:
    """Lugares da visita; `source` diz de onde vem o nome de cada panorama («label» ou «subtitle»)."""
    html = fetch(f"{base}/{slug}/")
    locales = re.findall(r"locale/([\w-]+)\.txt", html)
    locale = "pt" if "pt" in locales else (locales[0] if locales else "en")
    text = fetch(f"{base}/{slug}/locale/{locale}.txt")
    labels = dict(re.findall(r"^(" + PANORAMA_ID + r")\.label = (.+?)\s*$", text, re.M))
    subtitles = dict(re.findall(r"^(" + PANORAMA_ID + r")\.subtitle = (.+?)\s*$", text, re.M))
    names = labels if source == "label" else {key: subtitles.get(key, "") for key in labels}
    order = playlist_order(fetch(f"{base}/{slug}/script_general.js"))
    ranked = {key: index for index, key in enumerate(order)}
    keys = sorted(names, key=lambda key: (ranked.get(key, len(ranked)), names[key]))
    places: dict[str, dict] = {}
    for key in keys:
        label = names[key]
        if not is_human_label(label):
            continue
        name = place_name(label)
        if not name:
            continue
        place = places.setdefault(name.casefold(), {"room": name, "panorama_keys": [], "labels": []})
        place["panorama_keys"].append(key)
        place["labels"].append(label)
    return list(places.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="https://patrimonio360.webredirect.org:5500")
    parser.add_argument("--museus", nargs="+", default=["mnt", "mnsr", "mnaz", "mj", "cmag", "mnaa"])
    parser.add_argument("--pelo-subtitle", nargs="*", default=["cmag", "mnaa"], metavar="MUSEU",
                        help="visitas em que o nome do lugar é o subtitle do panorama, não o label")
    args = parser.parse_args()
    payload: dict = {
        "_descricao": (
            "Lugares das visitas 3DVista (nome humano do panorama sem os números, agrupado, pela ordem da "
            "mainPlayList; o nome vem do label ou, nas visitas com labels de ficheiro, do subtitle). "
            "Gerado por scripts/build_tour_locations.py; não editar à mão."
        ),
        "_gerado": {"em": dt.date.today().isoformat(), "base": args.base},
    }
    for slug in args.museus:
        source = "subtitle" if slug in args.pelo_subtitle else "label"
        payload[slug] = build_museum(args.base.rstrip("/"), slug, source)
        print(slug, len(payload[slug]), f"lugares (pelo {source}):", ", ".join(p["room"] for p in payload[slug]))
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("escrito", OUT)


if __name__ == "__main__":
    main()
