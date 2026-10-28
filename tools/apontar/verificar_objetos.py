"""O VLM verifica a identificação dos recortes: vê o recorte e as fotos das
candidatas e diz qual é a mesma peça, ou nenhuma.

    python tools/apontar/verificar_objetos.py mnt --raw /tmp/apontar/piloto/E/raw_mnt.jsonl \
        --out /tmp/apontar/piloto/E --debug /tmp/apontar/piloto/E/verificados

Entrada: o --raw do indexar_objetos_panorama.py corrido com --candidates 5
--save-crops --hotspot-candidates. Para cada recorte ainda não aceite cujo
melhor score chega a --min-score, ou que tem uma peça com hotspot entre as
candidatas, pergunta ao VLM (vLLM na :8005). Aceita a candidata escolhida com
confiança ≥ --min-confidence. Escreve panorama-objects-<museu>.json em --out
(a mesma forma do indexador, com `accepted_by: "vlm"`) e, com --debug, uma
folha por panorama com o recorte e a foto escolhida.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from indexar_objetos_panorama import bbox_iou  # noqa: E402

LETTERS = "ABCDE"


def fold(text: str) -> str:
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(ch))
PROMPT = (
    "A primeira imagem é um recorte de uma peça fotografada numa sala de museu (pode estar atrás de vidro, de lado, "
    "ou ser só parte da peça). A segunda imagem é uma grelha com fotografias de catálogo de peças candidatas, "
    "marcadas com as letras {letters} no canto vermelho. "
    "Qual das candidatas é EXATAMENTE a mesma peça física que está no recorte? Compara forma, decoração, cores, "
    "pormenores e inscrições; peças do mesmo tipo mas diferentes não contam. Se nenhuma for a mesma, responde none. "
    'Responde só com JSON: {{"match": "A" | "B" | … | "none", "confidence": 0.0 a 1.0, "reason": "poucas palavras"}}'
)


def data_url(path: Path, max_side: int = 640) -> str | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            from io import BytesIO
            buf = BytesIO(); im.save(buf, format="JPEG", quality=85)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def catalogue_path(root: Path, name: str | None) -> Path | None:
    if not name:
        return None
    for cand in (root / name, root / "Images" / Path(name).name):
        if cand.exists():
            return cand
    return None


def grid_image(photos: list[Path], cell: int = 360) -> str:
    """As candidatas numa só imagem (o vLLM aceita 3 imagens por pedido), cada
    uma com a sua letra, grande, no canto."""
    from io import BytesIO
    from PIL import ImageFont

    cols = 3 if len(photos) > 2 else len(photos)
    rows = (len(photos) + cols - 1) // cols
    grid = Image.new("RGB", (cols * (cell + 12), rows * (cell + 44)), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    for i, p in enumerate(photos):
        x, y = (i % cols) * (cell + 12) + 6, (i // cols) * (cell + 44) + 40
        try:
            with Image.open(p) as im:
                im = im.convert("RGB"); im.thumbnail((cell, cell))
                grid.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        except Exception:
            pass
        draw.rectangle([x, y - 38, x + 46, y - 2], fill=(220, 30, 30))
        draw.text((x + 10, y - 38), LETTERS[i], fill=(255, 255, 255), font=font)
    buf = BytesIO(); grid.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def ask(client, model: str, crop: Path, photos: list[Path]) -> tuple[str, float, str, float]:
    content = [{"type": "image_url", "image_url": {"url": data_url(crop, 512)}},
               {"type": "image_url", "image_url": {"url": grid_image(photos)}}]
    content.append({"type": "text", "text": PROMPT.format(letters=", ".join(LETTERS[: len(photos)]))})
    started = time.perf_counter()
    response = client.chat.completions.create(model=model, temperature=0.0, max_tokens=120,
                                              messages=[{"role": "user", "content": content}])
    text = response.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, flags=re.S)
    match, conf, reason = "none", 0.0, text[:80]
    if m:
        try:
            d = json.loads(m.group(0))
            match = str(d.get("match") or "none").strip().upper()[:1]
            conf = float(d.get("confidence") or 0.0)
            reason = str(d.get("reason") or "")[:80]
        except Exception:
            pass
    if match not in LETTERS[: len(photos)]:
        match = "none"
    return match, conf, reason, (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museum")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--debug", default=None)
    parser.add_argument("--min-score", type=float, default=0.60)
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8005/v1")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--images-root", default=None, help="pasta onde vivem as imagens das fichas (por omissão, IMAGE_ASSET_ROOT)")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.vlm_url, api_key="no-key", timeout=180.0, max_retries=1)
    settings = get_settings()
    root = Path(args.images_root or settings.IMAGE_ASSET_ROOT).expanduser()
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    # as fichas guardam «Images/…»: a raiz é a pasta que contém «Images»
    if root.name == "Images":
        root = root.parent
    if not (root / "Images").exists():
        print(f"ATENÇÃO: não encontro {root}/Images; o verificador não terá fotos para mostrar")
    rows = [json.loads(l) for l in open(args.raw, encoding="utf-8") if l.strip()]
    by_pano: dict[str, list[dict]] = {}
    for r in rows:
        by_pano.setdefault(r["panorama_key"], []).append(r)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"museum_slug": args.museum, "verified": True, "panoramas": {}}
    log = open(out_dir / f"verificacao_{args.museum}.jsonl", "w", encoding="utf-8")
    for key, segs in by_pano.items():
        started = time.perf_counter(); asked = accepted_vlm = 0
        for o in segs:
            if o.get("accepted_by") or not o.get("artifact_id") or not o.get("crop"):
                continue
            cands = [c for c in (o.get("candidates") or []) if c.get("image_name")]
            has_hot = any(c.get("hotspot") for c in cands)
            if (o.get("score") or 0.0) < args.min_score and not has_hot:
                continue
            # as 5 a mostrar: as 3 melhores do índice e 2 peças com hotspot, as que mais se parecem
            # com o tipo que o VLM deu ao recorte («vlm:leque» → títulos com «leque»)
            label_words = {w for w in re.split(r"[^a-z0-9]+", fold(str(o.get("prompt") or "").split(":", 1)[-1])) if len(w) >= 4}

            def affinity(c: dict) -> int:
                title_words = set(re.split(r"[^a-z0-9]+", fold(str(c.get("title") or ""))))
                return sum(1 for w in label_words if any(t.startswith(w[:5]) for t in title_words))

            scored = sorted([c for c in cands if c.get("score") is not None], key=lambda c: -c["score"])
            hot_all = sorted([c for c in cands if c.get("hotspot")], key=lambda c: (-affinity(c), -(c.get("score") or 0.0)))
            chosen = hot_all[:2] + [c for c in scored if not c.get("hotspot")]
            seen, cands5 = set(), []
            for c in chosen:
                if c["artifact_id"] not in seen:
                    seen.add(c["artifact_id"]); cands5.append(c)
            cands5 = cands5[: args.max_candidates]
            photos = [catalogue_path(root, c["image_name"]) for c in cands5]
            pairs = [(c, p) for c, p in zip(cands5, photos) if p is not None]
            if not pairs:
                continue
            try:
                match, conf, reason, ms = ask(client, args.vlm_model, Path(o["crop"]), [p for _, p in pairs])
            except Exception as exc:  # noqa: BLE001
                print(f"  VLM falhou: {exc}"); continue
            asked += 1
            decision = {"panorama_key": key, "crop": o["crop"], "top": o.get("inventory_number"), "score": o.get("score"),
                        "shown": [c["inventory_number"] for c, _ in pairs], "match": match, "confidence": conf, "reason": reason, "ms": round(ms)}
            if match != "none" and conf >= args.min_confidence:
                c = pairs[LETTERS.index(match)][0]
                o.update({"artifact_id": c["artifact_id"], "inventory_number": c["inventory_number"], "title": c.get("title"),
                          "image_name": c.get("image_name"), "accepted_by": "vlm", "vlm_confidence": conf, "vlm_reason": reason,
                          "knn_score": c.get("score")})
                accepted_vlm += 1
                decision["accepted"] = c["inventory_number"]
            log.write(json.dumps(decision, ensure_ascii=False) + "\n"); log.flush()
        # o mesmo fecho do indexador: por peça e por sítio, fica o melhor
        identified = [o for o in segs if o.get("accepted_by")]
        identified.sort(key=lambda o: (o.get("accepted_by") == "score", o.get("score") or 0.0), reverse=True)
        final: list[dict] = []
        for o in identified:
            if any(f["artifact_id"] == o["artifact_id"] for f in final):
                continue
            if any(bbox_iou(f["bbox"], o["bbox"]) > 0.5 for f in final):
                continue
            final.append(o)
        for i, o in enumerate(final, start=1):
            o["object_id"] = f"{key}:{i}"
        hot = {c["artifact_id"] for o in segs for c in (o.get("candidates") or []) if c.get("hotspot")}
        found = hot & {o["artifact_id"] for o in final}
        entry = {"segmented": len(segs), "asked_vlm": asked, "accepted_vlm": accepted_vlm, "identified": len(final),
                 "by": {k: sum(1 for o in final if o.get("accepted_by") == k) for k in ("score", "hotspot", "vlm")},
                 "hotspot_pieces": sorted(hot), "hotspot_pieces_found": sorted(found), "elapsed_s": round(time.perf_counter() - started, 1),
                 "objects": [{k: v for k, v in o.items() if k not in {"view", "candidates", "crop"}} for o in final]}
        payload["panoramas"][key] = entry
        print(f"  {key}: {len(segs)} recortes · VLM perguntado {asked}×, aceitou {accepted_vlm} · identificados {len(final)} {entry['by']}"
              f" · hotspots {len(found)}/{len(hot)} · {entry['elapsed_s']} s", flush=True)
        for o in final:
            if o.get("accepted_by") != "score":
                print(f"      {o['inventory_number']!s:14} {o.get('accepted_by'):7} knn {o.get('knn_score') if o.get('accepted_by') == 'vlm' else o.get('score')} "
                      f"{('conf ' + str(o.get('vlm_confidence'))) if o.get('accepted_by') == 'vlm' else ''}  {(o.get('title') or '')[:36]}  {o.get('vlm_reason', '')[:50]}")
        if args.debug:
            folder = Path(args.debug); folder.mkdir(parents=True, exist_ok=True)
            news = [o for o in final if o.get("accepted_by") != "score"]
            if news:
                cell = 220
                sheet = Image.new("RGB", (cell * 2 + 20, len(news) * (cell + 26)), (30, 30, 30)); d = ImageDraw.Draw(sheet)
                for i, o in enumerate(news):
                    y = i * (cell + 26)
                    for x, p in ((0, Path(o["crop"]) if o.get("crop") else None), (cell + 20, catalogue_path(root, o.get("image_name")))):
                        if p and p.exists():
                            im = Image.open(p).convert("RGB"); im.thumbnail((cell, cell)); sheet.paste(im, (x, y + 24))
                    d.text((4, y + 6), f"{o['inventory_number']}  {o.get('accepted_by')}  {(o.get('title') or '')[:30]}  {o.get('vlm_reason', '')[:40]}", fill=(255, 255, 255))
                sheet.save(folder / f"{args.museum}_{key[9:17]}.jpg", quality=80)
        (out_dir / f"panorama-objects-{args.museum}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
