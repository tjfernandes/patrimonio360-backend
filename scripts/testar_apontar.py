"""Simula o toque do «Apontar e perguntar» do frontend e mede cada fase.

    python scripts/testar_apontar.py --museum mnaa \
        --panorama panorama_83D8666B_91ED_0BA2_41D9_9A5E8973B146 --yaw 2.41 --pitch -2.5 \
        --expect-inventory "1874 Pint" --out /tmp/apontar
    python scripts/testar_apontar.py --museum mnaa --panorama … --yaw -25 --pitch 0 --expect-none
    python scripts/testar_apontar.py … --intent describe_point_crop --request-id point-…

Imprime cada evento SSE com o tempo desde o início, grava a vista projetada,
o crop e o cutout em --out (descarregados dos URLs do próprio backend) e
confirma: candidato 1 (--expect-inventory) com score ≥ --threshold, máscara
com ≥ 12 vértices dentro de 0..1, «decided» antes de 1,5 s e o texto a
começar antes de 2,5 s. Com --expect-none confirma chosen=null e o texto de
«não encontrei». Sai com 1 se alguma verificação falhar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="base do backend (sem /api/v1)")
    parser.add_argument("--museum", default="mnaa")
    parser.add_argument("--museum-name", default="Museu Nacional de Arte Antiga")
    parser.add_argument("--panorama", default="panorama_83D8666B_91ED_0BA2_41D9_9A5E8973B146")
    parser.add_argument("--yaw", type=float, default=2.41)
    parser.add_argument("--pitch", type=float, default=-2.5)
    parser.add_argument("--hfov", type=float, default=100.0)
    parser.add_argument("--viewport", default="999x759")
    parser.add_argument("--point", default="0.5,0.5", help="x,y em 0..1 da vista")
    parser.add_argument("--hotspot", default=None, help="hotspot_overlay_id (toque num hotspot)")
    parser.add_argument("--intent", default="point_and_ask", choices=["point_and_ask", "describe_point_crop"])
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--message", default=None)
    parser.add_argument("--language", default="pt")
    parser.add_argument("--session-id", default="teste-apontar")
    parser.add_argument("--out", default="/tmp/apontar")
    parser.add_argument("--expect-inventory", default=None)
    parser.add_argument("--expect-none", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    width, height = (int(v) for v in args.viewport.lower().split("x"))
    px, py = (float(v) for v in args.point.split(","))
    request_id = args.request_id or f"point-{int(time.time() * 1000)}"
    message = args.message or ("Descreve-me o que vês" if args.intent == "describe_point_crop" else "O que é isto?")
    body = {
        "museum_slug": args.museum, "museum_id": args.museum, "museum_name": args.museum_name, "language": args.language,
        "message": message, "conversation_id": f"{args.session_id}-{request_id}",
        "metadata": {
            "session_id": args.session_id, "intent": args.intent,
            "point": {
                "request_id": request_id, "panorama_key": args.panorama, "panorama_title": None,
                "camera": {"yaw": args.yaw, "pitch": args.pitch, "hfov": args.hfov, "vfov": None},
                "viewport": {"width": width, "height": height}, "point": {"x": px, "y": py},
                "ray": None, "hotspot_overlay_id": args.hotspot,
            },
        },
    }
    api = args.url.rstrip("/") + "/api/v1/chat"
    req = urllib.request.Request(api + "/messages/stream", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    print(f"=== {args.intent} · {args.museum} · {args.panorama} · câmara yaw {args.yaw} pitch {args.pitch} hfov {args.hfov} · ponto ({px}, {py})"
          + (f" · hotspot {args.hotspot}" if args.hotspot else ""))
    t0 = time.monotonic()
    events: list[tuple[float, str, dict]] = []
    first_token: float | None = None
    text_parts: list[str] = []
    result: dict | None = None
    event_name = None
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: ") and event_name:
                data = json.loads(line[6:])
                t = time.monotonic() - t0
                events.append((t, event_name, data))
                if event_name == "token":
                    if first_token is None:
                        first_token = t
                        print(f"  {t:6.2f} s  token      (primeiro)")
                    text_parts.append(data.get("delta", ""))
                elif event_name == "point":
                    payload = data.get("payload", {})
                    stage = payload.get("stage")
                    extra = ""
                    if stage == "segmented":
                        extra = f"mask {len(payload.get('mask') or [])} vértices · {payload.get('cutout_url')}"
                    elif stage == "candidates":
                        extra = " | ".join(f"{c.get('inventory_number')} {c.get('score')}" for c in payload.get("candidates") or [])
                    elif stage == "decided":
                        extra = f"chosen={payload.get('chosen')} hotspot_match={payload.get('hotspot_match')}"
                    elif stage == "captured":
                        extra = payload.get("crop_url") or ""
                    print(f"  {t:6.2f} s  point      {stage}: {extra}")
                elif event_name == "status":
                    print(f"  {t:6.2f} s  status     {data.get('key')}")
                elif event_name == "result":
                    result = data.get("payload")
                    print(f"  {t:6.2f} s  result")
                elif event_name == "error":
                    print(f"  {t:6.2f} s  error      {data}")
                event_name = None if event_name == "done" else event_name
    if result is None:
        print("sem result")
        return 1
    print("  texto:", " ".join(result["reply"].split())[:400])
    point = result.get("point") or {}
    print("  cartões:", [(a["inventory_number"], a["title"]) for a in result.get("artifact_results") or []])
    print("  navegação:", [t["inventory_id"] for t in result.get("navigation_targets") or []])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("view", "crop", "cutout"):
        url = f"{api}/point/{request_id}/{kind}.jpg"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                (out / f"{request_id}_{kind}.jpg").write_bytes(response.read())
        except Exception as exc:
            print(f"  (sem {kind}.jpg: {exc})")
    print(f"  imagens em {out}/{request_id}_{{view,crop,cutout}}.jpg")

    ok = True

    def check(condition: bool, label: str) -> None:
        nonlocal ok
        ok &= bool(condition)
        print(f"  {'✓' if condition else '✗'} {label}")

    if args.intent == "describe_point_crop":
        check(bool(result["reply"]), "texto da descrição")
        check(first_token is not None and first_token < 4.0, f"primeiro token {first_token if first_token else '—'} s < 4 s")
        return 0 if ok else 1

    decided = next((t for t, name, d in events if name == "point" and d.get("payload", {}).get("stage") == "decided"), None)
    segmented = next((d["payload"] for t, name, d in events if name == "point" and d.get("payload", {}).get("stage") == "segmented"), {})
    mask = segmented.get("mask") or []
    check(len(mask) >= 40 and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in mask), f"máscara com {len(mask)} vértices em 0..1 (≥ 40)")
    check(decided is not None and decided < 1.5, f"decided aos {decided if decided else '—'} s < 1,5 s")
    candidates = point.get("candidates") or []
    if args.expect_inventory:
        top = candidates[0] if candidates else {}
        check(top.get("inventory_number") == args.expect_inventory and float(top.get("score") or 0) >= args.threshold,
              f"candidato 1 = {top.get('inventory_number')} com score {top.get('score')} (esperado {args.expect_inventory} ≥ {args.threshold})")
        check(point.get("chosen") is not None, "chosen não é null")
        check(first_token is not None and first_token < 2.5, f"texto a começar aos {first_token if first_token else '—'} s < 2,5 s")
        cards = result.get("artifact_results") or []
        check(bool(cards) and cards[0].get("inventory_number") == args.expect_inventory, "o escolhido é o primeiro cartão")
    if args.expect_none:
        check(point.get("chosen") is None, f"chosen é null (melhor: {candidates[0].get('score') if candidates else '—'})")
        check("não encontrei" in result["reply"].lower() or "could not find" in result["reply"].lower(), "texto de «não encontrei»")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
