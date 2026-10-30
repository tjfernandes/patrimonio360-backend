"""O VLM diz onde estão as peças em cada vista de um panorama (caixas), para o
SAM recortar por caixa (indexar_objetos_panorama.py --boxes).

    python tools/apontar/caixas_vlm.py mnt --panorama panorama_… [--narrow] --out /tmp/apontar/piloto/caixas_mnt.jsonl --debug /tmp/apontar/piloto/caixas
    python tools/apontar/caixas_vlm.py mnaa --calibrar panorama_83D8… --yaw 2.41 --pitch -2.5   # uma vista, com as caixas desenhadas

As vistas são as mesmas do indexador (build_cameras). Cada vista vai ao VLM
(vLLM na :8005, API OpenAI) com o pedido de marcar todas as peças expostas,
uma por uma, incluindo as que estão dentro de vitrinas. As caixas ficam em
píxeis da vista. Com --debug grava cada vista com as caixas.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.point_view import Camera, CubeFaceStore, project_view  # noqa: E402
from indexar_objetos_panorama import build_cameras  # noqa: E402

PROMPT = (
    "Esta é uma vista de uma sala de museu. Marca TODAS as peças expostas que se veem inteiras ou quase inteiras: "
    "quadros, esculturas, mobiliário, trajes e vestidos em manequim, cerâmica, ourivesaria, joias, tapeçarias, "
    "e os objetos dentro de vitrinas, um por um. Não marques paredes, portas, janelas, bancos, extintores, "
    "etiquetas, nem a vitrina como um todo. Responde só com JSON, uma lista de objetos "
    '{"label": "tipo curto", "bbox_2d": [x1, y1, x2, y2]}, com as coordenadas em píxeis da imagem.'
)


def encode(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_ITEM = re.compile(r"\{[^{}]*?\"bbox_2d\"\s*:\s*\[([^\]]*)\][^{}]*\}", flags=re.S)
_LABEL = re.compile(r"\"label\"\s*:\s*\"([^\"]*)\"")


def parse_boxes(text: str, width: int, height: int, *, normalized: bool = True) -> list[dict]:
    """Lê os objetos um a um (aguenta JSON truncado pelo max_tokens). O Qwen3-VL
    dá as coordenadas em 0..1000 da imagem (calibrado na Sala 52 do MNAA,
    2026-10-28: caem exatamente nos quadros); `normalized=False` para px."""
    raw: list[tuple[str, list[float]]] = []
    for item in _ITEM.finditer(text):
        try:
            values = [float(v) for v in item.group(1).split(",")]
        except ValueError:
            continue
        if len(values) != 4:
            continue
        label = _LABEL.search(item.group(0))
        raw.append((label.group(1) if label else "?", values))
    boxes = []
    for label, (x0, y0, x1, y1) in raw:
        if normalized:
            x0, x1 = x0 * width / 1000, x1 * width / 1000
            y0, y1 = y0 * height / 1000, y1 * height / 1000
        x0, x1 = sorted((max(0.0, min(x0, width)), max(0.0, min(x1, width))))
        y0, y1 = sorted((max(0.0, min(y0, height)), max(0.0, min(y1, height))))
        if (x1 - x0) * (y1 - y0) < 0.0005 * width * height:
            continue
        boxes.append({"label": label[:40], "box": [round(x0), round(y0), round(x1), round(y1)]})
    return boxes


class Boxer:
    def __init__(self, args: argparse.Namespace) -> None:
        from openai import OpenAI

        self.args = args
        self.settings = get_settings()
        self.client = OpenAI(base_url=args.vlm_url, api_key="no-key", timeout=120.0, max_retries=1)
        self.model = args.vlm_model
        cache = self.settings.point_tiles_cache_dir_resolved
        self.faces = CubeFaceStore(base_url=self.settings.POINT_TOURS_BASE_URL, cache_dir=cache,
                                   tiles_per_side=int(self.settings.POINT_FACE_TILES), level=int(self.settings.POINT_FACE_LEVEL))
        self.zoom_faces = CubeFaceStore(base_url=self.settings.POINT_TOURS_BASE_URL, cache_dir=cache,
                                        tiles_per_side=int(self.settings.POINT_CROP_FACE_TILES), level=int(self.settings.POINT_CROP_FACE_LEVEL), memory_faces=12)

    def render(self, museum: str, panorama_key: str, camera: Camera) -> Image.Image:
        store = self.zoom_faces if camera.hfov < 60.0 else self.faces
        view = project_view(store, museum=museum, panorama_key=panorama_key, camera=camera, width=camera.width)
        return Image.fromarray(view.image)

    def ask(self, image: Image.Image) -> tuple[list[dict], str, float]:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=3000,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": encode(image)}}, {"type": "text", "text": PROMPT}]}],
        )
        text = response.choices[0].message.content or ""
        return parse_boxes(text, image.width, image.height), text, (time.perf_counter() - started) * 1000

    @staticmethod
    def draw(image: Image.Image, boxes: list[dict]) -> Image.Image:
        out = image.copy()
        d = ImageDraw.Draw(out)
        for b in boxes:
            x0, y0, x1, y1 = b["box"]
            d.rectangle([x0, y0, x1, y1], outline=(255, 211, 60), width=3)
            d.rectangle([x0, max(y0 - 14, 0), x0 + 7 * len(b["label"]) + 6, max(y0 - 14, 0) + 14], fill=(255, 211, 60))
            d.text((x0 + 3, max(y0 - 13, 0)), b["label"], fill=(0, 0, 0))
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("museum")
    parser.add_argument("--panorama", action="append", default=[])
    parser.add_argument("--panoramas-file", default=None, help="ficheiro com um panorama_key por linha")
    parser.add_argument("--calibrar", default=None, help="panorama_key: uma só vista (--yaw/--pitch) com as caixas desenhadas")
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--yaw-step", type=int, default=45)
    parser.add_argument("--pitches", default="0,-30")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--narrow", action="store_true")
    parser.add_argument("--narrow-hfov", type=float, default=45.0)
    parser.add_argument("--narrow-step", type=float, default=22.5)
    parser.add_argument("--narrow-pitches", default="0,-25,25")
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8005/v1")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--out", default=None, help="JSONL de saída (uma linha por vista)")
    parser.add_argument("--debug", default=None, help="pasta para as vistas com as caixas")
    parser.add_argument("--workers", type=int, default=4, help="pedidos ao VLM em paralelo")
    parser.add_argument("--skip-done", action="store_true", help="salta os panoramas que já têm todas as vistas no --out")
    args = parser.parse_args()

    boxer = Boxer(args)
    if args.calibrar:
        camera = Camera(yaw=args.yaw, pitch=args.pitch, hfov=args.hfov, width=args.width, height=int(round(args.width * 0.75)))
        image = boxer.render(args.museum, args.calibrar, camera)
        boxes, text, ms = boxer.ask(image)
        print(f"{len(boxes)} caixas em {ms:.0f} ms; resposta crua: {text[:600]}")
        for b in boxes:
            print("  ", b)
        folder = Path(args.debug or "/tmp/apontar/piloto"); folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"calibrar_{args.calibrar[:17]}_{args.yaw:+.0f}_{args.pitch:+.0f}.jpg"
        boxer.draw(image, boxes).save(path, quality=85); print("imagem:", path)
        return 0

    keys = list(args.panorama)
    if args.panoramas_file:
        keys += [l.strip() for l in Path(args.panoramas_file).read_text().splitlines() if l.strip()]
    if not keys:
        parser.error("indica --panorama, --panoramas-file ou --calibrar")
    cameras = build_cameras(args)
    if args.skip_done and args.out and Path(args.out).exists():
        counts: dict[str, int] = {}
        for line in Path(args.out).read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[json.loads(line)["panorama_key"]] = counts.get(json.loads(line)["panorama_key"], 0) + 1
        done = {k for k, n in counts.items() if n >= len(cameras)}
        keys = [k for k in keys if k not in done]
        print(f"{len(done)} panorama(s) já com caixas, saltados", flush=True)
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    print(f"{len(keys)} panorama(s) de {args.museum} · {len(cameras)} vistas cada · {args.workers} em paralelo", flush=True)
    from concurrent.futures import ThreadPoolExecutor

    def one(key: str, camera: Camera):
        image = boxer.render(args.museum, key, camera)
        boxes, _, _ = boxer.ask(image)
        return camera, image, boxes

    for index, key in enumerate(keys, start=1):
        started = time.perf_counter(); total = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(one, key, camera) for camera in cameras]
            for camera, future in zip(cameras, futures):
                try:
                    camera, image, boxes = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  {key} yaw {camera.yaw} pitch {camera.pitch}: FALHOU ({type(exc).__name__}: {exc})", flush=True)
                    continue
                total += len(boxes)
                if out:
                    out.write(json.dumps({"panorama_key": key, "view": {"yaw": camera.yaw, "pitch": camera.pitch, "hfov": camera.hfov,
                                          "width": camera.width, "height": camera.height}, "boxes": boxes}, ensure_ascii=False) + "\n")
                    out.flush()
                if args.debug and boxes:
                    folder = Path(args.debug) / args.museum / key; folder.mkdir(parents=True, exist_ok=True)
                    boxer.draw(image, boxes).save(folder / f"yaw{int(camera.yaw):+04d}_pitch{int(camera.pitch):+03d}_hfov{int(camera.hfov)}.jpg", quality=80)
        print(f"  [{index}/{len(keys)}] {time.strftime('%H:%M:%S')} {key}: {total} caixas em {len(cameras)} vistas · {time.perf_counter() - started:.0f} s", flush=True)
    if out:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
