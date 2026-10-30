"""«Apontar e perguntar»: o modelo de visão confirma a peça entre as melhores
candidatas do índice.

O kNN do Qwen3-VL-Embedding devolve peças parecidas; só o cosseno não
distingue duas casacas gémeas nem percebe que um painel pintado na parede não
é um leque. Dado o recorte e uma grelha com as fotos das candidatas (A, B, C),
o VLM (Qwen3-VL-8B no vLLM da :8005) diz qual é exatamente a mesma peça, ou
nenhuma. Medido offline em 10 panoramas (2026-10-28): a cobertura das peças
com hotspot passou de 1/67 para 31/72; as escolhas erradas eram decoração do
edifício com cosseno < 0,6, daí o travão POINT_VERIFY_MIN_SCORE.

Sem resposta do VLM (desligado, lento, erro) o fluxo segue como antes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import socket
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

LETTERS = "ABCDE"

# Par a par: o recorte e UMA candidata por pedido. Com a grelha de três, o
# modelo projetava os pormenores do recorte na candidata mais parecida (um
# medalhão oval com Ecce Homo «confirmado» num relicário hexagonal, 2026-10-28);
# a pergunta sim/não por candidata, com as três em paralelo, acertou nos quatro
# casos de teste e demora o mesmo (~0,4–0,7 s).
PROMPT_PAIR_PT = (
    "Duas imagens: a primeira é um recorte de uma peça fotografada numa sala de museu (pode estar atrás de vidro, de lado, "
    "ou ser só parte da peça); a segunda é a fotografia de catálogo de UMA peça candidata. É exatamente a mesma peça "
    "física? Compara a forma geral (oval, hexagonal, retangular…), o motivo representado (figura, cena, inscrição) e a "
    "decoração. Peças do mesmo tipo mas diferentes não contam. "
    'Responde só com JSON: {"same": true | false, "confidence": 0.0 a 1.0, "reason": "poucas palavras"}'
)

PROMPT_PT = (
    "A primeira imagem é um recorte de uma peça fotografada numa sala de museu (pode estar atrás de vidro, de lado, "
    "ou ser só parte da peça). A segunda imagem é uma grelha com fotografias de catálogo de peças candidatas, "
    "marcadas com as letras {letters} no canto vermelho. Qual das candidatas é EXATAMENTE a mesma peça física "
    "que está no recorte? Compara forma, decoração, cores, pormenores e inscrições; peças do mesmo tipo mas "
    "diferentes não contam. Se nenhuma for a mesma, responde none. "
    'Responde só com JSON: {{"match": "A" | "B" | "C" | "none", "confidence": 0.0 a 1.0, "reason": "poucas palavras"}}'
)


@dataclass(frozen=True)
class VerifyResult:
    status: str                 # «ok», «unavailable», «error»
    match: int | None           # índice da candidata escolhida, None = nenhuma
    confidence: float
    reason: str
    elapsed_ms: float

    def payload(self) -> dict[str, Any]:
        return {"status": self.status, "match": self.match, "confidence": round(self.confidence, 2),
                "reason": self.reason[:80], "elapsed_ms": round(self.elapsed_ms, 1)}


def resolve_image_path(root: Path | None, image_name: str | None) -> Path | None:
    """A foto da ficha no disco, como a rota /api/v1/chat/images a resolve:
    «Images/…» relativo à raiz, com o primeiro segmento igual ao nome da raiz
    retirado."""
    if root is None or not image_name:
        return None
    parts = [p for p in str(image_name).replace("\\", "/").split("/") if p and p not in {".", ".."}]
    if parts and parts[0].lower() == root.name.lower():
        parts = parts[1:]
    if not parts:
        return None
    path = (root / Path(*parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path if path.exists() else None


def _jpeg_data_url(image: Any, quality: int = 85) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_data_url(crop_jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(crop_jpeg).decode("ascii")


def grid_data_url(paths: list[Path], cell: int = 360) -> str:
    """As candidatas numa só imagem, cada uma com a sua letra, grande, no canto
    (o vLLM aceita poucas imagens por pedido)."""
    from PIL import Image, ImageDraw, ImageFont

    cols = 3 if len(paths) > 2 else max(len(paths), 1)
    rows = (len(paths) + cols - 1) // cols
    grid = Image.new("RGB", (cols * (cell + 12), max(rows, 1) * (cell + 44)), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for index, path in enumerate(paths):
        x, y = (index % cols) * (cell + 12) + 6, (index // cols) * (cell + 44) + 40
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell))
                grid.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        except Exception:  # noqa: BLE001
            pass
        draw.rectangle([x, y - 38, x + 46, y - 2], fill=(220, 30, 30))
        draw.text((x + 10, y - 38), LETTERS[index], fill=(255, 255, 255), font=font)
    return _jpeg_data_url(grid)


def photo_data_url(path: Path, max_side: int = 640) -> str | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            return _jpeg_data_url(im)
    except Exception:  # noqa: BLE001
        return None


def parse_pair(text: str) -> tuple[bool, float, str]:
    """{"same": true|false, "confidence": …, "reason": …}; sem JSON lê-se um «sim»/«true» solto."""
    found = re.search(r"\{.*\}", text or "", flags=re.S)
    if not found:
        return bool(re.search(r"\b(true|sim|yes)\b", (text or "").lower())), 0.0, (text or "")[:80]
    try:
        data = json.loads(found.group(0))
    except json.JSONDecodeError:
        return False, 0.0, (text or "")[:80]
    same = data.get("same")
    if isinstance(same, str):
        same = same.strip().lower() in {"true", "sim", "yes"}
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return bool(same), confidence, str(data.get("reason") or "")[:80]


def parse_answer(text: str, count: int) -> tuple[int | None, float, str]:
    found = re.search(r"\{.*\}", text or "", flags=re.S)
    if not found:
        return None, 0.0, (text or "")[:80]
    try:
        data = json.loads(found.group(0))
    except json.JSONDecodeError:
        return None, 0.0, (text or "")[:80]
    letter = str(data.get("match") or "none").strip().upper()[:1]
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(data.get("reason") or "")[:80]
    index = LETTERS.index(letter) if letter in LETTERS[:count] else None
    return index, confidence, reason


class PointVerifier:
    """Cliente do VLM para a confirmação; uma instância por PointAskService."""

    def __init__(self, *, base_url: str, model: str, api_key: str = "", timeout_seconds: float = 6.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "no-key"
        self.timeout = float(timeout_seconds)
        self._client: Any | None = None

    def _vlm(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
        return self._client

    def reachable(self, timeout: float = 0.5) -> bool:
        """No WSL um porto sem ninguém à escuta fica pendurado até ao timeout:
        liga-se ao porto antes do pedido."""
        parsed = urlparse(self.base_url)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    async def _pair(self, crop_url: str, photo: Path) -> tuple[bool, float, str]:
        photo_url = await asyncio.to_thread(photo_data_url, photo)
        if photo_url is None:
            return False, 0.0, "foto ilegível"
        response = await self._vlm().chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=100,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": crop_url}},
                {"type": "image_url", "image_url": {"url": photo_url}},
                {"type": "text", "text": PROMPT_PAIR_PT},
            ]}],
        )
        text = (response.choices[0].message.content or "") if response.choices else ""
        return parse_pair(text)

    async def verify(self, crop_jpeg: bytes, photos: list[Path]) -> VerifyResult:
        """Uma pergunta sim/não por candidata, todas em paralelo. Exatamente um
        «sim» → essa; vários → o de maior confiança (e, em empate, o primeiro,
        que é o de maior cosseno); nenhum → None."""
        started = time.perf_counter()
        if not photos:
            return VerifyResult("error", None, 0.0, "sem fotos das candidatas", 0.0)
        if not await asyncio.to_thread(self.reachable):
            return VerifyResult("unavailable", None, 0.0, f"VLM sem resposta em {self.base_url}", (time.perf_counter() - started) * 1000)
        crop_url = crop_data_url(crop_jpeg)
        results = await asyncio.gather(*(self._pair(crop_url, p) for p in photos), return_exceptions=True)
        answers: list[tuple[bool, float, str]] = []
        for item in results:
            if isinstance(item, BaseException):
                logger.warning("point.verify.failed: %s: %s", type(item).__name__, item)
                return VerifyResult("error", None, 0.0, f"{type(item).__name__}", (time.perf_counter() - started) * 1000)
            answers.append(item)
        yes = [(index, confidence, reason) for index, (same, confidence, reason) in enumerate(answers) if same]
        elapsed = (time.perf_counter() - started) * 1000
        if not yes:
            reasons = "; ".join(f"{LETTERS[i]}: {r}" for i, (_, _, r) in enumerate(answers) if r)[:80]
            return VerifyResult("ok", None, max((c for _, c, _ in answers), default=0.0), reasons, elapsed)
        index, confidence, reason = max(yes, key=lambda item: (item[1], -item[0]))
        if len(yes) > 1:
            reason = f"{len(yes)} candidatas «sim»; {reason}"
        return VerifyResult("ok", index, confidence, reason, elapsed)
