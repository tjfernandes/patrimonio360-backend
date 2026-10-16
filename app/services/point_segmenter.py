"""Recorte do objeto tocado («Apontar e perguntar»): SAM 3 com um ponto
positivo, máscara → polígono simplificado, crop e cutout.

O modelo (facebook/sam3, via transformers: Sam3TrackerModel, prompt por
ponto) carrega uma vez no arranque e é partilhado; cada toque corre uma
passagem do encoder (~40 ms na RTX 6000 depois do aquecimento). Sem CUDA ou
sem o modelo, `ready` fica False e o backend responde sem recorte.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    mask: np.ndarray          # (H, W) bool
    score: float              # iou previsto pelo SAM
    area_fraction: float      # área da máscara / área da imagem
    candidates: tuple[tuple[float, float], ...]  # (score, área) das 3 propostas, para o registo
    elapsed_ms: float


class PointSegmenter:
    """SAM 3 partilhado, uma passagem de cada vez (a GPU é uma)."""

    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any | None = None
        self._processor: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="point-sam")
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if self.ready:
            return True
        try:
            started = time.perf_counter()
            import torch
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            model = Sam3TrackerModel.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device).eval()
            processor = Sam3TrackerProcessor.from_pretrained(self.model_id)
            self._model, self._processor = model, processor
            loaded = time.perf_counter()
            # aquecimento: a primeira passagem compila os kernels (~600 ms)
            self._segment_sync(np.zeros((256, 256, 3), dtype=np.uint8), 128, 128)
            logger.info(
                "point.sam.ready model=%s device=%s load_ms=%d warmup_ms=%d",
                self.model_id, self.device, round((loaded - started) * 1000), round((time.perf_counter() - loaded) * 1000),
            )
            return True
        except Exception as exc:  # sem SAM o resto do backend arranca na mesma
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("point.sam.failed model=%s", self.model_id)
            self._model = self._processor = None
            return False

    async def segment(
        self, image: np.ndarray, x: int, y: int, *, min_area: float = 0.002, max_area: float = 0.30
    ) -> Segment:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._segment_sync, image, int(x), int(y), min_area, max_area)

    def _segment_sync(self, image: np.ndarray, x: int, y: int, min_area: float = 0.002, max_area: float = 0.30) -> Segment:
        if self._model is None or self._processor is None:
            raise RuntimeError("SAM não carregado")
        import torch

        started = time.perf_counter()
        pil = Image.fromarray(np.ascontiguousarray(image[..., :3]))
        with self._lock:
            inputs = self._processor(
                images=pil, input_points=[[[[x, y]]]], input_labels=[[[1]]], return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                outputs = self._model(**inputs, multimask_output=True)
            masks = self._processor.post_process_masks(outputs.pred_masks.float().cpu(), inputs["original_sizes"])[0][0]
            scores = outputs.iou_scores.float().cpu().numpy().ravel()
        masks_np = masks.numpy().astype(bool)
        total = float(masks_np.shape[-1] * masks_np.shape[-2])
        areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1) / total
        chosen = choose_mask(scores.tolist(), areas.tolist(), min_area=min_area, max_area=max_area)
        return Segment(
            mask=masks_np[chosen],
            score=float(scores[chosen]),
            area_fraction=float(areas[chosen]),
            candidates=tuple((float(s), float(a)) for s, a in zip(scores, areas)),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


@dataclass(frozen=True)
class ConceptHit:
    mask: np.ndarray
    score: float
    prompt: str
    area_fraction: float
    elapsed_ms: float


class ConceptSegmenter:
    """SAM 3 por conceito (texto): «artwork», «furniture», … dá a peça inteira
    que contém o ponto, enquanto o prompt por ponto numa pintura devolve o
    pormenor pintado (o cisne, a faca). Os prompts correm em lote, uma
    passagem por toque (~0,3 s na RTX 6000 com 4 prompts)."""

    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda", prompts: tuple[str, ...] = ("artwork",), threshold: float = 0.3) -> None:
        self.model_id = model_id
        self.device = device
        self.prompts = tuple(p.strip() for p in prompts if p.strip()) or ("artwork",)
        self.threshold = float(threshold)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if self.ready:
            return True
        try:
            started = time.perf_counter()
            import torch
            from transformers import Sam3Model, Sam3Processor

            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._model = Sam3Model.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device).eval()
            self._processor = Sam3Processor.from_pretrained(self.model_id)
            loaded = time.perf_counter()
            self.find_sync(np.zeros((778, 1024, 3), dtype=np.uint8), 512, 389)
            logger.info(
                "point.sam3.concept.ready prompts=%s load_ms=%d warmup_ms=%d",
                ",".join(self.prompts), round((loaded - started) * 1000), round((time.perf_counter() - loaded) * 1000),
            )
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("point.sam3.concept.failed model=%s", self.model_id)
            self._model = self._processor = None
            return False

    def find_sync(self, image: np.ndarray, x: int, y: int, *, min_area: float = 0.001, max_area: float = 0.60) -> ConceptHit | None:
        """A instância de maior score, entre todos os prompts, que contém o ponto."""
        if self._model is None or self._processor is None:
            return None
        import torch

        started = time.perf_counter()
        pil = Image.fromarray(np.ascontiguousarray(image[..., :3]))
        with self._lock:
            inputs = self._processor(images=[pil] * len(self.prompts), text=list(self.prompts), return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
            )
            best: ConceptHit | None = None
            for prompt, result in zip(self.prompts, results):
                scores = result["scores"].float().cpu().numpy().ravel()
                masks = result["masks"]
                for index in range(len(scores)):
                    mask = masks[index].cpu().numpy().astype(bool)
                    if not mask[y, x]:
                        continue
                    area = float(mask.mean())
                    if not (min_area <= area <= max_area):
                        continue
                    if best is None or float(scores[index]) > best.score:
                        best = ConceptHit(mask=mask, score=float(scores[index]), prompt=prompt, area_fraction=area, elapsed_ms=0.0)
        if best is None:
            return None
        return ConceptHit(mask=best.mask, score=best.score, prompt=best.prompt, area_fraction=best.area_fraction,
                          elapsed_ms=(time.perf_counter() - started) * 1000)


def choose_mask(scores: list[float], areas: list[float], *, min_area: float = 0.002, max_area: float = 0.30) -> int:
    """Qual das 3 propostas do SAM (encaixadas: o todo, uma parte, um pormenor).

    Quer-se o objeto inteiro: entre as propostas com área dentro de
    [min_area, max_area] da imagem e score de pelo menos metade do melhor,
    a maior. Sem nenhuma na janela, a de melhor score. Na vista geral o
    tecto limita a 30 % (não é a parede toda); na vista apertada à volta do
    objeto o objeto enche-a quase toda e a janela é outra."""
    if not scores:
        return 0
    best = max(scores)
    preferred = [i for i, a in enumerate(areas) if min_area <= a <= max_area and scores[i] >= 0.5 * best]
    if preferred:
        return max(preferred, key=lambda i: (areas[i], scores[i]))
    return max(range(len(scores)), key=lambda i: scores[i])


# --- máscara → polígono -------------------------------------------------------

def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def trace_contour(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contorno exterior (x, y) da maior componente, por rastreio de Moore no
    sentido horário. Só visita os píxeis da fronteira: poucos milhares."""
    component = largest_component(mask)
    if not component.any():
        return []
    padded = np.pad(component, 1)
    ys, xs = np.nonzero(padded)
    start = (int(ys[np.argmin(ys * padded.shape[1] + xs)]), int(xs[np.argmin(ys * padded.shape[1] + xs)]))
    # vizinhos de Moore no sentido horário, a começar a oeste
    neighbours = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    contour = [start]
    current, backtrack = start, 0
    limit = 4 * (padded.shape[0] + padded.shape[1]) * 4
    while len(contour) < limit:
        found = False
        for step in range(8):
            k = (backtrack + step) % 8
            dy, dx = neighbours[k]
            ny, nx = current[0] + dy, current[1] + dx
            if padded[ny, nx]:
                contour.append((ny, nx))
                current = (ny, nx)
                backtrack = (k + 5) % 8  # recomeça a partir do píxel antes do que entrou
                found = True
                break
        if not found or current == start:
            break
    if len(contour) > 1 and contour[-1] == start:
        contour.pop()
    return [(x - 1, y - 1) for y, x in contour]


def douglas_peucker(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    pts = np.asarray(points, dtype=np.float64)
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        start, end = pts[a], pts[b]
        segment = end - start
        length = np.hypot(*segment)
        chunk = pts[a + 1: b]
        if length < 1e-9:
            distances = np.hypot(*(chunk - start).T)
        else:
            relative = chunk - start
            distances = np.abs(segment[0] * relative[:, 1] - segment[1] * relative[:, 0]) / length
        index = int(np.argmax(distances))
        if distances[index] > epsilon:
            keep[a + 1 + index] = True
            stack.append((a, a + 1 + index))
            stack.append((a + 1 + index, b))
    return [tuple(p) for p in pts[keep]]


def mask_polygon(mask: np.ndarray, *, min_vertices: int = 12, max_vertices: int = 40) -> list[tuple[float, float]]:
    """Polígono do objeto em coordenadas 0..1 da imagem, com 12 a 40 vértices
    (Douglas-Peucker com o epsilon ajustado por bissecção)."""
    contour = trace_contour(mask)
    if len(contour) < 3:
        return []
    height, width = mask.shape
    closed = contour + [contour[0]]
    best = douglas_peucker(closed, 0.5)[:-1]
    if len(best) > max_vertices:
        low, high = 0.5, max(width, height) / 4.0
        for _ in range(24):
            mid = (low + high) / 2.0
            simplified = douglas_peucker(closed, mid)[:-1]
            if len(simplified) > max_vertices:
                low = mid
            else:
                best, high = simplified, mid
                if len(simplified) >= min_vertices:
                    break
    if len(best) < min_vertices and len(contour) >= min_vertices:
        # objeto quase retangular: reparte o contorno em pontos igualmente espaçados
        step = len(contour) / float(min_vertices)
        best = [contour[int(i * step)] for i in range(min_vertices)]
    return [(round(x / width, 4), round(y / height, 4)) for x, y in best]


# --- crop e cutout ------------------------------------------------------------

@dataclass(frozen=True)
class Cutout:
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 na imagem
    crop_jpeg: bytes
    cutout_jpeg: bytes


def crop_and_cutout(
    image: np.ndarray, mask: np.ndarray, *, margin: float = 0.08, dim: float = 0.35, quality: int = 85, whole: bool = False
) -> Cutout:
    """Caixa envolvente da máscara com `margin` de folga (ou a imagem inteira,
    com `whole`); o cutout escurece o fundo (fora da máscara) a `dim` do brilho."""
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if whole or len(xs) == 0:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        mx, my = int(math.ceil((x1 - x0) * margin)), int(math.ceil((y1 - y0) * margin))
        x0, y0, x1, y1 = max(0, x0 - mx), max(0, y0 - my), min(width, x1 + mx), min(height, y1 + my)
    crop = image[y0:y1, x0:x1]
    region = mask[y0:y1, x0:x1]
    shade = np.where(region[..., None], 1.0, dim)
    cutout = (crop.astype(np.float32) * shade).clip(0, 255).astype(np.uint8)
    return Cutout(box=(x0, y0, x1, y1), crop_jpeg=_jpeg(crop, quality), cutout_jpeg=_jpeg(cutout, quality))


def _jpeg(array: np.ndarray, quality: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()
