"""«Apontar e perguntar»: do toque no panorama à peça do acervo.

O frontend manda `metadata.point` (panorama, câmara, viewport, ponto 0..1,
hotspot próximo). Aqui:

1. reconstrói-se a vista das faces do cubo (point_view) e projeta-se o ponto;
2. o SAM 3 recorta o objeto tocado (point_segmenter): máscara → polígono,
   crop e cutout, guardados em memória por request_id durante 10 minutos;
3. o crop passa pelo Qwen3-VL-Embedding e vai ao kNN do índice de imagens,
   com prioridade às peças da visita (×1,15) e às que têm hotspot neste
   panorama (×1,3); se o toque caiu num hotspot, a peça dele é o candidato 1
   com 0,99, sem pesquisa;
4. acima do limiar, a peça escolhida segue o fluxo describe_artifact.

Cada fase manda um evento SSE `point` (captured, segmented, candidates,
decided). O áudio… não: as imagens nunca ficam no registo, só os tempos, os
candidatos e o escolhido.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import re
import socket
import threading
import time
from urllib.parse import urlparse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.services.point_segmenter import ConceptSegmenter, PointSegmenter, crop_and_cutout, mask_polygon
from app.services.point_view import Camera, CubeFaceStore, ProjectedView, mask_in_camera, point_in_view, project_view, ray_from_point

logger = logging.getLogger(__name__)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


@dataclass
class PointRequest:
    request_id: str
    panorama_key: str
    panorama_title: str | None
    camera: Camera
    point_x: float
    point_y: float
    ray_yaw: float | None
    ray_pitch: float | None
    hotspot_overlay_id: str | None

    @classmethod
    def parse(cls, raw: Any) -> "PointRequest | None":
        if not isinstance(raw, dict):
            return None
        try:
            camera = raw.get("camera") or {}
            viewport = raw.get("viewport") or {}
            point = raw.get("point") or {}
            ray = raw.get("ray") or {}
            request_id = str(raw.get("request_id") or "").strip()
            if not _REQUEST_ID.match(request_id):
                request_id = f"point-{int(time.time() * 1000)}"
            panorama_key = str(raw.get("panorama_key") or "").strip()
            width, height = int(viewport.get("width") or 0), int(viewport.get("height") or 0)
            hfov = float(camera.get("hfov") or 0)
            if not panorama_key or width <= 0 or height <= 0 or not (10 <= hfov <= 170):
                return None
            x, y = float(point.get("x")), float(point.get("y"))
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return None
            return cls(
                request_id=request_id,
                panorama_key=panorama_key,
                panorama_title=str(raw.get("panorama_title") or "").strip() or None,
                camera=Camera(yaw=float(camera.get("yaw") or 0), pitch=float(camera.get("pitch") or 0), hfov=hfov, width=width, height=height),
                point_x=x,
                point_y=y,
                ray_yaw=float(ray["yaw"]) if isinstance(ray.get("yaw"), (int, float)) else None,
                ray_pitch=float(ray["pitch"]) if isinstance(ray.get("pitch"), (int, float)) else None,
                hotspot_overlay_id=str(raw.get("hotspot_overlay_id") or "").strip() or None,
            )
        except (TypeError, ValueError, KeyError):
            return None


@dataclass
class PointAssets:
    crop_jpeg: bytes
    cutout_jpeg: bytes
    expires_at: float
    museum_slug: str = ""
    panorama_key: str = ""
    view_jpeg: bytes = b""


class PointAssetStore:
    """crop.jpg / cutout.jpg / view.jpg por request_id, em memória, durante o TTL."""

    def __init__(self, ttl_seconds: float = 600.0, max_items: int = 200) -> None:
        self.ttl = float(ttl_seconds)
        self.max_items = max_items
        self._items: dict[str, PointAssets] = {}
        self._lock = threading.Lock()

    def put(self, request_id: str, crop_jpeg: bytes, cutout_jpeg: bytes, *, museum_slug: str, panorama_key: str, view_jpeg: bytes = b"") -> None:
        with self._lock:
            self._sweep()
            self._items[request_id] = PointAssets(crop_jpeg, cutout_jpeg, time.monotonic() + self.ttl, museum_slug, panorama_key, view_jpeg)
            while len(self._items) > self.max_items:
                self._items.pop(next(iter(self._items)))

    def get(self, request_id: str) -> PointAssets | None:
        with self._lock:
            self._sweep()
            return self._items.get(request_id)

    def _sweep(self) -> None:
        now = time.monotonic()
        for key in [k for k, v in self._items.items() if v.expires_at <= now]:
            self._items.pop(key, None)


@dataclass
class Candidate:
    artifact_id: str
    inventory_number: str | None
    title: str | None
    score: float
    image_name: str | None
    in_tour: bool = False
    in_panorama: bool = False
    raw_score: float | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "inventory_number": self.inventory_number,
            "title": self.title,
            "score": round(self.score, 3),
            "image_name": self.image_name,
        }


@dataclass
class PointOutcome:
    request: PointRequest
    view: ProjectedView | None = None
    point_px: tuple[int, int] = (0, 0)
    ray: tuple[float, float] | None = None
    mask_polygon: list[tuple[float, float]] = field(default_factory=list)
    segment_score: float | None = None
    segment_area: float | None = None
    segment_candidates: tuple[tuple[float, float], ...] = ()
    segment_source: str | None = None   # «concept:artwork» ou «point»
    crop_url: str | None = None
    cutout_url: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    chosen: Candidate | None = None
    hotspot_match: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    crop_jpeg: bytes | None = None

    def decided_payload(self) -> dict[str, Any]:
        return {
            "stage": "decided",
            "request_id": self.request.request_id,
            "chosen": {"artifact_id": self.chosen.artifact_id, "score": round(self.chosen.score, 3)} if self.chosen else None,
            "hotspot_match": self.hotspot_match,
            "candidates": [c.payload() for c in self.candidates],
            "crop_url": self.crop_url,
            "cutout_url": self.cutout_url,
            "mask": [list(p) for p in self.mask_polygon],
        }

    def log_payload(self, threshold: float) -> dict[str, Any]:
        """Para o query_logger: sem imagens."""
        return {
            "request_id": self.request.request_id,
            "panorama_key": self.request.panorama_key,
            "panorama_title": self.request.panorama_title,
            "camera": {"yaw": self.request.camera.yaw, "pitch": self.request.camera.pitch, "hfov": self.request.camera.hfov,
                       "width": self.request.camera.width, "height": self.request.camera.height},
            "point": {"x": self.request.point_x, "y": self.request.point_y},
            "ray": {"yaw": round(self.ray[0], 2), "pitch": round(self.ray[1], 2)} if self.ray else None,
            "ray_frontend": {"yaw": self.request.ray_yaw, "pitch": self.request.ray_pitch},
            "hotspot_overlay_id": self.request.hotspot_overlay_id,
            "hotspot_match": self.hotspot_match,
            "faces": list(self.view.faces_used) if self.view else None,
            "segment": {"source": self.segment_source, "score": self.segment_score, "area": self.segment_area,
                        "proposals": [list(c) for c in self.segment_candidates], "vertices": len(self.mask_polygon)},
            "candidates": [{**c.payload(), "raw_score": c.raw_score, "in_tour": c.in_tour, "in_panorama": c.in_panorama} for c in self.candidates],
            "chosen": self.chosen.artifact_id if self.chosen else None,
            "threshold": threshold,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "error": self.error,
        }


def normalized_score(knn_score: float | None) -> float:
    """O OpenSearch (lucene, cosinesimil) devolve (1 + cos)/2; o score que o
    frontend mostra como «% de correspondência» é o cosseno, cortado a 0..1."""
    if not isinstance(knn_score, (int, float)):
        return 0.0
    return float(min(max(2.0 * float(knn_score) - 1.0, 0.0), 1.0))


class PointAskService:
    """O pipeline, partilhado pelo ChatService; o SAM e as faces vivem aqui."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.faces = CubeFaceStore(
            base_url=settings.POINT_TOURS_BASE_URL,
            cache_dir=settings.point_tiles_cache_dir_resolved,
            tiles_per_side=int(settings.POINT_FACE_TILES),
            level=int(settings.POINT_FACE_LEVEL),
        )
        # as faces no nível mais fino só para a vista apertada do recorte (uma face de 3072 px são 28 MB)
        self.zoom_faces = CubeFaceStore(
            base_url=settings.POINT_TOURS_BASE_URL,
            cache_dir=settings.point_tiles_cache_dir_resolved,
            tiles_per_side=int(settings.POINT_CROP_FACE_TILES),
            level=int(settings.POINT_CROP_FACE_LEVEL),
            memory_faces=12,
        )
        self.segmenter = PointSegmenter(model_id=settings.POINT_SAM_MODEL, device=settings.POINT_SAM_DEVICE)
        self.concepts = ConceptSegmenter(
            model_id=settings.POINT_SAM_MODEL, device=settings.POINT_SAM_DEVICE,
            prompts=tuple(settings.POINT_CONCEPT_PROMPTS.split(",")), threshold=float(settings.POINT_CONCEPT_THRESHOLD),
        )
        self.assets = PointAssetStore(ttl_seconds=float(settings.POINT_ASSET_TTL_SECONDS))
        self._vlm_client: Any | None = None

    @property
    def ready(self) -> bool:
        return self.segmenter.ready

    def load(self) -> bool:
        ok = self.segmenter.load()
        if ok and not self.concepts.load():
            logger.warning("point.sam3.concept unavailable: only the point prompt will be used")
        return ok

    # --- 1. a vista ---------------------------------------------------------

    async def render(self, request: PointRequest, *, museum_slug: str) -> ProjectedView:
        return await asyncio.to_thread(
            project_view, self.faces, museum=museum_slug, panorama_key=request.panorama_key,
            camera=request.camera, width=int(self.settings.POINT_VIEW_WIDTH),
        )

    # --- 2. o recorte -------------------------------------------------------

    async def cut(self, outcome: PointOutcome, *, museum_slug: str) -> None:
        """A máscara do objeto na vista geral vem do SAM 3 por conceito (a peça
        inteira que contém o ponto) e, sem conceito, do SAM por ponto. Depois
        projeta-se uma vista apertada à volta da máscara, com as faces no nível
        mais fino: é essa vista que vai ao embedding (crop) e, escurecida fora
        da máscara, ao visitante (cutout). O polígono fica na vista geral.
        Medido em 24 toques no MNAA (2026-10-16): o recorte apertado sobe o
        cosseno das peças certas de ~0,6 para ~0,9."""
        view = outcome.view
        assert view is not None
        request = outcome.request
        height, width = view.image.shape[:2]
        x = int(round(request.point_x * width))
        y = int(round(request.point_y * height))
        outcome.point_px = (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
        started = time.perf_counter()
        concept = await asyncio.to_thread(self.concepts.find_sync, view.image, *outcome.point_px) if self.concepts.ready else None
        outcome.timings_ms["sam"] = concept.elapsed_ms if concept else 0.0
        if concept is not None:
            mask = concept.mask
            outcome.segment_score, outcome.segment_area = concept.score, concept.area_fraction
            outcome.segment_candidates = ((concept.score, concept.area_fraction),)
            outcome.segment_source = f"concept:{concept.prompt}"
        else:
            segment = await self.segmenter.segment(view.image, *outcome.point_px, min_area=0.01, max_area=0.40)
            outcome.timings_ms["sam"] += segment.elapsed_ms
            mask = segment.mask
            outcome.segment_score, outcome.segment_area, outcome.segment_candidates = segment.score, segment.area_fraction, segment.candidates
            outcome.segment_source = "point"
        crop_image, crop_mask = view.image, mask
        try:
            zoom_camera = self._zoom_camera(request.camera, mask, margin=float(self.settings.POINT_CROP_MARGIN))
            zoom_started = time.perf_counter()
            zoom = await asyncio.to_thread(
                project_view, self.zoom_faces, museum=museum_slug, panorama_key=request.panorama_key,
                camera=zoom_camera, width=int(self.settings.POINT_CROP_WIDTH),
            )
            zh, zw = zoom.image.shape[:2]
            crop_image = zoom.image
            crop_mask = mask_in_camera(mask, request.camera, zoom_camera, zw, zh)
            outcome.timings_ms["zoom"] = (time.perf_counter() - zoom_started) * 1000
        except Exception:
            logger.warning("point.zoom.failed", exc_info=True)
        polygon, cutout, view_jpeg = await asyncio.to_thread(self._polygon_and_cutout, mask, crop_image, crop_mask, view.image)
        outcome.timings_ms["cutout"] = (time.perf_counter() - started) * 1000 - outcome.timings_ms["sam"] - outcome.timings_ms.get("zoom", 0.0)
        outcome.mask_polygon = polygon
        outcome.crop_jpeg = cutout.crop_jpeg
        self.assets.put(request.request_id, cutout.crop_jpeg, cutout.cutout_jpeg,
                        museum_slug=museum_slug, panorama_key=request.panorama_key, view_jpeg=view_jpeg)
        prefix = f"{self.settings.API_PREFIX}/chat/point/{request.request_id}"
        outcome.crop_url, outcome.cutout_url = f"{prefix}/crop.jpg", f"{prefix}/cutout.jpg"
        self._debug_dump(outcome, cutout.cutout_jpeg)

    @staticmethod
    def _zoom_camera(camera: Camera, mask: np.ndarray, *, margin: float = 0.16, min_fov: float = 3.0) -> Camera:
        """Câmara apertada à volta da máscara: o centro na direção do centro da
        caixa, o hfov a cobrir a caixa com folga, a proporção da própria caixa."""
        ys, xs = np.nonzero(mask)
        height, width = mask.shape
        if len(xs) == 0:
            raise ValueError("máscara vazia")
        x0, x1 = xs.min() / width, (xs.max() + 1) / width
        y0, y1 = ys.min() / height, (ys.max() + 1) / height
        corners = [ray_from_point(camera, x, y) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
        centre = ray_from_point(camera, (x0 + x1) / 2, (y0 + y1) / 2)
        yaw_span = max(abs(((yaw - centre[0]) + 180) % 360 - 180) for yaw, _ in corners) * 2
        pitch_span = max(abs(pitch - centre[1]) for _, pitch in corners) * 2
        hfov = min(max(yaw_span * (1 + margin), min_fov), 120.0)
        vfov = min(max(pitch_span * (1 + margin), min_fov), 120.0)
        # a altura em px da vista apertada segue a proporção angular (tan) da caixa
        aspect = math.tan(math.radians(vfov) / 2) / math.tan(math.radians(hfov) / 2)
        return Camera(yaw=centre[0], pitch=centre[1], hfov=hfov, width=1000, height=max(int(round(1000 * aspect)), 100))

    @staticmethod
    def _polygon_and_cutout(view_mask: np.ndarray, crop_image: np.ndarray, crop_mask: np.ndarray, view_image: np.ndarray):
        from app.services.point_segmenter import _jpeg

        # o crop é a vista apertada inteira (já tem a folga); o cutout escurece fora da máscara
        return mask_polygon(view_mask), crop_and_cutout(crop_image, crop_mask, margin=0.0, whole=True), _jpeg(view_image, 80)

    def _debug_dump(self, outcome: PointOutcome, cutout_jpeg: bytes) -> None:
        folder = (self.settings.POINT_DEBUG_DIR or "").strip()
        if not folder or outcome.view is None:
            return
        try:
            from PIL import Image, ImageDraw

            target = Path(folder).expanduser()
            target.mkdir(parents=True, exist_ok=True)
            stem = outcome.request.request_id
            view = Image.fromarray(outcome.view.image)
            draw = ImageDraw.Draw(view)
            px, py = outcome.point_px
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], outline=(255, 0, 0), width=3)
            if outcome.mask_polygon:
                w, h = view.size
                draw.polygon([(x * w, y * h) for x, y in outcome.mask_polygon], outline=(255, 211, 107), width=3)
            view.save(target / f"{stem}_vista.jpg", quality=85)
            (target / f"{stem}_crop.jpg").write_bytes(outcome.crop_jpeg or b"")
            (target / f"{stem}_cutout.jpg").write_bytes(cutout_jpeg)
        except Exception:
            logger.warning("point.debug_dump.failed", exc_info=True)

    # --- 3. a pesquisa ------------------------------------------------------

    def rank(
        self,
        hits: list[dict[str, Any]],
        *,
        in_tour_keys: set[str],
        panorama_keys: set[str],
        normalize_key,
    ) -> list[Candidate]:
        """Agrupa as imagens por peça (a melhor), normaliza o score e aplica os
        boosts da visita e do panorama; devolve os melhores por ordem."""
        best: dict[str, Candidate] = {}
        for hit in hits:
            artifact_id = str(hit.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            raw = normalized_score(hit.get("score"))
            inventory = str(hit.get("inventory_number") or hit.get("inventory") or "").strip() or None
            key = normalize_key(inventory) if inventory else ""
            in_panorama = bool(key) and key in panorama_keys
            in_tour = in_panorama or (bool(key) and key in in_tour_keys)
            score = raw
            if in_panorama:
                score *= float(self.settings.POINT_PANORAMA_BOOST)
            elif in_tour:
                score *= float(self.settings.POINT_IN_TOUR_BOOST)
            score = min(score, 1.0)
            current = best.get(artifact_id)
            if current is None or score > current.score:
                best[artifact_id] = Candidate(
                    artifact_id=artifact_id,
                    inventory_number=inventory,
                    title=str(hit.get("artifact_title") or hit.get("title") or "").strip() or None,
                    score=score,
                    image_name=str(hit.get("local_path") or hit.get("original_image_name") or "").replace("\\", "/").strip("/") or None,
                    in_tour=in_tour,
                    in_panorama=in_panorama,
                    raw_score=round(raw, 4),
                )
        ordered = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return ordered[: max(int(self.settings.POINT_CANDIDATES), 1)]

    # --- 5. «Descreve-me o que vês» -----------------------------------------

    def _vlm(self):
        if self._vlm_client is None:
            from openai import AsyncOpenAI

            self._vlm_client = AsyncOpenAI(
                api_key=self.settings.POINT_VLM_API_KEY or "no-key",
                base_url=self.settings.POINT_VLM_BASE_URL.rstrip("/"),
                timeout=float(self.settings.POINT_VLM_TIMEOUT_SECONDS),
                max_retries=0,
            )
        return self._vlm_client

    def _vlm_reachable(self, timeout: float = 1.0) -> bool:
        """Liga ao porto do VLM antes do pedido: no WSL um porto sem ninguém à
        escuta não recusa logo, fica pendurado até ao timeout de 30 s."""
        parsed = urlparse(self.settings.POINT_VLM_BASE_URL)
        host, port = parsed.hostname or "127.0.0.1", parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    async def describe_crop(self, crop_jpeg: bytes, *, prompt: str, system_prompt: str, on_delta=None) -> str:
        """Descrição do recorte pelo modelo de visão (vLLM, API OpenAI), em stream."""
        if not await asyncio.to_thread(self._vlm_reachable):
            raise ConnectionError(f"modelo de visão sem resposta em {self.settings.POINT_VLM_BASE_URL}")
        data_url = "data:image/jpeg;base64," + base64.b64encode(crop_jpeg).decode("ascii")
        stream = await self._vlm().chat.completions.create(
            model=self.settings.POINT_VLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": prompt}]},
            ],
            max_tokens=220,
            temperature=0.2,
            stream=True,
        )
        parts: list[str] = []
        async for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if delta:
                parts.append(delta)
                if on_delta is not None:
                    await on_delta(delta)
        return "".join(parts).strip()


def compute_ray(request: PointRequest) -> tuple[float, float]:
    return ray_from_point(request.camera, request.point_x, request.point_y)


def hotspot_in_view(request: PointRequest, yaw: float, pitch: float) -> tuple[float, float] | None:
    return point_in_view(request.camera, yaw, pitch)


@lru_cache(maxsize=1)
def get_point_ask_service() -> PointAskService:
    return PointAskService(get_settings())
