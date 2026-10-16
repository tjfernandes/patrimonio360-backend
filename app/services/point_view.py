"""Reconstrução da vista da visita no servidor («Apontar e perguntar»).

Não há captura de ecrã: a visita manda o panorama, a câmara (yaw, pitch, hfov,
como o PanoramaPlayer do 3DVista: yaw cresce para a direita, pitch para cima) e
o tamanho da vista, e aqui projeta-se a vista retilínea a partir das faces do
cubo publicadas em media/<panorama_key>_0/{f,b,l,r,u,d}/{nível}/{row}_{col}.webp.

Projeção: f = (W/2)/tan(hfov/2); o raio do píxel (u, v) em coordenadas de câmara
é ((u−W/2)/f, −(v−H/2)/f, 1), rodado pelo pitch (eixo x) e depois pelo yaw
(eixo y). A face e a coordenada no cubo saem do maior componente do vetor.
Tudo em NumPy, sem Python por píxel; as faces ficam em cache em disco (tiles)
e em memória (face montada).
"""

from __future__ import annotations

import logging
import math
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)

FACES = ("f", "r", "b", "l", "u", "d")
TILE = 512
# frente, direita, trás, esquerda, cima, baixo: (eixo da face, «direita» da face, «cima» da face)
_FACE_AXES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = {
    "f": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "r": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "b": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "l": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "u": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "d": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}


@dataclass(frozen=True)
class Camera:
    yaw: float      # graus, cresce para a direita
    pitch: float    # graus, cresce para cima
    hfov: float     # graus
    width: int      # px da vista
    height: int

    @property
    def focal(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov) / 2.0)

    @property
    def vfov(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.focal))


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Câmara → mundo: primeiro o pitch (eixo x), depois o yaw (eixo y)."""
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]], dtype=np.float64)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]], dtype=np.float64)
    return ry @ rx


def ray_directions(camera: Camera, width: int, height: int) -> np.ndarray:
    """Direções (H, W, 3) no mundo dos píxeis de uma imagem width×height com a
    proporção da vista (a focal escala com a largura)."""
    focal = camera.focal * (width / camera.width)
    us = (np.arange(width, dtype=np.float64) + 0.5 - width / 2.0) / focal
    vs = -(np.arange(height, dtype=np.float64) + 0.5 - height / 2.0) / focal
    x, y = np.meshgrid(us, vs)
    cam = np.stack([x, y, np.ones_like(x)], axis=-1)
    world = cam @ _rotation(camera.yaw, camera.pitch).T
    return world / np.linalg.norm(world, axis=-1, keepdims=True)


def ray_from_point(camera: Camera, x: float, y: float) -> tuple[float, float]:
    """(yaw, pitch) em graus da direção do ponto (x, y) em 0..1 da vista."""
    u = (x - 0.5) * camera.width / camera.focal
    v = -(y - 0.5) * camera.height / camera.focal
    d = _rotation(camera.yaw, camera.pitch) @ np.array([u, v, 1.0])
    d = d / np.linalg.norm(d)
    return math.degrees(math.atan2(d[0], d[2])), math.degrees(math.asin(float(np.clip(d[1], -1, 1))))


def point_in_view(camera: Camera, yaw: float, pitch: float) -> tuple[float, float] | None:
    """(x, y) em 0..1 da vista onde cai a direção (yaw, pitch); None se está atrás."""
    p, y_ = math.radians(pitch), math.radians(yaw)
    d = np.array([math.cos(p) * math.sin(y_), math.sin(p), math.cos(p) * math.cos(y_)])
    cam = _rotation(camera.yaw, camera.pitch).T @ d
    if cam[2] <= 1e-6:
        return None
    u = cam[0] / cam[2] * camera.focal
    v = cam[1] / cam[2] * camera.focal
    return float(u / camera.width + 0.5), float(0.5 - v / camera.height)


def face_coordinates(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Para cada direção: índice da face (ordem de FACES) e (s, t) em 0..1 na face."""
    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.full(x.shape, -1, dtype=np.int8)
    s = np.zeros(x.shape, dtype=np.float64)
    t = np.zeros(x.shape, dtype=np.float64)
    for index, name in enumerate(FACES):
        fwd, right, up = (np.array(v, dtype=np.float64) for v in _FACE_AXES[name])
        depth = directions @ fwd
        if name in ("f", "b"):
            chosen = (az >= ax) & (az >= ay) & (depth > 0)
        elif name in ("r", "l"):
            chosen = (ax > az) & (ax >= ay) & (depth > 0)
        else:
            chosen = (ay > ax) & (ay > az) & (depth > 0)
        if not chosen.any():
            continue
        d = np.where(chosen, depth, 1.0)
        s = np.where(chosen, (directions @ right) / d, s)
        t = np.where(chosen, (directions @ up) / d, t)
        face = np.where(chosen, index, face)
    return face, (s + 1.0) / 2.0, (1.0 - t) / 2.0


class CubeFaceStore:
    """Faces do cubo de cada panorama: tiles descarregados da visita publicada
    (ou de uma cópia local) para `cache_dir`, montados numa imagem por face e
    guardados em memória (LRU)."""

    def __init__(
        self,
        *,
        base_url: str,
        cache_dir: Path | str,
        tiles_per_side: int = 3,
        level: int = 1,
        memory_faces: int = 48,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.tiles_per_side = max(int(tiles_per_side), 1)
        self.level = int(level)
        self.timeout = timeout
        self._faces: OrderedDict[tuple[str, str, str], np.ndarray] = OrderedDict()
        self._memory_faces = max(int(memory_faces), 1)
        self._lock = threading.Lock()

    @property
    def face_size(self) -> int:
        return self.tiles_per_side * TILE

    def _tile_path(self, museum: str, panorama_key: str, face: str, level: int, row: int, col: int, ext: str) -> Path:
        return self.cache_dir / museum / "media" / f"{panorama_key}_0" / face / str(level) / f"{row}_{col}.{ext}"

    def _fetch_tile(self, museum: str, panorama_key: str, face: str, level: int, row: int, col: int) -> bytes | None:
        for ext in ("webp", "jpg"):
            path = self._tile_path(museum, panorama_key, face, level, row, col, ext)
            if path.is_file():
                return path.read_bytes()
        for ext in ("webp", "jpg"):
            url = f"{self.base_url}/{museum}/media/{panorama_key}_0/{face}/{level}/{row}_{col}.{ext}"
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as response:
                    data = response.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            path = self._tile_path(museum, panorama_key, face, level, row, col, ext)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return data
        return None

    def prefetch(self, museum: str, panorama_key: str, faces: tuple[str, ...] = FACES) -> int:
        """Descarrega os tiles das faces para o disco (sem montar); devolve quantos foram buscados."""
        n = self.tiles_per_side
        jobs = [(face, row, col) for face in faces for row in range(n) for col in range(n)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda j: self._fetch_tile(museum, panorama_key, j[0], self.level, j[1], j[2]) is not None, jobs))
        return sum(results)

    def _assemble(self, museum: str, panorama_key: str, face: str) -> np.ndarray:
        n = self.tiles_per_side
        canvas = np.zeros((n * TILE, n * TILE, 3), dtype=np.uint8)
        jobs = [(row, col) for row in range(n) for col in range(n)]
        # os tiles descarregam-se em paralelo: 36 por face no nível 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            tiles = list(pool.map(lambda rc: self._fetch_tile(museum, panorama_key, face, self.level, rc[0], rc[1]), jobs))
        for (row, col), data in zip(jobs, tiles):
            if data is None:
                raise FileNotFoundError(f"tile {face}/{self.level}/{row}_{col} de {museum}/{panorama_key}")
            tile = np.asarray(Image.open(BytesIO(data)).convert("RGB"))
            h, w = tile.shape[:2]
            canvas[row * TILE: row * TILE + h, col * TILE: col * TILE + w] = tile[:TILE, :TILE]
        return canvas

    def face(self, museum: str, panorama_key: str, face: str) -> np.ndarray:
        key = (museum, panorama_key, face)
        with self._lock:
            cached = self._faces.get(key)
            if cached is not None:
                self._faces.move_to_end(key)
                return cached
        image = self._assemble(museum, panorama_key, face)
        with self._lock:
            self._faces[key] = image
            self._faces.move_to_end(key)
            while len(self._faces) > self._memory_faces:
                self._faces.popitem(last=False)
        return image


def mask_in_camera(mask: np.ndarray, src: Camera, dst: Camera, width: int, height: int) -> np.ndarray:
    """Reamostra uma máscara (H×W, na vista `src`) para uma vista `dst` de
    width×height: cada píxel de `dst` olha na sua direção e vai buscar o valor
    ao píxel correspondente de `src` (fora da vista `src` fica False)."""
    src_h, src_w = mask.shape
    directions = ray_directions(dst, width, height).reshape(-1, 3)
    cam = directions @ _rotation(src.yaw, src.pitch)  # mundo → câmara de origem (R^T d)
    focal = src.focal * (src_w / src.width)
    z = np.where(cam[:, 2] > 1e-6, cam[:, 2], np.inf)
    px = np.rint(cam[:, 0] / z * focal + src_w / 2.0 - 0.5).astype(np.int64)
    py = np.rint(-cam[:, 1] / z * focal + src_h / 2.0 - 0.5).astype(np.int64)
    inside = (cam[:, 2] > 1e-6) & (px >= 0) & (px < src_w) & (py >= 0) & (py < src_h)
    result = np.zeros(len(directions), dtype=bool)
    result[inside] = mask[py[inside], px[inside]]
    return result.reshape(height, width)


@dataclass(frozen=True)
class ProjectedView:
    image: np.ndarray            # (H, W, 3) uint8
    camera: Camera
    faces_used: tuple[str, ...]
    elapsed_ms: float


def project_view(
    store: CubeFaceStore,
    *,
    museum: str,
    panorama_key: str,
    camera: Camera,
    width: int = 1024,
) -> ProjectedView:
    """A vista retilínea da câmara, com `width` px e a proporção da viewport."""
    started = time.perf_counter()
    width = max(int(width), 64)
    height = max(int(round(width * camera.height / camera.width)), 64)
    directions = ray_directions(camera, width, height)
    face_index, s, t = face_coordinates(directions)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    used: list[str] = []
    size = store.face_size
    for index, name in enumerate(FACES):
        mask = face_index == index
        if not mask.any():
            continue
        used.append(name)
        face_image = store.face(museum, panorama_key, name)
        rows = np.clip(t[mask] * size - 0.5, 0, size - 1)
        cols = np.clip(s[mask] * size - 0.5, 0, size - 1)
        for channel in range(3):
            output[..., channel][mask] = map_coordinates(
                face_image[..., channel], [rows, cols], order=1, mode="nearest"
            ).astype(np.uint8)
    return ProjectedView(image=output, camera=camera, faces_used=tuple(used), elapsed_ms=(time.perf_counter() - started) * 1000)
