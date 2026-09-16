"""Miniaturas das imagens do acervo servidas com `?w=` (CHAT_IMAGE_THUMBNAILS).

O widget mostra os resultados em mosaicos de ~240 px de largura, mas o backend
servia sempre o original. Medido a 2026-09-16 sobre 60 imagens aleatórias das
35 757 de `Images/`: mediana de 191 KB por original; a 480 px de largura, JPEG
q=82, o conjunto desce para 14 % dos bytes (mediana 34 KB), gerado em 7,5 ms
(mediana) na primeira vez e servido do disco a seguir.

A largura pedida é arredondada para cima para um conjunto fixo, para que a
cache em disco não cresça com larguras arbitrárias. Imagens que já são
estreitas, formatos não suportados ou falhas do Pillow devolvem None: a rota
serve então o original, como antes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

THUMBNAIL_WIDTHS: tuple[int, ...] = (160, 240, 320, 480, 640, 960, 1280)
RESIZABLE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
JPEG_QUALITY = 82


def snap_thumbnail_width(requested: int | None) -> int | None:
    """Menor largura suportada >= pedida; None quando não há miniatura a fazer."""
    if requested is None or requested <= 0:
        return None
    for width in THUMBNAIL_WIDTHS:
        if requested <= width:
            return width
    return None


def _cache_key(source: Path, width: int) -> str:
    stat = source.stat()
    raw = f"{source.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{width}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def thumbnail_path_for(source: Path, width: int, cache_dir: Path) -> Path:
    digest = _cache_key(source, width)
    return cache_dir / digest[:2] / f"{digest}.jpg"


def _atomic_write(target: Path, write) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False)
    try:
        with handle:
            write(handle)
        os.replace(handle.name, target)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def ensure_thumbnail(source: Path, width: int, cache_dir: Path) -> Path | None:
    """Caminho da miniatura em cache (gera-a se faltar) ou None para servir o original."""
    if source.suffix.lower() not in RESIZABLE_SUFFIXES:
        return None
    target = thumbnail_path_for(source, width, cache_dir)
    if target.is_file():
        return target
    # Marcador para imagens já estreitas: evita reabri-las a cada pedido.
    narrow_marker = target.with_suffix(".orig")
    if narrow_marker.is_file():
        return None

    from PIL import Image, ImageOps

    with Image.open(source) as opened:
        # O browser aplica a orientação EXIF ao original; a miniatura também.
        image = ImageOps.exif_transpose(opened)
        if image.width <= width:
            _atomic_write(narrow_marker, lambda handle: None)
            return None
        if image.mode in {"RGBA", "LA", "P"}:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        elif image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.thumbnail((width, 1_000_000), Image.Resampling.LANCZOS)
        _atomic_write(
            target,
            lambda handle: image.save(handle, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True),
        )
    return target
