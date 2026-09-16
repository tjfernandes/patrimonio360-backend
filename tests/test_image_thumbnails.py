"""Miniaturas `?w=` em /api/v1/chat/images (CHAT_IMAGE_THUMBNAILS).

    python -m unittest tests.test_image_thumbnails
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app.api.routes import chat as chat_routes  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.services.image_thumbnails import (  # noqa: E402
    ensure_thumbnail,
    snap_thumbnail_width,
    thumbnail_path_for,
)


def _write_image(path: Path, size: tuple[int, int], mode: str = "RGB", **save_kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = (120, 30, 40, 128) if mode == "RGBA" else (120, 30, 40)
    Image.new(mode, size, color).save(path, **save_kwargs)
    return path


class TestSnapWidth(unittest.TestCase):
    def test_rounds_up_to_supported_widths(self) -> None:
        self.assertEqual(snap_thumbnail_width(480), 480)
        self.assertEqual(snap_thumbnail_width(481), 640)
        self.assertEqual(snap_thumbnail_width(1), 160)
        self.assertIsNone(snap_thumbnail_width(None))
        self.assertIsNone(snap_thumbnail_width(0))
        self.assertIsNone(snap_thumbnail_width(5000))


class TestEnsureThumbnail(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cache = self.root / "cache"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_large_jpeg_is_resized_and_cached(self) -> None:
        source = _write_image(self.root / "a.jpg", (1600, 1200), quality=95)
        thumb = ensure_thumbnail(source, 480, self.cache)
        self.assertIsNotNone(thumb)
        with Image.open(thumb) as image:
            self.assertEqual(image.size, (480, 360))
            self.assertEqual(image.format, "JPEG")
        mtime = thumb.stat().st_mtime_ns
        self.assertEqual(ensure_thumbnail(source, 480, self.cache), thumb)
        self.assertEqual(thumb.stat().st_mtime_ns, mtime)
        self.assertEqual(list(self.cache.rglob("*.part")), [])

    def test_narrow_image_serves_original_and_leaves_marker(self) -> None:
        source = _write_image(self.root / "narrow.jpg", (300, 500))
        self.assertIsNone(ensure_thumbnail(source, 480, self.cache))
        self.assertTrue(thumbnail_path_for(source, 480, self.cache).with_suffix(".orig").is_file())
        self.assertIsNone(ensure_thumbnail(source, 480, self.cache))

    def test_png_with_alpha_becomes_rgb_jpeg(self) -> None:
        source = _write_image(self.root / "alpha.png", (1000, 1000), mode="RGBA")
        thumb = ensure_thumbnail(source, 160, self.cache)
        with Image.open(thumb) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (160, 160))

    def test_changed_source_gets_new_cache_entry(self) -> None:
        source = _write_image(self.root / "b.jpg", (1200, 600))
        first = ensure_thumbnail(source, 320, self.cache)
        _write_image(source, (1400, 700), quality=60)
        second = ensure_thumbnail(source, 320, self.cache)
        self.assertNotEqual(first, second)

    def test_unsupported_suffix_is_ignored(self) -> None:
        source = self.root / "file.gif"
        Image.new("RGB", (900, 900)).save(source)
        self.assertIsNone(ensure_thumbnail(source, 480, self.cache))


class TestImageRoute(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.images = root / "Images"
        _write_image(self.images / "mnt" / "big.jpg", (1600, 1000), quality=95)
        self.cache = root / "thumbs"
        app = FastAPI()
        app.include_router(chat_routes.router, prefix="/api/v1/chat")
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _settings(self, enabled: bool) -> Settings:
        return Settings(
            _env_file=None, IMAGE_ASSET_ROOT=str(self.images), CHAT_IMAGE_THUMBNAILS=enabled,
            IMAGE_THUMBNAIL_CACHE_DIR=str(self.cache), QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False,
        )

    def _get(self, enabled: bool, url: str):
        with mock.patch.object(chat_routes, "get_settings", return_value=self._settings(enabled)):
            return self.client.get(url)

    def test_flag_on_serves_thumbnail_with_cache_header(self) -> None:
        original = self._get(True, "/api/v1/chat/images/mnt/big.jpg")
        thumb = self._get(True, "/api/v1/chat/images/mnt/big.jpg?w=480&retry=1")
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.headers["content-type"], "image/jpeg")
        self.assertIn("max-age=86400", thumb.headers.get("cache-control", ""))
        self.assertLess(len(thumb.content), len(original.content))
        with Image.open(self.cache.rglob("*.jpg").__next__()) as image:
            self.assertEqual(image.width, 480)

    def test_flag_off_ignores_width(self) -> None:
        original = self._get(False, "/api/v1/chat/images/mnt/big.jpg")
        with_w = self._get(False, "/api/v1/chat/images/mnt/big.jpg?w=480")
        self.assertEqual(with_w.status_code, 200)
        self.assertEqual(with_w.content, original.content)
        self.assertNotIn("cache-control", with_w.headers)
        self.assertFalse(self.cache.exists())

    def test_path_traversal_still_rejected(self) -> None:
        response = self._get(True, "/api/v1/chat/images/..%2F..%2Fetc%2Fpasswd?w=160")
        self.assertIn(response.status_code, {400, 404})


if __name__ == "__main__":
    unittest.main()
