"""Confirmação da peça pelo modelo de visão (app/services/point_verify.py): leitura da resposta,
resolução das fotos e a grelha.

    python -m unittest tests.test_point_verify
"""
from __future__ import annotations

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.services.point_verify import PointVerifier, grid_data_url, parse_answer, resolve_image_path


class TestParseAnswer(unittest.TestCase):
    def test_json_plain_and_fenced(self) -> None:
        self.assertEqual(parse_answer('{"match": "B", "confidence": 0.9, "reason": "mesma moldura"}', 3), (1, 0.9, "mesma moldura"))
        self.assertEqual(parse_answer('```json\n{"match":"none","confidence":0.95,"reason":"peças diferentes"}\n```', 3), (None, 0.95, "peças diferentes"))

    def test_letter_outside_the_shown_ones_is_none(self) -> None:
        self.assertEqual(parse_answer('{"match": "E", "confidence": 0.9}', 3)[0], None)
        self.assertEqual(parse_answer("não sei", 3)[0], None)
        self.assertEqual(parse_answer('{"match": "a", "confidence": "0.8"}', 3), (0, 0.8, ""))


class TestImagesAndGrid(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "Images"
        (self.root / "museu" / "obj_1").mkdir(parents=True)
        Image.new("RGB", (40, 60), (200, 30, 30)).save(self.root / "museu" / "obj_1" / "a.jpg")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_resolve_strips_the_root_name_and_stays_inside(self) -> None:
        self.assertEqual(resolve_image_path(self.root, "Images/museu/obj_1/a.jpg"), (self.root / "museu/obj_1/a.jpg").resolve())
        self.assertEqual(resolve_image_path(self.root, "museu/obj_1/a.jpg"), (self.root / "museu/obj_1/a.jpg").resolve())
        self.assertIsNone(resolve_image_path(self.root, "Images/museu/obj_1/nao_existe.jpg"))
        self.assertIsNone(resolve_image_path(self.root, "../../etc/passwd"))
        self.assertIsNone(resolve_image_path(None, "Images/x.jpg"))
        self.assertIsNone(resolve_image_path(self.root, ""))

    def test_grid_is_a_jpeg_data_url(self) -> None:
        url = grid_data_url([self.root / "museu/obj_1/a.jpg"] * 3, cell=80)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        raw = base64.b64decode(url.split(",", 1)[1])
        self.assertTrue(raw.startswith(b"\xff\xd8"))


class TestVerifierWithoutServer(unittest.TestCase):
    def test_unreachable_port_is_unavailable_fast(self) -> None:
        verifier = PointVerifier(base_url="http://127.0.0.1:1/v1", model="m", timeout_seconds=1.0)
        result = asyncio.run(verifier.verify(b"\xff\xd8", [Path("/tmp/x.jpg")]))
        self.assertEqual(result.status, "unavailable")
        self.assertIsNone(result.match)

    def test_no_photos_is_an_error(self) -> None:
        verifier = PointVerifier(base_url="http://127.0.0.1:1/v1", model="m")
        self.assertEqual(asyncio.run(verifier.verify(b"\xff\xd8", [])).status, "error")


if __name__ == "__main__":
    unittest.main()
