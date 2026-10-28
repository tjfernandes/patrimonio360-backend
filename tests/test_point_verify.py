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

from app.services.point_verify import PointVerifier, VerifyResult, grid_data_url, parse_answer, parse_pair, resolve_image_path


class TestParsePair(unittest.TestCase):
    def test_json_and_loose(self) -> None:
        self.assertEqual(parse_pair('{"same": true, "confidence": 0.9, "reason": "mesma moldura"}'), (True, 0.9, "mesma moldura"))
        self.assertEqual(parse_pair('```json\n{"same": false, "confidence": 0.95, "reason": "hexagonal vs oval"}\n```'), (False, 0.95, "hexagonal vs oval"))
        self.assertEqual(parse_pair('{"same": "sim", "confidence": "0.7"}'), (True, 0.7, ""))
        self.assertEqual(parse_pair("Não, são peças diferentes.")[0], False)
        self.assertEqual(parse_pair("")[0], False)


class TestPairwiseDecision(unittest.TestCase):
    def _verifier(self, answers):
        verifier = PointVerifier(base_url="http://127.0.0.1:1/v1", model="m")
        verifier.reachable = lambda timeout=0.5: True   # type: ignore[method-assign]

        async def pair(crop_url, photo):
            return answers[photo.name]
        verifier._pair = pair   # type: ignore[method-assign]
        return verifier

    def test_single_yes_wins(self) -> None:
        verifier = self._verifier({"a.jpg": (False, 0.9, "oval vs hexagonal"), "b.jpg": (True, 0.8, "mesma cena"), "c.jpg": (False, 0.9, "x")})
        result = asyncio.run(verifier.verify(b"\xff\xd8", [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]))
        self.assertEqual((result.status, result.match, result.confidence), ("ok", 1, 0.8))

    def test_no_yes_is_none_with_reasons(self) -> None:
        verifier = self._verifier({"a.jpg": (False, 0.9, "oval vs hexagonal"), "b.jpg": (False, 0.95, "outra cena")})
        result = asyncio.run(verifier.verify(b"\xff\xd8", [Path("a.jpg"), Path("b.jpg")]))
        self.assertEqual((result.status, result.match), ("ok", None))
        self.assertIn("oval vs hexagonal", result.reason)

    def test_several_yes_takes_the_most_confident_then_the_first(self) -> None:
        verifier = self._verifier({"a.jpg": (True, 0.8, "x"), "b.jpg": (True, 0.95, "y"), "c.jpg": (True, 0.95, "z")})
        result = asyncio.run(verifier.verify(b"\xff\xd8", [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]))
        self.assertEqual(result.match, 1)
        self.assertTrue(result.reason.startswith("3 candidatas «sim»"))
        self.assertIsInstance(result, VerifyResult)


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
