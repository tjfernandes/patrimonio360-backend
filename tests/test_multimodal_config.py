"""Validação das flags multimodais (Fase 3, Etapa 2) — offline.

    python -m unittest tests.test_multimodal_config
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402


def make_settings(**overrides):
    # _env_file=None: valida apenas defaults + overrides, sem ler o .env local.
    return Settings(_env_file=None, **overrides)


class TestMultimodalDefaults(unittest.TestCase):
    def test_defaults_are_conservative_and_off(self) -> None:
        s = make_settings()
        self.assertEqual(s.MULTIMODAL_RETRIEVAL_MODE, "off")
        self.assertEqual(s.MULTIMODAL_RRF_K, 60)
        self.assertEqual(s.MULTIMODAL_ARTIFACT_WEIGHT, 1.0)
        self.assertEqual(s.MULTIMODAL_IMAGE_WEIGHT, 0.7)
        self.assertEqual(s.MULTIMODAL_MIN_IMAGE_SCORE, 0.64)
        self.assertEqual(s.MULTIMODAL_IMAGE_TOP_K, 30)
        self.assertFalse(s.MULTIMODAL_DEBUG)

    def test_mode_is_normalized(self) -> None:
        self.assertEqual(
            make_settings(MULTIMODAL_RETRIEVAL_MODE=" Intent ").MULTIMODAL_RETRIEVAL_MODE,
            "intent",
        )
        self.assertEqual(
            make_settings(MULTIMODAL_RETRIEVAL_MODE="ALWAYS").MULTIMODAL_RETRIEVAL_MODE,
            "always",
        )


class TestMultimodalValidation(unittest.TestCase):
    def test_invalid_mode_fails_at_startup(self) -> None:
        with self.assertRaises(Exception):
            make_settings(MULTIMODAL_RETRIEVAL_MODE="banana")

    def test_invalid_numbers_fail_at_startup(self) -> None:
        for overrides in (
            {"MULTIMODAL_RRF_K": 0},
            {"MULTIMODAL_ARTIFACT_WEIGHT": -0.1},
            {"MULTIMODAL_IMAGE_WEIGHT": -1},
            {"MULTIMODAL_MIN_IMAGE_SCORE": 1.5},
            {"MULTIMODAL_MIN_IMAGE_SCORE": -0.2},
            {"MULTIMODAL_IMAGE_TOP_K": 0},
            {"MULTIMODAL_IMAGE_TOP_K": 10_000},
        ):
            with self.assertRaises(Exception, msg=str(overrides)):
                make_settings(**overrides)


if __name__ == "__main__":
    unittest.main()
