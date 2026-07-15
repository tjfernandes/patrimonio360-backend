"""Wiring dos rerankers de 2a fase (texto 8B + VL 8B).

Testa os helpers do ChatService (gating por flag, fallback gracioso,
truncagem do pool alargado) e utilitarios do VLRerankerService sem
carregar modelos reais.
    python -m unittest tests.test_reranker_wiring
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.reranker import (  # noqa: E402
    VLRerankerService,
    _resolve_case_insensitive,
)


def _make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "OPENSEARCH_HOST": "localhost",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


def _make_service(settings: Settings, **service_overrides: Any) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = settings
    service._reranker_service = service_overrides.get("reranker_service")
    service._vl_reranker_service = service_overrides.get("vl_reranker_service")
    return service


class _FakeTextReranker:
    def __init__(self, *, scores: list[float] | None = None, error: Exception | None = None) -> None:
        self.scores = scores
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def rerank(
        self,
        *,
        query_text: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        self.calls.append({"query_text": query_text, "documents": documents, "top_k": top_k})
        if self.error is not None:
            raise self.error
        scored = []
        for index, doc in enumerate(documents):
            item = dict(doc)
            item["rerank_score"] = float(self.scores[index]) if self.scores else 0.0
            scored.append(item)
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored[:top_k]


class _FakeVLReranker:
    def __init__(self, *, scores: list[float] | None = None, error: Exception | None = None) -> None:
        self.scores = scores
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def rerank_image_hits(
        self,
        *,
        query_text: str,
        query_image_bytes: bytes | None,
        image_hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "query_text": query_text,
                "query_image_bytes": query_image_bytes,
                "image_hits": image_hits,
                "top_k": top_k,
            }
        )
        if self.error is not None:
            raise self.error
        scored = []
        for index, hit in enumerate(image_hits):
            item = dict(hit)
            item["vl_rerank_score"] = float(self.scores[index]) if self.scores else 0.0
            scored.append(item)
        scored.sort(key=lambda item: item["vl_rerank_score"], reverse=True)
        return scored[:top_k]


class TestMaybeRerankTextDocs(unittest.TestCase):
    def test_flag_off_keeps_first_stage_order(self) -> None:
        settings = _make_settings(CHAT_ENABLE_RERANKING=False)
        fake = _FakeTextReranker(scores=[0.1, 0.9])
        service = _make_service(settings, reranker_service=fake)
        docs = [{"artifact_id": "a"}, {"artifact_id": "b"}]

        result = asyncio.run(
            service._maybe_rerank_text_docs(query_text="q", docs=docs, top_k=2)
        )

        self.assertEqual(result, docs)
        self.assertEqual(fake.calls, [])

    def test_reorders_and_truncates_pool(self) -> None:
        settings = _make_settings(CHAT_ENABLE_RERANKING=True)
        fake = _FakeTextReranker(scores=[0.2, 0.9, 0.7])
        service = _make_service(settings, reranker_service=fake)
        docs = [{"artifact_id": "a"}, {"artifact_id": "b"}, {"artifact_id": "c"}]

        result = asyncio.run(
            service._maybe_rerank_text_docs(query_text="q", docs=docs, top_k=2)
        )

        self.assertEqual([doc["artifact_id"] for doc in result], ["b", "c"])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["top_k"], 2)

    def test_error_falls_back_to_truncated_first_stage(self) -> None:
        settings = _make_settings(CHAT_ENABLE_RERANKING=True)
        fake = _FakeTextReranker(error=RuntimeError("boom"))
        service = _make_service(settings, reranker_service=fake)
        docs = [{"artifact_id": "a"}, {"artifact_id": "b"}, {"artifact_id": "c"}]

        result = asyncio.run(
            service._maybe_rerank_text_docs(query_text="q", docs=docs, top_k=2)
        )

        # Pool alargado (3) trunca de volta ao page size (2) na ordem original.
        self.assertEqual([doc["artifact_id"] for doc in result], ["a", "b"])


class TestMaybeRerankImageHits(unittest.TestCase):
    def test_flag_off_keeps_first_stage_order(self) -> None:
        settings = _make_settings(CHAT_ENABLE_VL_RERANKING=False)
        fake = _FakeVLReranker(scores=[0.1, 0.9])
        service = _make_service(settings, vl_reranker_service=fake)
        hits = [{"image_id": "i1"}, {"image_id": "i2"}]

        result = asyncio.run(
            service._maybe_rerank_image_hits(
                query_text="q",
                query_image_bytes=b"img",
                image_hits=hits,
                top_k=2,
            )
        )

        self.assertEqual(result, hits)
        self.assertEqual(fake.calls, [])

    def test_reorders_and_truncates_pool(self) -> None:
        settings = _make_settings(CHAT_ENABLE_VL_RERANKING=True)
        fake = _FakeVLReranker(scores=[0.3, 0.95, 0.6])
        service = _make_service(settings, vl_reranker_service=fake)
        hits = [{"image_id": "i1"}, {"image_id": "i2"}, {"image_id": "i3"}]

        result = asyncio.run(
            service._maybe_rerank_image_hits(
                query_text="",
                query_image_bytes=b"img",
                image_hits=hits,
                top_k=2,
            )
        )

        self.assertEqual([hit["image_id"] for hit in result], ["i2", "i3"])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["query_image_bytes"], b"img")

    def test_error_falls_back_to_truncated_first_stage(self) -> None:
        settings = _make_settings(CHAT_ENABLE_VL_RERANKING=True)
        fake = _FakeVLReranker(error=RuntimeError("boom"))
        service = _make_service(settings, vl_reranker_service=fake)
        hits = [{"image_id": "i1"}, {"image_id": "i2"}, {"image_id": "i3"}]

        result = asyncio.run(
            service._maybe_rerank_image_hits(
                query_text="q",
                query_image_bytes=b"img",
                image_hits=hits,
                top_k=2,
            )
        )

        self.assertEqual([hit["image_id"] for hit in result], ["i1", "i2"])


class TestVLRerankerServiceOffline(unittest.TestCase):
    """Comportamentos que nao carregam o modelo."""

    def test_flag_off_short_circuits_without_model_load(self) -> None:
        settings = _make_settings(CHAT_ENABLE_VL_RERANKING=False)
        service = VLRerankerService(settings)
        hits = [{"image_id": "i1"}, {"image_id": "i2"}, {"image_id": "i3"}]

        result = asyncio.run(
            service.rerank_image_hits(
                query_text="q",
                query_image_bytes=b"img",
                image_hits=hits,
                top_k=2,
            )
        )

        self.assertEqual([hit["image_id"] for hit in result], ["i1", "i2"])
        self.assertIsNone(service._model)

    def test_empty_query_keeps_first_stage_order(self) -> None:
        settings = _make_settings(CHAT_ENABLE_VL_RERANKING=True)
        service = VLRerankerService(settings)
        hits = [{"image_id": "i1"}, {"image_id": "i2"}]

        result = service._rerank_sync(
            query_text="",
            query_image_bytes=None,
            image_hits=hits,
            top_k=2,
        )

        self.assertEqual([hit["image_id"] for hit in result], ["i1", "i2"])
        self.assertIsNone(service._model)

    def test_resolve_case_insensitive_recovers_mixed_case_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            target_dir = tmp_path / "mosteiro_dos_jeronimos" / "obj_MJ_IM001_1"
            target_dir.mkdir(parents=True)
            target = target_dir / "Foto.JPEG"
            target.write_bytes(b"fake")

            resolved = _resolve_case_insensitive(
                tmp_path,
                ["mosteiro_dos_jeronimos", "obj_mj_im001_1", "foto.jpeg"],
            )

            self.assertEqual(resolved, target)

    def test_resolve_case_insensitive_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                _resolve_case_insensitive(Path(tmp), ["nao_existe", "x.jpg"])
            )

    def test_resolve_hit_image_and_build_document(self) -> None:
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            self.skipTest("pillow nao instalado")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Images"
            folder = root / "museu" / "obj_X_1"
            folder.mkdir(parents=True)
            Image.new("RGB", (2048, 1024), color=(200, 10, 10)).save(folder / "peca.jpg")

            settings = _make_settings(
                CHAT_ENABLE_VL_RERANKING=True,
                IMAGE_ASSET_ROOT=str(root),
            )
            service = VLRerankerService(settings)

            image = service._resolve_hit_image(
                {"local_path": "Images/museu/obj_x_1/peca.jpg"}
            )
            self.assertIsNotNone(image)
            self.assertLessEqual(max(image.size), settings.VL_RERANKER_MAX_IMAGE_SIDE)

            document = service._build_document(
                {
                    "local_path": "Images/museu/obj_x_1/peca.jpg",
                    "artifact_title": "Peca",
                    "caption": "Vista",
                }
            )
            self.assertIn("image", document)
            self.assertEqual(document["text"], "Peca. Vista")

            missing = service._build_document(
                {"local_path": "Images/museu/nada.jpg", "artifact_title": "So texto"}
            )
            self.assertNotIn("image", missing)
            self.assertEqual(missing["text"], "So texto")


class TestRerankerConfig(unittest.TestCase):
    def test_defaults_are_off_and_resolved(self) -> None:
        settings = _make_settings()
        self.assertFalse(settings.CHAT_ENABLE_RERANKING)
        self.assertFalse(settings.CHAT_ENABLE_VL_RERANKING)
        self.assertEqual(
            settings.vl_reranker_model_resolved, "Qwen/Qwen3-VL-Reranker-8B"
        )
        self.assertIsNone(settings.reranker_model_revision_resolved)
        self.assertIsNone(settings.vl_reranker_model_revision_resolved)

    def test_revision_resolution(self) -> None:
        settings = _make_settings(
            RERANKER_MODEL_REVISION="  abc123  ",
            VL_RERANKER_MODEL_REVISION="",
        )
        self.assertEqual(settings.reranker_model_revision_resolved, "abc123")
        self.assertIsNone(settings.vl_reranker_model_revision_resolved)


if __name__ == "__main__":
    unittest.main()
