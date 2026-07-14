"""Orquestrador multimodal (Fase 3, Etapas 4/7) — offline com stubs.

    python -m unittest tests.test_multimodal_retrieval
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chat_service import ChatService  # noqa: E402
from app.services.retrieval.multimodal_retrieval import (  # noqa: E402
    MultimodalTextRetrieval,
)
from app.services.retrieval.visual_intent import decide_visual_intent  # noqa: E402


def make_settings(**overrides):
    base = dict(
        MULTIMODAL_RETRIEVAL_MODE="intent",
        MULTIMODAL_RRF_K=60,
        MULTIMODAL_ARTIFACT_WEIGHT=1.0,
        MULTIMODAL_IMAGE_WEIGHT=0.7,
        MULTIMODAL_MIN_IMAGE_SCORE=0.0,
        MULTIMODAL_IMAGE_TOP_K=30,
        MULTIMODAL_DEBUG=False,
        OPENSEARCH_INDEX_IMAGE="cultural_heritage_images_v4",
        QWEN_MULTIMODAL_EMBEDDING_MODEL_REVISION="rev123",
        multimodal_embedding_model_resolved="Qwen/Qwen3-VL-Embedding-8B",
        CHAT_RETRIEVAL_TOP_K=5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class StubProvider:
    def __init__(self, vector=None, error=None):
        self.vector = vector if vector is not None else [0.1, 0.2]
        self.error = error
        self.calls = 0

    async def embed_multimodal_text_query(self, text):
        self.calls += 1
        if self.error:
            raise self.error
        return self.vector


class StubGateway:
    def __init__(self, image_hits=None, search_error=None, by_ids=None, ids_error=None):
        self.image_hits = image_hits or []
        self.search_error = search_error
        self.by_ids = by_ids if by_ids is not None else {}
        self.ids_error = ids_error
        self.search_calls = []
        self.ids_calls = []

    async def search_similar_images_page(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.search_error:
            raise self.search_error
        return SimpleNamespace(results=list(self.image_hits), total=len(self.image_hits), query_body={})

    async def fetch_artifacts_by_ids(self, **kwargs):
        self.ids_calls.append(kwargs)
        if self.ids_error:
            raise self.ids_error
        out = []
        for artifact_id in kwargs.get("artifact_ids", []):
            doc = self.by_ids.get(artifact_id)
            if doc is not None:
                out.append(doc)
        return out


def img_hit(aid, image_id, score, museum="mnt", path=None):
    return {
        "artifact_id": aid,
        "image_id": image_id,
        "score": score,
        "museum_id": museum,
        "local_path": path or f"Images/x/{image_id}.jpg",
        "inventory_number": f"inv-{aid}",
    }


def art_doc(aid, museum="mnt", score=None):
    doc = {"artifact_id": aid, "museum_id": museum, "title": aid}
    if score is not None:
        doc["score"] = score
    return doc


def run(coro):
    return asyncio.run(coro)


DECISION = decide_visual_intent("peças azuis", mode="intent")


class TestOrchestrator(unittest.TestCase):
    def test_fuses_and_orders_by_weighted_rrf(self) -> None:
        # artifact_search: A1, A2 | t2i: A2 (img), A3 (img, só visual)
        gateway = StubGateway(
            image_hits=[img_hit("A2", "i2", 0.9), img_hit("A3", "i3", 0.8)],
            by_ids={"A3": art_doc("A3")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=StubProvider()
        )
        outcome = run(
            service.fuse_text_search(
                query_text="peças azuis",
                museum_slug="museu_nacional_do_traje",
                museum_id="mnt",
                artifact_docs=[art_doc("A1"), art_doc("A2")],
                router_decision=DECISION,
                trace_id="t1",
            )
        )
        self.assertIsNotNone(outcome)
        ids = [d["artifact_id"] for d in outcome.docs]
        # A2: 1/62 + 0.7/61 > A1: 1/61 > A3: 0.7/62
        self.assertEqual(ids, ["A2", "A1", "A3"])
        self.assertEqual(outcome.total, 3)
        # matched image preservada para os artefactos do ramo visual
        matched = {h["artifact_id"]: h["image_id"] for h in outcome.matched_image_hits}
        self.assertEqual(matched, {"A2": "i2", "A3": "i3"})
        # hidratação: A3 (só visual) foi buscado por artifact_id
        self.assertEqual(len(gateway.ids_calls), 1)
        self.assertEqual(gateway.ids_calls[0]["artifact_ids"], ["A3"])
        # provenance transporta os dois ramos para A2
        fused_a2 = next(r for r in outcome.fused if r.artifact_id == "A2")
        self.assertEqual(set(fused_a2.sources), {"artifact_search", "text_to_image"})

    def test_score_floor_drops_weak_hits_before_fusion(self) -> None:
        gateway = StubGateway(
            image_hits=[img_hit("A2", "i2", 0.9), img_hit("A3", "i3", 0.2)],
            by_ids={},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(MULTIMODAL_MIN_IMAGE_SCORE=0.5),
            opensearch_gateway=gateway,
            embedding_provider=StubProvider(),
        )
        outcome = run(
            service.fuse_text_search(
                query_text="peças azuis",
                museum_slug="s",
                museum_id="mnt",
                artifact_docs=[art_doc("A1")],
                router_decision=DECISION,
                trace_id="t2",
            )
        )
        ids = [d["artifact_id"] for d in outcome.docs]
        self.assertNotIn("A3", ids)
        dropped = outcome.diagnostics["text_to_image"]["dropped_by_floor"]
        self.assertEqual(dropped, [{"artifact_id": "A3", "score": 0.2}])

    def test_all_below_floor_returns_none_keeping_baseline(self) -> None:
        gateway = StubGateway(image_hits=[img_hit("A2", "i2", 0.1)])
        service = MultimodalTextRetrieval(
            settings=make_settings(MULTIMODAL_MIN_IMAGE_SCORE=0.5),
            opensearch_gateway=gateway,
            embedding_provider=StubProvider(),
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="s", museum_id="mnt",
                artifact_docs=[art_doc("A1")], router_decision=DECISION, trace_id="t3",
            )
        )
        self.assertIsNone(outcome)

    def test_search_failure_returns_none(self) -> None:
        gateway = StubGateway(search_error=RuntimeError("boom"))
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=StubProvider()
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="s", museum_id="mnt",
                artifact_docs=[art_doc("A1")], router_decision=DECISION, trace_id="t4",
            )
        )
        self.assertIsNone(outcome)

    def test_embed_failure_returns_none(self) -> None:
        service = MultimodalTextRetrieval(
            settings=make_settings(),
            opensearch_gateway=StubGateway(image_hits=[img_hit("A2", "i2", 0.9)]),
            embedding_provider=StubProvider(error=RuntimeError("no vl")),
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="s", museum_id="mnt",
                artifact_docs=[art_doc("A1")], router_decision=DECISION, trace_id="t5",
            )
        )
        self.assertIsNone(outcome)

    def test_hydration_failure_keeps_textual_branch_results(self) -> None:
        gateway = StubGateway(
            image_hits=[img_hit("A9", "i9", 0.9)], ids_error=RuntimeError("os down")
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=StubProvider()
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="s", museum_id="mnt",
                artifact_docs=[art_doc("A1")], router_decision=DECISION, trace_id="t6",
            )
        )
        self.assertIsNotNone(outcome)
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A1"])

    def test_museum_guard_drops_cross_museum_docs(self) -> None:
        gateway = StubGateway(
            image_hits=[img_hit("A8", "i8", 0.9, museum="mnaz")],
            by_ids={"A8": art_doc("A8", museum="mnaz")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=StubProvider()
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="s", museum_id="mnt",
                artifact_docs=[art_doc("A1")], router_decision=DECISION, trace_id="t7",
            )
        )
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A1"])

    def test_t2i_query_uses_configured_top_k_and_index_params(self) -> None:
        gateway = StubGateway(image_hits=[img_hit("A2", "i2", 0.9)])
        service = MultimodalTextRetrieval(
            settings=make_settings(MULTIMODAL_IMAGE_TOP_K=12),
            opensearch_gateway=gateway,
            embedding_provider=StubProvider(vector=[3.0, 4.0]),
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="slug", museum_id="mnt",
                artifact_docs=[art_doc("A1")], router_decision=DECISION, trace_id="t8",
            )
        )
        call = gateway.search_calls[0]
        self.assertEqual(call["page_size"], 12)
        self.assertEqual(call["retrieval_window_size"], 12)
        self.assertEqual(call["museum_id"], "mnt")
        diag = outcome.diagnostics["text_to_image"]
        self.assertEqual(diag["dim"], 2)
        self.assertAlmostEqual(diag["query_norm"], 5.0)
        self.assertEqual(diag["model"], "Qwen/Qwen3-VL-Embedding-8B")
        self.assertEqual(diag["revision"], "rev123")


class _SpyFuser:
    def __init__(self):
        self.calls = 0

    async def fuse_text_search(self, **kwargs):
        self.calls += 1
        return None


class TestChatServiceGating(unittest.TestCase):
    def _service(self, mode):
        service = ChatService.__new__(ChatService)
        service.settings = make_settings(MULTIMODAL_RETRIEVAL_MODE=mode)
        return service

    def _call(self, service, query="quem foi o autor?"):
        return run(
            ChatService._maybe_fuse_text_to_image(
                service,
                query_text=query,
                museum_slug="s",
                museum_id="mnt",
                artifact_docs=[art_doc("A1")],
                conversation_id="c",
                query_id="q",
            )
        )

    def test_off_short_circuits_without_router_or_service(self) -> None:
        service = self._service("off")
        self.assertIsNone(self._call(service, "peças azuis"))
        # nem sequer cria o serviço multimodal (nenhum modelo VL envolvido)
        self.assertFalse(hasattr(service, "_multimodal_text_retrieval"))

    def test_intent_factual_query_skips_visual_branch(self) -> None:
        service = self._service("intent")
        spy = _SpyFuser()
        service._multimodal_text_retrieval = spy
        self.assertIsNone(self._call(service, "quem foi o autor da peça?"))
        self.assertEqual(spy.calls, 0)

    def test_intent_visual_query_runs_branch(self) -> None:
        service = self._service("intent")
        spy = _SpyFuser()
        service._multimodal_text_retrieval = spy
        self._call(service, "peças azuis com padrões florais")
        self.assertEqual(spy.calls, 1)

    def test_always_runs_branch_even_for_factual(self) -> None:
        service = self._service("always")
        spy = _SpyFuser()
        service._multimodal_text_retrieval = spy
        self._call(service, "quem foi o autor da peça?")
        self.assertEqual(spy.calls, 1)


if __name__ == "__main__":
    unittest.main()
