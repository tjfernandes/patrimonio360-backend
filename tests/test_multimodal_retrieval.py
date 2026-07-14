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
        MULTIMODAL_I2I_WEIGHT=1.0,
        MULTIMODAL_IMAGE_TEXT_ARTIFACT_WEIGHT=0.5,
        MULTIMODAL_USE_MSEARCH=False,
        MULTIMODAL_IN_TOUR_MARGIN=0.0,
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


class TestTextToImageOnly(unittest.TestCase):
    """E7B — o ramo t2i é independente do resultado de artifact_search."""

    def test_empty_artifact_branch_with_t2i_results_yields_t2i_only(self) -> None:
        gateway = StubGateway(
            image_hits=[img_hit("A7", "i7", 0.9), img_hit("A8", "i8", 0.8)],
            by_ids={"A7": art_doc("A7"), "A8": art_doc("A8")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=StubProvider()
        )
        outcome = run(
            service.fuse_text_search(
                query_text="objetos com flores", museum_slug="s", museum_id="mnt",
                artifact_docs=[], router_decision=DECISION, trace_id="t2i-only",
            )
        )
        self.assertIsNotNone(outcome)
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A7", "A8"])
        matched = {h["artifact_id"]: h["image_id"] for h in outcome.matched_image_hits}
        self.assertEqual(matched, {"A7": "i7", "A8": "i8"})
        # hidratados por artifact_id, ordem do ramo visual preservada
        self.assertEqual(gateway.ids_calls[0]["artifact_ids"], ["A7", "A8"])

    def test_artifact_results_with_empty_t2i_keeps_baseline(self) -> None:
        gateway = StubGateway(image_hits=[])
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=StubProvider()
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="s", museum_id="mnt",
                artifact_docs=[art_doc("A1")], router_decision=DECISION, trace_id="tb",
            )
        )
        self.assertIsNone(outcome)  # caller mantém a baseline textual

    def test_both_branches_empty_returns_none(self) -> None:
        gateway = StubGateway(image_hits=[])
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=StubProvider()
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="s", museum_id="mnt",
                artifact_docs=[], router_decision=DECISION, trace_id="te",
            )
        )
        self.assertIsNone(outcome)

    def test_t2i_only_respects_museum_filter_guard(self) -> None:
        gateway = StubGateway(
            image_hits=[img_hit("A9", "i9", 0.9, museum="mnaz")],
            by_ids={"A9": art_doc("A9", museum="mnaz")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=StubProvider()
        )
        outcome = run(
            service.fuse_text_search(
                query_text="q", museum_slug="s", museum_id="mnt",
                artifact_docs=[], router_decision=DECISION, trace_id="tg",
            )
        )
        # único candidato é de outro museu -> guarda remove -> None (nunca
        # despromover a baseline para uma lista vazia)
        self.assertIsNone(outcome)


class StubGatewayImageText(StubGateway):
    """Stub com o ramo documental (search_relevant_context_page)."""

    def __init__(self, *, artifact_page=None, artifact_error=None, **kw):
        super().__init__(**kw)
        self.artifact_page = artifact_page or []
        self.artifact_error = artifact_error
        self.artifact_calls = []

    async def search_relevant_context_page(self, **kwargs):
        self.artifact_calls.append(kwargs)
        if self.artifact_error:
            raise self.artifact_error
        return SimpleNamespace(
            results=list(self.artifact_page), total=len(self.artifact_page), query_body={}
        )


class StubProviderImageText(StubProvider):
    def __init__(self, *, text_vector=None, **kw):
        super().__init__(**kw)
        self.text_vector = text_vector or [0.5, 0.5]
        self.text_calls = 0

    async def embed_text(self, text):
        self.text_calls += 1
        return self.text_vector


def run_image_text(service, *, i2i, run_t2i=True, temporal=None, text="com decoração floral"):
    return run(
        service.fuse_image_text_search(
            message_text=text,
            museum_slug="s",
            museum_id="mnt",
            i2i_hits=i2i,
            run_t2i=run_t2i,
            temporal_filter=temporal,
            router_decision=DECISION,
            trace_id="it",
        )
    )


class TestImageTextFusion(unittest.TestCase):
    def test_uses_all_three_branches_with_i2i_primary(self) -> None:
        gateway = StubGatewayImageText(
            image_hits=[img_hit("T1", "t1", 0.9)],          # t2i
            artifact_page=[art_doc("D1", score=10.0)],       # documental
            by_ids={"I1": art_doc("I1"), "T1": art_doc("T1")},
        )
        provider = StubProviderImageText()
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=provider
        )
        outcome = run_image_text(
            service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=True
        )
        ids = [d["artifact_id"] for d in outcome.docs]
        # i2i (peso 1.0) > t2i (0.7) > artifact (0.5), todos rank 1 nos seus ramos
        self.assertEqual(ids, ["I1", "T1", "D1"])
        self.assertEqual(provider.calls, 1)       # t2i embed
        self.assertEqual(provider.text_calls, 1)  # embed documental
        self.assertEqual(len(gateway.artifact_calls), 1)
        fused_i1 = next(r for r in outcome.fused if r.artifact_id == "I1")
        self.assertIn("image_to_image", fused_i1.sources)

    def test_debug_mode_logging_does_not_break_fusion(self) -> None:
        # Regressão: com MULTIMODAL_DEBUG=true o _log fazia trace_id duplicado
        # (payload contém trace_id) e a fusão inteira caía para o fallback.
        gateway = StubGatewayImageText(by_ids={"I1": art_doc("I1")})
        service = MultimodalTextRetrieval(
            settings=make_settings(MULTIMODAL_DEBUG=True),
            opensearch_gateway=gateway,
            embedding_provider=StubProviderImageText(),
        )
        outcome = run_image_text(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False)
        self.assertIsNotNone(outcome)
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["I1"])

    def test_without_visual_text_component_skips_t2i(self) -> None:
        gateway = StubGatewayImageText(by_ids={"I1": art_doc("I1")})
        provider = StubProviderImageText()
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=provider
        )
        outcome = run_image_text(
            service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False, text="do século XVIII"
        )
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["I1"])
        self.assertEqual(provider.calls, 0)  # sem embed t2i
        self.assertEqual(len(gateway.search_calls), 0)

    def test_temporal_filter_drops_out_of_window_and_unknown(self) -> None:
        in_window = art_doc("I1"); in_window.update(start_year=1701, end_year=1750)
        out_window = art_doc("I2"); out_window.update(start_year=1900, end_year=1920)
        unknown = art_doc("I3")  # sem anos -> include_unknown=False
        gateway = StubGatewayImageText(
            by_ids={"I1": in_window, "I2": out_window, "I3": unknown}
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(),
            opensearch_gateway=gateway,
            embedding_provider=StubProviderImageText(),
        )
        temporal = {"_temporal_interval": {"start_year": 1700, "end_year": 1799,
                                           "include_unknown": False}}
        outcome = run_image_text(
            service,
            i2i=[img_hit("I1", "a", 0.9), img_hit("I2", "b", 0.85), img_hit("I3", "c", 0.8)],
            run_t2i=False,
            temporal=temporal,
            text="apenas do século XVIII",
        )
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["I1"])
        self.assertEqual(outcome.diagnostics["fusion"]["dropped_temporal"], 2)

    def test_floor_applies_to_t2i_but_not_i2i(self) -> None:
        gateway = StubGatewayImageText(
            image_hits=[img_hit("T1", "t1", 0.2)],  # t2i fraco -> floor corta
            by_ids={"I1": art_doc("I1"), "T1": art_doc("T1")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(MULTIMODAL_MIN_IMAGE_SCORE=0.5),
            opensearch_gateway=gateway,
            embedding_provider=StubProviderImageText(),
        )
        outcome = run_image_text(
            service, i2i=[img_hit("I1", "i1", 0.3)], run_t2i=True  # i2i fraco MANTIDO
        )
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["I1"])

    def test_artifact_branch_failure_keeps_visual_results(self) -> None:
        gateway = StubGatewayImageText(
            artifact_error=RuntimeError("os down"),
            by_ids={"I1": art_doc("I1")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(),
            opensearch_gateway=gateway,
            embedding_provider=StubProviderImageText(),
        )
        outcome = run_image_text(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False)
        self.assertIsNotNone(outcome)
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["I1"])
        self.assertIn("error", outcome.diagnostics["artifact_search"])

    def test_matched_image_prefers_best_visual_rank(self) -> None:
        gateway = StubGatewayImageText(
            image_hits=[img_hit("I1", "t2i-img", 0.9)],
            by_ids={"I1": art_doc("I1")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(),
            opensearch_gateway=gateway,
            embedding_provider=StubProviderImageText(),
        )
        outcome = run_image_text(service, i2i=[img_hit("I1", "i2i-img", 0.95)], run_t2i=True)
        matched = {h["artifact_id"]: h["image_id"] for h in outcome.matched_image_hits}
        # ambos rank 1; desempate determinístico por nome de ramo (alfabético):
        # image_to_image < text_to_image -> vence a imagem do i2i.
        self.assertEqual(matched, {"I1": "i2i-img"})


class MsearchStubGateway(StubGatewayImageText):
    """Stub que expõe a superfície _msearch do gateway (E9)."""

    def __init__(self, *, msearch_responses=None, msearch_transport_error=None,
                 artifact_pipeline=False, **kw):
        super().__init__(**kw)
        self.msearch_responses = msearch_responses
        self.msearch_transport_error = msearch_transport_error
        self.artifact_pipeline = artifact_pipeline
        self.msearch_calls = []

    def build_similar_images_page_request(self, **kw):
        return {"index": "imgs", "body": {"kind": "t2i"}}

    def build_relevant_context_page_request(self, **kw):
        req = {"index": "arts", "body": {"kind": "art"}}
        if self.artifact_pipeline:
            req["search_pipeline"] = "nlp-search-pipeline"
        return req

    async def msearch_requests(self, requests):
        self.msearch_calls.append(requests)
        if self.msearch_transport_error:
            raise self.msearch_transport_error
        return self.msearch_responses

    def parse_similar_images_page_response(self, response, *, request):
        return SimpleNamespace(results=response["_hits"], total=len(response["_hits"]), query_body=request)

    def parse_relevant_context_page_response(self, response, *, request):
        return SimpleNamespace(results=response["_hits"], total=len(response["_hits"]), query_body=request)


class TestMsearchExecution(unittest.TestCase):
    def _fuse(self, gateway):
        service = MultimodalTextRetrieval(
            settings=make_settings(MULTIMODAL_USE_MSEARCH=True),
            opensearch_gateway=gateway,
            embedding_provider=StubProviderImageText(),
        )
        return run_image_text(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=True)

    def test_msearch_path_produces_same_shape_as_sequential(self) -> None:
        t2i_hits = [img_hit("T1", "t1", 0.9)]
        art_docs = [art_doc("D1", score=9.0)]
        gateway = MsearchStubGateway(
            msearch_responses=[{"_hits": t2i_hits}, {"_hits": art_docs}],
            by_ids={"I1": art_doc("I1"), "T1": art_doc("T1")},
        )
        outcome = self._fuse(gateway)
        self.assertEqual(len(gateway.msearch_calls), 1)
        self.assertEqual(outcome.diagnostics["execution"]["via"], "msearch")
        ids = [d["artifact_id"] for d in outcome.docs]
        # mesma ordem que o caminho sequencial equivalente (i2i > t2i > artifact)
        self.assertEqual(ids, ["I1", "T1", "D1"])

    def test_pipeline_branch_forces_sequential(self) -> None:
        # OpenSearch _msearch rejeita search_pipeline no header -> um ramo que o
        # exige nunca é batched; corre sequencial (sem round-trip inválido).
        gateway = MsearchStubGateway(
            artifact_pipeline=True,
            image_hits=[img_hit("T1", "t1", 0.9)],
            artifact_page=[art_doc("D1", score=9.0)],
            by_ids={"I1": art_doc("I1"), "T1": art_doc("T1")},
        )
        outcome = self._fuse(gateway)
        self.assertEqual(outcome.diagnostics["execution"]["via"], "sequential")
        self.assertEqual(len(gateway.msearch_calls), 0)  # nunca tentou msearch
        self.assertEqual(outcome.diagnostics["msearch"]["skipped"], "branch requires search_pipeline")
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["I1", "T1", "D1"])

    def test_branch_error_inside_msearch_keeps_other_branch(self) -> None:
        gateway = MsearchStubGateway(
            msearch_responses=[RuntimeError("t2i shard fail"), {"_hits": [art_doc("D1", score=9.0)]}],
            by_ids={"I1": art_doc("I1"), "D1": art_doc("D1")},
        )
        outcome = self._fuse(gateway)
        ids = [d["artifact_id"] for d in outcome.docs]
        self.assertEqual(ids, ["I1", "D1"])  # t2i caiu; i2i + documental vivos
        self.assertIn("error", outcome.diagnostics["text_to_image"])

    def test_transport_failure_falls_back_to_sequential(self) -> None:
        gateway = MsearchStubGateway(
            msearch_transport_error=RuntimeError("connection reset"),
            image_hits=[img_hit("T1", "t1", 0.9)],
            artifact_page=[art_doc("D1", score=9.0)],
            by_ids={"I1": art_doc("I1"), "T1": art_doc("T1")},
        )
        outcome = self._fuse(gateway)
        self.assertEqual(outcome.diagnostics["execution"]["via"], "sequential_fallback")
        ids = [d["artifact_id"] for d in outcome.docs]
        self.assertEqual(ids, ["I1", "T1", "D1"])
        # fallback usou as pesquisas separadas
        self.assertEqual(len(gateway.search_calls), 1)
        self.assertEqual(len(gateway.artifact_calls), 1)


class TestChatServiceImageTextGating(unittest.TestCase):
    def _service(self, mode):
        service = ChatService.__new__(ChatService)
        service.settings = make_settings(MULTIMODAL_RETRIEVAL_MODE=mode)
        return service

    def test_off_returns_none_without_side_effects(self) -> None:
        service = self._service("off")
        out = run(
            ChatService._maybe_fuse_image_text(
                service,
                message_text="com decoração floral",
                museum_slug="s",
                museum_id="mnt",
                i2i_hits=[img_hit("I1", "i1", 0.9)],
                conversation_id="c",
                query_id="q",
            )
        )
        self.assertIsNone(out)
        self.assertFalse(hasattr(service, "_multimodal_text_retrieval"))

    def test_empty_text_returns_none(self) -> None:
        service = self._service("intent")
        out = run(
            ChatService._maybe_fuse_image_text(
                service,
                message_text="   ",
                museum_slug="s",
                museum_id="mnt",
                i2i_hits=[img_hit("I1", "i1", 0.9)],
                conversation_id="c",
                query_id="q",
            )
        )
        self.assertIsNone(out)


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

    def test_llm_only_override_promotes_visual_queries_to_rag(self) -> None:
        service = self._service("intent")
        decision = {"mode": "llm_only", "intent": "overview"}
        out = ChatService._maybe_override_llm_only_for_visual(
            service, router_decision=decision, query_text="objetos com flores"
        )
        self.assertEqual(out["mode"], "rag")
        self.assertEqual(out["intent"], "visual_search")

    def test_llm_only_override_skips_factual_and_off(self) -> None:
        factual = {"mode": "llm_only", "intent": "overview"}
        out = ChatService._maybe_override_llm_only_for_visual(
            self._service("intent"),
            router_decision=factual,
            query_text="quem foi o diretor do museu?",
        )
        self.assertEqual(out["mode"], "llm_only")
        out_off = ChatService._maybe_override_llm_only_for_visual(
            self._service("off"),
            router_decision=factual,
            query_text="objetos com flores",
        )
        self.assertEqual(out_off["mode"], "llm_only")

    def test_llm_only_override_uses_intent_rules_even_in_always(self) -> None:
        # always não converte conversa em pesquisa: o probe usa regras intent.
        chatty = {"mode": "llm_only", "intent": "smalltalk"}
        out = ChatService._maybe_override_llm_only_for_visual(
            self._service("always"), router_decision=chatty, query_text="olá, tudo bem?"
        )
        self.assertEqual(out["mode"], "llm_only")
        visual = ChatService._maybe_override_llm_only_for_visual(
            self._service("always"),
            router_decision=chatty,
            query_text="peças com animais",
        )
        self.assertEqual(visual["mode"], "rag")

    def test_llm_only_override_never_touches_other_modes(self) -> None:
        for m in ("rag", "selected_artifact", "analytics"):
            decision = {"mode": m}
            out = ChatService._maybe_override_llm_only_for_visual(
                self._service("intent"),
                router_decision=decision,
                query_text="objetos com flores",
            )
            self.assertEqual(out["mode"], m)

    def test_intent_visual_runs_even_with_empty_artifact_docs(self) -> None:
        # E7B: t2i-only — o ramo visual corre mesmo com artifact_search vazio.
        service = self._service("intent")
        spy = _SpyFuser()
        service._multimodal_text_retrieval = spy
        run(
            ChatService._maybe_fuse_text_to_image(
                service,
                query_text="objetos com flores",
                museum_slug="s",
                museum_id="mnt",
                artifact_docs=[],
                conversation_id="c",
                query_id="q",
            )
        )
        self.assertEqual(spy.calls, 1)


if __name__ == "__main__":
    unittest.main()
