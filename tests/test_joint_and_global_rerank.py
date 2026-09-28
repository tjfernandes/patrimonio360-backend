"""E14 — ramo conjunto imagem+texto + rerank VL global (arquitetura, offline).

Cobre: routing por modalidade, elegibilidade do rerank global a TODOS os
ramos, inputs do reranker (query multimodal completa + candidatos limitados),
fallbacks, sobrevivência de provenance/matched image, dedup, restrições duras
pós-rerank e a regressão composicional "sapatos dourados + 'similar, but
black'" (B > A > C) com reranker mockado.

    python -m unittest tests.test_joint_and_global_rerank
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chat_service import ChatService  # noqa: E402
from app.services.reranker import VLRerankerService  # noqa: E402
from app.services.retrieval.multimodal_retrieval import (  # noqa: E402
    ARTIFACT_BRANCH,
    GLOBAL_VL_RERANK_DIAG_KEY,
    IMAGE_TO_IMAGE_BRANCH,
    JOINT_IMAGE_TEXT_BRANCH,
    TEXT_TO_IMAGE_BRANCH,
    MultimodalTextRetrieval,
    apply_global_vl_rerank,
    build_artifact_candidate_text,
    build_global_rerank_candidates,
)
from app.services.retrieval.visual_intent import decide_visual_intent  # noqa: E402


def make_settings(**overrides: Any) -> SimpleNamespace:
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
        MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED=True,
        MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K=7,
        MULTIMODAL_JOINT_IMAGE_TEXT_WEIGHT=1.0,
        MULTIMODAL_VL_RERANK_STAGE="global",
        MULTIMODAL_GLOBAL_VL_RERANK_TOP_N=16,
        OPENSEARCH_INDEX_IMAGE="cultural_heritage_images_v4",
        QWEN_MULTIMODAL_EMBEDDING_MODEL_REVISION="rev123",
        multimodal_embedding_model_resolved="Qwen/Qwen3-VL-Embedding-8B",
        CHAT_RETRIEVAL_TOP_K=5,
        CHAT_ENABLE_VL_RERANKING=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def img_hit(aid: str, image_id: str, score: float, museum: str = "mnt", path: str | None = None):
    return {
        "artifact_id": aid,
        "image_id": image_id,
        "score": score,
        "museum_id": museum,
        "local_path": path or f"Images/x/{image_id}.jpg",
        "inventory_number": f"inv-{aid}",
    }


def art_doc(aid: str, museum: str = "mnt", score: float | None = None, **fields: Any):
    doc: dict[str, Any] = {"artifact_id": aid, "museum_id": museum, "title": aid}
    if score is not None:
        doc["score"] = score
    doc.update(fields)
    return doc


def run(coro):
    return asyncio.run(coro)


DECISION = decide_visual_intent("peças azuis", mode="intent")


class QueueGateway:
    """Gateway stub: search_similar_images_page devolve páginas por ordem de
    chamada (a 1ª é o ramo conjunto, que corre antes do t2i sequencial)."""

    def __init__(self, *, search_pages=None, artifact_page=None, artifact_error=None,
                 by_ids=None, joint_error=None):
        self.search_pages = list(search_pages or [])
        self.artifact_page = artifact_page or []
        self.artifact_error = artifact_error
        self.by_ids = by_ids if by_ids is not None else {}
        self.joint_error = joint_error
        self.search_calls: list[dict[str, Any]] = []
        self.artifact_calls: list[dict[str, Any]] = []
        self.ids_calls: list[dict[str, Any]] = []

    async def search_similar_images_page(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.joint_error is not None and len(self.search_calls) == 1:
            raise self.joint_error
        hits = self.search_pages.pop(0) if self.search_pages else []
        return SimpleNamespace(results=list(hits), total=len(hits), query_body={})

    async def search_relevant_context_page(self, **kwargs):
        self.artifact_calls.append(kwargs)
        if self.artifact_error:
            raise self.artifact_error
        return SimpleNamespace(
            results=list(self.artifact_page), total=len(self.artifact_page), query_body={}
        )

    async def fetch_artifacts_by_ids(self, **kwargs):
        self.ids_calls.append(kwargs)
        out = []
        for artifact_id in kwargs.get("artifact_ids", []):
            doc = self.by_ids.get(artifact_id)
            if doc is not None:
                out.append(doc)
        return out


class JointProvider:
    def __init__(self, *, joint_vector=None, joint_error=None):
        self.joint_vector = joint_vector or [0.3, 0.4]
        self.joint_error = joint_error
        self.joint_calls: list[dict[str, Any]] = []
        self.t2i_calls = 0
        self.text_calls = 0

    async def embed_multimodal_joint_image_bytes(self, *, image_bytes: bytes, text: str):
        self.joint_calls.append({"image_bytes": image_bytes, "text": text})
        if self.joint_error:
            raise self.joint_error
        return self.joint_vector

    async def embed_multimodal_text_query(self, text: str):
        self.t2i_calls += 1
        return [0.1, 0.2]

    async def embed_text(self, text: str):
        self.text_calls += 1
        return [0.5, 0.5]


def fuse(service, *, i2i, run_t2i, text="com decoração floral", image_bytes=b"img-bytes",
         temporal=None):
    return run(
        service.fuse_image_text_search(
            message_text=text,
            museum_slug="s",
            museum_id="mnt",
            i2i_hits=i2i,
            run_t2i=run_t2i,
            temporal_filter=temporal,
            router_decision=DECISION,
            trace_id="jt",
            query_image_bytes=image_bytes,
        )
    )


class TestJointBranchRouting(unittest.TestCase):
    def test_image_plus_documentary_text_runs_joint_without_t2i(self) -> None:
        # Routing 2: i2i + joint + documental correm; t2i segue o probe (off).
        gateway = QueueGateway(
            search_pages=[[img_hit("J1", "j1", 0.8)]],  # única pesquisa = joint
            artifact_page=[art_doc("D1", score=9.0)],
            by_ids={"I1": art_doc("I1"), "J1": art_doc("J1")},
        )
        provider = JointProvider()
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=provider
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False,
                       text="do século XVIII")
        self.assertIsNotNone(outcome)
        sizes = outcome.diagnostics["fusion"]["branch_sizes"]
        self.assertEqual(
            set(sizes), {IMAGE_TO_IMAGE_BRANCH, JOINT_IMAGE_TEXT_BRANCH, ARTIFACT_BRANCH}
        )
        self.assertNotIn(TEXT_TO_IMAGE_BRANCH, sizes)
        self.assertEqual(provider.t2i_calls, 0)
        # joint viu exatamente os bytes e o texto originais (nunca substitui)
        self.assertEqual(
            provider.joint_calls,
            [{"image_bytes": b"img-bytes", "text": "do século XVIII"}],
        )
        # a pesquisa joint usou o top_k próprio do ramo
        self.assertEqual(gateway.search_calls[0]["page_size"], 7)

    def test_image_plus_appearance_text_runs_all_four_branches(self) -> None:
        # Routing 3: i2i + joint + t2i + documental — tudo chega à fusão.
        gateway = QueueGateway(
            search_pages=[[img_hit("J1", "j1", 0.8)], [img_hit("T1", "t1", 0.85)]],
            artifact_page=[art_doc("D1", score=9.0)],
            by_ids={"I1": art_doc("I1"), "J1": art_doc("J1"), "T1": art_doc("T1")},
        )
        provider = JointProvider()
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=provider
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=True)
        sizes = outcome.diagnostics["fusion"]["branch_sizes"]
        self.assertEqual(
            set(sizes),
            {IMAGE_TO_IMAGE_BRANCH, JOINT_IMAGE_TEXT_BRANCH, TEXT_TO_IMAGE_BRANCH, ARTIFACT_BRANCH},
        )
        self.assertEqual(outcome.diagnostics["fusion"]["candidates_before_dedup"], 4)
        ids = [d["artifact_id"] for d in outcome.docs]
        self.assertEqual(set(ids), {"I1", "J1", "T1", "D1"})

    def test_joint_disabled_flag_keeps_previous_topology(self) -> None:
        gateway = QueueGateway(
            search_pages=[[img_hit("T1", "t1", 0.85)]],
            artifact_page=[art_doc("D1", score=9.0)],
            by_ids={"I1": art_doc("I1"), "T1": art_doc("T1")},
        )
        provider = JointProvider()
        service = MultimodalTextRetrieval(
            settings=make_settings(MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED=False),
            opensearch_gateway=gateway,
            embedding_provider=provider,
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=True)
        self.assertEqual(provider.joint_calls, [])
        self.assertNotIn(JOINT_IMAGE_TEXT_BRANCH, outcome.diagnostics)
        self.assertNotIn(
            JOINT_IMAGE_TEXT_BRANCH, outcome.diagnostics["fusion"]["branch_sizes"]
        )

    def test_joint_without_image_bytes_is_skipped_never_substituted(self) -> None:
        # Sem bytes o ramo NÃO corre (nunca degrada para embedding só-texto).
        gateway = QueueGateway(by_ids={"I1": art_doc("I1")})
        provider = JointProvider()
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=provider
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False,
                       image_bytes=None)
        self.assertEqual(provider.joint_calls, [])
        self.assertEqual(
            outcome.diagnostics[JOINT_IMAGE_TEXT_BRANCH], {"skipped": "no_query_image_bytes"}
        )

    def test_joint_failure_is_isolated_from_other_branches(self) -> None:
        gateway = QueueGateway(
            artifact_page=[art_doc("D1", score=9.0)],
            by_ids={"I1": art_doc("I1")},
        )
        provider = JointProvider(joint_error=RuntimeError("vl down"))
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway, embedding_provider=provider
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False)
        self.assertIsNotNone(outcome)
        self.assertIn("error", outcome.diagnostics[JOINT_IMAGE_TEXT_BRANCH])
        ids = [d["artifact_id"] for d in outcome.docs]
        self.assertEqual(set(ids), {"I1", "D1"})

    def test_joint_search_failure_is_isolated(self) -> None:
        gateway = QueueGateway(
            joint_error=RuntimeError("shard fail"),
            artifact_page=[art_doc("D1", score=9.0)],
            by_ids={"I1": art_doc("I1")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway,
            embedding_provider=JointProvider(),
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False)
        self.assertIsNotNone(outcome)
        self.assertIn("error", outcome.diagnostics[JOINT_IMAGE_TEXT_BRANCH])
        self.assertEqual({d["artifact_id"] for d in outcome.docs}, {"I1", "D1"})

    def test_joint_only_artifact_is_hydrated_with_matched_image(self) -> None:
        gateway = QueueGateway(
            search_pages=[[img_hit("J9", "j9", 0.88)]],
            by_ids={"I1": art_doc("I1"), "J9": art_doc("J9")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway,
            embedding_provider=JointProvider(),
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False)
        matched = {h["artifact_id"]: h["image_id"] for h in outcome.matched_image_hits}
        self.assertEqual(matched["J9"], "j9")
        fused_j9 = next(r for r in outcome.fused if r.artifact_id == "J9")
        self.assertEqual(set(fused_j9.sources), {JOINT_IMAGE_TEXT_BRANCH})
        self.assertEqual(fused_j9.sources[JOINT_IMAGE_TEXT_BRANCH].rank, 1)

    def test_duplicate_images_of_same_artifact_collapse_to_one_result(self) -> None:
        # i2i e joint devolvem IMAGENS DIFERENTES do mesmo artefacto -> um só
        # cartão final, com a matched image do melhor rank (desempate por nome
        # de ramo, determinístico: image_to_image < joint_image_text).
        gateway = QueueGateway(
            search_pages=[[img_hit("I1", "joint-img", 0.9), img_hit("I1", "joint-img-2", 0.85)]],
            by_ids={"I1": art_doc("I1")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(), opensearch_gateway=gateway,
            embedding_provider=JointProvider(),
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i2i-img", 0.95)], run_t2i=False)
        self.assertEqual([d["artifact_id"] for d in outcome.docs].count("I1"), 1)
        matched = {h["artifact_id"]: h["image_id"] for h in outcome.matched_image_hits}
        self.assertEqual(matched, {"I1": "i2i-img"})

    def test_joint_weight_zero_removes_contribution_but_keeps_provenance(self) -> None:
        gateway = QueueGateway(
            search_pages=[[img_hit("J1", "j1", 0.9)]],
            by_ids={"I1": art_doc("I1"), "J1": art_doc("J1")},
        )
        service = MultimodalTextRetrieval(
            settings=make_settings(MULTIMODAL_JOINT_IMAGE_TEXT_WEIGHT=0.0),
            opensearch_gateway=gateway,
            embedding_provider=JointProvider(),
        )
        outcome = fuse(service, i2i=[img_hit("I1", "i1", 0.95)], run_t2i=False)
        fused_j1 = next(r for r in outcome.fused if r.artifact_id == "J1")
        self.assertEqual(fused_j1.fusion_score, 0.0)
        self.assertIn(JOINT_IMAGE_TEXT_BRANCH, fused_j1.sources)


class FakeArtifactVLReranker:
    """Duplo do VLRerankerService.rerank_artifact_candidates."""

    def __init__(self, *, scores_by_id: dict[str, float] | None = None,
                 error: Exception | None = None, ranked_override=None):
        self.scores_by_id = scores_by_id or {}
        self.error = error
        self.ranked_override = ranked_override
        self.calls: list[dict[str, Any]] = []

    async def rerank_artifact_candidates(self, *, query_text, query_image_bytes,
                                         candidates, top_k):
        self.calls.append(
            {
                "query_text": query_text,
                "query_image_bytes": query_image_bytes,
                "candidates": [dict(c) for c in candidates],
                "top_k": top_k,
            }
        )
        if self.error is not None:
            raise self.error
        if self.ranked_override is not None:
            return self.ranked_override
        scored = []
        for index, candidate in enumerate(candidates):
            item = dict(candidate)
            item["vl_rerank_score"] = float(
                self.scores_by_id.get(candidate["artifact_id"], 0.0)
            )
            item["retrieval_rank"] = index + 1
            item["document_had_image"] = bool(candidate.get("local_path"))
            scored.append(item)
        scored.sort(key=lambda item: item["vl_rerank_score"], reverse=True)
        return scored[:top_k]


def build_fused_outcome(service_settings=None, *, with_temporal=False):
    """Outcome real (via fuse_image_text_search) com candidatos de 4 ramos:
    A (i2i), B (joint), T (t2i), D (documental)."""
    docs = {
        "A": art_doc("A", title="Golden shoes", category="Footwear",
                     description="Pair of golden leather shoes."),
        "B": art_doc("B", title="Black shoes", category="Footwear",
                     description="Pair of black leather shoes."),
        "T": art_doc("T", title="Black dress", category="Costume",
                     description="Black silk dress."),
        "D": art_doc("D", title="Black fan", category="Accessories", score=9.0,
                     description="Black fan with dark decoration."),
    }
    if with_temporal:
        for doc in docs.values():
            doc.update(start_year=1701, end_year=1750)
    gateway = QueueGateway(
        search_pages=[[img_hit("B", "b1", 0.9)], [img_hit("T", "t1", 0.8)]],
        artifact_page=[docs["D"]],
        by_ids=docs,
    )
    service = MultimodalTextRetrieval(
        settings=service_settings or make_settings(),
        opensearch_gateway=gateway,
        embedding_provider=JointProvider(),
    )
    temporal = (
        {"_temporal_interval": {"start_year": 1700, "end_year": 1799,
                                "include_unknown": False}}
        if with_temporal
        else None
    )
    outcome = fuse(
        service,
        i2i=[img_hit("A", "a1", 0.95)],
        run_t2i=True,
        text="Find similar objects, but black.",
        image_bytes=b"golden-shoes-bytes",
        temporal=temporal,
    )
    assert outcome is not None
    return outcome


class TestGlobalVLRerank(unittest.TestCase):
    def test_candidates_come_from_every_active_branch(self) -> None:
        outcome = build_fused_outcome()
        candidates = build_global_rerank_candidates(outcome, top_n=16)
        ids = {c["artifact_id"] for c in candidates}
        self.assertEqual(ids, {"A", "B", "T", "D"})
        by_id = {c["artifact_id"]: c for c in candidates}
        # matched image retida no retrieval para os ramos visuais
        self.assertEqual(by_id["A"]["local_path"], "Images/x/a1.jpg")
        self.assertEqual(by_id["B"]["local_path"], "Images/x/b1.jpg")
        self.assertEqual(by_id["T"]["local_path"], "Images/x/t1.jpg")
        # documental sem matched image e sem image_paths -> candidato só-texto
        self.assertIsNone(by_id["D"]["local_path"])
        # texto limitado com evidência útil
        self.assertIn("Black shoes", by_id["B"]["text"])
        self.assertIn("category: Footwear", by_id["B"]["text"])

    def test_artifact_only_candidate_falls_back_to_representative_image(self) -> None:
        outcome = build_fused_outcome()
        for doc in outcome.docs:
            if doc["artifact_id"] == "D":
                doc["image_paths"] = ["Images/x/d-rep.jpg"]
        candidates = build_global_rerank_candidates(outcome, top_n=16)
        by_id = {c["artifact_id"]: c for c in candidates}
        self.assertEqual(by_id["D"]["local_path"], "Images/x/d-rep.jpg")

    def test_reranker_receives_full_multimodal_query_and_bounded_docs(self) -> None:
        outcome = build_fused_outcome()
        fake = FakeArtifactVLReranker(scores_by_id={"B": 0.9})
        run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="Find similar objects, but black.",
            query_image_bytes=b"golden-shoes-bytes", top_n=16, trace_id="t",
        ))
        self.assertEqual(len(fake.calls), 1)
        call = fake.calls[0]
        self.assertEqual(call["query_image_bytes"], b"golden-shoes-bytes")
        self.assertEqual(call["query_text"], "Find similar objects, but black.")
        self.assertEqual(len(call["candidates"]), 4)
        for candidate in call["candidates"]:
            self.assertIn("text", candidate)
            self.assertLess(len(candidate["text"]), 600)

    def test_compositional_black_shoes_regression_b_a_c(self) -> None:
        # imagem: sapatos dourados; texto: "similar, but black".
        # A = sapatos dourados (i2i), B = sapatos pretos (joint),
        # C(=T) = vestido preto (t2i). Reranker global promove B, mantém A,
        # despromove o vestido -> B > A > C.
        outcome = build_fused_outcome()
        fake = FakeArtifactVLReranker(
            scores_by_id={"B": 0.95, "A": 0.60, "T": 0.10, "D": 0.05}
        )
        result = run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="Find similar objects, but black.",
            query_image_bytes=b"golden-shoes-bytes", top_n=16, trace_id="t",
        ))
        ids = [d["artifact_id"] for d in result.docs]
        self.assertLess(ids.index("B"), ids.index("A"))
        self.assertLess(ids.index("A"), ids.index("T"))
        self.assertEqual(ids[0], "B")
        diag = result.diagnostics[GLOBAL_VL_RERANK_DIAG_KEY]
        self.assertEqual(diag["candidates"], 4)
        self.assertGreaterEqual(diag["positions_changed"], 1)

    def test_reranker_failure_preserves_rrf_order(self) -> None:
        outcome = build_fused_outcome()
        before_ids = [d["artifact_id"] for d in outcome.docs]
        before_scores = [r.fusion_score for r in outcome.fused]
        fake = FakeArtifactVLReranker(error=RuntimeError("cuda oom"))
        result = run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="q",
            query_image_bytes=b"img", top_n=16, trace_id="t",
        ))
        self.assertEqual([d["artifact_id"] for d in result.docs], before_ids)
        self.assertEqual([r.fusion_score for r in result.fused], before_scores)
        diag = result.diagnostics[GLOBAL_VL_RERANK_DIAG_KEY]
        self.assertEqual(diag["fallback"], "rrf_order")
        self.assertIn("error", diag)

    def test_invalid_reranker_output_falls_back_to_rrf(self) -> None:
        outcome = build_fused_outcome()
        before_ids = [d["artifact_id"] for d in outcome.docs]
        fake = FakeArtifactVLReranker(
            ranked_override=[{"artifact_id": "ZZZ", "vl_rerank_score": 1.0}]
        )
        result = run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="q",
            query_image_bytes=b"img", top_n=16, trace_id="t",
        ))
        self.assertEqual([d["artifact_id"] for d in result.docs], before_ids)
        diag = result.diagnostics[GLOBAL_VL_RERANK_DIAG_KEY]
        self.assertEqual(diag["fallback"], "rrf_order")

    def test_reranker_cannot_introduce_new_candidates(self) -> None:
        outcome = build_fused_outcome()
        fake = FakeArtifactVLReranker(
            ranked_override=[
                {"artifact_id": "EVIL", "vl_rerank_score": 1.0},
                {"artifact_id": "B", "vl_rerank_score": 0.9},
                {"artifact_id": "A", "vl_rerank_score": 0.8},
            ]
        )
        result = run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="q",
            query_image_bytes=b"img", top_n=16, trace_id="t",
        ))
        ids = [d["artifact_id"] for d in result.docs]
        self.assertNotIn("EVIL", ids)
        self.assertEqual(ids[0], "B")
        self.assertEqual(set(ids), {"A", "B", "T", "D"})

    def test_provenance_and_matched_images_survive_global_rerank(self) -> None:
        outcome = build_fused_outcome()
        fake = FakeArtifactVLReranker(scores_by_id={"B": 0.95, "A": 0.6})
        result = run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="q",
            query_image_bytes=b"img", top_n=16, trace_id="t",
        ))
        fused_b = next(r for r in result.fused if r.artifact_id == "B")
        self.assertIn(JOINT_IMAGE_TEXT_BRANCH, fused_b.sources)
        self.assertEqual(fused_b.matched_image_id, "b1")
        self.assertIsNotNone(fused_b.global_rerank)
        self.assertEqual(fused_b.global_rerank["position"], 1)
        self.assertIn("previous_position", fused_b.global_rerank)
        self.assertIn(fused_b.global_rerank["changed"], (True, False))
        self.assertEqual(fused_b.provenance()["global_rerank"]["position"], 1)
        # matched_image_hits realinhados com a nova ordem dos docs
        matched_ids = [h["artifact_id"] for h in result.matched_image_hits]
        doc_ids = [d["artifact_id"] for d in result.docs]
        self.assertEqual(
            matched_ids, [i for i in doc_ids if i in {"A", "B", "T"}]
        )
        # scores RRF preservados para análise
        self.assertTrue(all(r.fusion_score >= 0 for r in result.fused))

    def test_top_n_limits_rerank_and_keeps_tail_in_rrf_order(self) -> None:
        outcome = build_fused_outcome()
        before_ids = [d["artifact_id"] for d in outcome.docs]
        head_before = set(before_ids[:2])
        fake = FakeArtifactVLReranker(
            scores_by_id={aid: 1.0 - 0.1 * i for i, aid in enumerate(reversed(before_ids[:2]))}
        )
        result = run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="q",
            query_image_bytes=b"img", top_n=2, trace_id="t",
        ))
        self.assertEqual(len(fake.calls[0]["candidates"]), 2)
        after_ids = [d["artifact_id"] for d in result.docs]
        # cauda intacta na ordem RRF
        self.assertEqual(after_ids[2:], before_ids[2:])
        self.assertEqual(set(after_ids[:2]), head_before)
        # cauda sem anotação global_rerank
        for r in result.fused[2:]:
            self.assertIsNone(r.global_rerank)

    def test_hard_constraints_hold_after_global_rerank(self) -> None:
        outcome = build_fused_outcome(with_temporal=True)
        self.assertEqual(
            outcome.diagnostics["constraints"]["temporal_window"], [1700, 1799]
        )
        self.assertEqual(outcome.diagnostics["constraints"]["museum_id"], "mnt")
        fake = FakeArtifactVLReranker(scores_by_id={"B": 0.9, "A": 0.5})
        result = run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="q",
            query_image_bytes=b"img", top_n=16, trace_id="t",
        ))
        for doc in result.docs:
            self.assertEqual(doc["museum_id"], "mnt")
            self.assertTrue(1700 <= doc["end_year"] and doc["start_year"] <= 1799)
        self.assertEqual(
            result.diagnostics[GLOBAL_VL_RERANK_DIAG_KEY]["constraint_violations_dropped"], 0
        )

    def test_tampered_doc_violating_constraints_is_dropped_post_rerank(self) -> None:
        # Defesa em profundidade: se um doc violar as restrições registadas
        # (não deve acontecer numa permutação), é removido e registado.
        outcome = build_fused_outcome(with_temporal=True)
        for doc in outcome.docs:
            if doc["artifact_id"] == "T":
                doc["museum_id"] = "mnaz"
        fake = FakeArtifactVLReranker(scores_by_id={"B": 0.9})
        result = run(apply_global_vl_rerank(
            outcome, vl_reranker=fake, query_text="q",
            query_image_bytes=b"img", top_n=16, trace_id="t",
        ))
        ids = [d["artifact_id"] for d in result.docs]
        self.assertNotIn("T", ids)
        self.assertEqual(
            result.diagnostics[GLOBAL_VL_RERANK_DIAG_KEY]["constraint_violations_dropped"], 1
        )


class TestBoundedCandidateText(unittest.TestCase):
    def test_includes_useful_evidence_and_truncates_description(self) -> None:
        doc = art_doc(
            "X",
            title="Par de sapatos",
            category="Traje",
            creators=["Autor Um", "Autor Dois", "Autor Três"],
            support_or_material="Couro dourado",
            date_or_period="Século XVIII",
            museum_name="Museu Nacional do Traje",
            description="d" * 1000,
            search_text="NUNCA-INCLUIR" * 50,
        )
        text = build_artifact_candidate_text(doc)
        self.assertIn("Par de sapatos", text)
        self.assertIn("category: Traje", text)
        self.assertIn("author: Autor Um, Autor Dois", text)
        self.assertNotIn("Autor Três", text)
        self.assertIn("material: Couro dourado", text)
        self.assertIn("date: Século XVIII", text)
        self.assertIn("museum: Museu Nacional do Traje", text)
        self.assertNotIn("NUNCA-INCLUIR", text)
        self.assertLess(len(text), 600)

    def test_year_fallback_and_empty_doc(self) -> None:
        text = build_artifact_candidate_text(art_doc("Y", start_year=1720, end_year=1740))
        self.assertIn("date: 1720–1740", text)
        self.assertEqual(build_artifact_candidate_text({}), "no-document-text")


class TestVLRerankerArtifactCandidates(unittest.TestCase):
    def _settings(self, **overrides: Any):
        from app.core.config import Settings

        values: dict[str, Any] = {"OPENSEARCH_HOST": "localhost", "_env_file": None}
        values.update(overrides)
        return Settings(**values)

    def test_flag_off_short_circuits_without_model_load(self) -> None:
        service = VLRerankerService(self._settings(CHAT_ENABLE_VL_RERANKING=False))
        candidates = [{"artifact_id": "a", "text": "x"}, {"artifact_id": "b", "text": "y"}]
        result = run(service.rerank_artifact_candidates(
            query_text="q", query_image_bytes=b"img", candidates=candidates, top_k=1,
        ))
        self.assertEqual([c["artifact_id"] for c in result], ["a"])
        self.assertIsNone(service._model)

    def test_missing_candidate_image_degrades_to_text_only_document(self) -> None:
        service = VLRerankerService(self._settings(CHAT_ENABLE_VL_RERANKING=True))
        document = service._build_artifact_document(
            {"artifact_id": "a", "local_path": "Images/nao/existe.jpg", "text": "Título"}
        )
        self.assertNotIn("image", document)
        self.assertEqual(document["text"], "Título")
        empty = service._build_artifact_document({"artifact_id": "a"})
        self.assertEqual(empty["text"], "no-document-content")

    def test_sync_scoring_sorts_and_flags_image_failures(self) -> None:
        service = VLRerankerService(self._settings(CHAT_ENABLE_VL_RERANKING=True))
        service._score_pairs = lambda pairs: [0.2, 0.9]  # type: ignore[method-assign]
        candidates = [
            {"artifact_id": "a", "local_path": "Images/miss.jpg", "text": "A"},
            {"artifact_id": "b", "text": "B"},
        ]
        result = service._rerank_artifacts_sync(
            query_text="q", query_image_bytes=None, candidates=candidates, top_k=2,
        )
        self.assertEqual([c["artifact_id"] for c in result], ["b", "a"])
        self.assertEqual(result[0]["vl_rerank_score"], 0.9)
        self.assertFalse(result[0]["document_had_image"])
        # local_path presente mas sem pixels resolvíveis -> contado como falha
        self.assertEqual(result[0]["image_load_failures_in_pool"], 1)
        # sem query (nem texto nem imagem) -> ordem de entrada
        passthrough = service._rerank_artifacts_sync(
            query_text="", query_image_bytes=None, candidates=candidates, top_k=2,
        )
        self.assertEqual([c["artifact_id"] for c in passthrough], ["a", "b"])


class TestChatServiceStageWiring(unittest.TestCase):
    def _service(self, settings) -> ChatService:
        service = object.__new__(ChatService)
        service.settings = settings
        service._reranker_service = None
        service._vl_reranker_service = None
        return service

    def test_stage_resolution_respects_master_switch(self) -> None:
        service = self._service(make_settings(CHAT_ENABLE_VL_RERANKING=False,
                                              MULTIMODAL_VL_RERANK_STAGE="global"))
        self.assertEqual(service._resolve_vl_rerank_stage(), "off")
        service = self._service(make_settings(MULTIMODAL_VL_RERANK_STAGE="both"))
        self.assertEqual(service._resolve_vl_rerank_stage(), "both")
        service = self._service(make_settings(MULTIMODAL_VL_RERANK_STAGE="invalido"))
        self.assertEqual(service._resolve_vl_rerank_stage(), "local")

    def test_local_pass_matrix(self) -> None:
        # Routing 1: imagem-only em global corre o passe único no pool i2i.
        cases = {
            ("local", True): True,
            ("local", False): True,
            ("both", True): True,
            ("both", False): True,
            ("global", True): False,
            ("global", False): True,
            ("off", True): False,
            ("off", False): False,
        }
        for (stage, has_text), expected in cases.items():
            self.assertEqual(
                ChatService._local_vl_rerank_pass_enabled(stage, has_query_text=has_text),
                expected,
                msg=f"stage={stage} has_text={has_text}",
            )

    def test_enabled_override_skips_rerank_even_with_flag_on(self) -> None:
        service = self._service(make_settings())
        sentinel_calls: list[Any] = []

        class _NeverCalled:
            async def rerank_image_hits(self, **kwargs):
                sentinel_calls.append(kwargs)
                return []

        service._vl_reranker_service = _NeverCalled()
        hits = [{"image_id": "i1"}, {"image_id": "i2"}, {"image_id": "i3"}]
        result = run(service._maybe_rerank_image_hits(
            query_text="q", query_image_bytes=b"img", image_hits=hits, top_k=2,
            enabled=False,
        ))
        self.assertEqual([h["image_id"] for h in result], ["i1", "i2"])
        self.assertEqual(sentinel_calls, [])

    def test_global_wrapper_uses_fake_service_and_reorders(self) -> None:
        outcome = build_fused_outcome()
        fake = FakeArtifactVLReranker(scores_by_id={"B": 0.95, "A": 0.6})
        service = self._service(make_settings(MULTIMODAL_GLOBAL_VL_RERANK_TOP_N=16))
        service._vl_reranker_service = fake
        result = run(service._maybe_global_vl_rerank_outcome(
            outcome=outcome,
            query_text="Find similar objects, but black.",
            query_image_bytes=b"golden-shoes-bytes",
            conversation_id="c",
            query_id="q",
        ))
        self.assertEqual(result.docs[0]["artifact_id"], "B")
        self.assertEqual(len(fake.calls), 1)

    def test_global_wrapper_isolates_wiring_failures(self) -> None:
        outcome = build_fused_outcome()
        before_ids = [d["artifact_id"] for d in outcome.docs]

        class _BrokenService:
            async def rerank_artifact_candidates(self, **kwargs):
                raise RuntimeError("boom")

        service = self._service(make_settings())
        service._vl_reranker_service = _BrokenService()
        result = run(service._maybe_global_vl_rerank_outcome(
            outcome=outcome, query_text="q", query_image_bytes=b"img",
            conversation_id="c", query_id="q",
        ))
        self.assertEqual([d["artifact_id"] for d in result.docs], before_ids)


if __name__ == "__main__":
    unittest.main()
