"""Orquestração do retrieval multimodal (Fase 3, Etapa 4).

Combina o ramo textual existente (artifact_search — SEMPRE ativo; os docs
chegam já ordenados do caminho de produção) com o ramo texto→imagem
(Qwen3-VL em modo textual contra ``visual_embedding``), funde por artifact_id
com Weighted RRF e devolve docs hidratados na ordem fundida, preservando o
matched_image_id para as thumbnails.

Regras de segurança:
- nunca substitui o ramo textual: em qualquer falha do ramo visual devolve
  ``None`` e o chamador mantém a baseline;
- filtro de museu aplicado em ambos os ramos + guarda defensiva pós-fusão;
- floor de score aplicado ANTES da fusão (resultados visuais fracos não
  contaminam o ranking);
- nenhum vetor é registado nos logs (apenas dim + norma).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import log_event
from app.services.retrieval.fusion import (
    BranchHit,
    FusedResult,
    apply_score_floor,
    group_image_hits_by_artifact,
    weighted_rrf,
)
from app.services.retrieval.visual_intent import VisualIntentDecision

logger = logging.getLogger(__name__)

ARTIFACT_BRANCH = "artifact_search"
TEXT_TO_IMAGE_BRANCH = "text_to_image"


@dataclass
class MultimodalFusionOutcome:
    docs: list[dict[str, Any]]
    total: int
    fused: list[FusedResult]
    matched_image_hits: list[dict[str, Any]]
    router: VisualIntentDecision
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def debug_payload(self) -> dict[str, Any]:
        return {
            "router": self.router.as_dict(),
            "diagnostics": self.diagnostics,
            "fused": [result.provenance() for result in self.fused],
        }


class MultimodalTextRetrieval:
    """Ramo texto→imagem + fusão, reutilizando gateway/embeddings existentes."""

    def __init__(self, *, settings: Any, opensearch_gateway: Any, embedding_provider: Any) -> None:
        self.settings = settings
        self.gateway = opensearch_gateway
        self.embeddings = embedding_provider

    async def fuse_text_search(
        self,
        *,
        query_text: str,
        museum_slug: str,
        museum_id: str | None,
        artifact_docs: list[dict[str, Any]],
        router_decision: VisualIntentDecision,
        trace_id: str,
    ) -> MultimodalFusionOutcome | None:
        started = time.monotonic()
        diagnostics: dict[str, Any] = {
            "trace_id": trace_id,
            "branches": [ARTIFACT_BRANCH, TEXT_TO_IMAGE_BRANCH],
        }

        # --- ramo texto→imagem -------------------------------------------------
        t2i_hits, t2i_diag = await self._run_text_to_image(
            query_text=query_text,
            museum_slug=museum_slug,
            museum_id=museum_id,
            trace_id=trace_id,
        )
        diagnostics[TEXT_TO_IMAGE_BRANCH] = t2i_diag
        if t2i_hits is None:
            # Falha do ramo visual: o chamador mantém a baseline intacta.
            return None

        grouped = group_image_hits_by_artifact(t2i_hits)
        min_score = float(self.settings.MULTIMODAL_MIN_IMAGE_SCORE)
        kept, dropped = apply_score_floor(grouped, min_score=min_score)
        t2i_diag["grouped_artifacts"] = len(grouped)
        t2i_diag["kept_after_floor"] = len(kept)
        t2i_diag["dropped_by_floor"] = [
            {"artifact_id": h.artifact_id, "score": h.score} for h in dropped
        ]
        if not kept:
            # Sem evidência visual acima do floor: manter baseline (não é erro).
            diagnostics["outcome"] = "no_visual_evidence_above_floor"
            self._log(trace_id, "multimodal.t2i.empty", diagnostics)
            return None

        # --- ramo artifact_search (já executado pelo caminho de produção) ------
        artifact_hits: list[BranchHit] = []
        docs_by_id: dict[str, dict[str, Any]] = {}
        for index, doc in enumerate(artifact_docs, start=1):
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            docs_by_id[artifact_id] = doc
            raw_score = doc.get("score")
            artifact_hits.append(
                BranchHit(
                    artifact_id=artifact_id,
                    rank=index,
                    score=float(raw_score) if isinstance(raw_score, (int, float)) else None,
                )
            )

        # --- fusão --------------------------------------------------------------
        fused = weighted_rrf(
            {ARTIFACT_BRANCH: artifact_hits, TEXT_TO_IMAGE_BRANCH: kept},
            weights={
                ARTIFACT_BRANCH: float(self.settings.MULTIMODAL_ARTIFACT_WEIGHT),
                TEXT_TO_IMAGE_BRANCH: float(self.settings.MULTIMODAL_IMAGE_WEIGHT),
            },
            rrf_k=int(self.settings.MULTIMODAL_RRF_K),
        )

        # --- hidratação por artifact_id (só os que faltam) ----------------------
        missing_ids = [r.artifact_id for r in fused if r.artifact_id not in docs_by_id]
        if missing_ids:
            try:
                fetched = await self.gateway.fetch_artifacts_by_ids(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    artifact_ids=missing_ids,
                    top_k=len(missing_ids),
                )
            except Exception as exc:  # falha parcial: segue só com o ramo textual hidratado
                log_event(
                    logger,
                    logging.WARNING,
                    "multimodal.hydration.error",
                    trace_id=trace_id,
                    missing=len(missing_ids),
                    error=exc,
                )
                fetched = []
            for doc in fetched:
                artifact_id = str(doc.get("artifact_id") or "").strip()
                if artifact_id:
                    docs_by_id[artifact_id] = doc
            diagnostics["hydration"] = {
                "requested": len(missing_ids),
                "returned": len(fetched),
                "missing_after": [i for i in missing_ids if i not in docs_by_id],
            }

        resolved_museum = (museum_id or "").strip()
        ordered_docs: list[dict[str, Any]] = []
        matched_hits_by_image: dict[str, dict[str, Any]] = {
            str(h.get("image_id") or ""): h for h in t2i_hits
        }
        matched_image_hits: list[dict[str, Any]] = []
        kept_fused: list[FusedResult] = []
        for result in fused:
            doc = docs_by_id.get(result.artifact_id)
            if doc is None:
                continue  # não hidratado (registado em diagnostics.hydration)
            if resolved_museum and str(doc.get("museum_id") or "").strip() != resolved_museum:
                # Guarda defensiva: zero mistura entre museus.
                continue
            ordered_docs.append(doc)
            kept_fused.append(result)
            if result.matched_image_id:
                hit = matched_hits_by_image.get(result.matched_image_id)
                if hit is not None:
                    matched_image_hits.append(hit)

        if not ordered_docs:
            # Nunca despromover a baseline: sem docs utilizáveis pós-guardas,
            # o chamador mantém o que tinha (mesmo que também vazio).
            diagnostics["outcome"] = "empty_after_guards"
            self._log(trace_id, "multimodal.fusion.empty", diagnostics)
            return None

        diagnostics["fusion"] = {
            "rrf_k": int(self.settings.MULTIMODAL_RRF_K),
            "weights": {
                ARTIFACT_BRANCH: float(self.settings.MULTIMODAL_ARTIFACT_WEIGHT),
                TEXT_TO_IMAGE_BRANCH: float(self.settings.MULTIMODAL_IMAGE_WEIGHT),
            },
            "artifact_branch_size": len(artifact_hits),
            "t2i_branch_size": len(kept),
            "fused_size": len(ordered_docs),
            "t2i_only_added": sum(
                1 for r in kept_fused if ARTIFACT_BRANCH not in r.sources
            ),
        }
        diagnostics["latency_ms"] = {
            "total": round((time.monotonic() - started) * 1000, 1),
            TEXT_TO_IMAGE_BRANCH: t2i_diag.get("latency_ms"),
        }
        diagnostics["outcome"] = "fused"
        self._log(trace_id, "multimodal.fusion.finish", diagnostics)

        return MultimodalFusionOutcome(
            docs=ordered_docs,
            total=len(ordered_docs),
            fused=kept_fused,
            matched_image_hits=matched_image_hits,
            router=router_decision,
            diagnostics=diagnostics,
        )

    async def _run_text_to_image(
        self,
        *,
        query_text: str,
        museum_slug: str,
        museum_id: str | None,
        trace_id: str,
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        diag: dict[str, Any] = {}
        top_k = int(self.settings.MULTIMODAL_IMAGE_TOP_K)
        started = time.monotonic()
        try:
            embedding = await self.embeddings.embed_multimodal_text_query(query_text)
        except Exception as exc:
            diag["error"] = f"embed: {exc}"
            self._log(trace_id, "multimodal.t2i.embed.error", diag, level=logging.WARNING)
            return None, diag
        norm = math.sqrt(sum(x * x for x in embedding)) if embedding else 0.0
        diag.update(
            {
                "model": self.settings.multimodal_embedding_model_resolved,
                "revision": (self.settings.QWEN_MULTIMODAL_EMBEDDING_MODEL_REVISION or "unpinned"),
                "dim": len(embedding or []),
                "query_norm": round(norm, 5),
                "index": self.settings.OPENSEARCH_INDEX_IMAGE,
                "field": "visual_embedding",
                "k": top_k,
                "size": top_k,
                "score_floor": float(self.settings.MULTIMODAL_MIN_IMAGE_SCORE),
            }
        )
        try:
            page = await self.gateway.search_similar_images_page(
                museum_slug=museum_slug,
                museum_id=museum_id,
                image_embedding=embedding,
                from_offset=0,
                page_size=top_k,
                retrieval_window_size=top_k,
            )
        except Exception as exc:
            diag["error"] = f"search: {exc}"
            self._log(trace_id, "multimodal.t2i.search.error", diag, level=logging.WARNING)
            return None, diag
        hits = list(page.results or [])
        diag["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        diag["hits"] = len(hits)
        diag["top_image_ids"] = [str(h.get("image_id") or "") for h in hits[:5]]
        diag["top_artifact_ids"] = [str(h.get("artifact_id") or "") for h in hits[:5]]
        diag["top_scores"] = [
            round(float(h.get("score")), 5)
            for h in hits[:5]
            if isinstance(h.get("score"), (int, float))
        ]
        return hits, diag

    def _log(
        self,
        trace_id: str,
        event: str,
        payload: dict[str, Any],
        *,
        level: int = logging.INFO,
    ) -> None:
        if bool(getattr(self.settings, "MULTIMODAL_DEBUG", False)):
            log_event(logger, level, event, trace_id=trace_id, **payload)
            return
        log_event(
            logger,
            level,
            event,
            trace_id=trace_id,
            **{
                key: value
                for key, value in payload.items()
                if key in {"outcome", "error", "latency_ms", "fusion", "hits"}
            },
        )
