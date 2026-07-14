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
    promote_in_tour_within_margin,
    weighted_rrf,
)
from app.services.retrieval.visual_intent import VisualIntentDecision

logger = logging.getLogger(__name__)

ARTIFACT_BRANCH = "artifact_search"
TEXT_TO_IMAGE_BRANCH = "text_to_image"
IMAGE_TO_IMAGE_BRANCH = "image_to_image"


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

        return await self._assemble_outcome(
            fused=fused,
            docs_by_id=docs_by_id,
            museum_slug=museum_slug,
            museum_id=museum_id,
            hits_by_image_id={str(h.get("image_id") or ""): h for h in t2i_hits},
            temporal_filter=None,
            router_decision=router_decision,
            diagnostics=diagnostics,
            trace_id=trace_id,
            started=started,
            branch_sizes={
                ARTIFACT_BRANCH: len(artifact_hits),
                TEXT_TO_IMAGE_BRANCH: len(kept),
            },
        )

    async def fuse_image_text_search(
        self,
        *,
        message_text: str,
        museum_slug: str,
        museum_id: str | None,
        i2i_hits: list[dict[str, Any]],
        run_t2i: bool,
        temporal_filter: dict[str, Any] | None,
        router_decision: VisualIntentDecision,
        trace_id: str,
    ) -> MultimodalFusionOutcome | None:
        """Etapa 8 — imagem+texto: i2i (principal) + t2i (quando o texto tem
        componente visual) + artifact_search documental, fundidos por
        artifact_id. Filtros estruturados conhecidos (temporal) aplicam-se como
        FILTROS: no ramo de artefactos via `_temporal_interval` e, pós-fusão,
        como predicado sobre start/end_year (nunca via embedding visual).
        Falha de um ramo nunca apaga os resultados válidos dos restantes."""
        started = time.monotonic()
        diagnostics: dict[str, Any] = {"trace_id": trace_id, "mode": "image_text"}

        branches: dict[str, list[BranchHit]] = {}
        hits_by_image_id: dict[str, dict[str, Any]] = {}

        # --- i2i (ramo principal; hits já vêm da pesquisa do endpoint) --------
        i2i_grouped = group_image_hits_by_artifact(i2i_hits)
        branches[IMAGE_TO_IMAGE_BRANCH] = i2i_grouped
        hits_by_image_id.update({str(h.get("image_id") or ""): h for h in i2i_hits})
        diagnostics[IMAGE_TO_IMAGE_BRANCH] = {
            "hits": len(i2i_hits),
            "grouped_artifacts": len(i2i_grouped),
        }

        # --- embeddings dos ramos textuais (t2i opcional; documental sempre) --
        top_k = int(self.settings.MULTIMODAL_IMAGE_TOP_K)
        t2i_diag: dict[str, Any] = {}
        t2i_embedding: list[float] | None = None
        if run_t2i:
            try:
                t2i_embedding = await self.embeddings.embed_multimodal_text_query(
                    message_text
                )
                norm = math.sqrt(sum(x * x for x in t2i_embedding or []))
                t2i_diag = {
                    "model": self.settings.multimodal_embedding_model_resolved,
                    "dim": len(t2i_embedding or []),
                    "query_norm": round(norm, 5),
                    "k": top_k,
                    "score_floor": float(self.settings.MULTIMODAL_MIN_IMAGE_SCORE),
                }
            except Exception as exc:
                t2i_diag = {"error": f"embed: {exc}"}
            diagnostics[TEXT_TO_IMAGE_BRANCH] = t2i_diag

        doc_embedding: list[float] | None = None
        try:
            doc_embedding = await self.embeddings.embed_text(message_text)
        except Exception as exc:
            diagnostics[ARTIFACT_BRANCH] = {"error": f"embed: {exc}"}

        # --- execução: _msearch quando há 2 ramos; senão/fallback sequencial --
        t2i_hits: list[dict[str, Any]] | None = None
        artifact_page = None
        executed_via = "sequential"
        use_msearch = bool(getattr(self.settings, "MULTIMODAL_USE_MSEARCH", True))
        artifact_request_kwargs = dict(
            museum_slug=museum_slug,
            museum_id=museum_id,
            query_text=message_text,
            lexical_query=None,
            query_embedding=doc_embedding or [],
            from_offset=0,
            page_size=top_k,
            filters=dict(temporal_filter or {}),
            sort={},
            retrieval_window_size=top_k,
        )
        if use_msearch and t2i_embedding is not None and doc_embedding is not None:
            t2i_request = self.gateway.build_similar_images_page_request(
                museum_slug=museum_slug,
                museum_id=museum_id,
                image_embedding=t2i_embedding,
                from_offset=0,
                page_size=top_k,
                retrieval_window_size=top_k,
            )
            artifact_request = self.gateway.build_relevant_context_page_request(
                **artifact_request_kwargs
            )
            # OpenSearch _msearch does NOT accept `search_pipeline` in the
            # per-request metadata header. The artifact hybrid branch requires
            # the nlp-search-pipeline, so this pair cannot be batched — run it
            # sequentially (correct results, no invalid round-trip). _msearch
            # stays available for pipeline-free multi-branch batching.
            requires_pipeline = any(
                r.get("search_pipeline") for r in (t2i_request, artifact_request)
            )
            if requires_pipeline:
                diagnostics["msearch"] = {"skipped": "branch requires search_pipeline"}
                use_msearch = False
        if use_msearch and t2i_embedding is not None and doc_embedding is not None:
            try:
                msearch_started = time.monotonic()
                responses = await self.gateway.msearch_requests(
                    [t2i_request, artifact_request]
                )
                executed_via = "msearch"
                diagnostics["msearch"] = {
                    "latency_ms": round((time.monotonic() - msearch_started) * 1000, 1)
                }
                if isinstance(responses[0], Exception):
                    t2i_diag["error"] = f"search: {responses[0]}"
                else:
                    t2i_page = self.gateway.parse_similar_images_page_response(
                        responses[0], request=t2i_request
                    )
                    t2i_hits = list(t2i_page.results or [])
                if isinstance(responses[1], Exception):
                    diagnostics[ARTIFACT_BRANCH] = {"error": f"search: {responses[1]}"}
                else:
                    artifact_page = self.gateway.parse_relevant_context_page_response(
                        responses[1], request=artifact_request
                    )
            except Exception as exc:
                # Falha transport-level do _msearch: fallback para queries
                # separadas sem perder nenhum ramo.
                executed_via = "sequential_fallback"
                diagnostics["msearch"] = {"error": str(exc)[:160]}
                self._log(
                    trace_id,
                    "multimodal.msearch.fallback",
                    diagnostics["msearch"],
                    level=logging.WARNING,
                )
        if executed_via != "msearch":
            if t2i_embedding is not None and "error" not in t2i_diag:
                try:
                    page = await self.gateway.search_similar_images_page(
                        museum_slug=museum_slug,
                        museum_id=museum_id,
                        image_embedding=t2i_embedding,
                        from_offset=0,
                        page_size=top_k,
                        retrieval_window_size=top_k,
                    )
                    t2i_hits = list(page.results or [])
                except Exception as exc:
                    t2i_diag["error"] = f"search: {exc}"
            if doc_embedding is not None and artifact_page is None and "error" not in diagnostics.get(ARTIFACT_BRANCH, {}):
                try:
                    artifact_page = await self.gateway.search_relevant_context_page(
                        **artifact_request_kwargs
                    )
                except Exception as exc:
                    diagnostics[ARTIFACT_BRANCH] = {"error": str(exc)[:160]}
        diagnostics["execution"] = {"via": executed_via}

        # --- ramo t2i: agrupar + floor -----------------------------------------
        if t2i_hits is not None:
            grouped = group_image_hits_by_artifact(t2i_hits)
            kept, dropped = apply_score_floor(
                grouped, min_score=float(self.settings.MULTIMODAL_MIN_IMAGE_SCORE)
            )
            t2i_diag["hits"] = len(t2i_hits)
            t2i_diag["kept_after_floor"] = len(kept)
            t2i_diag["dropped_by_floor"] = [
                {"artifact_id": h.artifact_id, "score": h.score} for h in dropped
            ]
            branches[TEXT_TO_IMAGE_BRANCH] = kept
            hits_by_image_id.update({str(h.get("image_id") or ""): h for h in t2i_hits})

        # --- ramo documental ----------------------------------------------------
        artifact_docs_by_id: dict[str, dict[str, Any]] = {}
        if artifact_page is not None:
            artifact_docs = list(artifact_page.results or [])
            artifact_hits: list[BranchHit] = []
            for index, doc in enumerate(artifact_docs, start=1):
                artifact_id = str(doc.get("artifact_id") or "").strip()
                if not artifact_id:
                    continue
                artifact_docs_by_id[artifact_id] = doc
                raw_score = doc.get("score")
                artifact_hits.append(
                    BranchHit(
                        artifact_id=artifact_id,
                        rank=index,
                        score=float(raw_score)
                        if isinstance(raw_score, (int, float))
                        else None,
                    )
                )
            branches[ARTIFACT_BRANCH] = artifact_hits
            diagnostics[ARTIFACT_BRANCH] = {"hits": len(artifact_hits)}
        elif "error" in diagnostics.get(ARTIFACT_BRANCH, {}):
            self._log(
                trace_id,
                "multimodal.image_text.artifact.error",
                diagnostics[ARTIFACT_BRANCH],
                level=logging.WARNING,
            )

        if not any(branches.values()):
            return None

        fused = weighted_rrf(
            branches,
            weights={
                IMAGE_TO_IMAGE_BRANCH: float(self.settings.MULTIMODAL_I2I_WEIGHT),
                TEXT_TO_IMAGE_BRANCH: float(self.settings.MULTIMODAL_IMAGE_WEIGHT),
                ARTIFACT_BRANCH: float(
                    self.settings.MULTIMODAL_IMAGE_TEXT_ARTIFACT_WEIGHT
                ),
            },
            rrf_k=int(self.settings.MULTIMODAL_RRF_K),
        )

        outcome = await self._assemble_outcome(
            fused=fused,
            docs_by_id=artifact_docs_by_id,
            museum_slug=museum_slug,
            museum_id=museum_id,
            hits_by_image_id=hits_by_image_id,
            temporal_filter=temporal_filter,
            router_decision=router_decision,
            diagnostics=diagnostics,
            trace_id=trace_id,
            started=started,
            branch_sizes={name: len(hits) for name, hits in branches.items()},
        )
        return outcome

    async def _assemble_outcome(
        self,
        *,
        fused: list[FusedResult],
        docs_by_id: dict[str, dict[str, Any]],
        museum_slug: str,
        museum_id: str | None,
        hits_by_image_id: dict[str, dict[str, Any]],
        temporal_filter: dict[str, Any] | None,
        router_decision: VisualIntentDecision,
        diagnostics: dict[str, Any],
        trace_id: str,
        started: float,
        branch_sizes: dict[str, int],
    ) -> MultimodalFusionOutcome | None:
        missing_ids = [r.artifact_id for r in fused if r.artifact_id not in docs_by_id]
        if missing_ids:
            try:
                fetched = await self.gateway.fetch_artifacts_by_ids(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    artifact_ids=missing_ids,
                    top_k=len(missing_ids),
                )
            except Exception as exc:
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
            }

        resolved_museum = (museum_id or "").strip()
        window = None
        if temporal_filter and isinstance(temporal_filter.get("_temporal_interval"), dict):
            interval = temporal_filter["_temporal_interval"]
            window = (interval.get("start_year"), interval.get("end_year"))

        ordered_docs: list[dict[str, Any]] = []
        matched_image_hits: list[dict[str, Any]] = []
        kept_fused: list[FusedResult] = []
        dropped_temporal = 0
        for result in fused:
            doc = docs_by_id.get(result.artifact_id)
            if doc is None:
                continue
            if resolved_museum and str(doc.get("museum_id") or "").strip() != resolved_museum:
                continue
            if window is not None:
                start = doc.get("start_year")
                end = doc.get("end_year")
                if not isinstance(start, int) and not isinstance(end, int):
                    dropped_temporal += 1
                    continue  # include_unknown=False: sem anos -> fora do filtro
                doc_start = start if isinstance(start, int) else end
                doc_end = end if isinstance(end, int) else start
                if doc_end < window[0] or doc_start > window[1]:
                    dropped_temporal += 1
                    continue
            ordered_docs.append(doc)
            kept_fused.append(result)
            if result.matched_image_id:
                hit = hits_by_image_id.get(result.matched_image_id)
                if hit is not None:
                    matched_image_hits.append(hit)

        if not ordered_docs:
            diagnostics["outcome"] = "empty_after_guards"
            self._log(trace_id, "multimodal.fusion.empty", diagnostics)
            return None

        # E10: preferência in_tour pós-fusão (margem pequena e desligável).
        margin = float(getattr(self.settings, "MULTIMODAL_IN_TOUR_MARGIN", 0.0) or 0.0)
        if margin > 0:
            in_tour_map = {
                str(doc.get("artifact_id") or "").strip(): bool(doc.get("in_tour"))
                for doc in ordered_docs
            }
            reordered, promotions = promote_in_tour_within_margin(
                kept_fused, in_tour_by_artifact=in_tour_map, margin=margin
            )
            if promotions:
                docs_map = {
                    str(d.get("artifact_id") or "").strip(): d for d in ordered_docs
                }
                kept_fused = reordered
                ordered_docs = [docs_map[r.artifact_id] for r in kept_fused]
            diagnostics["in_tour_promotions"] = promotions

        diagnostics["fusion"] = {
            "rrf_k": int(self.settings.MULTIMODAL_RRF_K),
            "branch_sizes": branch_sizes,
            "fused_size": len(ordered_docs),
            "dropped_temporal": dropped_temporal,
        }
        diagnostics["latency_ms"] = {
            "total": round((time.monotonic() - started) * 1000, 1),
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
            safe_payload = {k: v for k, v in payload.items() if k != "trace_id"}
            log_event(logger, level, event, trace_id=trace_id, **safe_payload)
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
