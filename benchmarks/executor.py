from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from app.schemas.chat import ResponseFormatObject
from benchmarks.loaders import BenchmarkCase
from benchmarks.metrics import hit_at_k, score_ranking, selected_artifact_hit
from benchmarks.runner import build_skip_result
from benchmarks.service_factory import BenchmarkServiceBundle
from benchmarks.variants import VariantSpec


_SINGLE_PATH_VARIANTS = {"bm25_only", "dense_only"}

_SELECTOR_SYSTEM_PROMPT = (
    "És um avaliador offline de retrieval para um assistente de museu.\n"
    "Recebes a query do utilizador e uma lista ordenada de candidatos já recuperados.\n"
    "Escolhe exatamente um artifact_id da lista que melhor responde ao pedido.\n"
    "Nunca inventes artifact_ids.\n"
    "Nunca devolvas texto fora do JSON.\n"
    'Devolve JSON estrito no formato {"selected_artifact_id":"artifact_xxx"}.\n'
    "Se vários candidatos forem plausíveis, escolhe apenas um."
)


class OfflineBenchmarkExecutor:
    def __init__(
        self,
        *,
        variant: VariantSpec,
        services: BenchmarkServiceBundle,
        enable_warmup: bool = True,
        include_multimodal_warmup: bool = True,
        include_multiview_worker_warmup: bool = False,
        enable_assistant_selection: bool = True,
    ) -> None:
        self.variant = variant
        self.services = services
        self.repo_root = Path(__file__).resolve().parents[2]
        self.enable_warmup = enable_warmup
        self.include_multimodal_warmup = include_multimodal_warmup
        self.include_multiview_worker_warmup = include_multiview_worker_warmup
        self.enable_assistant_selection = enable_assistant_selection and services.llm_service.enabled
        self._warmup_completed = False

    async def execute_case(self, case: BenchmarkCase) -> dict[str, Any]:
        if case.mode == "text_single":
            return await self._execute_text_case(case)
        if case.mode == "text_multi":
            return await self._execute_text_case(case)
        if case.mode == "rewriting_pair":
            return await self._execute_rewriting_pair(case)
        if case.mode == "image":
            return await self._execute_image_case(case)
        if case.mode == "text_to_image":
            return await self._execute_text_to_image_case(case)
        if case.mode == "image_text":
            return await self._execute_image_text_case(case)
        if case.mode == "model_3d":
            return await self._execute_model_case(case)
        raise ValueError(f"Unsupported benchmark mode '{case.mode}'.")

    def close(self) -> None:
        self.services.close()

    async def warmup(self) -> None:
        if self._warmup_completed or not self.enable_warmup:
            return
        await self.services.warmup(
            include_multimodal=self.include_multimodal_warmup,
            include_multiview_worker=self.include_multiview_worker_warmup,
        )
        self._warmup_completed = True

    def _result_template(self, case: BenchmarkCase) -> dict[str, Any]:
        return {
            "case_id": case.case_id,
            "museum_id": case.museum_id,
            "mode": case.mode,
            "variant": self.variant.name,
            "status": "scored",
            "skip_reason": None,
            "error": None,
            "notes": case.notes,
            "query": case.query,
            "q1": case.q1,
            "q2": case.q2,
            "rewritten_query": case.rewritten_query,
            "message": case.message,
            "input_path": str(case.input_path) if case.input_path is not None else None,
            "input_id": case.input_id,
            "input_used": case.input_used(),
            "ranking_artifact_ids": [],
            "q1_ranking_artifact_ids": [],
            "retrieval_top_1_artifact_id": None,
            "selected_artifact_id": None,
            "selected_artifact_rank": None,
            "selected_hit": None,
            "assistant_selection_error": None,
            "target_artifact": case.target_artifact,
            "relevant_artifacts": list(case.relevant_artifacts),
            "recall_at_1": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "hit_at_5": None,
            "hit_at_10": None,
            "precision_at_5": None,
            "mrr": None,
            "ndcg_at_5": None,
            "ndcg_at_10": None,
            "ranking_image_ids": [],
            "image_hit_at_1": None,
            "image_hit_at_5": None,
            "latency_first_ms": None,
            "latency_final_ms": None,
            "result_count": 0,
        }

    def _artifact_rank(
        self,
        ranking: list[str],
        artifact_id: str | None,
    ) -> int | None:
        selected = (artifact_id or "").strip()
        if not selected:
            return None
        try:
            return ranking.index(selected) + 1
        except ValueError:
            return None

    def _assistant_error(self) -> str | None:
        error = (self.services.llm_service.last_error or "").strip()
        return error or None

    def _selection_query_text(self, case: BenchmarkCase) -> str:
        if case.mode in {"text_single", "text_multi"}:
            return (case.query or "").strip()
        if case.mode == "rewriting_pair":
            q1 = (case.q1 or "").strip()
            q2 = (case.q2 or "").strip()
            if q1 and q2:
                return f"Contexto anterior: {q1}\nFollow-up atual: {q2}"
            return q2 or q1
        if case.mode == "image":
            return (case.message or self.services.settings.CHAT_IMAGE_DEFAULT_MESSAGE).strip()
        if case.mode == "model_3d":
            return (case.message or self.services.settings.CHAT_MODEL_DEFAULT_MESSAGE).strip()
        return (case.input_used() or "").strip()

    def _rewriting_benchmark_query(self, case: BenchmarkCase) -> str:
        raw_follow_up = (case.q2 or "").strip()
        rewritten = (case.rewritten_query or "").strip()
        if self.variant.name == "full" and rewritten:
            return rewritten
        return raw_follow_up

    def _selector_candidates_payload(
        self,
        docs: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for rank, doc in enumerate(docs, start=1):
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            candidate = {
                "rank": rank,
                "artifact_id": artifact_id,
            }
            for field_name in (
                "title",
                "inventory",
                "description",
                "museum_id",
                "museum_name",
                "author",
                "support",
                "initial_year",
                "final_year",
                "initial_century",
                "final_century",
            ):
                value = doc.get(field_name)
                if value in (None, ""):
                    continue
                candidate[field_name] = value
            candidates.append(candidate)
        return candidates

    def _extract_selected_artifact_id(
        self,
        response_payload: object,
        raw_text: str | None,
        allowed_ids: set[str],
    ) -> str | None:
        candidate_id: str | None = None
        if isinstance(response_payload, dict):
            for field_name in ("selected_artifact_id", "artifact_id", "selected_id"):
                raw_value = response_payload.get(field_name)
                if isinstance(raw_value, str) and raw_value.strip():
                    candidate_id = raw_value.strip()
                    break
        if not candidate_id and raw_text:
            match = re.search(r"artifact_[A-Za-z0-9_]+", raw_text)
            if match:
                candidate_id = match.group(0)
        if candidate_id and candidate_id in allowed_ids:
            return candidate_id
        return None

    async def _select_with_llm(
        self,
        *,
        case: BenchmarkCase,
        docs: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        if not self.enable_assistant_selection:
            return None, None
        if not docs:
            return None, None

        candidates = self._selector_candidates_payload(docs)
        if not candidates:
            return None, None

        allowed_ids = {str(candidate["artifact_id"]) for candidate in candidates}
        query_text = self._selection_query_text(case)
        selector_prompt = "\n".join(
            [
                "Seleciona o melhor candidato para o pedido do utilizador.",
                f"mode: {case.mode}",
                f"museum_id: {case.museum_id}",
                f"user_query: {query_text or 'sem texto adicional'}",
                "candidate_results_json:",
                json.dumps(candidates, ensure_ascii=False, indent=2),
                "Responde apenas com JSON estrito.",
            ]
        )

        self.services.llm_service.reset_tracking()
        try:
            response = await self.services.llm_service.generate(
                message=selector_prompt,
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=_SELECTOR_SYSTEM_PROMPT,
            )
        except Exception as exc:
            return None, f"assistant_selection_failed: {exc}"

        selected_artifact_id = self._extract_selected_artifact_id(
            response.parsed_json,
            response.text,
            allowed_ids,
        )
        if selected_artifact_id is None:
            return None, "assistant_selection_invalid_output"

        return selected_artifact_id, self._assistant_error()

    def _finalize_result(
        self,
        *,
        case: BenchmarkCase,
        ranking: list[str],
        latency_final_ms: float,
        latency_first_ms: float | None = None,
        q1_ranking: list[str] | None = None,
        selected_artifact_id: str | None = None,
        assistant_selection_error: str | None = None,
    ) -> dict[str, Any]:
        result = self._result_template(case)
        result["ranking_artifact_ids"] = ranking
        result["q1_ranking_artifact_ids"] = list(q1_ranking or [])
        result["retrieval_top_1_artifact_id"] = ranking[0] if ranking else None
        result["selected_artifact_id"] = selected_artifact_id
        result["selected_artifact_rank"] = self._artifact_rank(ranking, selected_artifact_id)
        result["selected_hit"] = selected_artifact_hit(selected_artifact_id, case.scoring_targets)
        result["assistant_selection_error"] = assistant_selection_error
        result["latency_first_ms"] = latency_first_ms
        result["latency_final_ms"] = latency_final_ms
        result["result_count"] = len(ranking)
        result.update(
            score_ranking(
                ranking,
                relevant_artifacts=case.scoring_targets,
                mode=case.mode,
            )
        )
        return result

    async def _retrieve_docs(
        self,
        *,
        museum_id: str,
        query: str,
        filters: dict[str, object] | None = None,
        sort: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        if self.variant.name in _SINGLE_PATH_VARIANTS:
            return await self._retrieve_docs_single_path(
                museum_id=museum_id,
                query=query,
                filters=filters,
                sort=sort,
            )
        _, _, docs = await self.services.chat_service._retrieve_context(
            museum_slug=museum_id,
            museum_id=museum_id,
            query=query,
            filters=filters or {},
            sort=sort or {},
        )
        return docs

    async def _retrieve_docs_single_path(
        self,
        *,
        museum_id: str,
        query: str,
        filters: dict[str, object] | None = None,
        sort: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        settings = self.services.settings
        mode = self.variant.name
        query_text = (query or "").strip()
        if not query_text:
            return []
        query_embedding: list[float] = []
        if mode == "dense_only":
            query_embedding = await self.services.embedding_provider.embed_text(query_text)
        page = await self.services.opensearch_gateway.search_relevant_context_page(
            museum_slug=museum_id,
            museum_id=museum_id,
            query_text=query_text,
            lexical_query=query_text,
            query_embedding=query_embedding,
            from_offset=0,
            page_size=max(int(settings.CHAT_RETRIEVAL_CANDIDATES), 10),
            filters=dict(filters or {}),
            sort=dict(sort or {}),
            retrieval_window_size=max(int(settings.CHAT_RETRIEVAL_PAGINATION_WINDOW), 1),
            retrieval_mode=mode,
        )
        return page.results

    def _ranking_from_docs(self, docs: list[dict[str, object]]) -> list[str]:
        ranking: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id or artifact_id in seen:
                continue
            seen.add(artifact_id)
            ranking.append(artifact_id)
        return ranking

    def _resolve_input_path(self, case: BenchmarkCase) -> Path | None:
        if case.input_path is not None:
            return case.input_path

        input_id = (case.input_id or "").strip()
        if not input_id:
            return None

        candidate = Path(input_id)
        if candidate.is_absolute() and candidate.exists():
            return candidate.resolve()
        if candidate.exists():
            return candidate.resolve()

        search_roots = [
            self.repo_root / "assets" / "images",
            self.repo_root / "assets" / "test",
            self.repo_root / "assets" / "test" / "mnt",
            self.repo_root / "assets" / "test" / "mnaz",
            self.repo_root / "assets" / "test" / "mj",
        ]
        for root in search_roots:
            direct = (root / input_id).resolve()
            if direct.exists():
                return direct
        for root in search_roots:
            if not root.exists():
                continue
            matches = list(root.rglob(input_id))
            if matches:
                return matches[0].resolve()
        return None

    async def _execute_text_case(self, case: BenchmarkCase) -> dict[str, Any]:
        start = time.perf_counter()
        docs = await self._retrieve_docs(
            museum_id=case.museum_id,
            query=case.query or "",
        )
        ranking = self._ranking_from_docs(docs)
        selected_artifact_id, assistant_selection_error = await self._select_with_llm(
            case=case,
            docs=docs,
        )
        return self._finalize_result(
            case=case,
            ranking=ranking,
            latency_final_ms=(time.perf_counter() - start) * 1000.0,
            selected_artifact_id=selected_artifact_id,
            assistant_selection_error=assistant_selection_error,
        )

    async def _execute_rewriting_pair(self, case: BenchmarkCase) -> dict[str, Any]:
        start = time.perf_counter()
        q1_docs = await self._retrieve_docs(
            museum_id=case.museum_id,
            query=case.q1 or "",
        )
        q1_ranking = self._ranking_from_docs(q1_docs)
        benchmark_query = self._rewriting_benchmark_query(case)
        docs = await self._retrieve_docs(
            museum_id=case.museum_id,
            query=benchmark_query,
            filters={},
            sort={},
        )
        ranking = self._ranking_from_docs(docs)
        selected_artifact_id, assistant_selection_error = await self._select_with_llm(
            case=case,
            docs=docs,
        )

        return self._finalize_result(
            case=case,
            ranking=ranking,
            q1_ranking=q1_ranking,
            latency_final_ms=(time.perf_counter() - start) * 1000.0,
            selected_artifact_id=selected_artifact_id,
            assistant_selection_error=assistant_selection_error,
        )

    def _visual_search_top_k(self, *, excluded_count: int) -> int:
        return (
            max(
                self.services.settings.CHAT_IMAGE_RETRIEVAL_TOP_K,
                self.services.settings.CHAT_IMAGE_ARTIFACT_TOP_K,
                10,
            )
            + excluded_count
        )

    def _apply_leave_self_out(
        self,
        case: BenchmarkCase,
        image_hits: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        excluded = {image_id for image_id in case.exclude_image_ids if image_id}
        if not excluded:
            return image_hits
        return [
            hit
            for hit in image_hits
            if str(hit.get("image_id") or "").strip() not in excluded
        ]

    async def _relevant_image_ids(self, case: BenchmarkCase) -> list[str]:
        targets = case.scoring_targets
        if not targets:
            return []
        try:
            image_docs = await self.services.opensearch_gateway.fetch_images_by_artifact_ids(
                museum_slug=case.museum_id,
                museum_id=case.museum_id,
                artifact_ids=targets,
                per_artifact=16,
                max_total=64,
            )
        except Exception:
            return []
        excluded = {image_id for image_id in case.exclude_image_ids if image_id}
        relevant: list[str] = []
        for doc in image_docs:
            image_id = str(doc.get("image_id") or "").strip()
            if image_id and image_id not in excluded:
                relevant.append(image_id)
        return relevant

    async def _score_visual_case(
        self,
        *,
        case: BenchmarkCase,
        image_hits: list[dict[str, object]],
        start: float,
    ) -> dict[str, Any]:
        image_hits = self._apply_leave_self_out(case, image_hits)
        image_ranking: list[str] = []
        seen_image_ids: set[str] = set()
        for hit in image_hits:
            image_id = str(hit.get("image_id") or "").strip()
            if not image_id or image_id in seen_image_ids:
                continue
            seen_image_ids.add(image_id)
            image_ranking.append(image_id)

        artifact_ids = self._ranking_from_docs(image_hits)
        artifact_docs = await self.services.opensearch_gateway.fetch_artifacts_by_ids(
            museum_slug=case.museum_id,
            museum_id=case.museum_id,
            artifact_ids=artifact_ids,
            top_k=max(self.services.settings.CHAT_IMAGE_ARTIFACT_TOP_K, 10),
        )
        ranking = self._ranking_from_docs(artifact_docs)
        selected_artifact_id, assistant_selection_error = await self._select_with_llm(
            case=case,
            docs=artifact_docs,
        )
        result = self._finalize_result(
            case=case,
            ranking=ranking,
            latency_final_ms=(time.perf_counter() - start) * 1000.0,
            selected_artifact_id=selected_artifact_id,
            assistant_selection_error=assistant_selection_error,
        )
        result["ranking_image_ids"] = image_ranking
        relevant_image_ids = await self._relevant_image_ids(case)
        if relevant_image_ids:
            result["image_hit_at_1"] = hit_at_k(image_ranking, relevant_image_ids, 1)
            result["image_hit_at_5"] = hit_at_k(image_ranking, relevant_image_ids, 5)
        return result

    async def _execute_image_case(self, case: BenchmarkCase) -> dict[str, Any]:
        if self.variant.name in _SINGLE_PATH_VARIANTS:
            return build_skip_result(
                case,
                self.variant,
                reason="variant_not_applicable",
                error="Single-path text variants do not apply to image cases.",
            )
        input_path = self._resolve_input_path(case)
        if input_path is None or not input_path.exists():
            return build_skip_result(
                case,
                self.variant,
                reason="unresolved_input_id",
                error=f"Could not resolve image input '{case.input_id}'.",
            )

        image_bytes = input_path.read_bytes()
        message = case.message or self.services.settings.CHAT_IMAGE_DEFAULT_MESSAGE
        start = time.perf_counter()
        image_embedding = await self.services.embedding_provider.embed_multimodal_image_bytes(
            image_bytes=image_bytes,
            text=message,
        )
        image_hits = await self.services.opensearch_gateway.search_similar_images(
            museum_slug=case.museum_id,
            museum_id=case.museum_id,
            image_embedding=image_embedding,
            top_k=self._visual_search_top_k(excluded_count=len(case.exclude_image_ids)),
        )
        result = await self._score_visual_case(case=case, image_hits=image_hits, start=start)
        result["input_path"] = str(input_path)
        result["input_used"] = str(input_path)
        return result

    async def _execute_text_to_image_case(self, case: BenchmarkCase) -> dict[str, Any]:
        if self.variant.name in _SINGLE_PATH_VARIANTS:
            return build_skip_result(
                case,
                self.variant,
                reason="variant_not_applicable",
                error="Single-path text variants do not apply to text-to-image cases.",
            )
        query_text = (case.query or "").strip()
        if not query_text:
            return build_skip_result(
                case,
                self.variant,
                reason="missing_query",
                error="text_to_image case has no query text.",
            )
        start = time.perf_counter()
        query_embedding = await self.services.embedding_provider.embed_multimodal_text_query(
            query_text
        )
        image_hits = await self.services.opensearch_gateway.search_similar_images(
            museum_slug=case.museum_id,
            museum_id=case.museum_id,
            image_embedding=query_embedding,
            top_k=self._visual_search_top_k(excluded_count=len(case.exclude_image_ids)),
        )
        return await self._score_visual_case(case=case, image_hits=image_hits, start=start)

    async def _execute_image_text_case(self, case: BenchmarkCase) -> dict[str, Any]:
        if self.variant.name in _SINGLE_PATH_VARIANTS:
            return build_skip_result(
                case,
                self.variant,
                reason="variant_not_applicable",
                error="Single-path text variants do not apply to image+text cases.",
            )
        input_path = self._resolve_input_path(case)
        if input_path is None or not input_path.exists():
            return build_skip_result(
                case,
                self.variant,
                reason="unresolved_input_id",
                error=f"Could not resolve image input '{case.input_id}'.",
            )
        query_text = (case.query or "").strip()
        if not query_text:
            return build_skip_result(
                case,
                self.variant,
                reason="missing_query",
                error="image_text case has no query text.",
            )

        image_bytes = input_path.read_bytes()
        start = time.perf_counter()
        joint_embedding = await self.services.embedding_provider.embed_multimodal_joint_image_bytes(
            image_bytes=image_bytes,
            text=query_text,
        )
        image_hits = await self.services.opensearch_gateway.search_similar_images(
            museum_slug=case.museum_id,
            museum_id=case.museum_id,
            image_embedding=joint_embedding,
            top_k=self._visual_search_top_k(excluded_count=len(case.exclude_image_ids)),
        )
        result = await self._score_visual_case(case=case, image_hits=image_hits, start=start)
        result["input_path"] = str(input_path)
        result["input_used"] = str(input_path)
        return result

    async def _execute_model_case(self, case: BenchmarkCase) -> dict[str, Any]:
        if self.variant.name in _SINGLE_PATH_VARIANTS:
            return build_skip_result(
                case,
                self.variant,
                reason="variant_not_applicable",
                error="Single-path text variants do not apply to 3D model cases.",
            )
        input_path = self._resolve_input_path(case)
        if input_path is None or not input_path.exists():
            return build_skip_result(
                case,
                self.variant,
                reason="unresolved_input_id",
                error=f"Could not resolve model input '{case.input_id}'.",
            )

        model_bytes = input_path.read_bytes()
        first_latency_ms: float | None = None
        start = time.perf_counter()

        async def _progress_cb(_: str, __: dict[str, object]) -> None:
            nonlocal first_latency_ms
            if first_latency_ms is None:
                first_latency_ms = (time.perf_counter() - start) * 1000.0

        retrieval = await self.services.model_retrieval_service.retrieve(
            museum_slug=case.museum_id,
            museum_id=case.museum_id,
            model_bytes=model_bytes,
            file_name=input_path.name,
            progress_cb=_progress_cb,
        )
        ranking = self._ranking_from_docs(retrieval.artifact_docs)
        selected_artifact_id, assistant_selection_error = await self._select_with_llm(
            case=case,
            docs=retrieval.artifact_docs,
        )
        result = self._finalize_result(
            case=case,
            ranking=ranking,
            latency_first_ms=first_latency_ms,
            latency_final_ms=(time.perf_counter() - start) * 1000.0,
            selected_artifact_id=selected_artifact_id,
            assistant_selection_error=assistant_selection_error,
        )
        result["input_path"] = str(input_path)
        result["input_used"] = str(input_path)
        return result
