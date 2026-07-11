from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any, Protocol

from benchmarks.loaders import BenchmarkCase, BenchmarkSuite
from benchmarks.variants import VariantSpec


class BenchmarkExecutorProtocol(Protocol):
    async def execute_case(self, case: BenchmarkCase) -> dict[str, Any]:
        ...

    async def warmup(self) -> Any:
        ...

    def close(self) -> Any:
        ...


def _base_result(case: BenchmarkCase, variant: VariantSpec) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "museum_id": case.museum_id,
        "mode": case.mode,
        "variant": variant.name,
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


def build_skip_result(
    case: BenchmarkCase,
    variant: VariantSpec,
    *,
    reason: str,
    error: str | None = None,
) -> dict[str, Any]:
    result = _base_result(case, variant)
    result["status"] = "skipped"
    result["skip_reason"] = reason
    result["error"] = error
    return result


def build_error_result(
    case: BenchmarkCase,
    variant: VariantSpec,
    *,
    error: str,
) -> dict[str, Any]:
    result = _base_result(case, variant)
    result["status"] = "error"
    result["error"] = error
    return result


async def run_benchmark_suite(
    suite: BenchmarkSuite,
    *,
    variants: list[VariantSpec],
    executor_factory: Callable[[VariantSpec], BenchmarkExecutorProtocol],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for variant in variants:
        executor: BenchmarkExecutorProtocol | None = None
        if variant.supported:
            executor = executor_factory(variant)
            warmup = getattr(executor, "warmup", None)
            if callable(warmup):
                warmup_result = warmup()
                if inspect.isawaitable(warmup_result):
                    await warmup_result

        try:
            for case in suite.cases:
                if not case.enabled:
                    results.append(build_skip_result(case, variant, reason="disabled_case"))
                    continue
                if case.is_incomplete:
                    results.append(
                        build_skip_result(
                            case,
                            variant,
                            reason="incomplete_case",
                            error=case.incomplete_reason,
                        )
                    )
                    continue
                if not variant.supported:
                    results.append(
                        build_skip_result(
                            case,
                            variant,
                            reason="unsupported_variant",
                            error=variant.unsupported_reason,
                        )
                    )
                    continue
                try:
                    result = await executor.execute_case(case)
                except Exception as exc:
                    results.append(build_error_result(case, variant, error=str(exc)))
                    continue
                results.append(result)
        finally:
            if executor is not None:
                close_result = executor.close()
                if inspect.isawaitable(close_result):
                    await close_result

    return results
