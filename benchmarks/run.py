from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.loaders import load_benchmark_suite
from benchmarks.report import (
    build_markdown_report,
    build_summary,
    write_markdown_report,
    write_results_csv,
    write_results_json,
    write_summary_csv,
    write_summary_json,
)
from benchmarks.runner import run_benchmark_suite
from benchmarks.variants import VARIANT_SPECS, resolve_variants



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline retrieval benchmark cases.")
    parser.add_argument(
        "--cases",
        default="benchmarks/cases/benchmark_cases.json",
        help="Path to benchmark cases JSON file.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Benchmark variant to run. Repeat the flag for multiple variants.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_runs",
        help="Directory where the run folder will be created.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable benchmark prewarm and measure raw cold start latency.",
    )
    parser.add_argument(
        "--no-assistant-selection",
        action="store_true",
        help=(
            "Do not run the benchmark LLM selector over retrieval candidates. "
            "Retrieval metrics still run normally."
        ),
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available variants and exit.",
    )
    return parser.parse_args()


def _print_variants() -> None:
    for name in sorted(VARIANT_SPECS):
        variant = VARIANT_SPECS[name]
        support_label = "supported" if variant.supported else "placeholder"
        print(f"{name}: {support_label} - {variant.description}")


async def _main_async(args: argparse.Namespace) -> int:
    suite = load_benchmark_suite(args.cases)
    variants = resolve_variants(args.variant)
    generated_at = datetime.now(timezone.utc).isoformat()
    supported_variant_names = [variant.name for variant in variants if variant.supported]
    case_modes = {case.mode for case in suite.cases if case.enabled and not case.is_incomplete}

    output_root = Path(args.output_dir).resolve()
    run_dir = output_root / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    if supported_variant_names:
        from benchmarks.executor import OfflineBenchmarkExecutor
        from benchmarks.service_factory import build_service_bundle

        executor_factory = lambda variant: OfflineBenchmarkExecutor(
            variant=variant,
            services=build_service_bundle(
                variant,
                enable_assistant_selection=not args.no_assistant_selection,
            ),
            enable_warmup=not args.no_warmup,
            include_multimodal_warmup=bool({"image", "model_3d"} & case_modes),
            include_multiview_worker_warmup="model_3d" in case_modes,
            enable_assistant_selection=not args.no_assistant_selection,
        )
    else:
        executor_factory = lambda variant: None

    results = await run_benchmark_suite(
        suite,
        variants=variants,
        executor_factory=executor_factory,
    )
    summary = build_summary(results)
    run_metadata = {
        "generated_at": generated_at,
        "cases_path": str(suite.source_path),
        "schema_version": suite.schema_version,
        "variants": [variant.name for variant in variants],
        "notes": suite.notes,
        "run_dir": str(run_dir),
        "warmup_enabled": not args.no_warmup,
        "assistant_selection_enabled": not args.no_assistant_selection,
    }

    write_results_json(run_dir / "results.json", run_metadata=run_metadata, results=results)
    write_results_csv(run_dir / "results.csv", results)
    write_summary_json(run_dir / "summary.json", summary)
    write_summary_csv(run_dir / "summary.csv", summary)
    write_markdown_report(
        run_dir / "report.md",
        build_markdown_report(
            run_metadata=run_metadata,
            summary=summary,
            results=results,
        ),
    )

    overall = summary.get("overall_counts", [{}])[0]
    print(f"Run directory: {run_dir}")
    print(
        "Cases="
        f"{overall.get('total_cases', 0)} "
        f"scored={overall.get('scored_cases', 0)} "
        f"skipped={overall.get('skipped_cases', 0)} "
        f"errors={overall.get('error_cases', 0)}"
    )
    return 0


def main() -> int:
    args = _parse_args()
    if args.list_variants:
        _print_variants()
        return 0
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
