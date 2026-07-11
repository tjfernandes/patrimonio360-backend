from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


_RESULT_FIELDS = [
    "case_id",
    "museum_id",
    "mode",
    "variant",
    "status",
    "skip_reason",
    "error",
    "notes",
    "query",
    "q1",
    "q2",
    "rewritten_query",
    "message",
    "input_path",
    "input_id",
    "input_used",
    "ranking_artifact_ids",
    "q1_ranking_artifact_ids",
    "retrieval_top_1_artifact_id",
    "selected_artifact_id",
    "selected_artifact_rank",
    "selected_hit",
    "assistant_selection_error",
    "target_artifact",
    "relevant_artifacts",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "hit_at_5",
    "hit_at_10",
    "precision_at_5",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
    "ranking_image_ids",
    "image_hit_at_1",
    "image_hit_at_5",
    "latency_first_ms",
    "latency_final_ms",
    "result_count",
]

_SUMMARY_METRICS = [
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "hit_at_5",
    "hit_at_10",
    "precision_at_5",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
    "image_hit_at_1",
    "image_hit_at_5",
    "selected_hit",
    "latency_first_ms",
    "latency_final_ms",
]

_SINGLE_TARGET_MODES = ("text_single", "image", "model_3d", "text_to_image", "image_text")
_TEXT_MULTI_MODES = ("text_multi",)


def _average(records: list[dict[str, Any]], field_name: str) -> float | None:
    values = [
        float(record[field_name])
        for record in records
        if record.get("status") == "scored" and record.get(field_name) is not None
    ]
    if not values:
        return None
    return sum(values) / float(len(values))


def _aggregate_records(
    records: list[dict[str, Any]],
    *,
    section: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    scored_records = [record for record in records if record.get("status") == "scored"]
    skipped_records = [record for record in records if record.get("status") == "skipped"]
    error_records = [record for record in records if record.get("status") == "error"]
    row: dict[str, Any] = {
        "section": section,
        "group_type": group_type,
        "group_value": group_value,
        "total_cases": len(records),
        "scored_cases": len(scored_records),
        "skipped_cases": len(skipped_records),
        "error_cases": len(error_records),
    }
    for field_name in _SUMMARY_METRICS:
        row[field_name] = _average(records, field_name)
    return row


def _build_section_summary(
    results: list[dict[str, Any]],
    *,
    section_name: str,
    allowed_modes: tuple[str, ...],
    include_by_mode: bool,
) -> dict[str, list[dict[str, Any]]]:
    section_records = [record for record in results if record.get("mode") in allowed_modes]
    section: dict[str, list[dict[str, Any]]] = {
        "overall": [
            _aggregate_records(
                section_records,
                section=section_name,
                group_type="overall",
                group_value=section_name,
            )
        ],
        "by_variant": [],
        "by_mode": [],
        "by_museum": [],
    }

    variant_values = sorted(
        {str(record.get("variant") or "") for record in section_records if record.get("variant")}
    )
    for variant_name in variant_values:
        variant_records = [record for record in section_records if record.get("variant") == variant_name]
        section["by_variant"].append(
            _aggregate_records(
                variant_records,
                section=section_name,
                group_type="variant",
                group_value=variant_name,
            )
        )

    if include_by_mode:
        mode_values = sorted(
            {str(record.get("mode") or "") for record in section_records if record.get("mode")}
        )
        for mode in mode_values:
            mode_records = [record for record in section_records if record.get("mode") == mode]
            section["by_mode"].append(
                _aggregate_records(
                    mode_records,
                    section=section_name,
                    group_type="mode",
                    group_value=mode,
                )
            )

    museum_values = sorted(
        {str(record.get("museum_id") or "") for record in section_records if record.get("museum_id")}
    )
    for museum_id in museum_values:
        museum_records = [record for record in section_records if record.get("museum_id") == museum_id]
        section["by_museum"].append(
            _aggregate_records(
                museum_records,
                section=section_name,
                group_type="museum",
                group_value=museum_id,
            )
        )

    return section


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_counts": [
            _aggregate_records(
                results,
                section="overall_counts",
                group_type="overall",
                group_value="all",
            )
        ],
        "single_target_family": _build_section_summary(
            results,
            section_name="single_target_family",
            allowed_modes=_SINGLE_TARGET_MODES,
            include_by_mode=True,
        ),
        "text_multi": _build_section_summary(
            results,
            section_name="text_multi",
            allowed_modes=_TEXT_MULTI_MODES,
            include_by_mode=False,
        ),
    }


def _flatten_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(list(summary.get("overall_counts", [])))
    for section_name in ("single_target_family", "text_multi"):
        section = summary.get(section_name, {})
        for bucket_name in ("overall", "by_variant", "by_mode", "by_museum"):
            rows.extend(list(section.get(bucket_name, [])))
    return rows


def write_results_json(
    output_path: Path,
    *,
    run_metadata: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "run_metadata": run_metadata,
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_results_csv(output_path: Path, results: list[dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["ranking_artifact_ids"] = "|".join(result.get("ranking_artifact_ids", []))
            row["q1_ranking_artifact_ids"] = "|".join(result.get("q1_ranking_artifact_ids", []))
            row["relevant_artifacts"] = "|".join(result.get("relevant_artifacts", []))
            row["ranking_image_ids"] = "|".join(result.get("ranking_image_ids") or [])
            row = {key: value for key, value in row.items() if key in _RESULT_FIELDS}
            writer.writerow(row)


def write_summary_json(output_path: Path, summary: dict[str, Any]) -> None:
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary_csv(output_path: Path, summary: dict[str, Any]) -> None:
    rows = _flatten_summary(summary)
    fieldnames = [
        "section",
        "group_type",
        "group_value",
        "total_cases",
        "scored_cases",
        "skipped_cases",
        "error_cases",
        *_SUMMARY_METRICS,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def build_rewriting_delta_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_rows = {
        record["case_id"]: record
        for record in results
        if record.get("variant") == "full" and record.get("mode") == "rewriting_pair"
    }
    no_rewriting_rows = {
        record["case_id"]: record
        for record in results
        if record.get("variant") == "no_rewriting" and record.get("mode") == "rewriting_pair"
    }

    rows: list[dict[str, Any]] = []
    for case_id in sorted(set(full_rows) & set(no_rewriting_rows)):
        full_row = full_rows[case_id]
        no_rewriting_row = no_rewriting_rows[case_id]
        rows.append(
            {
                "case_id": case_id,
                "full_recall_at_5": full_row.get("recall_at_5"),
                "no_rewriting_recall_at_5": no_rewriting_row.get("recall_at_5"),
                "delta_recall_at_5": None
                if full_row.get("recall_at_5") is None or no_rewriting_row.get("recall_at_5") is None
                else float(full_row["recall_at_5"]) - float(no_rewriting_row["recall_at_5"]),
                "full_mrr": full_row.get("mrr"),
                "no_rewriting_mrr": no_rewriting_row.get("mrr"),
                "delta_mrr": None
                if full_row.get("mrr") is None or no_rewriting_row.get("mrr") is None
                else float(full_row["mrr"]) - float(no_rewriting_row["mrr"]),
                "full_top_1": (full_row.get("ranking_artifact_ids") or [None])[0],
                "no_rewriting_top_1": (no_rewriting_row.get("ranking_artifact_ids") or [None])[0],
                "full_selected": full_row.get("selected_artifact_id"),
                "no_rewriting_selected": no_rewriting_row.get("selected_artifact_id"),
                "full_latency_final_ms": full_row.get("latency_final_ms"),
                "no_rewriting_latency_final_ms": no_rewriting_row.get("latency_final_ms"),
            }
        )
    return rows


def _append_single_target_tables(lines: list[str], section: dict[str, list[dict[str, Any]]]) -> None:
    lines.append("## Text Single, Image, 3D")

    lines.append("### By Mode")
    lines.append("| Mode | Scored | Recall@1 | Recall@5 | Recall@10 | MRR | LLM Final Selection | Avg Final ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in section.get("by_mode", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("group_value")),
                    str(row.get("scored_cases", 0)),
                    _format_metric(row.get("recall_at_1")),
                    _format_metric(row.get("recall_at_5")),
                    _format_metric(row.get("recall_at_10")),
                    _format_metric(row.get("mrr")),
                    _format_metric(row.get("selected_hit")),
                    _format_metric(row.get("latency_final_ms")),
                ]
            )
            + " |"
        )
    lines.append("")

    if section.get("by_museum"):
        lines.append("### By Museum")
        lines.append("| Museum | Scored | Recall@1 | Recall@5 | Recall@10 | MRR | LLM Final Selection | Avg Final ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in section["by_museum"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("group_value")),
                        str(row.get("scored_cases", 0)),
                        _format_metric(row.get("recall_at_1")),
                        _format_metric(row.get("recall_at_5")),
                        _format_metric(row.get("recall_at_10")),
                        _format_metric(row.get("mrr")),
                        _format_metric(row.get("selected_hit")),
                        _format_metric(row.get("latency_final_ms")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if section.get("by_variant"):
        lines.append("### By Variant")
        lines.append("| Variant | Scored | Recall@1 | Recall@5 | Recall@10 | MRR | LLM Final Selection | Avg Final ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in section["by_variant"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("group_value")),
                        str(row.get("scored_cases", 0)),
                        _format_metric(row.get("recall_at_1")),
                        _format_metric(row.get("recall_at_5")),
                        _format_metric(row.get("recall_at_10")),
                        _format_metric(row.get("mrr")),
                        _format_metric(row.get("selected_hit")),
                        _format_metric(row.get("latency_final_ms")),
                    ]
                )
                + " |"
            )
        lines.append("")


def _append_text_multi_tables(lines: list[str], section: dict[str, list[dict[str, Any]]]) -> None:
    lines.append("## Text Multi")

    overall_row = section.get("overall", [{}])[0]
    lines.append("### Overall")
    lines.append("| Scored | Hit@5 | Precision@5 | nDCG@5 | nDCG@10 | LLM Final Selection | Avg Final ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        "| "
        + " | ".join(
            [
                str(overall_row.get("scored_cases", 0)),
                _format_metric(overall_row.get("hit_at_5")),
                _format_metric(overall_row.get("precision_at_5")),
                _format_metric(overall_row.get("ndcg_at_5")),
                _format_metric(overall_row.get("ndcg_at_10")),
                _format_metric(overall_row.get("selected_hit")),
                _format_metric(overall_row.get("latency_final_ms")),
            ]
        )
        + " |"
    )
    lines.append("")

    if section.get("by_museum"):
        lines.append("### By Museum")
        lines.append("| Museum | Scored | Hit@5 | Precision@5 | nDCG@5 | nDCG@10 | LLM Final Selection | Avg Final ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in section["by_museum"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("group_value")),
                        str(row.get("scored_cases", 0)),
                        _format_metric(row.get("hit_at_5")),
                        _format_metric(row.get("precision_at_5")),
                        _format_metric(row.get("ndcg_at_5")),
                        _format_metric(row.get("ndcg_at_10")),
                        _format_metric(row.get("selected_hit")),
                        _format_metric(row.get("latency_final_ms")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if section.get("by_variant"):
        lines.append("### By Variant")
        lines.append("| Variant | Scored | Hit@5 | Precision@5 | nDCG@5 | nDCG@10 | LLM Final Selection | Avg Final ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in section["by_variant"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("group_value")),
                        str(row.get("scored_cases", 0)),
                        _format_metric(row.get("hit_at_5")),
                        _format_metric(row.get("precision_at_5")),
                        _format_metric(row.get("ndcg_at_5")),
                        _format_metric(row.get("ndcg_at_10")),
                        _format_metric(row.get("selected_hit")),
                        _format_metric(row.get("latency_final_ms")),
                    ]
                )
                + " |"
            )
        lines.append("")


def build_markdown_report(
    *,
    run_metadata: dict[str, Any],
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    lines: list[str] = ["# Offline Benchmark Report", ""]
    lines.append("## Run")
    lines.append(f"- Generated at: {run_metadata.get('generated_at')}")
    lines.append(f"- Cases file: `{run_metadata.get('cases_path')}`")
    lines.append(f"- Variants: {', '.join(run_metadata.get('variants', []))}")
    lines.append(
        f"- LLM candidate selection: {'enabled' if run_metadata.get('assistant_selection_enabled') else 'disabled'}"
    )
    lines.append(
        "- Se o selector LLM falhar, os campos `selected_*` ficam vazios e o detalhe aparece em `assistant_selection_error` no JSON/CSV."
    )
    lines.append(
        "- `Avg Final ms` corresponde à média por pedido da latência end-to-end até ao fim da execução do benchmark case."
    )
    lines.append(
        "- Retrieval query behavior: exact current production pipeline; the benchmark does not change OpenSearch query construction."
    )
    lines.append("")

    overall_counts = summary.get("overall_counts", [{}])[0]
    lines.append("## Overall Counts")
    lines.append("| Total | Scored | Skipped | Errors | Avg Final ms |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        "| "
        + " | ".join(
            [
                str(overall_counts.get("total_cases", 0)),
                str(overall_counts.get("scored_cases", 0)),
                str(overall_counts.get("skipped_cases", 0)),
                str(overall_counts.get("error_cases", 0)),
                _format_metric(overall_counts.get("latency_final_ms")),
            ]
        )
        + " |"
    )
    lines.append("")

    _append_single_target_tables(lines, summary.get("single_target_family", {}))
    _append_text_multi_tables(lines, summary.get("text_multi", {}))

    rewriting_rows = build_rewriting_delta_rows(results)
    lines.append("## Rewriting Delta")
    if not rewriting_rows:
        lines.append(
            "No comparable `full` vs `no_rewriting` rewriting-pair rows were produced in this run."
        )
    else:
        lines.append(
            "| Case | Full top-1 | Full selected | Full Avg Final ms | No rewriting top-1 | No rewriting selected | No rewriting Avg Final ms | Delta Recall@5 | Delta MRR |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rewriting_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("case_id")),
                        str(row.get("full_top_1")),
                        str(row.get("full_selected")),
                        _format_metric(row.get("full_latency_final_ms")),
                        str(row.get("no_rewriting_top_1")),
                        str(row.get("no_rewriting_selected")),
                        _format_metric(row.get("no_rewriting_latency_final_ms")),
                        _format_metric(row.get("delta_recall_at_5")),
                        _format_metric(row.get("delta_mrr")),
                    ]
                )
                + " |"
            )
    lines.append("")

    skipped_rows = [record for record in results if record.get("status") == "skipped"]
    if skipped_rows:
        lines.append("## Skipped Cases")
        for row in skipped_rows:
            lines.append(
                f"- `{row.get('variant')}` / `{row.get('case_id')}`: "
                f"{row.get('skip_reason')} ({row.get('error') or 'no extra detail'})"
            )
        lines.append("")

    error_rows = [record for record in results if record.get("status") == "error"]
    if error_rows:
        lines.append("## Errors")
        for row in error_rows:
            lines.append(f"- `{row.get('variant')}` / `{row.get('case_id')}`: {row.get('error')}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(output_path: Path, markdown_text: str) -> None:
    output_path.write_text(markdown_text, encoding="utf-8")
