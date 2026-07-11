from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

CaseMode = Literal[
    "text_single",
    "text_multi",
    "rewriting_pair",
    "image",
    "model_3d",
    "text_to_image",
    "image_text",
]
_VALID_CASE_MODES = {
    "text_single",
    "text_multi",
    "rewriting_pair",
    "image",
    "model_3d",
    "text_to_image",
    "image_text",
}


def _clean_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_artifact_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        artifact_id = _clean_string(item)
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        cleaned.append(artifact_id)
    return cleaned


def _resolve_path(base_dir: Path, raw_value: Any) -> Path | None:
    value = _clean_string(raw_value)
    if not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _join_reasons(reasons: list[str]) -> str | None:
    unique_reasons: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        unique_reasons.append(reason)
    if not unique_reasons:
        return None
    return "; ".join(unique_reasons)


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    museum_id: str
    mode: CaseMode
    enabled: bool = True
    notes: str | None = None
    query: str | None = None
    q1: str | None = None
    q2: str | None = None
    rewritten_query: str | None = None
    message: str | None = None
    input_path: Path | None = None
    input_id: str | None = None
    target_artifact: str | None = None
    relevant_artifacts: list[str] = field(default_factory=list)
    exclude_image_ids: list[str] = field(default_factory=list)
    incomplete_reason: str | None = None

    @property
    def is_incomplete(self) -> bool:
        return self.incomplete_reason is not None

    @property
    def scoring_targets(self) -> list[str]:
        if self.relevant_artifacts:
            return list(self.relevant_artifacts)
        if self.target_artifact:
            return [self.target_artifact]
        return []

    @property
    def uses_ndcg(self) -> bool:
        return self.mode == "text_multi" or len(self.relevant_artifacts) > 1

    def input_used(self) -> str | None:
        if self.mode in {"text_single", "text_multi", "text_to_image"}:
            return self.query
        if self.mode == "rewriting_pair":
            return self.q2
        if self.input_path is not None:
            return str(self.input_path)
        return self.input_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "museum_id": self.museum_id,
            "mode": self.mode,
            "enabled": self.enabled,
            "notes": self.notes,
            "query": self.query,
            "q1": self.q1,
            "q2": self.q2,
            "rewritten_query": self.rewritten_query,
            "message": self.message,
            "input_path": str(self.input_path) if self.input_path is not None else None,
            "input_id": self.input_id,
            "target_artifact": self.target_artifact,
            "relevant_artifacts": list(self.relevant_artifacts),
            "exclude_image_ids": list(self.exclude_image_ids),
            "incomplete_reason": self.incomplete_reason,
        }


@dataclass(slots=True)
class BenchmarkSuite:
    schema_version: int
    source_path: Path
    notes: str | None
    cases: list[BenchmarkCase]


def _parse_case(raw_case: dict[str, Any], *, base_dir: Path) -> BenchmarkCase:
    case_id = _clean_string(raw_case.get("case_id"))
    museum_id = _clean_string(raw_case.get("museum_id"))
    mode_raw = _clean_string(raw_case.get("mode"))
    if not case_id:
        raise ValueError("Benchmark case is missing a valid 'case_id'.")
    if not museum_id:
        raise ValueError(f"Benchmark case '{case_id}' is missing a valid 'museum_id'.")
    if mode_raw not in _VALID_CASE_MODES:
        raise ValueError(
            f"Benchmark case '{case_id}' has invalid mode '{mode_raw}'. "
            f"Expected one of {sorted(_VALID_CASE_MODES)}."
        )

    reasons: list[str] = []
    target_artifact = _clean_string(raw_case.get("target_artifact"))
    relevant_artifacts = _clean_artifact_list(raw_case.get("relevant_artifacts"))
    if not target_artifact and not relevant_artifacts:
        reasons.append("missing_ground_truth")

    query = None
    q1 = None
    q2 = None
    rewritten_query = None
    message = _clean_string(raw_case.get("message"))
    input_path = None
    input_id = None

    if mode_raw in {"text_single", "text_multi", "text_to_image"}:
        query = _clean_string(raw_case.get("query"))
        if not query:
            reasons.append("missing_query")
    elif mode_raw == "rewriting_pair":
        q1 = _clean_string(raw_case.get("q1"))
        q2 = _clean_string(raw_case.get("q2"))
        rewritten_query = _clean_string(raw_case.get("rewritten_query"))
        if not q1:
            reasons.append("missing_q1")
        if not q2:
            reasons.append("missing_q2")
    elif mode_raw == "image":
        input_path = _resolve_path(base_dir, raw_case.get("image_path"))
        input_id = _clean_string(raw_case.get("image_id"))
        if input_path is None and not input_id:
            reasons.append("missing_image_input")
    elif mode_raw == "image_text":
        query = _clean_string(raw_case.get("query"))
        input_path = _resolve_path(base_dir, raw_case.get("image_path"))
        input_id = _clean_string(raw_case.get("image_id"))
        if not query:
            reasons.append("missing_query")
        if input_path is None and not input_id:
            reasons.append("missing_image_input")
    elif mode_raw == "model_3d":
        input_path = _resolve_path(base_dir, raw_case.get("model_path"))
        input_id = _clean_string(raw_case.get("model_id"))
        if input_path is None and not input_id:
            reasons.append("missing_model_input")

    if input_path is not None and not input_path.exists():
        reasons.append("missing_input_file")

    return BenchmarkCase(
        case_id=case_id,
        museum_id=museum_id,
        mode=mode_raw,
        enabled=bool(raw_case.get("enabled", True)),
        notes=_clean_string(raw_case.get("notes")),
        query=query,
        q1=q1,
        q2=q2,
        rewritten_query=rewritten_query,
        message=message,
        input_path=input_path,
        input_id=input_id,
        target_artifact=target_artifact,
        relevant_artifacts=relevant_artifacts,
        exclude_image_ids=_clean_artifact_list(raw_case.get("exclude_image_ids")),
        incomplete_reason=_join_reasons(reasons),
    )


def load_benchmark_suite(cases_path: str | Path) -> BenchmarkSuite:
    source_path = Path(cases_path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_cases = payload.get("cases")
        notes = _clean_string(payload.get("notes"))
        schema_version = int(payload.get("schema_version", 1))
    elif isinstance(payload, list):
        raw_cases = payload
        notes = None
        schema_version = 1
    else:
        raise ValueError("Benchmark cases file must contain either an object or a list.")

    if not isinstance(raw_cases, list):
        raise ValueError("Benchmark cases payload must contain a 'cases' list.")

    base_dir = source_path.parent
    cases: list[BenchmarkCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("Every benchmark case must be an object.")
        cases.append(_parse_case(raw_case, base_dir=base_dir))

    return BenchmarkSuite(
        schema_version=schema_version,
        source_path=source_path,
        notes=notes,
        cases=cases,
    )

