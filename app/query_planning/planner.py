from __future__ import annotations

import re
from typing import Any, Protocol

from app.prompts.query_planner_prompts import (
    ANALYTICS_PLANNER_SYSTEM_PROMPT,
    build_analytics_planner_prompt,
)
from app.query_planning.models import (
    CompiledOpenSearchDSL,
    ExistsFilter,
    GroupBucket,
    QueryExecutionResult,
    QueryMode,
    QueryPlan,
    QuerySchema,
    RangeFilter,
    TermFilter,
    TermsFilter,
)
from app.schemas.chat import ResponseFormatObject


class _SupportsGenerate(Protocol):
    async def generate(
        self,
        *,
        message: str,
        response_format: ResponseFormatObject,
        system_prompt: str | None = None,
        model_override: str | None = None,
    ) -> Any:
        ...


class QueryPlanningError(Exception):
    """Raised when query planning fails."""


class QueryCompileError(QueryPlanningError):
    """Raised when a plan cannot be compiled into valid OpenSearch DSL."""


class QueryExecutionError(QueryPlanningError):
    """Raised when execution against OpenSearch fails."""


def classify_query(question: str, schema: QuerySchema) -> QueryMode:
    text = _normalize_text(question)
    if not text:
        return "rag"

    analytic_score = 0
    descriptive_score = 0

    if re.search(r"\b(quant[oa]s?|numero|n[uú]mero|total|contagem|contar)\b", text):
        analytic_score += 3
    if re.search(r"\b(lista|listar|quais|enumera|mostra)\b", text):
        analytic_score += 2
    if re.search(r"\b(agrupa|agrupar|distribui[cç][aã]o)\b", text):
        analytic_score += 1
    if re.search(r"\b(existe|existem|h[aá]|tem)\b", text):
        analytic_score += 1
    if re.search(r"^\s*(existe|existem|h[aá]|tem)\b", text):
        analytic_score += 2

    if re.search(r"\b(explica|hist[oó]ria|contexto|interpreta|significa|porque)\b", text):
        descriptive_score += 3
    if re.search(r"\b(como|quando|quem)\b", text):
        descriptive_score += 1

    # Presence of explicit facetable/textual nouns with analytic verbs is usually analytical.
    if analytic_score > 0:
        matched_schema_terms = 0
        for field_name in schema.facetable_fields + schema.text_fields:
            token = field_name.split(".")[0].replace("_", " ")
            if token and token in text:
                matched_schema_terms += 1
        if matched_schema_terms:
            analytic_score += 1

    return "structured" if analytic_score >= descriptive_score + 2 else "rag"


async def plan_query(
    question: str,
    schema: QuerySchema,
    *,
    llm_service: _SupportsGenerate | None = None,
    model_override: str | None = None,
) -> QueryPlan:
    planner_llm = llm_service
    if planner_llm is None:
        from app.services.llm_service import get_llm_service

        planner_llm = get_llm_service()

    schema_payload = _schema_for_prompt(schema)
    planner_prompt = build_analytics_planner_prompt(
        question=question,
        schema_payload=schema_payload,
        output_schema=_planner_output_schema(),
    )

    try:
        response = await planner_llm.generate(
            message=planner_prompt,
            response_format=ResponseFormatObject(type="json_object"),
            system_prompt=ANALYTICS_PLANNER_SYSTEM_PROMPT,
            model_override=model_override,
        )
    except Exception as exc:
        raise QueryPlanningError(str(exc)) from exc

    parsed = response.parsed_json
    if not isinstance(parsed, dict):
        raise QueryPlanningError("Planner did not return a JSON object.")

    try:
        plan = QueryPlan.model_validate(parsed)
    except Exception as exc:
        raise QueryPlanningError(f"Planner JSON is invalid: {exc}") from exc

    if plan.query_text is None:
        plan.query_text = _extract_core_terms(question)
    return plan


def compile_query(plan: QueryPlan, schema: QuerySchema) -> CompiledOpenSearchDSL:
    _validate_plan_against_schema(plan, schema)
    bool_query = _compile_bool_query(plan=plan, schema=schema)

    if plan.operation in {"count", "exists"}:
        return CompiledOpenSearchDSL(
            endpoint="_count",
            index=schema.index_name,
            body={"query": bool_query},
        )

    if plan.operation == "list":
        list_spec = plan.list_spec
        if list_spec is None:
            raise QueryCompileError("list operation requires list_spec.")
        body: dict[str, Any] = {
            "size": max(list_spec.limit, 1),
            "query": bool_query,
        }
        if list_spec.fields:
            body["_source"] = list_spec.fields
        if list_spec.sort:
            body["sort"] = [
                {sort_clause.field: {"order": sort_clause.order}}
                for sort_clause in list_spec.sort
            ]
        return CompiledOpenSearchDSL(
            endpoint="_search",
            index=schema.index_name,
            body=body,
        )

    if plan.operation == "group_by":
        group_by = plan.group_by
        if group_by is None:
            raise QueryCompileError("group_by operation requires group_by spec.")
        order_map = {
            "count_desc": {"_count": "desc"},
            "count_asc": {"_count": "asc"},
            "term_asc": {"_key": "asc"},
            "term_desc": {"_key": "desc"},
        }
        body = {
            "size": 0,
            "query": bool_query,
            "aggs": {
                "group_by": {
                    "terms": {
                        "field": group_by.field,
                        "size": group_by.size,
                        "order": order_map[group_by.order],
                    }
                }
            },
        }
        return CompiledOpenSearchDSL(
            endpoint="_search",
            index=schema.index_name,
            body=body,
        )

    raise QueryCompileError(f"Unsupported operation: {plan.operation}")


def execute_query(
    plan: QueryPlan,
    dsl: CompiledOpenSearchDSL,
    client: Any,
) -> QueryExecutionResult:
    try:
        if dsl.endpoint == "_count":
            response = client.count(index=dsl.index, body=dsl.body)
            count = int(response.get("count", 0))
            if plan.operation == "exists":
                return QueryExecutionResult(
                    operation="exists",
                    count=count,
                    exists=count > 0,
                    total=count,
                )
            return QueryExecutionResult(
                operation="count",
                count=count,
                total=count,
            )

        if dsl.endpoint == "_search":
            response = client.search(index=dsl.index, body=dsl.body)
            if plan.operation == "list":
                hits = response.get("hits", {}).get("hits", [])
                total = _extract_total_hits(response)
                items = [hit.get("_source", {}) or {} for hit in hits]
                return QueryExecutionResult(
                    operation="list",
                    items=items,
                    total=total,
                )

            if plan.operation == "group_by":
                buckets = (
                    response.get("aggregations", {})
                    .get("group_by", {})
                    .get("buckets", [])
                )
                groups = [
                    GroupBucket(
                        key=str(bucket.get("key", "")),
                        doc_count=int(bucket.get("doc_count", 0)),
                    )
                    for bucket in buckets
                ]
                return QueryExecutionResult(
                    operation="group_by",
                    groups=groups,
                    total=sum(bucket.doc_count for bucket in groups),
                )

            raise QueryExecutionError(
                f"Unsupported operation for _search endpoint: {plan.operation}"
            )

        raise QueryExecutionError(f"Unsupported endpoint: {dsl.endpoint}")
    except QueryExecutionError:
        raise
    except Exception as exc:
        raise QueryExecutionError(str(exc)) from exc


def _extract_total_hits(response: dict[str, Any]) -> int:
    total_obj = response.get("hits", {}).get("total", 0)
    if isinstance(total_obj, dict):
        return int(total_obj.get("value", 0))
    return int(total_obj or 0)


def _validate_plan_against_schema(plan: QueryPlan, schema: QuerySchema) -> None:
    field_names = set(schema.fields.keys())

    for clause in plan.filters:
        field_name = clause.field
        if field_name not in field_names:
            raise QueryCompileError(f"Unknown filter field: {field_name}")
        if isinstance(clause, RangeFilter) and not schema.is_numeric(field_name):
            raise QueryCompileError(
                f"Range filter requires numeric field, got '{field_name}'."
            )

    if plan.group_by and not schema.is_facetable(plan.group_by.field):
        raise QueryCompileError(f"group_by field is not facetable: {plan.group_by.field}")

    if plan.list_spec:
        for field in plan.list_spec.fields:
            if field not in field_names:
                raise QueryCompileError(f"Unknown list field: {field}")
        for sort_clause in plan.list_spec.sort:
            if sort_clause.field not in field_names:
                raise QueryCompileError(f"Unknown sort field: {sort_clause.field}")


def _compile_bool_query(plan: QueryPlan, schema: QuerySchema) -> dict[str, Any]:
    filter_clauses: list[dict[str, Any]] = []
    must_clauses: list[dict[str, Any]] = []
    should_clauses: list[dict[str, Any]] = []

    for default_filter in schema.default_filters:
        filter_clauses.append(_compile_filter(default_filter))
    for clause in plan.filters:
        filter_clauses.append(_compile_filter(clause))

    text_query = (plan.query_text or "").strip()
    semantic_query = (plan.semantic_query or "").strip()
    text_fields = schema.text_fields or [
        name for name, spec in schema.fields.items() if spec.text
    ]

    if text_query and text_fields:
        must_clauses.append(
            {
                "multi_match": {
                    "query": text_query,
                    "fields": text_fields,
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        )

    if semantic_query and text_fields and semantic_query != text_query:
        should_clauses.append(
            {
                "multi_match": {
                    "query": semantic_query,
                    "fields": text_fields,
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        )

    if not filter_clauses and not must_clauses and not should_clauses:
        return {"match_all": {}}

    bool_query: dict[str, Any] = {}
    if filter_clauses:
        bool_query["filter"] = filter_clauses
    if must_clauses:
        bool_query["must"] = must_clauses
    if should_clauses:
        bool_query["should"] = should_clauses
        bool_query["minimum_should_match"] = 0 if must_clauses else 1
    return {"bool": bool_query}


def _compile_filter(clause: TermFilter | TermsFilter | RangeFilter | ExistsFilter) -> dict[str, Any]:
    if isinstance(clause, TermFilter):
        return {"term": {clause.field: clause.value}}
    if isinstance(clause, TermsFilter):
        return {"terms": {clause.field: clause.values}}
    if isinstance(clause, RangeFilter):
        payload = clause.range.model_dump(exclude_none=True)
        return {"range": {clause.field: payload}}
    if isinstance(clause, ExistsFilter):
        return {"exists": {"field": clause.field}}
    raise QueryCompileError(f"Unsupported filter clause: {type(clause)!r}")


def _schema_for_prompt(schema: QuerySchema) -> dict[str, Any]:
    return {
        "index_name": schema.index_name,
        "fields": {
            name: field.model_dump(mode="json")
            for name, field in schema.fields.items()
        },
        "facetable_fields": schema.facetable_fields,
        "text_fields": schema.text_fields,
        "semantic_fields": schema.semantic_fields,
        "default_filters": [clause.model_dump(mode="json") for clause in schema.default_filters],
    }


def _planner_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mode", "operation", "confidence", "filters"],
        "properties": {
            "mode": {"type": "string", "enum": ["structured"]},
            "operation": {
                "type": "string",
                "enum": ["count", "list", "group_by", "exists"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "query_text": {"type": ["string", "null"]},
            "semantic_query": {"type": ["string", "null"]},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["term", "terms", "range", "exists"],
                        },
                        "field": {"type": "string"},
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                            ]
                        },
                        "values": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "boolean"},
                                ]
                            },
                        },
                        "range": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "gt": {"type": "number"},
                                "gte": {"type": "number"},
                                "lt": {"type": "number"},
                                "lte": {"type": "number"},
                            },
                        },
                    },
                    "required": ["kind", "field"],
                },
            },
            "list_spec": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "sort": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["field", "order"],
                            "properties": {
                                "field": {"type": "string"},
                                "order": {"type": "string", "enum": ["asc", "desc"]},
                            },
                        },
                    },
                },
            },
            "group_by": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "field": {"type": "string"},
                    "size": {"type": "integer", "minimum": 1, "maximum": 100},
                    "order": {
                        "type": "string",
                        "enum": ["count_desc", "count_asc", "term_asc", "term_desc"],
                    },
                },
            },
        },
    }


def _normalize_text(value: str) -> str:
    lowered = (value or "").strip().lower()
    return re.sub(r"\s+", " ", lowered)


def _extract_core_terms(question: str) -> str:
    text = _normalize_text(question)
    if not text:
        return ""
    cleaned = re.sub(
        r"\b(quant[oa]s?|numero|n[uú]mero|total|lista|listar|quais|exist[ea]m?|ha|h[aá]|tem)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^\w\sÀ-ÿ]", " ", cleaned, flags=re.UNICODE)
    tokens = [token for token in cleaned.split() if len(token) > 2]
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return " ".join(deduped[:10]).strip()
