import json
from typing import Any


ANALYTICS_PLANNER_SYSTEM_PROMPT = (
    "És um planner de queries para OpenSearch num assistente de museu.\n"
    "Objetivo: converter perguntas analíticas em JSON compacto para um compilador determinístico.\n"
    "Nunca devolvas OpenSearch DSL.\n"
    "Nunca devolvas texto fora do JSON.\n"
    "Usa apenas campos presentes no schema fornecido.\n"
    "Operações válidas: count, list, group_by, exists.\n"
    "Se a confiança for baixa, reduz o plano ao mínimo e baixa confidence."
)


def build_analytics_planner_prompt(
    *,
    question: str,
    schema_payload: dict[str, Any],
    output_schema: dict[str, Any],
) -> str:
    return (
        "Produz um plano estruturado para a pergunta do utilizador.\n"
        "Responde em JSON estrito e compacto de acordo com `output_schema_json`.\n\n"
        f"schema_json: {json.dumps(schema_payload, ensure_ascii=True)}\n"
        f"output_schema_json: {json.dumps(output_schema, ensure_ascii=True)}\n"
        f"question: {question}\n"
    )
