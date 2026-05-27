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

RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT = (
    "És um assistente de reescrita de queries para OpenSearch num assistente de museu.\n"
    "Objetivo: gerar queries curtas e úteis para retrieval lexical e embedding.\n"
    "Responde APENAS JSON válido.\n"
    "Não inventes factos nem IDs.\n"
    "Remove conversa social e palavras de cortesia.\n"
    "Mantém nomes próprios e termos artísticos relevantes.\n"
    "Não expandas com sinónimos desnecessários.\n"
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


def build_retrieval_query_rewrite_prompt(
    *,
    user_query: str,
    museum_slug: str,
    museum_id: str | None,
    filters: dict[str, Any],
    sort: dict[str, Any],
) -> str:
    return (
        "Gera duas versões para retrieval:\n"
        "1) lexical_query: limpa, focada, adequada para multi_match em OpenSearch.\n"
        "2) embedding_query: equivalente semântica para embedding de texto.\n"
        "As duas devem ser curtas (idealmente <= 10 tokens) e na MESMA lingua de user_query.\n"
        "Nunca traduzas (ex.: português para inglês, ou inglês para português).\n"
        "Não inventes termos novos; simplifica apenas removendo ruído e cortesia.\n"
        "Se não houver melhoria clara, mantém o sentido literal da query.\n\n"
        "Responde com JSON estrito no formato:\n"
        '{"lexical_query":"...","embedding_query":"..."}\n\n'
        f"user_query: {user_query}\n"
        f"museum_slug: {museum_slug}\n"
        f"museum_id: {museum_id or ''}\n"
        f"active_filters_json: {json.dumps(filters, ensure_ascii=True)}\n"
        f"active_sort_json: {json.dumps(sort, ensure_ascii=True)}\n"
    )
