import json
from typing import Any


ROUTER_SYSTEM_PROMPT_PT = (
    "Encaminha pedidos do assistente virtual de museu.\n"
    "- Usa rag para pedidos factuais sobre museu/artefactos, pesquisa, localizacao, cronologia, categorias e colecao. Tambem rag se mencionar algum artefacto ou objeto.\n"
    "- Se intent for search ou refine, mode TEM de ser rag e needs_retrieval TEM de ser true.\n"
    "- Nunca devolvas mode=llm_only com intent=search/refine.\n"
    "- Usa llm_only para saudacoes, conversa geral, ou pedidos sem dados suportados.\n"
    "- Politica de precedencia: CURRENT_MESSAGE > MEDIA_ATUAL > EXPLICIT_STATE > RECENT_HISTORY_AUX > ROLLING_SUMMARY_AUX.\n"
    "- Default: nao usar contexto anterior para construir query.\n"
    "- So usar history/rolling_summary para query quando houver follow-up claro.\n"
    "- rolling_summary e apenas auxiliar; nao cria filtros novos por si.\n"
    "Regras para rewritten_query:\n"
    "- rewritten_query e a pesquisa autocontida do utilizador, na MESMA lingua da mensagem.\n"
    "- Num follow-up claro, resolve as referencias com o contexto: 'e de linho?' depois de 'encontra vestidos de seda' vira 'vestidos de linho'.\n"
    "- Se a mensagem ja for autocontida, copia-a sem alteracoes.\n"
    "- Se a mensagem for saudacao ou conversa (llm_only), copia a mensagem TAL E QUAL; nunca inventes uma pergunta.\n"
    "- NUNCA acrescentes o nome do museu, slug ou codigo (ex.: mnt, mnaz, mnsr, mj) a rewritten_query; a pesquisa ja esta limitada ao museu."
)

ROUTER_SYSTEM_PROMPT_EN = (
    "Route requests for a museum virtual assistant.\n"
    "- Use rag for factual requests about museum/artifacts, search, location, chronology, categories, and collection. Also use rag if an artifact/object is mentioned.\n"
    "- If intent is search or refine, mode MUST be rag and needs_retrieval MUST be true.\n"
    "- Never return mode=llm_only with intent=search/refine.\n"
    "- Use llm_only for greetings, small talk, or requests without grounded data.\n"
    "- Source precedence policy: CURRENT_MESSAGE > CURRENT_MEDIA > EXPLICIT_STATE > RECENT_HISTORY_AUX > ROLLING_SUMMARY_AUX.\n"
    "- Default: do not use previous context to build the query.\n"
    "- Use history/rolling_summary for query building only when there is a clear follow-up.\n"
    "- rolling_summary is auxiliary only and must not create new filters by itself.\n"
    "Rules for rewritten_query:\n"
    "- rewritten_query is the user's self-contained search request, in the SAME language as the message.\n"
    "- On a clear follow-up, resolve references using context: 'and in linen?' after 'find silk dresses' becomes 'linen dresses'.\n"
    "- If the message is already self-contained, copy it unchanged.\n"
    "- If the message is a greeting or small talk (llm_only), copy the message AS IS; never invent a question.\n"
    "- NEVER add the museum name, slug, or code (e.g. mnt, mnaz, mnsr, mj) to rewritten_query; the search is already scoped to the museum."
)

LEXICAL_QUERY_SYSTEM_PROMPT_PT = (
    "Transforma mensagens de utilizador em queries lexicais limpas para OpenSearch.\n"
    "Suporta portugues e ingles.\n"
    "Devolve apenas JSON estrito com a chave lexical_query.\n"
    "Mantem apenas termos pesquisaveis do acervo: objeto, tipo, material, tecnica, estilo, tema, epoca, local, autor, referencia/inventario.\n"
    "Remove intencao conversacional, verbos de pedido, pronomes, artigos, preposicoes, marcadores de cortesia e nomes/codigos do museu quando nao forem o alvo.\n"
    "Nao traduzas, nao expandas sinonimos e nao inventes termos.\n"
    "Nao uses operadores de query, aspas, parenteses, wildcards ou pontuacao.\n"
    "Se o pedido ja estiver limpo, devolve a mesma query."
)

LEXICAL_QUERY_SYSTEM_PROMPT_EN = (
    "Transform user messages into clean lexical queries for OpenSearch.\n"
    "Support Portuguese and English.\n"
    "Return strict JSON only with the key lexical_query.\n"
    "Keep only searchable collection terms: object, type, material, technique, style, subject, period, place, maker, reference/inventory.\n"
    "Remove conversational intent, request verbs, pronouns, articles, prepositions, politeness markers, and museum names/codes when they are not the target.\n"
    "Do not translate, expand synonyms, or invent terms.\n"
    "Do not use query operators, quotes, parentheses, wildcards, or punctuation.\n"
    "If the request is already clean, return the same query."
)

# Backward compatibility: existing imports expect this symbol.
ROUTER_SYSTEM_PROMPT = ROUTER_SYSTEM_PROMPT_PT


def _normalize_language(language: str | None) -> str:
    value = (language or "").strip().lower()
    return "en" if value == "en" else "pt"


def get_router_system_prompt(*, language: str | None = None) -> str:
    if _normalize_language(language) == "en":
        return ROUTER_SYSTEM_PROMPT_EN
    return ROUTER_SYSTEM_PROMPT_PT


def get_lexical_query_system_prompt(*, language: str | None = None) -> str:
    if _normalize_language(language) == "en":
        return LEXICAL_QUERY_SYSTEM_PROMPT_EN
    return LEXICAL_QUERY_SYSTEM_PROMPT_PT


def build_lexical_query_prompt(
    *,
    query: str,
    museum_slug: str,
    museum_id: str | None,
    language: str | None = None,
) -> str:
    resolved_language = _normalize_language(language)

    if resolved_language == "en":
        return (
            "Build one clean OpenSearch lexical query from CURRENT_MESSAGE.\n"
            "Return JSON exactly like {\"lexical_query\":\"...\"}.\n"
            "Rules:\n"
            "- Keep the original language of useful terms.\n"
            "- Remove words that only mean find/search/show/tell/want/please/about/in/of/the/a.\n"
            "- Remove museum_slug and museum_id unless the user is explicitly searching those codes as collection data.\n"
            "- Keep qualifiers that change the object, such as child, religious, Marian, blue, silk, 18th century.\n"
            "- Prefer 2 to 8 terms; max 12 terms.\n\n"
            "Examples:\n"
            "CURRENT_MESSAGE: encontra vestido de crian\u00e7a\n"
            "{\"lexical_query\":\"vestido crian\u00e7a\"}\n"
            "CURRENT_MESSAGE: mostra-me por favor azulejos com iconografia mariana no museu mnaz\n"
            "{\"lexical_query\":\"azulejos iconografia mariana\"}\n"
            "CURRENT_MESSAGE: find a child's dress please\n"
            "{\"lexical_query\":\"child dress\"}\n"
            "CURRENT_MESSAGE: show blue and white tiles from the 18th century\n"
            "{\"lexical_query\":\"blue white tiles 18th century\"}\n\n"
            "MUSEUM_CONTEXT:\n"
            f"- museum_slug: {museum_slug}\n"
            f"- museum_id: {museum_id or 'unknown'}\n\n"
            "CURRENT_MESSAGE:\n"
            f"{query}\n"
        )

    return (
        "Cria uma query lexical limpa para OpenSearch a partir de CURRENT_MESSAGE.\n"
        "Devolve JSON exatamente no formato {\"lexical_query\":\"...\"}.\n"
        "Regras:\n"
        "- Mantem a lingua original dos termos uteis.\n"
        "- Remove palavras que so significam encontrar/procurar/mostrar/dizer/querer/por favor/sobre/em/de/o/a.\n"
        "- Remove museum_slug e museum_id exceto se o utilizador estiver explicitamente a procurar esses codigos como dados do acervo.\n"
        "- Mantem qualificadores que mudam o objeto, como crian\u00e7a, religioso, mariana, azul, seda, seculo XVIII.\n"
        "- Prefere 2 a 8 termos; maximo 12 termos.\n\n"
        "Exemplos:\n"
        "CURRENT_MESSAGE: encontra vestido de crian\u00e7a\n"
        "{\"lexical_query\":\"vestido crian\u00e7a\"}\n"
        "CURRENT_MESSAGE: mostra-me por favor azulejos com iconografia mariana no museu mnaz\n"
        "{\"lexical_query\":\"azulejos iconografia mariana\"}\n"
        "CURRENT_MESSAGE: find a child's dress please\n"
        "{\"lexical_query\":\"child dress\"}\n"
        "CURRENT_MESSAGE: show blue and white tiles from the 18th century\n"
        "{\"lexical_query\":\"blue white tiles 18th century\"}\n\n"
        "MUSEUM_CONTEXT:\n"
        f"- museum_slug: {museum_slug}\n"
        f"- museum_id: {museum_id or 'desconhecido'}\n\n"
        "CURRENT_MESSAGE:\n"
        f"{query}\n"
    )


def build_router_user_prompt(
    *,
    museum_slug: str,
    museum_name: str | None,
    rolling_summary: str,
    filters_state: dict[str, Any],
    sort_state: dict[str, Any],
    history_lines: list[str],
    current_user_message: str,
    router_schema: dict[str, Any] | None = None,
    context_policy_hint: dict[str, Any] | None = None,
    conversation_state_aux: dict[str, Any] | None = None,
    language: str | None = None,
) -> str:
    resolved_language = _normalize_language(language)
    history_text = "\n".join(history_lines) if history_lines else "- vazio"
    schema_text = json.dumps(router_schema, ensure_ascii=True) if router_schema else "{}"
    context_hint_text = json.dumps(context_policy_hint or {}, ensure_ascii=True)
    state_aux_text = json.dumps(conversation_state_aux or {}, ensure_ascii=True)

    if resolved_language == "en":
        history_text = "\n".join(history_lines) if history_lines else "- empty"
        return (
            "Conversation router for a museum assistant.\n"
            "Return strict JSON that follows TARGET_SCHEMA_JSON.\n"
            "If uncertain, assume an autonomous new search (not a follow-up).\n"
            "rolling_summary/history are AUXILIARY and not source-of-truth.\n\n"
            "TARGET_SCHEMA_JSON:\n"
            f"{schema_text}\n\n"
            "MUSEUM_CONTEXT:\n"
            f"- museum_slug: {museum_slug}\n"
            f"- museum_name: {museum_name or 'unknown'}\n\n"
            "SOURCE_PRIORITY:\n"
            "1) CURRENT_MESSAGE\n"
            "2) CURRENT_MEDIA (if present)\n"
            "3) EXPLICIT_STATE_FILTERS / EXPLICIT_STATE_SORT\n"
            "4) RECENT_HISTORY_AUX (only if clear follow-up)\n"
            "5) ROLLING_SUMMARY_AUX (last resort)\n\n"
            "CURRENT_MESSAGE:\n"
            f"{current_user_message}\n\n"
            "EXPLICIT_STATE_FILTERS:\n"
            f"{json.dumps(filters_state, ensure_ascii=True)}\n\n"
            "EXPLICIT_STATE_SORT:\n"
            f"{json.dumps(sort_state, ensure_ascii=True)}\n\n"
            "CONTEXT_POLICY_HINT:\n"
            f"{context_hint_text}\n\n"
            "CONVERSATION_STATE_AUX:\n"
            f"{state_aux_text}\n\n"
            "ROLLING_SUMMARY_AUX:\n"
            f"{rolling_summary or 'empty'}\n\n"
            "RECENT_HISTORY_AUX:\n"
            f"{history_text}\n"
        )

    return (
        "Router de conversa para assistente de museu.\n"
        "Devolve JSON estrito de acordo com TARGET_SCHEMA_JSON.\n"
        "Se houver duvida, assume nova pesquisa autonoma (nao follow-up).\n"
        "rolling_summary/history sao AUXILIARES; nao sao fonte de verdade.\n\n"
        "TARGET_SCHEMA_JSON:\n"
        f"{schema_text}\n\n"
        "MUSEUM_CONTEXT:\n"
        f"- museum_slug: {museum_slug}\n"
        f"- museum_name: {museum_name or 'desconhecido'}\n\n"
        "PRIORIDADE_DE_FONTES:\n"
        "1) CURRENT_MESSAGE\n"
        "2) MEDIA_ATUAL (se existir)\n"
        "3) EXPLICIT_STATE_FILTERS / EXPLICIT_STATE_SORT\n"
        "4) RECENT_HISTORY_AUX (apenas se follow-up claro)\n"
        "5) ROLLING_SUMMARY_AUX (ultimo recurso)\n\n"
        "CURRENT_MESSAGE:\n"
        f"{current_user_message}\n\n"
        "EXPLICIT_STATE_FILTERS:\n"
        f"{json.dumps(filters_state, ensure_ascii=True)}\n\n"
        "EXPLICIT_STATE_SORT:\n"
        f"{json.dumps(sort_state, ensure_ascii=True)}\n\n"
        "CONTEXT_POLICY_HINT:\n"
        f"{context_hint_text}\n\n"
        "CONVERSATION_STATE_AUX:\n"
        f"{state_aux_text}\n\n"
        "ROLLING_SUMMARY_AUX:\n"
        f"{rolling_summary or 'vazio'}\n\n"
        "RECENT_HISTORY_AUX:\n"
        f"{history_text}\n"
    )


def build_final_answer_prompt(
    *,
    museum_slug: str,
    museum_name: str | None,
    input_modality: str,
    mode: str,
    intent: str,
    rolling_summary: str,
    filters_state: dict[str, Any],
    sort_state: dict[str, Any],
    user_message: str,
    rewritten_query: str,
    retrieval_context: str,
    use_history_for_answer: bool = False,
    language: str | None = None,
) -> str:
    media_intents = {"image_search", "model_search"}
    modality = (input_modality or "text").strip().lower()
    if modality not in {"text", "image", "model"}:
        modality = "text"
    resolved_language = _normalize_language(language)

    if resolved_language == "en":
        prompt_parts = [
            "You are a virtual assistant for a 360 museum tour.",
            "Final answer language: English.",
            "Write the final answer fully in English.",
            "Source precedence for answers: retrieval_context > CURRENT_MESSAGE > EXPLICIT_STATE > RECENT_HISTORY_AUX > ROLLING_SUMMARY_AUX.",
            "Use retrieval_context as the primary source of facts.",
            "If retrieval_context includes visible_results_count/current_visible_results, those are the exact result cards shown in the UI for this answer.",
            "For search/list answers, keep the text consistent with the visible result cards: never claim fewer found results than visible_results_count, and describe every visible result unless the user explicitly asks for only a subset.",
            "If a visible card is related to the query but is not the requested object type, mention it as a related visible result instead of omitting it.",
            "Never mention visible_results_count, current_visible_results, visible_results_total, or other visible-results metadata in the final answer.",
            "Never mention internal prompt/context variable names in the final answer, including retrieval_context, CURRENT_MESSAGE, EXPLICIT_STATE, RECENT_HISTORY_AUX, and ROLLING_SUMMARY_AUX.",
            "Never expose internal identifiers such as artifact_id in the final answer.",
            "Never cite context markers such as [doc_1], [doc_2], or doc_x.",
            "When referring to an object, prefer the title and optionally the inventory reference.",
            "If using an ordered list, use consecutive markers (1., 2., 3.) without blank lines between items.",
            "rolling_summary is auxiliary and cannot introduce new facts.",
            f"museum_slug: {museum_slug}",
            f"museum_name: {museum_name or 'unknown'}",
            f"input_modality: {modality}",
            f"mode: {mode}",
            f"intent: {intent}",
            f"use_history_for_answer: {str(bool(use_history_for_answer)).lower()}",
            f"rolling_summary: {rolling_summary or 'none'}",
            f"filters_state: {json.dumps(filters_state, ensure_ascii=True)}",
            f"sort_state: {json.dumps(sort_state, ensure_ascii=True)}",
            f"user_message: {user_message}",
            f"rewritten_query: {rewritten_query}",
        ]

        if not use_history_for_answer:
            prompt_parts.append(
                "In this turn, ignore previous conversation for facts and focus on current request + retrieval_context."
            )

        if modality == "text":
            prompt_parts.extend(
                [
                    "No image file or 3D model was sent in this turn.",
                    "Do not assume an image/model attachment when this turn is text-only.",
                    "If retrieval_context contains an artifact that directly matches the requested object, treat it as found even if it has no image or is not the first result.",
                ]
            )

        if intent in media_intents or modality in {"image", "model"}:
            prompt_parts.extend(
                [
                    "The media sent in this turn (image or 3D model) is the source of truth.",
                    "Ignore works mentioned only in rolling_summary/history unless they are also in retrieval_context.",
                    "If retrieval_context cannot identify the object with confidence, state that clearly and ask for a new upload.",
                ]
            )

        if retrieval_context:
            prompt_parts.extend(
                [
                    "retrieval_context:",
                    retrieval_context,
                    "If retrieval_context is present, do not use rolling_summary to contradict retrieved facts.",
                ]
            )
        else:
            prompt_parts.append(
                "No retrieval_context available. Answer cautiously; if factual data is missing, state that clearly."
            )

        return "\n".join(prompt_parts)

    prompt_parts = [
        "Es um assistente virtual para uma visita 360 ao museu.",
        "Idioma final da resposta: portugues.",
        "Escreve a resposta final integralmente em portugues.",
        "Politica de precedencia para resposta: retrieval_context > CURRENT_MESSAGE > EXPLICIT_STATE > RECENT_HISTORY_AUX > ROLLING_SUMMARY_AUX.",
        "Usa retrieval_context como fonte principal de factos.",
        "Se retrieval_context incluir visible_results_count/current_visible_results, esses sao os cartoes exatos mostrados na UI para esta resposta.",
        "Em respostas de pesquisa/lista, mantem o texto consistente com os cartoes visiveis: nunca digas que foram encontrados menos resultados do que visible_results_count, e descreve todos os resultados visiveis salvo se o utilizador pedir explicitamente apenas um subconjunto.",
        "Se um cartao visivel estiver relacionado com a pesquisa mas nao for o tipo de objeto pedido, menciona-o como resultado relacionado visivel em vez de o omitir.",
        "Nunca menciones visible_results_count, current_visible_results, visible_results_total ou outro metadado interno dos resultados visiveis na resposta final.",
        "Nunca exponhas variáveis como retrieval_context internos na resposta final.",
        "Nunca exponhas identificadores internos como artifact_id na resposta final.",
        "Nunca cites marcadores de contexto como [doc_1], [doc_2] ou doc_x.",
        "Ao referires um objeto, privilegia o titulo e opcionalmente a referencia de inventario.",
        "Se usares lista numerada, usa marcadores consecutivos (1., 2., 3.) sem linhas em branco entre itens.",
        "rolling_summary e auxiliar e nao pode introduzir factos novos.",
        "Nao referir nomes de variaveis ou constantes do contexto ou deste prompt, por exemplo retrieval_context, CURRENT_MESSAGE, EXPLICIT_STATE, RECENT_HISTORY_AUX, ROLLING_SUMMARY_AUX.",
        f"museum_slug: {museum_slug}",
        f"museum_name: {museum_name or 'desconhecido'}",
        f"input_modality: {modality}",
        f"mode: {mode}",
        f"intent: {intent}",
        f"use_history_for_answer: {str(bool(use_history_for_answer)).lower()}",
        f"rolling_summary: {rolling_summary or 'nenhum'}",
        f"filters_state: {json.dumps(filters_state, ensure_ascii=True)}",
        f"sort_state: {json.dumps(sort_state, ensure_ascii=True)}",
        f"user_message: {user_message}",
        f"rewritten_query: {rewritten_query}",
    ]

    if not use_history_for_answer:
        prompt_parts.append(
            "Neste turno, ignora contexto anterior para factos e foco no pedido atual + retrieval_context."
        )

    if modality == "text":
        prompt_parts.extend(
            [
                "Neste turno nao foi enviado ficheiro de imagem nem modelo 3D.",
                "Nao assumes imagem/modelo anexado quando apenas existe texto.",
                "Se retrieval_context contem um artefacto que corresponde diretamente ao objeto pedido, trata-o como encontrado mesmo que nao tenha imagem ou nao seja o primeiro resultado.",
            ]
        )

    if intent in media_intents or modality in {"image", "model"}:
        prompt_parts.extend(
            [
                "O media enviado neste turno (imagem ou modelo 3D) e a fonte de verdade.",
                "Ignora obras mencionadas apenas em rolling_summary/history, exceto se tambem estiverem em retrieval_context.",
                "Se retrieval_context nao identificar o objeto com confianca, diz isso de forma clara e pede novo upload.",
            ]
        )

    if retrieval_context:
        prompt_parts.extend(
            [
                "retrieval_context:",
                retrieval_context,
                "Se retrieval_context estiver presente, nao uses rolling_summary para contrariar factos recuperados.",
            ]
        )
    else:
        prompt_parts.append(
            "Sem retrieval_context disponivel. Responde com cautela; se faltarem dados factuais, diz isso claramente."
        )

    return "\n".join(prompt_parts)
