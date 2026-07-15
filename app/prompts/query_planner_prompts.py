import json
from typing import Any


FOLLOW_UP_RESOLUTION_SYSTEM_PROMPT = (
    "Resolves referências em follow-ups de pesquisa de um assistente de museu.\n"
    "A tua única tarefa é reescrever a mensagem atual como pesquisa autocontida,\n"
    "usando as mensagens anteriores do utilizador para resolver o que está implícito.\n"
    "Responde APENAS JSON válido.\n"
)


def build_follow_up_resolution_prompt(
    *,
    current_message: str,
    previous_user_messages: list[str],
) -> str:
    previous_lines = "\n".join(f"- {message}" for message in previous_user_messages) or "- (vazio)"
    return (
        "Reescreve a MENSAGEM_ATUAL como uma pesquisa autocontida, resolvendo referências\n"
        "com as MENSAGENS_ANTERIORES do utilizador.\n\n"
        "Regras obrigatórias:\n"
        "- Usa APENAS palavras da MENSAGEM_ATUAL e das MENSAGENS_ANTERIORES; nunca inventes termos novos.\n"
        "- Mantém a língua da MENSAGEM_ATUAL; nunca traduzas.\n"
        "- Nunca acrescentes nomes de museus, slugs ou códigos.\n"
        "- Se a MENSAGEM_ATUAL já for autocontida, ou não for uma pesquisa, copia-a tal e qual.\n\n"
        "Exemplos:\n"
        "MENSAGENS_ANTERIORES:\n- encontra brincos de ouro\nMENSAGEM_ATUAL: e de prata?\n"
        "output: {\"resolved_query\":\"brincos de prata\"}\n"
        "MENSAGENS_ANTERIORES:\n- painéis com pássaros\nMENSAGEM_ATUAL: e com flores?\n"
        "output: {\"resolved_query\":\"painéis com flores\"}\n"
        "MENSAGENS_ANTERIORES:\n- mostra vestidos de noiva\nMENSAGEM_ATUAL: há mais?\n"
        "output: {\"resolved_query\":\"vestidos de noiva\"}\n"
        "MENSAGENS_ANTERIORES:\n- azulejos azuis\nMENSAGEM_ATUAL: quero ver esculturas\n"
        "output: {\"resolved_query\":\"quero ver esculturas\"}\n\n"
        "Responde apenas JSON:\n"
        "{\"resolved_query\":\"...\"}\n\n"
        "MENSAGENS_ANTERIORES:\n"
        f"{previous_lines}\n"
        f"MENSAGEM_ATUAL: {current_message}\n"
    )


RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT = (
    "És um extrator de expressões de pesquisa para um índice de acervo museológico.\n"
    "A tua tarefa NÃO é responder ao utilizador.\n"
    "A tua tarefa NÃO é resumir a pergunta.\n"
    "A tua tarefa é extrair a expressão nominal concreta que deve ser pesquisada no índice.\n"
    "Responde APENAS JSON válido.\n"
)


def build_retrieval_query_rewrite_prompt(
    *,
    user_query: str,
    filters: dict[str, Any],
    sort: dict[str, Any],
) -> str:
    return (
        "És um extrator de expressões de pesquisa para um índice de acervo museológico.\n\n"
        "A tua tarefa NÃO é responder ao utilizador.\n"
        "A tua tarefa NÃO é resumir a pergunta.\n"
        "A tua tarefa é extrair a expressão nominal concreta que deve ser pesquisada no índice.\n\n"
        "Regras obrigatórias:\n"
        "- Mantém a expressão do utilizador na mesma língua.\n"
        "- Nunca traduzas.\n"
        "- Nunca substituas por sinónimos.\n"
        "- Nunca mudes singular/plural.\n"
        "- Nunca removas palavras internas de expressões compostas: \"de\", \"do\", \"da\", \"dos\", \"das\" devem ficar quando fazem parte do nome.\n"
        "- Nao incluas anos, datas, seculos, decadas ou periodos historicos quando forem restricoes temporais; isso e filtro de metadata.\n"
        "- Nao incluas frases de scope como \"nesta visita virtual\", \"na visita 360\" ou \"no tour\"; isso e filtro de metadata.\n"
        "- A query extraida sera usada tanto para BM25 como para embeddings; deve ser curta, nominal e pesquisavel.\n"
        "- Remove intencoes de resposta, ranking ou planeamento: \"porque sao importantes\", \"para Portugal\", \"nao devo perder\", \"imperdivel\", \"recomendados\", \"15 minutos\", \"tempo limitado\".\n"
        "- Remove molduras interrogativas: \"quem sao\", \"quais sao\", \"se eu tiver\", \"o que\", \"porque\", \"explica\".\n"
        "- Remove apenas intenção/conversa: \"consegues\", \"podes\", \"encontra\", \"existem\", \"há\", \"quero\", \"procuro\".\n"
        "- Remove termos genéricos de enquadramento: \"museu\", \"acervo\", \"coleção\", \"peças\", \"objetos\", exceto se forem a única coisa pesquisável.\n"
        "- Se existir uma expressão específica de objeto, material, tema, autor ou título, usa essa expressão.\n"
        "- Prefere a expressão específica mais longa em vez de uma palavra isolada.\n"
        "- Se não tiveres certeza, copia a expressão mais literal da pergunta; não inventes uma query nova.\n"
        "- NUNCA acrescentes palavras que não estão na user_query — em particular nomes de museus, slugs ou códigos (mnt, mnaz, mnsr, mj) e localidades.\n"
        "- Se a user_query não contém NADA pesquisável (saudação, agradecimento, conversa), devolve {\"lexical_query\":\"\"} — nunca inventes uma pesquisa.\n\n"
        "Exemplos:\n"
        "user_query: Encontra brincos\n"
        "output: {\"lexical_query\":\"brincos\"}\n"
        "user_query: Olá\n"
        "output: {\"lexical_query\":\"\"}\n"
        "user_query: obrigado, gostei muito\n"
        "output: {\"lexical_query\":\"\"}\n"
        "user_query: vestidos pretos\n"
        "output: {\"lexical_query\":\"vestidos pretos\"}\n"
        "user_query: Consegues encontrar vestidos de noiva?\n"
        "output: {\"lexical_query\":\"vestidos de noiva\"}\n"
        "user_query: existem fatos de banho no acervo do museu?\n"
        "output: {\"lexical_query\":\"fatos de banho\"}\n"
        "user_query: há túmulos de reis?\n"
        "output: {\"lexical_query\":\"túmulos de reis\"}\n"
        "user_query: e o túmulo de Fernando Pessoa\n"
        "output: {\"lexical_query\":\"túmulo de Fernando Pessoa\"}\n\n"
        "user_query: Quem são as personalidades sepultadas nos Jerónimos e porque são importantes para Portugal?\n"
        "output: {\"lexical_query\":\"personalidades sepultadas nos Jerónimos\"}\n"
        "user_query: Se eu tiver apenas 15 minutos para uma visita virtual, quais são os objetos que não devo perder?\n"
        "output: {\"lexical_query\":\"objetos\"}\n\n"
        "Contraexemplos proibidos:\n"
        "- \"fatos de banho\" nunca deve virar \"banhos\".\n"
        "- \"vestidos de noiva\" nunca deve virar \"vestidos\".\n"
        "- \"túmulo de Fernando Pessoa\" nunca deve virar \"Fernando Pessoa\" se o objeto \"túmulo\" foi pedido.\n"
        "- Nunca incluir \"museu\", \"acervo\" ou \"coleção\" quando há termos específicos.\n\n"
        "Responde apenas JSON:\n"
        "{\"lexical_query\":\"...\"}\n\n"
        f"user_query: {user_query}\n"
        f"active_filters_json: {json.dumps(filters, ensure_ascii=True)}\n"
        f"active_sort_json: {json.dumps(sort, ensure_ascii=True)}\n"
    )
