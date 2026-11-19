import json
from typing import Any


FOLLOW_UP_RESOLUTION_SYSTEM_PROMPT = (
    "Resolves referências em follow-ups de pesquisa de um assistente de museu.\n"
    "A tua única tarefa é reescrever a mensagem atual como pesquisa autocontida,\n"
    "usando as mensagens anteriores do utilizador para resolver o que está implícito.\n"
    "Responde APENAS JSON válido.\n"
)


def build_request_reading_prompt(
    *,
    museum_name: str | None,
    history_lines: list[str],
    message: str,
) -> str:
    """Leitura do pedido (CHAT_REQUEST_READING). Redação medida a 2026-11-16 em
    120 casos × 3 execuções (tests/request_reading_cases.py): seguimento e
    língua 120/120, pedido 118/120. Qualquer alteração repete essa medição.
    """
    history = "\n".join(line for line in history_lines if line.strip()) or "(nenhuma)"
    museum = (museum_name or "").strip() or "museu"
    return (
        f"És o pré-processador de um guia de museu (visita virtual 360° ao {museum}). "
        "Lê a MENSAGEM do visitante, tendo em conta a CONVERSA ANTERIOR.\n\n"
        f"CONVERSA ANTERIOR:\n{history}\n\nMENSAGEM: {message}\n\n"
        'Responde só com JSON: {"lingua": "pt|en|es|outra", "seguimento": true|false, "pedido": "..."}\n\n'
        "seguimento: true quando a mensagem só se percebe com a conversa anterior "
        "(pronomes como «o», «ela», «delas», «it»; «e…?», «só as…»).\n"
        "pedido: a mensagem reescrita de forma completa, na língua da mensagem, com as referências resolvidas pela conversa anterior "
        "(«Quem o fez?» depois de se falar da Custódia de Belém → «Quem fez a Custódia de Belém?»). Se a mensagem já é completa, copia-a. "
        "Não acrescentes nada que não esteja na mensagem ou na conversa."
    )


def build_author_name_prompt(*, message: str, words: str) -> str:
    """Ficha do autor (CHAT_AUTHOR_CARD): `words` são as palavras da mensagem que
    estão no nome de um autor das peças mostradas. São o nome de um artista?
    Só a resposta A junta a ficha.

    Redação medida a 2026-11-18 (tests/author_card_cases.py, seis museus): 71
    de 74 mensagens com o nome de um autor recebem A; 6 de 71 perguntas feitas
    para enganar também («Que autor tem mais obras?», «Onde fica a sala do
    Germain?»). O prompt não diz o nome do autor no catálogo: com ele, o modelo
    respondia A a quase tudo (13 de 71, e a 34 de 129 mensagens reais em vez de
    19). Pedir ao modelo para escolher o autor numa lista também não serviu:
    com «Gregório Lopes» em primeiro, escolhia o segundo. Pelo código do
    serviço, só com os autores dos cartões: 68 de 74 e 1 de 71
    (docs/ficha-do-autor.md). Qualquer alteração repete a medição.
    """
    return (
        f"Mensagem de um visitante do museu:\n«{message}»\n\n"
        f"Na mensagem, «{words}» é o nome de quê?\n"
        "A) de um artista ou de uma oficina (quem fez obras)\n"
        "B) de um santo, de uma figura religiosa ou de quem está representado numa obra\n"
        "C) de um rei, de uma rainha ou de outra figura histórica\n"
        "D) de um lugar, de uma sala ou de um museu\n"
        "E) de uma peça\n"
        "F) não é um nome próprio\n"
        'Responde só com JSON: {"resposta": "A" a "F"}'
    )


IMPLICIT_ARTIFACT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWYZ"  # sem o X, que é a peça aberta na visita
IMPLICIT_ARTIFACT_OPEN_LETTER = "X"


def build_implicit_artifact_prompt(*, entities: list[dict], open_piece: dict | None, message: str) -> str:
    """Peça implícita pelo modelo (CHAT_IMPLICIT_ARTIFACT_MODEL): de que peça do ecrã fala
    a mensagem? `entities` são as peças mostradas, pela ordem do ecrã (`title`,
    `inventory_number`, `set_name`); `open_piece` é a peça aberta na visita quando não é
    uma delas. O modelo responde com a letra, ou «nenhuma».

    Redação medida a 2026-11-19 (tests/implicit_artifact_cases.py): com a posição e o n.º
    de inventário resolvidos antes por procura exata e as regras como rede, 67 de 68 casos
    de afinação e 57 de 59 de validação (regras sozinhas: 61 e 52). O que a medição ensinou:
    - a lista vai por letras: com números, o modelo respondia com o n.º de inventário;
    - o campo «quantas» vem primeiro: sem ele, uma pesquisa no plural com as palavras de um
      cartão («Retratos de família») era lida como esse cartão;
    - a peça aberta na visita vai à parte, com a letra X: no fim da lista, era «o último»;
    - muitas condições na resposta «nenhuma» fazem o modelo recusar referências claras;
    - uma só pergunta para tudo (peça nomeada, peça em foco, pesquisa) dá 50 a 61 de 68: o
      modelo confunde pesquisas com perguntas sobre a peça em foco. Essa parte fica nas regras.
    Qualquer alteração repete a medição (docs/peca-implicita-pelo-modelo.md).
    """

    def line(letter: str, entity: dict) -> str:
        text = f"{letter}) " + (f"«{entity['title']}»" if entity.get("title") else "peça sem título")
        if entity.get("inventory_number"):
            text += f" (n.º {entity['inventory_number']})"
        if entity.get("set_name"):
            text += f" — do conjunto «{entity['set_name']}»"
        return text

    listed = "\n".join(line(letter, entity) for letter, entity in zip(IMPLICIT_ARTIFACT_LETTERS, entities))
    if open_piece is not None:
        listed += ("\n" if listed else "") + "Fora desta lista, o visitante tem aberta na visita a peça:\n"
        listed += line(IMPLICIT_ARTIFACT_OPEN_LETTER, open_piece)
    return (
        "Um visitante de um museu fala com um assistente. O ecrã mostra estas peças, por esta ordem:\n"
        f"{listed}\n\nMensagem do visitante:\n«{message}»\n\n"
        "Primeiro: a mensagem fala de uma peça só, ou de várias peças (um tipo de peças)?\n"
        "Depois: se fala de uma só e diz qual é, pelo nome, pelo número de inventário ou pela posição no ecrã, a letra dessa peça. "
        "Senão, «nenhuma»: fala de várias, é uma pergunta sem nome de peça, ou é outro assunto.\n"
        'Responde só com JSON: {"quantas": "uma" ou "várias", "peca": "letra" ou "nenhuma"}'
    )


def build_author_only_prompt(*, message: str, words: str) -> str:
    """Só peças do autor (CHAT_AUTHOR_ONLY_RESULTS): a mensagem diz o nome de um
    artista (`words`, já confirmado por build_author_name_prompt). O visitante
    quer só as obras dele? Só a resposta A restringe os cartões ao autor.

    Redação medida a 2026-11-19 (tests/author_card_cases.py, seis museus): as 74
    mensagens que pedem um autor recebem A; das 31 que dizem o nome de um autor
    mas querem também outros («obras de Cranach e de Holbein», «pintores
    parecidos com Silva Porto», «sem ser as da Josefa de Óbidos»), 2 recebem A.
    Outras redações deram pior: «Que obras quer o visitante?» e «O visitante
    quer ver só obras de X?» deixavam passar 13 e 16 das 31; sem o texto entre
    parênteses na resposta A, 3 «Quem foi X?» em 74 recebiam B. Qualquer
    alteração repete a medição (docs/ficha-do-autor.md).
    """
    return (
        f"Mensagem de um visitante do museu:\n«{message}»\n\n"
        f"«{words}» é o nome de um artista. Para responder a esta mensagem, que peças do museu se devem mostrar?\n"
        f"A) só peças feitas por «{words}» (o visitante quer as obras dele ou saber quem foi)\n"
        f"B) peças de «{words}» e de outro artista que a mensagem também nomeia\n"
        f"C) peças de artistas parecidos com «{words}», do mesmo tempo ou do mesmo género\n"
        f"D) peças de outros artistas, e não de «{words}»\n"
        'Responde só com JSON: {"resposta": "A", "B", "C" ou "D"}'
    )


def build_listing_request_prompt(*, message: str) -> str:
    """O visitante pede uma lista do que existe no museu? (CHAT_CATALOGUE_LISTINGS)

    As letras B a H são listagens; ver LISTING_FACETS em catalogue_analytics.py.
    Redação medida a 2026-11-17 (tests/listing_request_cases.py), junto com
    build_listing_guard_prompt: 63 de 70 listagens reconhecidas; 192 de 198
    pedidos normais e 169 de 198 pesquisas do tipo «que peças de X há?» passam
    sem ser apanhados (os outros ficam para a comparação com os valores do
    catálogo, em chat_service). Outras redações deram pior: com uma lista de
    características em vez de respostas por letra, o modelo escolhia uma para
    quase tudo (124 de 169 pedidos normais apanhados). O prompt não leva o nome
    do museu: com «Museu Nacional do Traje» ou «do Azulejo», «Que autores há
    neste museu?» era lida como pedido de peças. Qualquer alteração repete a
    medição.
    """
    return (
        f"Mensagem de um visitante do museu:\n«{message}»\n\n"
        "Que resposta espera o visitante?\n"
        "A) peças do museu (uma ou várias), ou informação sobre uma peça\n"
        "B) os nomes dos autores que o museu tem\n"
        "C) os tipos de peças que o museu tem\n"
        "D) os materiais de que as peças do museu são feitas\n"
        "E) as técnicas usadas nas peças do museu\n"
        "F) os locais ou países de onde as peças do museu vêm\n"
        "G) os séculos ou épocas das peças do museu\n"
        "H) os estilos das peças do museu\n"
        "I) outra coisa\n"
        'Responde só com JSON: {"resposta": "A" a "I"}'
    )


def build_listing_guard_prompt(*, message: str) -> str:
    """Segunda pergunta das listagens: a mensagem nomeia alguma coisa (então quer
    peças ou fala de uma peça) ou só pergunta quais existem? Só a resposta B
    deixa passar. Sem ela, «De que século é a Salomé?» e «Há esculturas em
    madeira?» eram lidas como listagens. Medida com build_listing_request_prompt.
    """
    return (
        f"Mensagem de um visitante do museu:\n«{message}»\n\n"
        "A mensagem nomeia alguma coisa?\n"
        "A) sim: uma peça, um autor, um material, um estilo, um século, um local, uma sala ou um tema "
        "(como «Custódia de Belém», «Bosch», «madeira», «barroco», «século XVI», «Japão»), ou aponta para uma peça («esta», «ela», «este quadro»)\n"
        "B) não: só pergunta quais existem (que autores, que tipos de peças, que materiais, que estilos…)\n"
        "C) nenhuma das duas\n"
        'Responde só com JSON: {"nomeia": "A", "B" ou "C"}'
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
        "- NUNCA acrescentes palavras que não estão na user_query — em particular nomes de museus, slugs ou códigos (mnt, mnaz, mnsr, mj, cmag, mnaa) e localidades.\n"
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
