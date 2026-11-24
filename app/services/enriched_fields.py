"""O assunto de uma contagem lido pelos campos enriquecidos (CHAT_ENRICHED_FIELDS).

«Quantas cadeiras há?» contava as fichas em que a palavra «cadeira» aparece no título,
na técnica ou na descrição: entram pinturas com uma cadeira e cadeirais. Com os campos
enriquecidos (tools/enriquecimento) conta-se o que a peça é: tipo de objeto, material,
técnica, função, país de origem e tema (`enr_temas_palavras` tem as palavras com que
se pede cada tema: «religiosa», «animal», «paisagem»; `enr_foto_palavras`, o que o
modelo de visão viu na fotografia: pessoas, animais, letras).

Aqui só se decide se o assunto se lê por inteiro com os valores que o museu tem nesses
campos. É uma comparação com dados, sem modelo: se sobrar uma palavra que não é valor
de nenhuma lista, devolve None e a contagem fica como era.
"""
from __future__ import annotations

import itertools
import re
import unicodedata

# campos em que se procura o assunto, por esta ordem
SUBJECT_FIELDS = ("enr_tipo", "enr_materiais", "enr_tecnicas", "enr_funcao", "enr_origem_pais", "enr_temas_palavras",
                  "enr_foto_palavras")
# Na pesquisa (CHAT_ENRICHED_SEARCH_FILTER) só servem de filtro os campos que cobrem quase todas as peças: o tipo
# (99%) e o que se vê na fotografia (99%). O material (21% nas esculturas) e a origem (33%) deixavam de fora peças
# certas: em «peças vindas do Japão», os biombos Namban, cuja ficha não tem centro de produção.
SEARCH_FIELDS = ("enr_tipo", "enr_foto_palavras")
# palavras de um pedido que não dizem o que se procura
_REQUEST_WORDS = frozenset(
    "mostra mostrame me ha tens tem existem existe quero queria ver procuro procura encontra encontrar que quais qual peca pecas obra "
    "obras objeto objetos algum alguma alguns algumas onde aparece aparecem se ve veem tenham tenha show find are there any with".split()
)
# como um visitante chama aos tipos do catálogo
_TYPE_SYNONYMS = {"quadro": "pintura", "quadros": "pinturas", "tela": "pintura", "telas": "pinturas", "estatua": "escultura",
                  "estatuas": "esculturas"}
MAX_PHRASE_TOKENS = 4
# palavras de ligação: não contam nem no assunto nem nos valores («pintura a óleo» lê-se em «pinturas óleo»,
# que é como o assunto chega depois de a analítica lhe tirar as palavras vazias)
_CONNECTIVES = frozenset("a o as os de do da dos das e em no na nos nas com para por".split())
# plural → singular, à portuguesa («texteis» tanto pode ser «textil» como «textel»: tentam-se as duas)
_PLURALS = (("oes", ("ao",)), ("aes", ("ao",)), ("ais", ("al",)), ("eis", ("el", "il")), ("ois", ("ol",)), ("uis", ("ul",)),
            ("ns", ("m",)), ("res", ("r",)), ("zes", ("z",)), ("ses", ("s",)), ("s", ("",)))


def fold(text: object) -> str:
    plain = unicodedata.normalize("NFKD", str(text or "").casefold())
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def word_forms(token: str) -> list[str]:
    """A palavra e os singulares possíveis, do mais provável para o menos."""
    forms = [token]
    for ending, replacements in _PLURALS:
        if token.endswith(ending) and len(token) > len(ending) + 2:
            forms.extend(token[: -len(ending)] + replacement for replacement in replacements)
    # «peças bordadas» → «bordado»: só nos particípios, para «prata» nunca virar «prato»
    forms.extend(form[:-1] + "o" for form in list(forms) if form.endswith(("ada", "ida")))
    return list(dict.fromkeys(forms))


def match_subject(subject: str, vocabulary: dict[str, list[str]]) -> list[dict[str, list[str]]] | None:
    """As condições que leem o assunto todo, ou None.

    Cada condição é {campo: [valores]}: a peça tem de cumprir todas as condições, e
    cumpre uma se tiver um dos valores em qualquer dos campos dela («tapeçaria» é
    tipo de objeto e é técnica). `vocabulary` são os valores que o museu tem em cada
    campo, como estão no índice.
    """
    tokens = [token for token in fold(subject).split() if token not in _CONNECTIVES]
    if not tokens:
        return None
    labels: dict[str, dict[str, list[str]]] = {}
    for field in SUBJECT_FIELDS:
        for value in vocabulary.get(field) or []:
            key = " ".join(token for token in fold(value).split() if token not in _CONNECTIVES)
            if key:
                labels.setdefault(key, {}).setdefault(field, []).append(str(value))
    clauses: list[dict[str, list[str]]] = []
    position = 0
    while position < len(tokens):
        found: dict[str, list[str]] | None = None
        for size in range(min(MAX_PHRASE_TOKENS, len(tokens) - position), 0, -1):
            # cada palavra pode vir no plural («pinturas óleo», «serviços mesa»)
            for candidate in itertools.product(*(word_forms(token) for token in tokens[position: position + size])):
                found = labels.get(" ".join(candidate))
                if found:
                    break
            if found:
                position += size
                break
        if not found:
            return None
        clauses.append({field: list(values) for field, values in found.items()})
    return clauses


def match_search(query: str, vocabulary: dict[str, list[str]]) -> list[dict[str, list[str]]] | None:
    """As condições de um pedido de pesquisa que junta um tipo de peça ao que se vê nela («pinturas com animais»,
    «esculturas com inscrições»), ou None. Só o tipo («mostra-me cadeiras») não chega: aí a pesquisa de hoje já
    devolve as peças certas, e um tipo de uma palavra só («cruz», «painel») tanto é a peça como o assunto dela."""
    tokens = [_TYPE_SYNONYMS.get(token, token) for token in fold(query).split() if token not in _REQUEST_WORDS]
    clauses = match_subject(" ".join(tokens), {field: vocabulary.get(field) or [] for field in SEARCH_FIELDS})
    if not clauses:
        return None
    fields = {field for clause in clauses for field in clause}
    return clauses if {"enr_tipo", "enr_foto_palavras"} <= fields else None


def describe(clauses: list[dict[str, list[str]]]) -> str:
    """Para o registo: «enr_tipo=cadeira E enr_materiais=prata»."""
    return " E ".join(" ou ".join(f"{field}={','.join(values)}" for field, values in clause.items()) for clause in clauses)
