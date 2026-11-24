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


def describe(clauses: list[dict[str, list[str]]]) -> str:
    """Para o registo: «enr_tipo=cadeira E enr_materiais=prata»."""
    return " E ".join(" ou ".join(f"{field}={','.join(values)}" for field, values in clause.items()) for clause in clauses)
