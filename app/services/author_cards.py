"""Ficha do autor no chat (CHAT_AUTHOR_CARD): funções puras.

O visitante diz o nome de um autor («obras de Gregório Lopes», «quem foi
Soares dos Reis?»). Daqui sai o autor das peças mostradas cujo nome tem
palavras da mensagem (o candidato); quem decide se essas palavras são mesmo
o nome de um artista é o modelo de pré-processamento
(build_author_name_prompt). Nada aqui decide.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

# O que o catálogo junta ao nome entre parênteses ou parênteses retos: datas e
# locais («Keil, Maria (Silves, 1914 - Lisboa 2012)»). Não entra na comparação
# com a mensagem, senão «azulejos de Lisboa» propunha dezenas de autores.
_BRACKETED = re.compile(r"[(\[][^)\]]*[)\]]?")
_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*")


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").casefold())
    folded = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def name_tokens(name: str) -> list[str]:
    """Palavras do nome, sem o que vem entre parênteses e sem números."""
    return [token for token in fold(_BRACKETED.sub(" ", name or "")).split() if not token.isdigit()]


@dataclass
class AuthorGroup:
    """Um autor do museu. O catálogo escreve o mesmo autor de várias maneiras
    («Sequeira, Domingos António de» e «Sequeira, Domingos António de
    (1768-1837)»); as fichas com as mesmas palavras no nome ficam juntas."""

    key: str
    name: str
    ids: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    count: int = 0
    tokens: frozenset[str] = frozenset()


@dataclass
class AuthorDirectory:
    groups: list[AuthorGroup]
    # em quantos autores do museu aparece cada palavra
    frequency: dict[str, int]


def build_author_directory(rows: list[dict]) -> AuthorDirectory:
    """rows: [{"id", "name", "count"}], como vêm da agregação por
    creator_ids, do autor com mais peças para o que tem menos. O nome que fica
    é o da grafia com mais peças."""
    by_key: dict[str, AuthorGroup] = {}
    best: dict[str, int] = {}
    for row in rows:
        name = str(row.get("name") or "").strip()
        author_id = str(row.get("id") or "").strip()
        tokens = name_tokens(name)
        if not name or not author_id or not tokens:
            continue
        key = " ".join(sorted(tokens))
        count = int(row.get("count") or 0)
        group = by_key.get(key)
        if group is None:
            group = by_key[key] = AuthorGroup(key=key, name=name, tokens=frozenset(tokens))
        if count > best.get(key, -1):
            best[key] = count
            group.name = name
        group.ids.append(author_id)
        group.names.append(name)
        group.count += count
    groups = sorted(by_key.values(), key=lambda g: (-g.count, g.name))
    frequency: dict[str, int] = {}
    for group in groups:
        for token in group.tokens:
            frequency[token] = frequency.get(token, 0) + 1
    return AuthorDirectory(groups=groups, frequency=frequency)


def author_candidates(
    message: str,
    directory: AuthorDirectory,
    *,
    among: list[AuthorGroup] | None = None,
    limit: int = 1,
) -> list[AuthorGroup]:
    """Os autores cujo nome tem palavras da mensagem, do mais parecido para o
    menos; com `among`, só entre esses (os autores das peças mostradas). Uma
    palavra conta pelo que distingue: «António» está em dezenas de nomes e
    pesa pouco, «Pousão» está num e pesa muito. Palavras com menos de três
    letras não contam. É uma procura na lista de autores, não uma decisão: a
    seguir pergunta-se ao modelo. Medido a 2026-11-18 em 74 mensagens com o
    nome de um autor (seis museus): o autor certo vem em primeiro em todas."""
    total = len(directory.groups)
    if not total:
        return []
    words = {token for token in fold(message).split() if len(token) >= 3 and not token.isdigit()}
    if not words:
        return []

    def weight(token: str) -> float:
        return math.log(1 + total / max(directory.frequency.get(token, 1), 1))

    scored: list[tuple[float, float, int, AuthorGroup]] = []
    for group in directory.groups if among is None else among:
        matched = words & group.tokens
        if not matched:
            continue
        got = sum(weight(token) for token in matched)
        own = sum(weight(token) for token in group.tokens if len(token) >= 3) or 1.0
        scored.append((got, got / own, group.count, group))
    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3].name))
    return [group for _got, _share, _count, group in scored[: max(int(limit), 1)]]


def words_in_message(message: str, group: AuthorGroup) -> str:
    """As palavras da mensagem que estão no nome do autor, como o visitante as
    escreveu («Há paisagens do Porto?» e «Silva Porto, António Carvalho de» →
    «Porto»). Se entre a primeira e a última só houver palavras curtas, vai o
    troço inteiro («Soares dos Reis», «Josefa de Óbidos»)."""
    pieces = [(match.group(0), fold(match.group(0))) for match in _WORD.finditer(message or "")]
    marks = [len(folded) >= 3 and folded in group.tokens for _original, folded in pieces]
    if not any(marks):
        return ""
    first, last = marks.index(True), len(marks) - 1 - marks[::-1].index(True)
    span = pieces[first : last + 1]
    if all(marks[first + offset] or len(folded) <= 3 for offset, (_original, folded) in enumerate(span)):
        return " ".join(original for original, _folded in span)
    return " ".join(original for (original, _folded), mark in zip(pieces, marks) if mark)


_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_TIME_SUFFIX = re.compile(r"\s+\d{1,2}:\d{2}(:\d{2})?$")


def clean_life_date(value: object) -> str | None:
    """A data de nascimento ou morte como o catálogo a escreve, sem a hora que
    algumas trazem agarrada («1922-03-05 00:00:00» → «1922-03-05»)."""
    text = _TIME_SUFFIX.sub("", str(value or "").strip())
    return text or None


def life_year(value: object) -> int | None:
    """O ano (quatro dígitos) de uma data escrita de qualquer maneira; None se
    não tiver exatamente um («Século XV (?)», vazio)."""
    years = _YEAR.findall(str(value or ""))
    return int(years[0]) if len(set(years)) == 1 else None


def merged_author_fields(docs: list[dict]) -> dict[str, str | None]:
    """Os dados do autor no índice de autores. Quando o catálogo tem o autor em
    mais de uma ficha, fica o primeiro valor preenchido de cada campo."""

    def first(*keys: str) -> str | None:
        for doc in docs:
            for key in keys:
                text = str(doc.get(key) or "").strip()
                if text:
                    return text
        return None

    return {
        "activity": first("activity", "atividade"),
        "birth_date": clean_life_date(first("birth_date", "data_nascimento")),
        "death_date": clean_life_date(first("death_date", "data_obito")),
        "birth_place": first("birth_place", "local_nascimento"),
        "death_place": first("death_place", "local_obito"),
        "biography": first("biography", "biografia"),
        "url": first("url"),
    }
