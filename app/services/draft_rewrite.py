"""O rascunho da resposta e o que a AMÁLIA reescreve (CHAT_DRAFT_THEN_REWRITE).

Numa resposta com lista, quem decide o que se diz é o pré-processamento: o
modelo de pré-processamento escreve uma frase de abertura e uma frase muito
breve por peça (cada uma num pedido só com a ficha dessa peça), e o serviço
monta o rascunho — a abertura e uma linha por cartão, «N. **nome** – frase»,
pela ordem dos cartões. A AMÁLIA recebe o rascunho e só o reescreve.

`DraftRewriteMerger` junta as duas coisas: lê o que a AMÁLIA escreve, linha a
linha e por ordem, e fica com a linha dela quando é a linha N do rascunho
reescrita (o mesmo número e o mesmo nome a negrito, com uma frase); senão
fica a do rascunho. O que ela escrever a mais não entra. O texto só cresce:
o que já saiu nunca muda, e por isso o streaming envia-o à medida que sai e
o texto final é exatamente o que foi enviado.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from app.services.author_cards import fold

_ITEM = re.compile(r"^(?P<number>\d+)[.)]\s+(?P<body>\S.*)$")
_BOLD_NAME = re.compile(r"^(?:\*\*|__)(?P<name>.+?)(?:\*\*|__)(?P<rest>.*)$")
_OPENING_MAX_CHARS = 300
_SEPARATOR = re.compile(r"^\s*(?:\([^()]{1,40}\)\s*)?(?:[–—-]|:)\s*(?P<sentence>\S.*)$")


@dataclass
class AnswerDraft:
    """O rascunho: a abertura e uma linha por cartão, pela ordem dos cartões."""

    opening: str
    names: list[str]          # o nome de cada peça, como vai a negrito
    lines: list[str]          # «N. **nome** – frase»
    # a abertura já é a final (a dos números do catálogo, numa resposta por autor): a da AMÁLIA não a troca
    fixed_opening: bool = False
    info: dict[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join([self.opening, "", *self.lines])


class DraftRewriteMerger:
    """As linhas finais da resposta, decididas uma a uma e por ordem."""

    def __init__(self, draft: AnswerDraft, *, clean: Callable[[str], str] | None = None) -> None:
        self.draft = draft
        self.clean = clean or (lambda line: line)
        self.out: list[str] = [draft.opening, ""] if draft.fixed_opening else []
        self.next = 0             # o próximo cartão por decidir
        self.rewritten = 0        # linhas em que ficou a da AMÁLIA
        self.done = False

    def _open(self, opening: str | None = None) -> None:
        if not self.out:
            self.out = [opening or self.draft.opening, ""]

    def _rewritten_line(self, position: int, number: int, body: str) -> str | None:
        """A linha da AMÁLIA para o cartão `position`, se for a do rascunho reescrita."""
        body = self.clean(body).strip()
        name = _BOLD_NAME.match(body)
        if name is None or fold(name["name"]) != fold(self.draft.names[position]):
            return None
        sentence = _SEPARATOR.match(name["rest"])
        if sentence is None or not fold(sentence["sentence"]):
            return None
        return f"{number}. {body}"

    def feed(self, line: str) -> None:
        """Uma linha completa do que a AMÁLIA escreveu."""
        text = (line or "").strip()
        if not text or self.done:
            return
        item = _ITEM.match(text)
        if item is None:
            # antes da lista: a abertura dela (a primeira linha). Depois da lista começar não entra mais nada
            if not self.out and self.next == 0:
                opening = self.clean(text).strip()
                # uma abertura que já fala das peças (com negrito, ou um parágrafo) repetia a lista: fica a do rascunho
                usable = bool(opening) and "**" not in opening and len(opening) <= _OPENING_MAX_CHARS
                self._open(opening if usable else None)
            return
        number = int(item["number"])
        position = number - 1
        if position < self.next or position >= len(self.draft.lines):
            return
        self._open()
        while self.next < position:   # ela saltou linhas: ficam as do rascunho
            self.out.append(self.draft.lines[self.next])
            self.next += 1
        rewritten = self._rewritten_line(position, number, item["body"])
        self.out.append(rewritten or self.draft.lines[position])
        self.rewritten += rewritten is not None
        self.next += 1

    def finish(self) -> None:
        """Acabou: as linhas que ela não escreveu ficam as do rascunho."""
        if self.done:
            return
        self._open()
        self.out.extend(self.draft.lines[self.next:])
        self.next = len(self.draft.lines)
        self.done = True

    def text(self) -> str:
        """O texto decidido até agora (acaba numa mudança de linha enquanto houver mais para vir)."""
        if not self.out:
            return ""
        joined = "\n".join(self.out)
        return joined if (self.done or self.next >= len(self.draft.lines)) else joined + "\n"


def merge_rewrite(draft: AnswerDraft, rewritten_text: str, *, clean: Callable[[str], str] | None = None) -> tuple[str, int]:
    """O texto final e em quantas linhas ficou a da AMÁLIA."""
    merger = DraftRewriteMerger(draft, clean=clean)
    for line in (rewritten_text or "").split("\n"):
        merger.feed(line)
    merger.finish()
    return merger.text(), merger.rewritten
