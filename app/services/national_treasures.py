"""Tesouros Nacionais: a lista por museu (app/config/national_treasures.json) e
os pedidos sobre eles («tesouros nacionais de pintura», «isto é tesouro
nacional?»).

A marca não existe no RAIZ nem no índice: vem do selo nas páginas dos POIs das
visitas publicadas. Por isso vive num ficheiro de configuração, relido quando
muda, e nunca no índice. Os n.ºs comparam-se sem maiúsculas, espaços nem
pontuação («A. 3908» = «A 3908»; «114/34» = «114/ 34»).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "national_treasures.json"

LIST, QUESTION = "list", "question"

_TREASURE_PHRASE = re.compile(
    r"\btesouros?\s+nacion(?:al|ais)\b"
    r"|\bnational\s+treasures?\b"
    r"|\bclassificad[ao]s?(?:\s+como\s+tesouros?\s+nacion(?:al|ais))?\b"
)
# «A Fuga para o Egipto é tesouro nacional?», «isto é tesouro nacional?»,
# «está classificada?», «Is this a national treasure?»
_QUESTION = re.compile(
    r"\b(?:e|sao|sera|seria|foi|eram|esta|estao)\s+(?:um\s+|uma\s+)?(?:tesouros?\s+nacion|classificad)"
    r"|^\s*(?:is|are|was)\b.*\bnational\s+treasures?\b"
    r"|\b(?:is|are|was)\s+(?:(?:this|it|that|they)\s+)?(?:an?\s+)?national\s+treasures?\b"
)
# Palavras que não restringem a lista («mostra-me os tesouros nacionais que há»).
_NOT_SUBJECT = frozenset(
    "o a os as um uma uns umas de do da dos das em no na nos nas que qual quais quem e ou com sem por para ha existe existem "
    "tem temos tens sao es este esta estes estas deste desta neste nesta esse essa isso isto aqui ca la mais muito me diz "
    "diga mostra mostrar mostre encontra encontrar procura procurar lista listar indica indicar sabes sabe podes pode "
    "posso quero queria gostava ver conhecer saber todas todos toda todo tambem entao ai quantas quantos onde como "
    "the a an of in on at to for and or is are was were there be this that these those which what who how me show tell "
    "find list give do does can could i we you please all any some its it has have from with by "
    "peca pecas objeto objetos objecto objectos obra obras artefacto artefactos item itens coisa coisas exemplar exemplares "
    "piece pieces object objects work works artifact artifacts item items "
    "museu museus acervo colecao catalogo collection museum visita tour percurso virtual visit "
    "tesouro tesouros nacional nacionais classificada classificado classificadas classificados classificacao "
    "national treasure treasures classified "
    "importantes importante principais principal interessantes famosos famosas valiosos valiosas raros raras melhores "
    "existentes expostos expostas bonitos bonitas mais "
    # época: vai para o filtro temporal, nunca para o assunto
    "seculo seculos sec secs century centuries epoca periodo ano anos".split()
)
_ROMAN = re.compile(r"^(?=[mdclxvi])m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$")
# Assunto → radicais a procurar na categoria, título, técnica e material.
_SYNONYMS = {
    "quadro": ("pintu",), "quadros": ("pintu",), "painting": ("pintu",), "paintings": ("pintu",),
    "joia": ("ouriv",), "joias": ("ouriv",), "jewel": ("ouriv",), "jewels": ("ouriv",), "goldsmith": ("ouriv",),
    "prata": ("ouriv", "prata"), "ouro": ("ouriv", "ouro"),
    "tecido": ("texte",), "tecidos": ("texte",), "textil": ("texte",), "textile": ("texte",), "textiles": ("texte",),
    "tapecaria": ("texte", "tapet"), "tapecarias": ("texte", "tapet"), "tapete": ("tapet",), "tapetes": ("tapet",),
    "movel": ("mobil",), "moveis": ("mobil",), "mobilia": ("mobil",), "furniture": ("mobil",),
    "sculpture": ("escul",), "sculptures": ("escul",), "estatua": ("escul",), "estatuas": ("escul",),
    "ceramic": ("ceram",), "ceramics": ("ceram",), "loica": ("ceram",), "faianca": ("ceram", "faian"), "porcelana": ("ceram", "porce"),
    "moeda": ("numis", "solid", "moeda"), "moedas": ("numis", "solid", "moeda"), "coin": ("numis", "solid"), "coins": ("numis", "solid"),
    "inscricao": ("epigr", "inscr"), "inscricoes": ("epigr", "inscr"), "inscription": ("epigr", "inscr"),
}
NO_MATCH_ID = "__nenhum_tesouro_nacional__"


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").casefold())
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", stripped).strip()


def inventory_key(inventory: str | None) -> str:
    """«114/ 34 Pin MNSR» → «11434pinmnsr»; «A. 3908» → «a3908»."""
    return fold(inventory or "").replace(" ", "")


def _expand(item: str) -> list[str]:
    """«A 3776 a A 3784» → A 3776 … A 3784; o resto fica como está."""
    match = re.fullmatch(r"\s*(?P<p>[^\d]*?)\s*(?P<a>\d+)\s+a\s+(?P<p2>[^\d]*?)\s*(?P<b>\d+)\s*", item or "")
    if match and inventory_key(match["p"]) == inventory_key(match["p2"]):
        start, end = int(match["a"]), int(match["b"])
        if start <= end <= start + 500:
            return [f"{match['p'].strip()} {n}".strip() for n in range(start, end + 1)]
    return [item.strip()] if (item or "").strip() else []


class NationalTreasureRegistry:
    """A lista por museu, relida quando o ficheiro muda."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._by_museum: dict[str, list[str]] = {}
        self._keys: dict[str, frozenset[str]] = {}

    def _reload_if_changed(self) -> None:
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return
        with self._lock:
            if mtime == self._mtime:
                return
            by_museum: dict[str, list[str]] = {}
            if mtime is not None:
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception as exc:  # um ficheiro estragado não pode derrubar o chat
                    logger.warning("national_treasures: %s ilegível (%s); lista vazia", self.path, exc)
                    raw = {}
                for museum, items in (raw.items() if isinstance(raw, dict) else []):
                    if museum.startswith("_") or not isinstance(items, list):
                        continue
                    by_museum[museum.strip().lower()] = [x for item in items for x in _expand(str(item))]
            self._by_museum = by_museum
            self._keys = {m: frozenset(inventory_key(i) for i in items) for m, items in by_museum.items()}
            self._mtime = mtime

    @property
    def version(self) -> float | None:
        self._reload_if_changed()
        return self._mtime

    def inventories(self, museum_id: str | None) -> list[str]:
        self._reload_if_changed()
        return list(self._by_museum.get((museum_id or "").strip().lower(), []))

    def keys(self, museum_id: str | None) -> frozenset[str]:
        self._reload_if_changed()
        return self._keys.get((museum_id or "").strip().lower(), frozenset())

    def is_treasure(self, museum_id: str | None, inventories: Iterable[str | None]) -> bool:
        """True se algum dos n.ºs (um cartão agrupado tem vários) está na lista do museu."""
        keys = self.keys(museum_id)
        return bool(keys) and any(inventory_key(i) in keys for i in inventories if i)


_REGISTRY: NationalTreasureRegistry | None = None


def get_registry(path: str | None = None) -> NationalTreasureRegistry:
    global _REGISTRY
    wanted = Path(path) if path else DEFAULT_PATH
    if _REGISTRY is None or _REGISTRY.path != wanted:
        _REGISTRY = NationalTreasureRegistry(wanted)
    return _REGISTRY


@dataclass(frozen=True)
class TreasureIntent:
    kind: str                       # LIST («tesouros nacionais de pintura») ou QUESTION («é tesouro nacional?»)
    subject_tokens: tuple[str, ...]  # o que restringe a lista («pintura»); vazio = todos


def parse_treasure_intent(message: str) -> TreasureIntent | None:
    folded = fold(message)
    if not folded or not _TREASURE_PHRASE.search(folded):
        return None
    if _QUESTION.search(folded):
        return TreasureIntent(QUESTION, ())
    rest = _TREASURE_PHRASE.sub(" ", folded)
    tokens = tuple(t for t in rest.split() if t not in _NOT_SUBJECT and not t.isdigit() and not _ROMAN.match(t))
    return TreasureIntent(LIST, tokens)


def strip_treasure_words(text: str) -> str:
    """Tira «tesouros nacionais», «classificadas» da pergunta para a pesquisa
    se concentrar no resto («A Fuga para o Egipto é tesouro nacional?» →
    «A Fuga para o Egipto é ?»)."""
    cleaned = re.sub(
        r"(?i)\btesouros?\s+nacion(?:al|ais)\b|\bnational\s+treasures?\b|\bclassificad[ao]s?\b(?:\s+como)?",
        " ",
        text or "",
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def subject_matches(tokens: Iterable[str], doc: dict) -> bool:
    """Cada termo do assunto tem de aparecer (por radical, ou sinónimo) na
    categoria, título, técnica ou material da ficha."""
    words = fold(" ".join(str(doc.get(k) or "") for k in ("category", "super_category", "title", "technique", "support_or_material"))).split()
    for token in tokens:
        stems = _SYNONYMS.get(token) or (token[:5] if len(token) >= 5 else token,)
        if not any(word.startswith(stem) for stem in stems for word in words):
            return False
    return True
