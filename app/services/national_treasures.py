"""Tesouros Nacionais: a lista por museu e os pedidos sobre eles («tesouros
nacionais de pintura», «isto é tesouro nacional?»).

A marca não existe no RAIZ nem no índice: é o selo nas páginas dos POIs das
visitas publicadas, escrito como `nationalTreasure` nos mapas dos POIs
(poi_tours/panorama-overlays-inventory-<museu>.json, ver app/services/tour_pois.py),
que são a fonte única. Os n.ºs são os do catálogo (inventoryIds) e comparam-se
sem maiúsculas, espaços nem pontuação («A. 3908» = «A 3908»; «114/34» = «114/ 34»).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.services.tour_pois import TourPoiStore, get_tour_poi_store

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


class NationalTreasureRegistry:
    """Os n.ºs do catálogo com `nationalTreasure` nos mapas dos POIs, por museu."""

    def __init__(self, store: TourPoiStore | Path | str | None = None) -> None:
        if store is None:
            store = get_tour_poi_store()
        self.store = store if isinstance(store, TourPoiStore) else TourPoiStore(Path(store))
        self._built_for: tuple | None | bool = False
        self._by_museum: dict[str, list[str]] = {}
        self._keys: dict[str, frozenset[str]] = {}

    def _refresh(self) -> None:
        version = self.store.version
        if version == self._built_for:
            return
        by_museum: dict[str, list[str]] = {}
        for museum in self.store.museums():
            seen: set[str] = set()
            numbers: list[str] = []
            for entry in self.store.entries(museum) or []:
                if entry.get("nationalTreasure") is not True:
                    continue
                ids = entry.get("inventoryIds")
                for number in ([ids] if isinstance(ids, str) else ids or []):
                    key = inventory_key(str(number))
                    if key and key not in seen:
                        seen.add(key)
                        numbers.append(str(number).strip())
            by_museum[museum] = numbers
        self._by_museum = by_museum
        self._keys = {museum: frozenset(inventory_key(n) for n in numbers) for museum, numbers in by_museum.items()}
        self._built_for = version

    @property
    def version(self) -> tuple | None:
        return self.store.version

    def inventories(self, museum_id: str | None) -> list[str]:
        self._refresh()
        return list(self._by_museum.get((museum_id or "").strip().lower(), []))

    def keys(self, museum_id: str | None) -> frozenset[str]:
        self._refresh()
        return self._keys.get((museum_id or "").strip().lower(), frozenset())

    def is_treasure(self, museum_id: str | None, inventories: Iterable[str | None]) -> bool:
        """True se algum dos n.ºs (um cartão agrupado tem vários) é Tesouro Nacional no mapa do museu."""
        keys = self.keys(museum_id)
        return bool(keys) and any(inventory_key(i) in keys for i in inventories if i)


_REGISTRY: NationalTreasureRegistry | None = None


def get_registry() -> NationalTreasureRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = NationalTreasureRegistry()
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
