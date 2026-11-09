"""Intenção `plan_route` por regras e interpretação do tema em facetas.

Regras primeiro (sem LLM): o pedido é um percurso se tiver uma das palavras
de percurso («percurso», «roteiro», «visita guiada», «leva-me», «mostra-me …
pela visita», «tour of», «route», «itinerary»…). As facetas saem do texto
dobrado por correspondência com o vocabulário visual fechado
(app/config/visual_vocabulary.json) e por expressões regulares para séculos,
intervalos de anos, número de paragens e orçamento de tempo. O LLM
(ROUTE_THEME_LLM) só acrescenta o que as regras não apanharam, e só valores
do vocabulário; sem LLM o percurso sai na mesma.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VOCABULARY_PATH = Path(__file__).resolve().parents[2] / "config" / "visual_vocabulary.json"

FACET_KINDS = ("colors", "pigments", "motifs", "iconography", "materials", "techniques", "styles")
# chave da faceta -> chave no vocabulário (listas fechadas)
VOCAB_KEYS = {
    "colors": "colors",
    "pigments": "pigment_hints",
    "motifs": "motifs",
    "iconography": "iconography",
    "materials": "materials_visual",
    "techniques": "techniques_visual",
    "styles": "style_period_visual",
}
FACET_LABELS_PT = {"colors": "cor", "pigments": "pigmento", "centuries": "século", "motifs": "motivo", "iconography": "iconografia",
                   "materials": "material", "techniques": "técnica", "styles": "estilo", "years": "período", "free_text": "tema"}

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
          "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20, "xxi": 21}
_ROMAN_BY_NUMBER = {v: k.upper() for k, v in _ROMAN.items()}

ROUTE_REQUEST_RE = re.compile(
    r"\b(?:percursos?|roteiros?|itinerarios?|itinerary|itineraries|visita guiada|visitas guiadas|guided tour|"
    r"leva[- ]?me|leve[- ]?me|guia[- ]?me|conduz[- ]?me|acompanha[- ]?me|take me|guide me|walk me|show me around|"
    r"tour of|a tour|route|routes|passeio tematico|passeio)\b"
)
# «mostra-me … pela visita / pelo museu / pelas salas»
ROUTE_SHOW_RE = re.compile(r"\b(?:mostra|mostre|ensina|apresenta)[- ]?me\b.*\b(?:pela visita|pelo museu|pelas salas|pela exposicao|ao longo da visita|na visita)\b")
_NEGATIVE_RE = re.compile(r"\b(?:ver na tour|onde estou|em que sala)\b")

_REQUEST_WORDS_RE = re.compile(
    r"\b(?:faz|faz-me|faze-me|fazes|podes fazer|cria|cria-me|prepara|prepara-me|sugere|sugere-me|quero|queria|gostava de|gostaria de|"
    r"da-me|da me|monta|monta-me|organiza|planeia|planeia-me|mostra-me|mostra me|leva-me|leva me|leve-me|guia-me|"
    r"leva|leve|levas|guia|conduz|acompanha|mostra|mostre|ensina|apresenta|take|guide|walk|show|"
    r"um|uma|o|a|de|do|da|dos|das|em|no|na|nos|nas|com|sobre|por|para|pelo|pela|pelos|pelas|ao|aos|que|e|"
    r"percurso|percursos|roteiro|roteiros|itinerario|visita guiada|visita|guiada|passeio|tematico|tematica|tema|temático|temática|"
    r"museu|exposicao|salas|sala|pecas|peca|obras|obra|ver|veja|vejo|visitar|conhecer|dedicado|dedicada|ligado|ligada|relacionado|relacionada|os|as|"
    r"make|create|plan|give|me|i|want|would|like|to|see|a|an|the|of|on|about|through|around|tour|route|itinerary|guided|please|por favor|"
    r"minutos|minuto|minutes|minute|min|paragens|paragem|stops|stop)\b"
)


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").casefold())
    folded = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", folded)).strip()


@dataclass
class ThemeFacets:
    theme_text: str = ""
    colors: list[str] = field(default_factory=list)
    pigments: list[str] = field(default_factory=list)
    centuries: list[int] = field(default_factory=list)
    year_range: tuple[int, int] | None = None
    motifs: list[str] = field(default_factory=list)
    iconography: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    techniques: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    free_text: str = ""
    stops_requested: int | None = None
    time_budget_min: int | None = None
    must_include: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    rooms: list[str] = field(default_factory=list)
    national_treasure: bool = False   # «tesouros nacionais»: só peças com o selo (filtro duro)
    source: str = "rules"

    def has_visual(self) -> bool:
        return any(getattr(self, kind) for kind in FACET_KINDS)

    def is_temporal(self) -> bool:
        return bool(self.centuries or self.year_range)

    def year_window(self, tolerance: int = 0) -> tuple[int, int] | None:
        """(ano inicial, ano final) do tema, com a tolerância dos filtros duros."""
        if self.year_range:
            start, end = self.year_range
        elif self.centuries:
            start = (min(self.centuries) - 1) * 100
            end = max(self.centuries) * 100 - 1
        else:
            return None
        return start - tolerance, end + tolerance

    def labels(self) -> list[dict[str, str]]:
        """Facetas para o cartão do percurso: [{kind, label}]."""
        out: list[dict[str, str]] = []
        for value in self.colors:
            out.append({"kind": "color", "label": value})
        for value in self.pigments:
            out.append({"kind": "pigment", "label": value})
        for century_number in self.centuries:
            out.append({"kind": "century", "label": f"século {_ROMAN_BY_NUMBER.get(century_number, century_number)}"})
        if self.year_range and not self.centuries:
            out.append({"kind": "years", "label": f"{self.year_range[0]}–{self.year_range[1]}"})
        for kind, singular in (("motifs", "motif"), ("iconography", "iconography"), ("materials", "material"), ("techniques", "technique"), ("styles", "style")):
            for value in getattr(self, kind):
                out.append({"kind": singular, "label": value})
        for value in self.rooms:
            out.append({"kind": "room", "label": value})
        if self.national_treasure:
            out.append({"kind": "national_treasure", "label": "Tesouro Nacional"})
        if not out and self.free_text:
            out.append({"kind": "theme", "label": self.free_text})
        return out

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["year_range"] = list(self.year_range) if self.year_range else None
        return data


@lru_cache(maxsize=1)
def load_vocabulary(path: str | None = None) -> dict[str, list[str]]:
    """Listas fechadas do vocabulário visual, por chave, em texto dobrado -> forma canónica."""
    try:
        payload = json.loads(Path(path or _VOCABULARY_PATH).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — sem vocabulário ficam só as regras temporais
        return {}
    lists: dict[str, list[str]] = {}
    for key, value in payload.items():
        if str(key).startswith("_") or not isinstance(value, list):
            continue
        terms: list[str] = []
        for item in value:
            if isinstance(item, str):
                terms.append(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("term") or item.get("label")
                if name:
                    terms.append(str(name))
                for alias in item.get("aliases") or item.get("synonyms") or []:
                    if isinstance(alias, str):
                        terms.append(f"{alias}={name}")
        lists[str(key)] = terms
    return lists


def _term_index(terms: list[str]) -> dict[str, str]:
    """texto dobrado do termo (ou alias) -> forma canónica."""
    index: dict[str, str] = {}
    for term in terms:
        if "=" in term:
            alias, canonical = term.split("=", 1)
            index[fold(alias)] = canonical.strip().lower()
        else:
            index[fold(term)] = term.strip().lower()
    return index


def is_route_request(text: str) -> bool:
    folded = fold(text)
    if not folded or _NEGATIVE_RE.search(folded):
        return False
    return bool(ROUTE_REQUEST_RE.search(folded) or ROUTE_SHOW_RE.search(folded))


def _match_terms(folded: str, index: dict[str, str]) -> tuple[list[str], str]:
    """(termos canónicos encontrados, texto sem eles). Termos mais longos primeiro,
    para «azul cobalto» ganhar a «azul»; o que bate sai do texto, para a lista
    seguinte não o voltar a contar."""
    padded = f" {folded} "
    found: list[str] = []
    for term in sorted(index, key=len, reverse=True):
        if term and f" {term} " in padded:
            canonical = index[term]
            if canonical not in found:
                found.append(canonical)
            padded = padded.replace(f" {term} ", " ")
    return found, re.sub(r"\s+", " ", padded).strip()


def _centuries(folded: str) -> list[int]:
    found: list[int] = []
    token = r"([ivx]+|[0-9]{1,2})"
    for pattern in (rf"\b(?:seculo|seculos|sec|secs|century|centuries)\s+{token}\b", rf"\b{token}(?:o|a|st|nd|rd|th)?\s+(?:seculo|century)\b"):
        for match in re.finditer(pattern, folded):
            raw = match.group(1)
            value = _ROMAN.get(raw) if raw.isalpha() else int(raw)
            if value and 1 <= value <= 21 and value not in found:
                found.append(value)
    # «dos séculos XVI a XVIII», «entre o século XV e o XVII»
    span = re.search(rf"\b(?:seculos?|secs?)\s+{token}\s+(?:a|e|ao|até|ate|to|and)\s+(?:o\s+)?(?:seculo\s+)?{token}\b", folded)
    if span:
        a = _ROMAN.get(span.group(1)) if span.group(1).isalpha() else int(span.group(1))
        b = _ROMAN.get(span.group(2)) if span.group(2).isalpha() else int(span.group(2))
        if a and b and 1 <= a <= 21 and 1 <= b <= 21:
            for value in range(min(a, b), max(a, b) + 1):
                if value not in found:
                    found.append(value)
    for word, value in (("quinhentista", 16), ("seiscentista", 17), ("setecentista", 18), ("oitocentista", 19), ("novecentista", 20), ("medieval", 0), ("medievais", 0)):
        if re.search(rf"\b{word}s?\b", folded):
            if value and value not in found:
                found.append(value)
            if value == 0:
                for v in (12, 13, 14, 15):
                    if v not in found:
                        found.append(v)
    return sorted(found)


def _year_range(folded: str) -> tuple[int, int] | None:
    match = re.search(r"\b(?:entre|de|from|between)\s+(1[0-9]{3}|20[0-9]{2})\s+(?:e|a|and|to)\s+(1[0-9]{3}|20[0-9]{2})\b", folded)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        return (min(a, b), max(a, b))
    match = re.search(r"\b(?:anos|decada de|decada|the)\s+(1[0-9]{2}0|20[0-9]0)s?\b", folded)
    if match:
        start = int(match.group(1))
        return (start, start + 9)
    return None


def rule_facets(text: str, *, vocabulary: dict[str, list[str]] | None = None) -> ThemeFacets:
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    folded = fold(text)
    facets = ThemeFacets(theme_text=(text or "").strip())
    # pigmentos antes das cores: «azul-cobalto» é um pigmento e não conta também como «azul»
    remaining = folded
    for kind in ("pigments", "colors", "techniques", "materials", "styles", "iconography", "motifs"):
        matched, remaining = _match_terms(remaining, _term_index(vocab.get(VOCAB_KEYS[kind], [])))
        setattr(facets, kind, matched)
    # «século xvii» também está na lista de estilos do vocabulário: a época fica só em centuries
    facets.styles = [s for s in facets.styles if not re.match(r"^(seculo|sec)\b", fold(s))]
    facets.centuries = _centuries(folded)
    facets.year_range = _year_range(folded)
    facets.national_treasure = bool(re.search(r"\btesouros? naciona(?:l|is)\b|\bnational treasures?\b", folded))
    stops = re.search(r"\b([0-9]{1,2})\s+(?:paragens|paragem|pecas|peca|obras|obra|stops|stop|pieces|objetos|objectos)\b", folded)
    if stops:
        facets.stops_requested = int(stops.group(1))
    budget = re.search(r"\b([0-9]{1,3})\s*(?:min|mins|minutos|minuto|minutes|minute)\b", folded)
    if budget:
        facets.time_budget_min = int(budget.group(1))
    half = re.search(r"\b(?:meia hora|half an hour)\b", folded)
    if half and not facets.time_budget_min:
        facets.time_budget_min = 30
    hour = re.search(r"\b(?:uma hora|one hour|an hour)\b", folded)
    if hour and not facets.time_budget_min:
        facets.time_budget_min = 60
    facets.free_text = strip_request_words(text)
    return facets


def strip_request_words(text: str) -> str:
    """«Faz-me um percurso sobre azul-cobalto no século XVII» -> «azul-cobalto no século XVII»."""
    raw = (text or "").strip().rstrip("?!. ")
    folded_tokens = fold(raw).split()
    kept: list[str] = []
    # mantém as palavras do pedido original (com acentos) que não são palavras de pedido
    original_tokens = re.findall(r"[^\s]+", raw)
    folded_originals = [fold(t) for t in original_tokens]
    count_words = {"paragens", "paragem", "pecas", "peca", "obras", "obra", "stops", "stop", "pieces", "objetos", "min", "mins", "minutos", "minuto", "minutes", "minute"}
    for position, (token, folded_token) in enumerate(zip(original_tokens, folded_originals)):
        if not folded_token:
            continue
        # «5 paragens», «20 minutos»: o número cai com a palavra que o segue
        if folded_token.isdigit() and position + 1 < len(folded_originals) and folded_originals[position + 1] in count_words:
            continue
        # «faz-me» dobra em duas palavras («faz me»): o token cai se todas forem palavras de pedido
        words = folded_token.split()
        if words and all(_REQUEST_WORDS_RE.fullmatch(w) and w not in {"seculo", "seculos"} for w in words):
            continue
        kept.append(token.strip(",;:"))
    theme = " ".join(kept).strip()
    if not theme and folded_tokens:
        theme = raw
    return theme


def theme_title(facets: ThemeFacets, language: str | None = None) -> str:
    text = facets.free_text or facets.theme_text
    text = text.strip()
    if not text:
        text = "Percurso" if not (language or "").startswith("en") else "Route"
    return text[:1].upper() + text[1:] if text else text


# Quantos valores sem apoio no pedido o LLM pode acrescentar por faceta quando
# as regras não apanharam nada (frases leigas: «bichos» -> animais).
LLM_UNGROUNDED_PER_KIND = 2
_TIME_MENTION_RE = re.compile(r"\d|\bsec\b|\bseculos?\b|\bcentur|\banos?\b|\bantig")


def _lay_values(folded_message: str, vocab: dict[str, Any]) -> set[str]:
    """Valores canónicos que os termos leigos do vocabulário presentes no pedido autorizam."""
    out: set[str] = set()
    padded = f" {folded_message} "
    for entry in vocab.get("lay_keywords", []) or []:
        if not isinstance(entry, dict):
            continue
        term = fold(str(entry.get("term", "")))
        if term and f" {term} " in padded:
            for key in VOCAB_KEYS.values():
                for value in entry.get(key, []) or []:
                    out.add(str(value).strip().lower())
    return out


def _grounded(canonical: str, message_tokens: set[str], lay_values: set[str]) -> bool:
    """O valor do LLM tem apoio no pedido: vem de um termo leigo presente, ou todas as
    suas palavras com 4+ letras aparecem no pedido (ou são prefixo/radical de uma)."""
    if canonical in lay_values:
        return True
    tokens = [t for t in fold(canonical).split() if len(t) >= 4]
    if not tokens:
        return False
    for token in tokens:
        if token in message_tokens:
            continue
        if any((len(token) >= 5 and word.startswith(token)) or (len(word) >= 5 and token.startswith(word)) for word in message_tokens):
            continue
        return False
    return True


def merge_llm_facets(
    facets: ThemeFacets,
    parsed: dict[str, Any] | None,
    *,
    vocabulary: dict[str, list[str]] | None = None,
    message: str | None = None,
) -> ThemeFacets:
    """Junta ao que as regras apanharam o que o LLM devolveu — só valores do vocabulário e só o que falta.

    Com o `message` em mão, o LLM só acrescenta valores com apoio no pedido (o 9B
    «completa» o tema com meia dúzia de facetas que o visitante não disse); sem apoio
    só entram quando as regras não apanharam nada, e no máximo LLM_UNGROUNDED_PER_KIND
    por faceta. Séculos e anos do LLM só entram se o pedido fala de época."""
    if not isinstance(parsed, dict):
        return facets
    vocab = vocabulary if vocabulary is not None else load_vocabulary()
    data = parsed.get("facets") if isinstance(parsed.get("facets"), dict) else parsed
    ground = message is not None
    rules_found = facets.has_visual() or facets.is_temporal() or facets.national_treasure
    folded_message = fold(message or "")
    message_tokens = set(folded_message.split())
    lay_values = _lay_values(folded_message, vocab) if ground else set()
    for kind, key in VOCAB_KEYS.items():
        index = _term_index(vocab.get(key, []))
        values = data.get(kind) if isinstance(data, dict) else None
        if not isinstance(values, list):
            continue
        current = list(getattr(facets, kind))
        ungrounded = 0
        for value in values:
            canonical = index.get(fold(str(value)))
            if not canonical or canonical in current or (kind == "colors" and canonical in facets.pigments):
                continue
            if ground and not _grounded(canonical, message_tokens, lay_values):
                if rules_found or ungrounded >= LLM_UNGROUNDED_PER_KIND:
                    continue
                ungrounded += 1
            current.append(canonical)
        setattr(facets, kind, current)
    time_ok = not ground or bool(_TIME_MENTION_RE.search(folded_message))
    centuries = data.get("centuries") if isinstance(data, dict) else None
    if isinstance(centuries, list) and not facets.centuries and time_ok:
        clean = sorted({int(c) for c in centuries if isinstance(c, (int, float)) and 1 <= int(c) <= 21})
        facets.centuries = clean
    years = data.get("year_range") if isinstance(data, dict) else None
    if isinstance(years, (list, tuple)) and len(years) == 2 and not facets.year_range and time_ok and all(isinstance(y, (int, float)) for y in years):
        facets.year_range = (int(min(years)), int(max(years)))
    if facets.stops_requested is None and isinstance(parsed.get("stops_requested"), (int, float)):
        facets.stops_requested = int(parsed["stops_requested"])
    if facets.time_budget_min is None and isinstance(parsed.get("time_budget_min"), (int, float)):
        facets.time_budget_min = int(parsed["time_budget_min"])
    for key in ("must_include", "avoid"):
        values = parsed.get(key)
        if isinstance(values, list):
            setattr(facets, key, [str(v).strip() for v in values if str(v).strip()][:10])
    theme_text = parsed.get("theme_text")
    if isinstance(theme_text, str) and theme_text.strip() and len(theme_text.strip()) <= 120:
        facets.free_text = theme_text.strip()
    facets.source = "rules+llm"
    return facets
