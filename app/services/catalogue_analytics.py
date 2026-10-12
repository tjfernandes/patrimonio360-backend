"""Perguntas analíticas sobre o acervo, reconhecidas por regras (CHAT_CATALOGUE_ANALYTICS).

«Qual é a peça mais antiga?», «quantos vestidos há nesta visita?», «de que
século é a maioria das peças?», «que autor tem mais obras?». O LLM não sabe
calcular isto a partir de oito cartões; o OpenSearch sabe. Aqui só se decide
se a pergunta é destas, de que tipo, com que assunto («vestidos») e com que
âmbito (a visita ou o museu). A pesquisa e a resposta ficam no serviço.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

OLDEST, NEWEST, COUNT, CENTURIES, TOP_CREATORS = "oldest", "newest", "count", "centuries", "top_creators"

_TOUR_SCOPE = re.compile(
    r"\b(?:nesta|neste|nessa|nesse|na|no|desta|deste|da|do)\s+(?:visita|tour|percurso)(?:\s+(?:virtual|360|digital|guiad[ao]))?\b"
    r"|\b(?:visita|tour|percurso)\s+(?:virtual|360|digital)\b"
    r"|\b(?:in\s+|of\s+|on\s+)?(?:this|the)\s+(?:virtual\s+|360\s+)?(?:tour|visit)\b"
)
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (TOP_CREATORS, re.compile(
        r"\b(?:autor|autores|autora|autoras|artista|artistas|criador|criadores|pintor|pintores|author|authors|artist|artists|maker|makers)\b"
        r".*\b(?:mais\s+(?:obras|pecas|objetos|trabalhos|representad\w*)|most\s+(?:works|pieces|objects|represented)|more\s+works)\b"
        r"|\b(?:mais\s+(?:obras|pecas|objetos)|most\s+(?:works|pieces))\b.*\b(?:autor|autores|artista|artistas|author|authors|artist|artists)\b"
    )),
    (CENTURIES, re.compile(
        r"\b(?:de|em|a)\s+que\s+secul\w*\b.*\b(?:maioria|maior\s+parte|mais\s+pecas|predomin\w*|mais\s+comum)\b"
        r"|\b(?:maioria|maior\s+parte)\b.*\bsecul\w*\b"
        r"|\b(?:distribui\w*|repartid\w*|divid\w*|contag\w*)\b.*\bsecul\w*\b"
        r"|\bpor\s+secul\w*\b"
        r"|\b(?:which|what)\s+century\b.*\b(?:most|majority)\b|\bby\s+century\b"
    )),
    (OLDEST, re.compile(r"\bmais\s+(?:antig[ao]s?|velh[ao]s?|remot[ao]s?)\b|\b(?:oldest|earliest|most\s+ancient)\b")),
    (NEWEST, re.compile(r"\bmais\s+(?:recentes?|nov[ao]s?|modern[ao]s?|atuais?|atual)\b|\b(?:newest|most\s+recent|latest)\b")),
    (COUNT, re.compile(r"^\s*(?:quantas|quantos|how\s+many)\b|\b(?:numero|total|contagem)\s+de\s+(?:pecas|objetos|obras|artefactos)\b|\bnumber\s+of\b")),
)
# Nunca é analítica: lugares («nesta sala»), semelhança, posição.
_NOT_ANALYTICS = re.compile(
    r"\b(?:sala|salas|zona|zonas|piso|pisos|room|rooms|area|areas|onde|where|parecid\w*|semelhante\w*|similar|igual|iguais|like\s+this)\b"
)
_STOPWORDS = frozenset(
    # ligação, determinantes, verbos de pedido
    "o a os as um uma uns umas de do da dos das em no na nos nas que qual quais quem e ou com sem por para ha existe existem "
    "existir tem temos tens sao es este esta estes estas deste desta neste nesta esse essa isso isto aqui ca la mais menos muito "
    "me diz diga dizer mostra mostrar mostre encontra encontrar procura procurar lista listar indica indicar sabes sabe podes "
    "pode posso podemos quero queria gostava ver conhecer saber dizes apresenta todas todos toda todo tambem entao ai "
    "the a an of in on at to for and or is are was were there be this that these those which what who how many much me show "
    "tell find list give do does can could i we you please all any some its it has have had from with by as "
    # nomes genéricos da peça e do acervo
    "peca pecas objeto objetos objecto objectos obra obras artefacto artefactos artefato artefatos item itens coisa coisas "
    "exemplar exemplares piece pieces object objects work works artifact artifacts item items thing things "
    "museu museus acervo colecao coleccao catalogo collection museum catalogue inventory inventario "
    "visita visitas tour percurso virtual digital guiada guiado visit "
    # as próprias palavras das perguntas
    "antiga antigo antigas antigos velha velho velhas velhos remota remoto recente recentes nova novo novas novos moderna "
    "moderno modernas modernos atual atuais oldest earliest ancient newest recent latest most quantas quantos numero total "
    "contagem number maioria maior parte predomina predominante comum seculo seculos century centuries distribuicao distribuição "
    "repartidas divididas por autor autores autora autoras artista artistas criador criadores pintor pintores author authors "
    "artist artists maker makers representados representadas represented trabalhos".split()
)


@dataclass(frozen=True)
class AnalyticsQuery:
    kind: str
    subject: str      # o que restringe a contagem ou a ordenação («vestidos»); vazio = tudo
    in_tour: bool     # «nesta visita»


def fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").casefold())
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", stripped).strip()


def parse_catalogue_analytics(message: str) -> AnalyticsQuery | None:
    """Tipo, assunto e âmbito de uma pergunta analítica; None se não for uma."""
    folded = fold(message)
    if not folded or _NOT_ANALYTICS.search(folded):
        return None
    kind = next((name for name, pattern in _PATTERNS if pattern.search(folded)), None)
    if kind is None:
        return None
    in_tour = bool(_TOUR_SCOPE.search(folded))
    rest = _TOUR_SCOPE.sub(" ", folded)
    subject = " ".join(token for token in rest.split() if token not in _STOPWORDS and not token.isdigit())
    return AnalyticsQuery(kind=kind, subject=subject, in_tour=in_tour)


def roman(century: int) -> str:
    """18 → XVIII; -1 → I a.C. é tratado por quem chama."""
    value = abs(century)
    out = ""
    for arabic, letters in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while value >= arabic:
            out += letters
            value -= arabic
    return out


def century_of(year: int) -> int:
    return (year - 1) // 100 + 1 if year > 0 else -((-year - 1) // 100 + 1)


def century_label(century: int, *, english: bool = False) -> str:
    if english:
        return f"{roman(century)} BC" if century < 0 else roman(century)
    return f"{roman(century)} a.C." if century < 0 else roman(century)
