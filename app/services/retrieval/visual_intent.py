"""Router determinístico de intenção visual (Fase 3, Etapa 6).

Decide se uma pesquisa textual beneficia do ramo texto→imagem. Sem LLM: regras
lexicais testáveis em PT e EN, com decisão + razão + regra + confiança.

Comportamento por modo (imposto pelo chamador e reforçado aqui):
- off    -> o router nem deve ser chamado (defensivo: devolve TEXT_ONLY).
- intent -> a decisão manda.
- always -> ramo visual sempre autorizado; o router não bloqueia.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

TEXT_ONLY = "TEXT_ONLY"
TEXT_AND_VISUAL = "TEXT_AND_VISUAL"


@dataclass(frozen=True)
class VisualIntentDecision:
    decision: str
    use_visual: bool
    reason: str
    rule: str
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "use_visual": self.use_visual,
            "reason": self.reason,
            "rule": self.rule,
            "confidence": self.confidence,
        }


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = stripped.casefold()
    # Pontuação vira espaço para as fronteiras por espaço funcionarem em
    # "pretos, de couro" / "shoes in black." (hífens incluídos: "mostra-me"
    # -> "mostra me", coberto pelas regras).
    lowered = re.sub(r"[^\w\s]+", " ", lowered, flags=re.UNICODE)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return f" {lowered} "


# Sinais factuais/administrativos/biográficos/cronológicos — não ativam o ramo
# visual por si; só perdem se existir um sinal visual explícito mais forte.
_FACTUAL_PATTERNS: list[tuple[str, str]] = [
    (r" quem ", "pt_quem"),
    (r" quando ", "pt_quando"),
    (r" quantos | quantas ", "pt_quantos"),
    (r" onde ", "pt_onde"),
    (r" porque | porquê ", "pt_porque"),
    (r" horario| horários| horario de abertura| bilhete| preco| preço| entrada ", "pt_admin"),
    (r" biografia| nasceu | morreu | viveu ", "pt_biografia"),
    (r" diretor| diretora| fundado| fundacao| fundação ", "pt_institucional"),
    (r" numero de inventario| número de inventário| inventario ", "pt_inventario"),
    (r" who | when | how many | where | why ", "en_wh"),
    (r" opening hours| ticket| price| admission ", "en_admin"),
    (r" biography| was born | died ", "en_biography"),
]

# Sinais visuais: similaridade, cor, forma/padrão/motivo, representação.
_VISUAL_PATTERNS: list[tuple[str, str, float]] = [
    # similaridade visual explícita
    (r" parecid| semelhant| similar | identic| como est[ae] | mesmo estilo| do mesmo genero| do mesmo género| like this | resembl| looks like ", "similarity", 0.9),
    # construção explícita "de cor X" / "cor de X" / "da cor X" (qualquer cor)
    (r" de cor | cor de | da cor | de cores | com a cor | in colou?r ", "color_phrase", 0.85),
    # cores (PT + EN) — flexões género/número completas
    # fronteiras fechadas onde o prefixo geraria falsos positivos:
    # " azul" aberto apanhava "azulejos" (fatal no MNAZ); " rosa" apanhava "rosário".
    (
        r" azul | azuis | vermelh| verdes? | amarel| dourad| pratead| pret[oa]s? "
        r"| negr[oa]s? | branc| rosas? | rose[oa]s? | rox[oa]s? | castanh| laranja"
        r"| turquesa| cinzent| beges? | violeta| lilas ",
        "color_pt",
        0.8,
    ),
    (
        r" blue | red | green | yellow | golden | gold | silver | black | white "
        r"| pink | purple | brown | orange | gray | grey | beige | turquoise ",
        "color_en",
        0.8,
    ),
    # forma / padrão / motivo / composição
    (r" forma | formato | circular| redond| quadrad| retangular| oval | esferic| geometric| simetric | round | rounded | square | shaped? | spherical| symmetric", "shape", 0.8),
    (r" padrao| padrão| padroes| padrões| motivo| listras| riscas | as riscas| axadrezado| estampad| decorac| decoração| ornament| relevo ", "pattern_pt", 0.75),
    (r" pattern| motif | striped | checkered | decorated | ornament| composition ", "pattern_en", 0.75),
    (r" floral | flores | flor | folhagem| vegetalista ", "floral", 0.75),
    (r" animais | animal | passaro| passaros| aves? | cavalo| leao| leoes| peixes? | serpente| dragao| animals? | birds? | horses? | lions? | fish ", "animals", 0.75),
    # representação / conteúdo pictórico
    (r" imagens de | imagem de | figuras de | figura de | representando | que representem| que representam| que mostrem| que mostram| retratando| depicting | images of | with figures ", "depiction", 0.7),
    (r" visualmente| aparencia| aparência| aspecto | aspeto ", "visual_explicit", 0.85),
    (r" mostra-me | mostra me | show me ", "show_me", 0.65),
]


def decide_visual_intent(query: str, *, mode: str) -> VisualIntentDecision:
    normalized_mode = (mode or "").strip().lower()
    if normalized_mode == "always":
        return VisualIntentDecision(
            decision=TEXT_AND_VISUAL,
            use_visual=True,
            reason="mode=always executa o ramo visual em todas as pesquisas textuais",
            rule="mode_always",
            confidence=1.0,
        )
    if normalized_mode == "off":
        return VisualIntentDecision(
            decision=TEXT_ONLY,
            use_visual=False,
            reason="mode=off: router não devia ser executado (guarda defensiva)",
            rule="mode_off",
            confidence=1.0,
        )

    folded = _fold(query)

    visual_hit: tuple[str, str, float] | None = None
    for pattern, rule, confidence in _VISUAL_PATTERNS:
        if re.search(pattern, folded):
            visual_hit = (pattern, rule, confidence)
            break

    factual_rule: str | None = None
    for pattern, rule in _FACTUAL_PATTERNS:
        if re.search(pattern, folded):
            factual_rule = rule
            break

    # "mostra-me" sozinho é fraco; não chega para vencer um sinal factual.
    weak_visual = visual_hit is not None and visual_hit[1] == "show_me"
    if visual_hit and not (factual_rule and weak_visual):
        _, rule, confidence = visual_hit
        return VisualIntentDecision(
            decision=TEXT_AND_VISUAL,
            use_visual=True,
            reason=f"sinal visual detetado ({rule})"
            + (f"; sinal factual coexistente ({factual_rule}) não bloqueia" if factual_rule else ""),
            rule=rule,
            confidence=confidence,
        )
    if factual_rule:
        return VisualIntentDecision(
            decision=TEXT_ONLY,
            use_visual=False,
            reason=f"pesquisa factual/administrativa ({factual_rule}) sem sinal visual forte",
            rule=factual_rule,
            confidence=0.85,
        )
    return VisualIntentDecision(
        decision=TEXT_ONLY,
        use_visual=False,
        reason="sem sinais visuais; ramo visual não deve ser ativado por defeito",
        rule="default_no_visual_signal",
        confidence=0.6,
    )
