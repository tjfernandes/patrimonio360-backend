from functools import lru_cache
import asyncio
from dataclasses import dataclass
import difflib
import json
import logging
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse
from uuid import uuid4

from app.core.config import Settings, get_settings
from app.core.logging import log_event, log_json_event
from app.prompts import (
    build_final_answer_prompt,
    build_router_user_prompt,
    get_router_system_prompt,
)
from app.services.catalogue_analytics import (
    CENTURIES,
    COUNT,
    NEWEST,
    OLDEST,
    TOP_CREATORS,
    AnalyticsQuery,
    century_label,
    century_of,
    parse_catalogue_analytics,
)
from app.prompts.chat_prompts import (
    build_describe_artifact_prompt,
    build_final_answer_prompt_v2,
    build_tour_context_prompt,
    card_selection_instruction,
)
from app.prompts.query_planner_prompts import (
    FOLLOW_UP_RESOLUTION_SYSTEM_PROMPT,
    RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT,
    build_follow_up_resolution_prompt,
    build_retrieval_query_rewrite_prompt,
)
from app.schemas.chat import (
    ArtifactImageResult,
    ArtifactResult,
    ChatHealthResponse,
    ChatImageMessageRequest,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatModelMessageRequest,
    ChatRegenerateRequest,
    ChatResultsPageRequest,
    ChatResultsPageResponse,
    ChatSearchScope,
    ImageMatchResult,
    ResponseFormatObject,
    TourNavigationTarget,
    ChatSubjectArtifact,
    TourRoomPiece,
    TourRoomTarget,
    ArtifactDetailContextResponse,
    AuthorEntity,
    SetEntityWithArtifacts,
    ExhibitionEntityWithArtifacts,
    RelatedArtifactItem,
    RelatedArtifactsPageResponse,
)
from app.services.embeddings import EmbeddingProvider, get_embedding_provider
from app.services.chat_session_store import (
    ChatSessionState,
    ChatSessionStore,
    get_chat_session_store,
)
from app.services.chat_i18n import normalize_language, translate
from app.services.llm_service import LLMService, LLMServiceError, get_llm_service
from app.services.opensearch_client import OpenSearchGateway, get_opensearch_gateway
from app.services.model_retrieval import ModelRetrievalService, get_model_retrieval_service
from app.services.query_logger import QueryLogger, get_query_logger, utc_timestamp
from app.services.tour_navigation import TourNavigationService, get_tour_navigation_service

StatusCallback = Callable[[dict[str, Any]], Awaitable[None]]
logger = logging.getLogger(__name__)

# Perguntas "fala-me do museu" (visao geral). Padroes aplicados ao texto
# folded (_fold_query_text: casefold, sem acentos, sem pontuacao), ancorados
# a frase completa para nunca capturar pesquisas de objetos ("vestidos no
# museu"). Uma cauda so e aceite se for parte do nome do museu atual.
# «Onde estou?» e variantes (CHAT_TOUR_LOCATION_ANSWERS). Texto dobrado, sem
# pontuação; a cauda só admite palavras vazias de contexto («agora», «no
# museu»), para «onde estão os vestidos» ou «que sala tem chapéus» irem à
# pesquisa como sempre.
_TOUR_LOCATION_TAIL = r"(?:\s+(?:agora|neste\s+momento|exatamente|na\s+visita|no\s+museu|no\s+mosteiro|right\s+now|now|exactly|in\s+the\s+tour|in\s+the\s+museum))*"
_TOUR_LOCATION_PATTERNS: tuple[str, ...] = (
    r"^(?:e\s+)?onde\s+(?:e\s+que\s+)?(?:eu\s+)?(?:estou|me\s+encontro|me\s+situo|estamos|nos\s+encontramos)(?:\s+(?:localizad[oa]|situad[oa]|posicionad[oa]|localizados|situados))?" + _TOUR_LOCATION_TAIL + r"$",
    r"^(?:em\s+)?(?:que|qual)\s+(?:parte|zona|sitio|lugar|ponto)\s+(?:do\s+(?:museu|mosteiro)|da\s+visita)\s+(?:e\s+que\s+)?(?:eu\s+)?(?:estou|me\s+encontro|estamos)" + _TOUR_LOCATION_TAIL + r"$",
    r"^onde\s+(?:e\s+que\s+)?(?:fica|e)\s+(?:isto|isso|aqui|este\s+sitio|este\s+lugar)" + _TOUR_LOCATION_TAIL + r"$",
    r"^(?:em\s+)?(?:que|qual)\s+(?:sala|zona|espaco|parte|lugar|sitio|local)(?:\s+do\s+(?:museu|mosteiro))?\s+(?:e\s+que\s+)?(?:eu\s+)?(?:estou|me\s+encontro|estamos|e\s+esta|e\s+este)" + _TOUR_LOCATION_TAIL + r"$",
    r"^(?:que|qual)\s+(?:sala|zona|espaco|lugar|sitio|local)\s+e\s+(?:esta|este|aqui)" + _TOUR_LOCATION_TAIL + r"$",
    r"^qual\s+e\s+(?:a\s+sala|a\s+zona|o\s+espaco|o\s+lugar|o\s+sitio|o\s+local)\s+(?:em\s+que|onde)\s+(?:eu\s+)?(?:estou|me\s+encontro|estamos)" + _TOUR_LOCATION_TAIL + r"$",
    r"^(?:o\s+)?que\s+(?:e\s+que\s+)?(?:ha|existe|se\s+pode\s+ver|posso\s+ver|podemos\s+ver|tem|se\s+encontra)\s+(?:aqui|nesta\s+sala|neste\s+espaco|nesta\s+zona|neste\s+sitio|neste\s+local|por\s+aqui|a\s+minha\s+volta|nesta\s+parte(?:\s+do\s+(?:museu|mosteiro))?)" + _TOUR_LOCATION_TAIL + r"$",
    r"^(?:que|quais)\s+(?:pecas|objetos|obras|artefactos)\s+(?:e\s+que\s+)?(?:ha|existem|estao|posso\s+ver|se\s+podem\s+ver|tem)\s+(?:aqui|nesta\s+sala|neste\s+espaco|nesta\s+zona|por\s+aqui|a\s+minha\s+volta)" + _TOUR_LOCATION_TAIL + r"$",
    r"^(?:o\s+)?que\s+(?:e\s+que\s+)?(?:esta|se\s+ve|vejo)\s+(?:aqui|nesta\s+sala|a\s+minha\s+volta|neste\s+espaco)" + _TOUR_LOCATION_TAIL + r"$",
    r"^where\s+am\s+i" + _TOUR_LOCATION_TAIL + r"$",
    r"^(?:which|what)\s+(?:room|area|space|part(?:\s+of\s+the\s+(?:museum|monastery))?)\s+(?:am\s+i\s+in|is\s+this)" + _TOUR_LOCATION_TAIL + r"$",
    r"^what(?:\s+is|\s*'?s)\s+(?:in\s+)?(?:here|in\s+this\s+room|in\s+this\s+area|around\s+me|around\s+here)" + _TOUR_LOCATION_TAIL + r"$",
    r"^what\s+can\s+i\s+see\s+(?:here|in\s+this\s+room|in\s+this\s+area|around\s+me)" + _TOUR_LOCATION_TAIL + r"$",
    r"^(?:which|what)\s+(?:objects|pieces|works|artifacts)\s+are\s+(?:here|in\s+this\s+room|in\s+this\s+area)" + _TOUR_LOCATION_TAIL + r"$",
)

# Palavras que podem sobrar numa pergunta sobre uma zona da visita depois de
# retirar o nome da zona; qualquer outra («vestidos») faz dela uma pesquisa.
# Inclui os verbos de pedido («encontra a sala cantonal», «leva-me ao pátio»).
_TOUR_ROOM_QUESTION_WORDS = frozenset(
    "e a o as os ao aos à na no nas nos da do de das dos em que qual quais onde fica ficam é sao existe existem "
    "existir ha tem temos tens estao "
    "esta este essa esse isso isto aqui la leva leve levar levas me vai vamos ir irmos quero queria posso podemos "
    "ver visitar conhecer mostra mostrar mostre sobre fala fale conta conte diz diga um uma pouco mais pecas objetos "
    "obras artefactos encontro encontrar encontra encontre encontras procura procurar procure procuro pesquisa "
    "pesquisar pesquise abre abrir abra navega navegar navegue localiza localizar localize situa situada situado "
    "chegar chego como para ate museu mosteiro palacio visita virtual agora entao então por favor gostava gostaria "
    "de ir ver o que sala salas zona zonas piso lugar local sitio espaco ali quero "
    "what is in the there room area take go to show tell about where can i see are which objects pieces works "
    "lets let s and visit please me us how do get find search look for locate open navigate bring".split()
)
# Lista das zonas, por estrutura e não por lista fechada de frases (medido a
# 2026-09-17 em 33 formulações: a lista fechada apanhava 13, o LLM local 18–25
# conforme o prompt — falhava até «Lista as salas»). Regra: um nome de zona no
# PLURAL (ou «mapa») é o assunto do pedido — antes dele pode vir qualquer verbo
# («listas», «podes enumerar», «indica-me»), desde que não seja «X de salas»;
# depois dele só palavras de enquadramento («do museu», «posso visitar»). Pedir
# peças, ou qualquer outra palavra a seguir («salas com vestidos»), é pesquisa.
_TOUR_AREA_PLURALS = frozenset(
    "salas zonas espacos divisoes areas lugares sitios locais partes alas pisos andares recintos compartimentos "
    "rooms spaces places zones sections floors mapa map".split()
)
_TOUR_PIECES_WORDS = frozenset(
    "pecas peca objetos objeto obras obra artefactos artefacto itens colecao colecoes acervo "
    "objects object pieces piece works items collection".split()
)
_TOUR_LIST_TAIL_WORDS = _TOUR_ROOM_QUESTION_WORDS | frozenset(
    "outras outros todas todos restantes existentes disponiveis deste desta neste nesta destas destes diferentes "
    "varias varios palacio edificio convento igreja casa parque jardim exposicao percurso tour explorar percorrer "
    "passear andar entrar descobrir seguir a seguir depois dentro ca pode podes podem se da tenho "
    "lista listas listar listagem enumera enumerar indica indicar list quantas quantos how many "
    "other all available this museum building palace visit explore walk around next inside here of does do have has any".split()
)
# Todas as peças da visita («que peças posso ver nesta visita?»): tem de nomear
# a visita, pedir peças em geral (ou «o que posso ver»), e nada mais — qualquer
# outra palavra («vestidos», «azuis», «1850») faz dela uma pesquisa.
_TOUR_SCOPE_WORDS = frozenset("visita visitas tour percurso visit".split())
_TOUR_PIECES_LIST_WORDS = (
    _TOUR_ROOM_QUESTION_WORDS
    | _TOUR_PIECES_WORDS
    | _TOUR_LIST_TAIL_WORDS
    | _TOUR_SCOPE_WORDS
    | frozenset(
        "consigo conseguimos observar assinaladas assinalados destacadas destacados marcadas marcados incluidas "
        "incluidos fazem parte expostas expostos aparecem apresenta apresentadas apresentados ola bom dia boa tarde "
        "noite todas todos pecas peças pieces available included highlighted shown displayed featured can see".split()
    )
)
# Peça selecionada: o que faz de uma mensagem um pedido SOBRE ela. Tudo por
# regras (o LLM local não é fiável a decidir intenções — medição de 2026-09-17).
_SELECTED_REFERENCE_WORDS = frozenset(
    "este esta isto isso esse essa aquele aquela aquilo deste desta disto disso desse dessa daquele daquela "
    "neste nesta nisto nisso nesse nessa dele dela lhe "
    "this that it its".split()
)
_SELECTED_ATTRIBUTE_STEMS: tuple[str, ...] = (
    "materia", "feit", "fabric", "data", "datac", "ano", "anos", "secul", "epoca", "quando", "idade", "antig",
    "quem", "autor", "criador", "criou", "fez", "desenh", "pintou", "pintor", "artista", "assin", "usava", "usou",
    "usad", "usar", "pertenc", "dono", "dona", "propriet", "dimens", "tamanho", "mede", "medida", "altura",
    "largura", "compriment", "peso", "pesa", "tecnic", "origem", "origin", "provenien", "veio", "vem", "histor",
    "conta", "fala", "descrev", "descric", "explic", "resum", "serve", "servia", "func", "utilidade", "utiliz",
    "cor", "cores", "estado", "conserv", "restaur", "expost", "guardad", "reserva", "signific", "simbol",
    "represent", "inscri", "decora", "bordad", "motivo", "detalh", "pormenor", "inform", "curios", "valor",
    "vale", "rar", "importan", "famos", "doad", "doou", "adquir", "comprad", "inventar", "numero", "categoria",
    "tipo", "nome", "chama", "titulo", "designa", "estilo", "moda", "ocasi", "cerimon", "ler", "diz", "saber",
    "made", "date", "year", "century", "when", "old", "who", "maker", "artist", "wore", "worn", "owner",
    "size", "technique", "tell", "describ", "explain", "detail", "meaning", "color", "colour", "condition",
    "story", "purpose", "name", "style",
)
# «peça», «objeto», «obra»: nomes genéricos para a peça em foco. Não são um
# assunto próprio («O que é esta peça?» é sobre a peça aberta, não uma pesquisa
# por «peça»). Só no singular: «peças de 1850» é uma pesquisa.
_SELECTED_GENERIC_NOUNS = frozenset(
    "peca objeto objecto obra artefacto artefato item "
    "piece object artifact artefact artwork".split()
)
_SELECTED_SEARCH_VERBS = frozenset(
    "encontra encontrar encontre encontras procura procurar procure procuro pesquisa pesquisar pesquise mostra "
    "mostrar mostre lista listar listas liste quero queria gostava gostaria ver apresenta sugere recomenda "
    "ha existe existem tem tens temos find search show list look want see recommend suggest are is there any".split()
)
_SELECTED_INTERROGATIVES = frozenset(
    "que qual quais quem quando onde como porque porquê quanto quanta quantos quantas e era foi sera pode podes "
    "what which who when where how why is was does did can could".split()
)
_SELECTED_FILLER_WORDS = frozenset(
    "a o as os um uma uns umas de da do das dos em na no nas nos por para com sem e ou mas se me te nos lhe "
    "mais muito pouco bem so tambem ainda ja agora entao por favor pf ola sim nao sobre acerca algo alguma algum "
    "coisa coisas me diz diga dizer dá da podes pode poderia consegues sabes sabe "
    "the an of in on for with and or but to me please some something anything about more tell".split()
)

# Conversa sem pedido: saudações, cortesia, vocativos e enchimento. Uma
# mensagem só com estas palavras não vai à pesquisa.
_SMALLTALK_WORDS = frozenset(
    "ola oi hey hello hi hej alo viva ei ola bom boa dia tarde noite bons dias tardes noites tudo bem bom otimo "
    "fixe porreiro top boas bemvindo bem vindo vinda "
    "obrigado obrigada obrigadinho obrigadinha agradecido agradecida valeu brigado thanks thank you thx ty cheers "
    "adeus tchau xau ate logo breve amanha ja depois bye goodbye see later "
    "como estas esta estao vai vais anda andas sentes te sente vao correm as coisas isso por ai contigo consigo "
    "voces tu voce a o e um uma e que quem sou es sim nao ok okay okey certo claro pois entao enfim ah oh eh uau "
    "wow hmm hum hm ha ha haha lol xd kk desculpa desculpe perdao lamento sorry please favor "
    "irmao irma mano mana amigo amiga amigos parceiro parceira colega chefe pessoal malta gente bro sis mate "
    "dude man buddy friend folks amalia there very much "
    "muito mesmo bastante tambem ainda hoje agora aqui la ca eu me mim meu minha nos nosso nossa "
    "good morning afternoon evening night how are doing everything all fine great nice cool yes yep nope "
    "i am im its it is this that so well right".split()
)
# Referências a um cartão visível, sem clicar em nada.
_ORDINAL_WORDS: dict[str, int] = {
    "primeiro": 0, "primeira": 0, "segundo": 1, "segunda": 1, "terceiro": 2, "terceira": 2,
    "quarto": 3, "quarta": 3, "quinto": 4, "quinta": 4, "sexto": 5, "sexta": 5, "setimo": 6, "setima": 6,
    "oitavo": 7, "oitava": 7, "nono": 8, "nona": 8, "decimo": 9, "decima": 9,
    "ultimo": -1, "ultima": -1, "penultimo": -2, "penultima": -2,
    "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4, "sixth": 5, "seventh": 6, "eighth": 7,
    "ninth": 8, "tenth": 9, "last": -1,
}
_REFERENCE_DETERMINERS = frozenset(
    "o a os as esse essa este esta aquele aquela do da desse dessa deste desta daquele daquela no na nesse nessa "
    "neste nesta naquele naquela pelo pela ao the that this".split()
)
# pedir peças com estes verbos é pesquisa, mesmo com «o vestido»
_REFERENCE_SEARCH_VERBS = frozenset(
    "encontra encontrar encontre procura procurar procure procuro pesquisa pesquisar pesquise lista listar liste "
    "mostra mostrar mostre ha existe existem find search show list any".split()
)

# O modelo escolhe os cartões (CHAT_MODEL_SELECTS_CARDS): bloco final
# «[[cartões: doc_1, doc_3]]» / «[[cards: …]]», fechado ou cortado no fim.
_CARD_SELECTION_BLOCK = re.compile(r"\[\[\s*(?:cart[oõ]es|cards)\s*:\s*([^\[\]]*?)\s*\]\]", re.IGNORECASE)
_CARD_SELECTION_OPEN_TAIL = re.compile(r"\[\[\s*(?:cart[oõ]es|cards)\s*:\s*([^\[\]]*)$", re.IGNORECASE)
# Pedido de uma peça pelo nome: «encontra a fuga para o egito nesta visita» →
# o nome é o que sobra depois dos verbos de pesquisa, do âmbito da visita e
# das palavras de ligação; começa por um determinante no singular.
_NAME_QUERY_STOPWORDS = (
    _SELECTED_SEARCH_VERBS
    | _SELECTED_FILLER_WORDS
    | _SELECTED_INTERROGATIVES
    | _REFERENCE_DETERMINERS
    | frozenset("visita virtual tour museu museum aqui here amalia".split())
)
_NAME_QUERY_SINGULAR_DETERMINERS = frozenset(
    "o a este esta esse essa aquele aquela the this that".split()
)
_TITLE_STOPWORDS = _SELECTED_FILLER_WORDS | _REFERENCE_DETERMINERS

_SMALLTALK_THANKS_WORDS = frozenset("obrigado obrigada obrigadinho obrigadinha agradecido agradecida valeu brigado thanks thank thx ty cheers".split())
_SMALLTALK_BYE_WORDS = frozenset("adeus tchau xau bye goodbye".split())
_SMALLTALK_PREFIX = re.compile(r"^(?:(?:ola|oi|hey|hello|hi|boas|bom dia|boa tarde|boa noite|amalia|obrigado|obrigada|ok|entao|e|por favor)\s+)*")
_SMALLTALK_PATTERNS: tuple[str, ...] = (
    r"quem (?:es|e) (?:tu|voce|voces|a amalia|amalia|o assistente)(?: afinal)?",
    r"(?:o )?que (?:podes|pode|sabes|sabe|consegues|consegue|e que podes|e que sabes)(?: tu)? fazer(?: por mim)?",
    r"(?:o que|que|como|em que) (?:e que )?(?:me )?(?:podes|pode|posso|consegues|consegue) (?:te )?(?:ajudar|perguntar|pedir)(?: me)?",
    r"(?:podes|pode|consegues|conseguem) (?:me )?ajudar(?: me)?",
    r"(?:preciso|precisava|precisamos) de ajuda",
    r"ajuda(?: me)?",
    r"who are you",
    r"what can you do(?: for me)?",
    r"(?:can|could) you help(?: me)?",
    r"i need help",
)

_TOUR_WHERE_TO_GO_PATTERN = re.compile(
    r"^(?:e\s+)?(?:(?:por|para|a|ate)\s+)?(?:a?onde|where)\s+(?:e\s+que\s+|mais\s+|else\s+)?"
    r"(?:eu\s+)?(?:posso|podemos|pode|se\s+pode|da\s+para|devo|can\s+i|can\s+we|could\s+i|should\s+i)\s+"
    r"(?:ir|andar|passear|visitar|explorar|entrar|circular|go|walk|visit|explore)\b"
)

_MUSEUM_OVERVIEW_PATTERNS: tuple[str, ...] = (
    r"^(?:por favor\s+)?(?:fala|conta|diz)(?:\s+me)?(?:\s+mais)?(?:\s+um\s+(?:pouco|bocado))?\s+(?:sobre|do|da|acerca\s+de)\s+(?:o\s+|este\s+|esta\s+)?(?:museu|mosteiro)(?P<tail>.*)$",
    r"^o\s+que\s+e\s+(?:este|o)\s+(?:museu|mosteiro)(?P<tail>.*)$",
    r"^(?:qual\s+e\s+a\s+)?historia\s+(?:deste|do)\s+(?:museu|mosteiro)(?P<tail>.*)$",
    r"^apresenta(?:\s+me)?\s+(?:o|este)\s+(?:museu|mosteiro)(?P<tail>.*)$",
    r"^sobre\s+(?:o|este)\s+(?:museu|mosteiro)(?P<tail>.*)$",
    r"^que\s+(?:museu|mosteiro)\s+e\s+este(?P<tail>.*)$",
    r"^informacoes?\s+sobre\s+(?:o|este)\s+(?:museu|mosteiro)(?P<tail>.*)$",
    r"^(?:please\s+)?tell\s+me(?:\s+more)?\s+about\s+(?:this|the)\s+(?:museum|monastery)(?P<tail>.*)$",
    r"^what\s+is\s+this\s+(?:museum|monastery)(?:\s+about)?(?P<tail>.*)$",
    r"^(?:what\s+is\s+the\s+)?history\s+of\s+(?:this|the)\s+(?:museum|monastery)(?P<tail>.*)$",
    r"^about\s+(?:this|the)\s+(?:museum|monastery)(?P<tail>.*)$",
)


@lru_cache(maxsize=1)
def _load_museum_profiles() -> dict[str, dict[str, object]]:
    path = Path(__file__).resolve().parents[1] / "data" / "museum_profiles.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - ficheiro ausente/corrompido
        return {}
    return payload if isinstance(payload, dict) else {}


@dataclass(frozen=True, slots=True)
class TemporalQuery:
    start_year: int | None
    end_year: int | None
    expression: str | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SearchMuseum:
    museum_id: str
    museum_slug: str
    museum_name: str | None
    aliases: tuple[str, ...]


_SEARCH_MUSEUMS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "search_museums.json"


@lru_cache(maxsize=1)
def _load_search_museums() -> tuple[SearchMuseum, ...]:
    fallback = [
        {
            "museum_id": "mnt",
            "museum_slug": "mnt",
            "museum_name": "Museu Nacional do Traje",
            "aliases": ["museu nacional do traje", "museu do traje", "nacional do traje", "mnt"],
        },
        {
            "museum_id": "mnaz",
            "museum_slug": "mnaz",
            "museum_name": "Museu Nacional do Azulejo",
            "aliases": ["museu nacional do azulejo", "museu do azulejo", "nacional do azulejo", "mnaz"],
        },
        {
            "museum_id": "mj",
            "museum_slug": "mj",
            "museum_name": "Mosteiro dos Jeronimos",
            "aliases": ["mosteiro dos jeronimos", "jeronimos", "jeronimo", "mj"],
        },
        {
            "museum_id": "mnsr",
            "museum_slug": "mnsr",
            "museum_name": "Museu Nacional Soares dos Reis",
            "aliases": ["museu nacional soares dos reis", "museu soares dos reis", "soares dos reis", "soares dos resi", "mnsr"],
        },
        {
            "museum_id": "cmag",
            "museum_slug": "cmag",
            "museum_name": "Casa-Museu Dr. Anastácio Gonçalves",
            "aliases": ["casa museu dr anastacio goncalves", "casa museu anastacio goncalves", "anastacio goncalves", "cmag"],
        },
        {
            "museum_id": "mnaa",
            "museum_slug": "mnaa",
            "museum_name": "Museu Nacional de Arte Antiga",
            "aliases": ["museu nacional de arte antiga", "museu de arte antiga", "arte antiga", "janelas verdes", "mnaa"],
        },
    ]
    try:
        raw = json.loads(_SEARCH_MUSEUMS_CONFIG_PATH.read_text(encoding="utf-8"))
        entries = raw if isinstance(raw, list) else fallback
    except Exception:
        entries = fallback

    museums: list[SearchMuseum] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        museum_id = str(entry.get("museum_id") or "").strip()
        museum_slug = str(entry.get("museum_slug") or museum_id).strip()
        if not museum_id or not museum_slug:
            continue
        raw_aliases = entry.get("aliases")
        aliases = tuple(
            str(alias or "").strip()
            for alias in (raw_aliases if isinstance(raw_aliases, list) else [])
            if str(alias or "").strip()
        )
        museums.append(
            SearchMuseum(
                museum_id=museum_id,
                museum_slug=museum_slug,
                museum_name=str(entry.get("museum_name") or "").strip() or None,
                aliases=aliases or (museum_slug, museum_id),
            )
        )
    return tuple(museums)


_TOUR_SCOPE_PATTERNS: tuple[str, ...] = (
    r"\b(?:nesta|neste|nessa|nesse|na|no|desta|deste|da|do)\s+(?:visita|tour|percurso)(?:\s+(?:virtual|360|digital|guiada|guiado))?\b",
    r"\b(?:visita|tour|percurso)\s+(?:virtual|360|digital)\b",
    r"\b(?:in\s+)?this\s+(?:virtual\s+|360\s+|digital\s+)?(?:tour|visit)\b",
    r"\b(?:in\s+the\s+)?(?:virtual|360|digital)\s+(?:tour|visit)\b",
)

_INVENTORY_PREFIXES: tuple[str, ...] = (
    "cmag",
    "mj",
    "mnaa",
    "mnac",
    "mnaj",
    "mnar",
    "mnap",
    "mnaz",
    "mnmc",
    "mnsr",
    "mnt",
)

# Quick win SRCH-04/LLM-01 (CHAT_INVENTORY_GRAMMAR_V2): gramática dos números
# de inventário reais, levantada do export congelado (benchmark_dataset/data/
# corpus/artifacts.jsonl, 2026-09-15):
#   MNSR  «793 Mob MNSR», «255 Cer CMP/ MNSR», «1 Pin CMP (DHL)/ MNSR», «91/ 51 Pin MNSR»
#   MNAz  «MNAz 1 Az», «MNAz 1151 Proj», «MNAz 706 Cer», «MNAz 12/3 Az»
#   MJ    «MJ_ES032/1», «MJ_SN_10», «MJ_IM01/2»
#   MNT   «1234», «1234/5», «MNT CF 4» (só via marcadores; um número solto é ambíguo)
# Os padrões correm sobre o texto folded (minúsculas, sem acentos, qualquer
# pontuação vira um espaço), pelo que «CMP/ MNSR» surge como «cmp mnsr».
_INVENTORY_TYPE_CODES_MNSR: tuple[str, ...] = (
    "cer", "pin", "vid", "esc", "our", "grav", "div", "mob", "lap", "obj",
    "lit", "des", "tex", "leq", "min",
)
_INVENTORY_TYPE_CODES_MNAZ: tuple[str, ...] = ("az", "proj", "cer", "tec", "esc")
_INVENTORY_V2_MNSR_NUMBER_FIRST = re.compile(
    r"\b(?P<num>\d{1,6}(?: \d{1,6})?) (?P<type>" + "|".join(_INVENTORY_TYPE_CODES_MNSR) + r")"
    r"(?: (?P<coll>cmp|cmfc)(?: dhl)?)? mnsr\b"
)
_INVENTORY_V2_MNSR_PREFIX_FIRST = re.compile(
    r"\bmnsr (?P<num>\d{1,6}) (?P<type>" + "|".join(_INVENTORY_TYPE_CODES_MNSR) + r")\b"
)
_INVENTORY_V2_MNAZ = re.compile(
    r"\bmnaz (?P<num>\d{1,6}(?: \d{1,6})?) (?P<type>" + "|".join(_INVENTORY_TYPE_CODES_MNAZ) + r")\b"
)
_INVENTORY_V2_MJ = re.compile(
    r"\bmj (?P<kind>es|im)(?P<num>\d{1,4})(?: (?P<sub>\d{1,3}))?\b|\bmj sn (?P<sn>\d{1,4})\b"
)

_EXPLICIT_INVENTORY_MARKERS: tuple[tuple[str, ...], ...] = (
    ("numero", "de", "inventario"),
    ("n", "de", "inventario"),
    ("inventario",),
    ("referencia",),
    ("ref",),
)

_ARTIFACT_REFERENCE_MARKERS: set[str] = {
    "artefacto",
    "artefactos",
    "artefato",
    "artefatos",
    "item",
    "itens",
    "objeto",
    "objetos",
    "peca",
    "pecas",
}

_ARTIFACT_REFERENCE_NUMBER_CONNECTORS: set[str] = {
    "n",
    "no",
    "nr",
    "num",
    "numero",
}


_HARDCODED_HISTORICAL_PERIODS: tuple[tuple[tuple[str, ...], TemporalQuery], ...] = (
    (
        (
            "periodo pombalino",
            "epoca pombalina",
            "era pombalina",
        ),
        TemporalQuery(1750, 1777, "periodo pombalino", 1.0),
    ),
    (
        (
            "periodo joanino",
            "periodo joanino de d joao v",
        ),
        TemporalQuery(1706, 1750, "periodo joanino", 1.0),
    ),
    (
        (
            "periodo manuelino",
            "epoca manuelina",
        ),
        TemporalQuery(1495, 1521, "periodo manuelino", 1.0),
    ),
    (
        (
            "periodo sebastianista",
            "periodo sebastianistas",
            "sebastianista",
            "sebastianistas",
            "crise de sucessao",
        ),
        TemporalQuery(1578, 1580, "periodo sebastianista", 1.0),
    ),
    (
        ("periodo miguelista", "miguelista", "miguelistas"),
        TemporalQuery(1828, 1834, "periodo miguelista", 1.0),
    ),
)

_ROMAN_CENTURY_NUMERALS: dict[str, int] = {
    "xv": 15,
    "xvi": 16,
    "xvii": 17,
    "xviii": 18,
    "xix": 19,
    "xx": 20,
    "xxi": 21,
    "xxii": 22,
    "xxiii": 23,
    "xxiv": 24,
    "xxv": 25,
    "xxvi": 26,
    "xxvii": 27,
    "xxviii": 28,
    "xxix": 29,
    "xxx": 30,
}



_PT_QUERY_LANGUAGE_HINTS: set[str] = {
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "ao",
    "aos",
    "para",
    "por",
    "com",
    "sem",
    "sobre",
    "que",
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "encontra",
    "encontrar",
    "procura",
    "procurar",
    "mostra",
    "museu",
    "peca",
    "pecas",
    "quero",
    "ver",
    "crianca",
    "criancas",
    "infantil",
    "infantis",
    "traje",
    "trajes",
    "vestuario",
    "vestuarios",
    "roupa",
    "roupas",
    "vestido",
    "vestidos",
    "chapeu",
    "chapeus",
    "cabeca",
    "arte",
    "retrato",
    "retratos",
    "azulejo",
    "azulejos",
}

_EN_QUERY_LANGUAGE_HINTS: set[str] = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "with",
    "without",
    "about",
    "find",
    "search",
    "show",
    "museum",
    "piece",
    "pieces",
    "child",
    "children",
    "dress",
    "dresses",
    "hat",
    "hats",
    "headgear",
    "headwear",
    "tile",
    "tiles",
}

class ChatService:
    """Chat service.

    Route-facing service layer.
    Uses a hybrid chat pipeline with optional RAG retrieval.
    """

    def __init__(
        self,
        settings: Settings,
        opensearch_gateway: OpenSearchGateway,
        embedding_provider: EmbeddingProvider,
        model_retrieval_service: ModelRetrievalService,
        tour_navigation_service: TourNavigationService,
        llm_service: LLMService,
        session_store: ChatSessionStore,
        query_logger: QueryLogger | None = None,
        reranker_service: Any | None = None,
        vl_reranker_service: Any | None = None,
    ) -> None:
        self.settings = settings
        self.opensearch_gateway = opensearch_gateway
        self.embedding_provider = embedding_provider
        self.model_retrieval_service = model_retrieval_service
        self.tour_navigation_service = tour_navigation_service
        self.llm_service = llm_service
        self.session_store = session_store
        self.query_logger = query_logger or get_query_logger()
        self._reranker_service = reranker_service
        self._vl_reranker_service = vl_reranker_service

    def _get_reranker_service(self) -> Any:
        if self._reranker_service is None:
            from app.services.reranker import get_reranker_service

            self._reranker_service = get_reranker_service()
        return self._reranker_service

    def _get_vl_reranker_service(self) -> Any:
        if self._vl_reranker_service is None:
            from app.services.reranker import get_vl_reranker_service

            self._vl_reranker_service = get_vl_reranker_service()
        return self._vl_reranker_service

    async def _maybe_rerank_text_docs(
        self,
        *,
        query_text: str,
        docs: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Second-stage rerank of the text retrieval pool (graceful fallback).

        Returns ``docs`` untouched when reranking is disabled or fails; the
        first-stage (hybrid search) order is always a valid answer.
        """
        effective_top_k = max(int(top_k), 1)
        if not self.settings.CHAT_ENABLE_RERANKING or not docs:
            return docs[:effective_top_k]
        started_at = time.monotonic()
        try:
            reranked = await self._get_reranker_service().rerank(
                query_text=query_text,
                documents=docs,
                top_k=effective_top_k,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.retrieval.rerank.error",
                doc_count=len(docs),
                duration_ms=round((time.monotonic() - started_at) * 1000, 1),
                error=exc,
            )
            # First-stage order stays valid; truncate the widened pool back
            # to the requested page size so downstream shape is unchanged.
            return docs[:effective_top_k]
        log_event(
            logger,
            logging.INFO,
            "chat.retrieval.rerank.finish",
            doc_count=len(docs),
            kept=len(reranked),
            duration_ms=round((time.monotonic() - started_at) * 1000, 1),
        )
        return reranked

    def _resolve_vl_rerank_stage(self) -> str:
        """Posição do rerank VL (local/global/both/off) na rota de imagem.

        CHAT_ENABLE_VL_RERANKING continua a ser o master switch: desligado,
        o stage é sempre "off" (nenhum modelo VL de rerank é tocado)."""
        if not self.settings.CHAT_ENABLE_VL_RERANKING:
            return "off"
        stage = str(
            getattr(self.settings, "MULTIMODAL_VL_RERANK_STAGE", "local") or "local"
        ).strip().lower()
        return stage if stage in {"off", "local", "global", "both"} else "local"

    @staticmethod
    def _local_vl_rerank_pass_enabled(stage: str, *, has_query_text: bool) -> bool:
        """O passe local (pool i2i pré-fusão) corre em local/both. Em global
        com pedido SÓ-imagem também corre: sem fusão, o pool i2i É o pool
        global, e este passe único preserva o comportamento imagem-only."""
        if stage in {"local", "both"}:
            return True
        return stage == "global" and not has_query_text

    async def _maybe_rerank_image_hits(
        self,
        *,
        query_text: str,
        query_image_bytes: bytes | None,
        image_hits: list[dict[str, Any]],
        top_k: int,
        enabled: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Second-stage visual rerank of i2i candidates (graceful fallback).

        ``enabled`` sobrepõe o gate por flag quando o chamador já resolveu o
        stage (ex.: stage=global adia o rerank para o pool fundido); None
        mantém o comportamento histórico guiado por CHAT_ENABLE_VL_RERANKING.
        Desativado, devolve o pool truncado ao page size (shape inalterado).
        """
        effective_top_k = max(int(top_k), 1)
        if enabled is None:
            enabled = bool(self.settings.CHAT_ENABLE_VL_RERANKING)
        if not enabled or not image_hits:
            return image_hits[:effective_top_k]
        started_at = time.monotonic()
        try:
            reranked = await self._get_vl_reranker_service().rerank_image_hits(
                query_text=query_text,
                query_image_bytes=query_image_bytes,
                image_hits=image_hits,
                top_k=effective_top_k,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.image.rerank.error",
                hit_count=len(image_hits),
                duration_ms=round((time.monotonic() - started_at) * 1000, 1),
                error=exc,
            )
            return image_hits[:effective_top_k]
        log_event(
            logger,
            logging.INFO,
            "chat.image.rerank.finish",
            hit_count=len(image_hits),
            kept=len(reranked),
            duration_ms=round((time.monotonic() - started_at) * 1000, 1),
        )
        return reranked

    async def _maybe_global_vl_rerank_outcome(
        self,
        *,
        outcome: Any,
        query_text: str,
        query_image_bytes: bytes | None,
        conversation_id: str,
        query_id: str,
    ) -> Any:
        """Rerank VL GLOBAL do pool fundido (todas as origens), pós-RRF.

        Avalia a query multimodal completa (imagem carregada + texto) contra o
        top-N fundido — candidatos vindos de i2i, joint, t2i e artifact_search.
        Em qualquer falha a ordem RRF fica intacta (fallback já dentro de
        apply_global_vl_rerank); aqui apenas isolamos exceções de wiring."""
        from app.services.retrieval.multimodal_retrieval import apply_global_vl_rerank

        top_n = max(
            int(getattr(self.settings, "MULTIMODAL_GLOBAL_VL_RERANK_TOP_N", 16)), 1
        )
        started_at = time.monotonic()
        try:
            return await apply_global_vl_rerank(
                outcome,
                vl_reranker=self._get_vl_reranker_service(),
                query_text=query_text,
                query_image_bytes=query_image_bytes,
                top_n=top_n,
                trace_id=query_id,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.image.global_rerank.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round((time.monotonic() - started_at) * 1000, 1),
                error=exc,
            )
            return outcome

    def health(self) -> ChatHealthResponse:
        return ChatHealthResponse(
            llm_provider=self.settings.LLM_PROVIDER,
            llm_base_url=self.settings.llm_base_url_resolved,
            llm_text_model=self.settings.llm_model_resolved,
            llm_json_model=self.settings.llm_model_resolved,
            text_embedding_model=self.settings.text_embedding_model_resolved,
            multimodal_embedding_model=self.settings.multimodal_embedding_model_resolved,
        )

    async def _emit_status(
        self,
        status_cb: StatusCallback | None,
        key: str,
        *,
        language: str | None = None,
        **fields: object,
    ) -> None:
        if status_cb is None:
            return
        payload: dict[str, Any] = {
            "type": "status",
            "key": key,
            "message": translate(key, language, **fields),
            **fields,
        }
        try:
            await status_cb(payload)
        except Exception:
            pass

    def _sync_state_language(
        self,
        state: ChatSessionState,
        requested_language: str | None,
    ) -> str:
        language = normalize_language(requested_language or state.language)
        state.language = language
        return language

    def _final_system_prompt(self, system_prompt: str | None, language: str | None) -> str:
        language_guard = translate("llm.final_language_guard", language)
        base_prompt = (system_prompt or "").strip()
        if not base_prompt:
            return language_guard
        return f"{base_prompt}\n\n{language_guard}"

    def _resolve_results_page_size(self, requested_page_size: int | None, *, default_size: int) -> int:
        base = max(int(default_size), 1)
        if requested_page_size is None:
            return base
        return max(1, min(int(requested_page_size), 50))

    def _text_results_default_page_size(self) -> int:
        return max(int(self.settings.CHAT_RETRIEVAL_RESULTS_PAGE_SIZE), 1)

    def _text_retrieval_window_size(self, *, minimum: int = 1) -> int:
        configured = int(getattr(self.settings, "CHAT_RETRIEVAL_PAGINATION_WINDOW", 0) or 0)
        return max(configured, int(minimum), 1)

    def _image_retrieval_window_size(self, *, minimum: int = 1) -> int:
        configured = int(
            getattr(self.settings, "CHAT_IMAGE_RETRIEVAL_PAGINATION_WINDOW", 0) or 0
        )
        return max(configured, int(minimum), 1)

    def _retrieval_window_from_request(self, request: dict[str, Any]) -> int | None:
        try:
            value = int(request.get("retrieval_window_size") or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _bounded_retrieval_total(self, total: int, retrieval_window_size: int | None) -> int:
        resolved_total = max(int(total), 0)
        if retrieval_window_size is None:
            return resolved_total
        return min(resolved_total, max(int(retrieval_window_size), 1))

    def _paginate_retrieval_results(
        self,
        *,
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        navigation_targets: list[TourNavigationTarget],
        page: int,
        page_size: int | None,
        default_page_size: int,
        total_override: int | None = None,
    ) -> tuple[
        list[ArtifactResult],
        list[ImageMatchResult],
        list[TourNavigationTarget],
        int,
        int,
        int,
        bool,
    ]:
        resolved_page = max(int(page), 1)
        resolved_page_size = self._resolve_results_page_size(
            page_size,
            default_size=default_page_size,
        )
        available_total = len(artifact_results) if artifact_results else len(image_matches)
        reported_total = (
            max(int(total_override), available_total, 0)
            if total_override is not None
            else available_total
        )
        if available_total <= 0:
            start = (resolved_page - 1) * resolved_page_size
            return (
                [],
                [],
                [],
                resolved_page,
                resolved_page_size,
                reported_total,
                start < reported_total,
            )

        start = (resolved_page - 1) * resolved_page_size
        if start >= available_total:
            return [], [], [], resolved_page, resolved_page_size, reported_total, start < reported_total

        end = min(start + resolved_page_size, available_total)
        has_more = end < reported_total

        artifact_slice: list[ArtifactResult] = []
        if artifact_results:
            artifact_slice = artifact_results[start:end]

        def _norm(value: str | None) -> str:
            return str(value or "").strip().casefold()

        if artifact_slice:
            artifact_ids = {
                str(artifact.artifact_id or "").strip()
                for artifact in artifact_slice
                if str(artifact.artifact_id or "").strip()
            }
            inventories = {
                _norm(artifact.inventory_number)
                for artifact in artifact_slice
                if _norm(artifact.inventory_number)
            }
            image_slice = [
                match
                for match in image_matches
                if (
                    str(match.artifact_id or "").strip() in artifact_ids
                    or _norm(match.inventory) in inventories
                )
            ]
            if not image_slice and image_matches:
                image_slice = image_matches[start:end]
            navigation_slice = [
                target
                for target in navigation_targets
                if _norm(target.inventory_id) in inventories
            ]
        else:
            image_slice = image_matches[start:end]
            image_inventories = {
                _norm(match.inventory)
                for match in image_slice
                if _norm(match.inventory)
            }
            if image_inventories:
                navigation_slice = [
                    target
                    for target in navigation_targets
                    if _norm(target.inventory_id) in image_inventories
                ]
            else:
                navigation_slice = navigation_targets[start:end]

        return (
            artifact_slice,
            image_slice,
            navigation_slice,
            resolved_page,
            resolved_page_size,
            reported_total,
            has_more,
        )

    def _reported_results_total(self, request: dict[str, Any], page_total: int) -> int:
        try:
            stored_total = int(request.get("results_total") or 0)
        except (TypeError, ValueError):
            stored_total = 0
        reported_total = max(stored_total, int(page_total), 0)
        retrieval_window_size = self._retrieval_window_from_request(request)
        reported_total = self._bounded_retrieval_total(
            reported_total,
            retrieval_window_size,
        )
        request["results_total"] = reported_total
        return reported_total

    def _cache_last_retrieval_results(
        self,
        *,
        state: ChatSessionState,
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        navigation_targets: list[TourNavigationTarget],
        default_page_size: int,
        retrieval_request: dict[str, Any] | None = None,
    ) -> str:
        results_request = dict(retrieval_request or {})
        results_request_id = str(results_request.get("results_request_id") or "").strip()
        if not results_request_id:
            results_request_id = str(uuid4())
        results_request["results_request_id"] = results_request_id

        artifact_payload = [
            artifact.model_dump(mode="json")
            for artifact in artifact_results
        ]
        image_payload = [
            match.model_dump(mode="json")
            for match in image_matches
        ]
        navigation_payload = [
            target.model_dump(mode="json")
            for target in navigation_targets
        ]
        resolved_default_page_size = max(int(default_page_size), 1)

        state.last_paged_artifact_results = artifact_payload
        state.last_paged_image_matches = image_payload
        state.last_paged_navigation_targets = navigation_payload
        state.last_paged_results_default_page_size = resolved_default_page_size
        state.last_paged_retrieval_request = results_request

        paged_results_by_request_id = getattr(state, "paged_results_by_request_id", None)
        if not isinstance(paged_results_by_request_id, dict):
            paged_results_by_request_id = {}
            state.paged_results_by_request_id = paged_results_by_request_id

        if results_request_id in paged_results_by_request_id:
            paged_results_by_request_id.pop(results_request_id, None)
        paged_results_by_request_id[results_request_id] = {
            "artifact_results": artifact_payload,
            "image_matches": image_payload,
            "navigation_targets": navigation_payload,
            "default_page_size": resolved_default_page_size,
            "retrieval_request": results_request,
        }
        while len(paged_results_by_request_id) > 20:
            oldest_key = next(iter(paged_results_by_request_id))
            paged_results_by_request_id.pop(oldest_key, None)

        return results_request_id

    def _build_paged_results(
        self,
        *,
        state: ChatSessionState,
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        navigation_targets: list[TourNavigationTarget],
        page: int,
        page_size: int | None,
        default_page_size: int,
        total_override: int | None = None,
        retrieval_request: dict[str, Any] | None = None,
    ) -> tuple[
        list[ArtifactResult],
        list[ImageMatchResult],
        list[TourNavigationTarget],
        int,
        int,
        int,
        bool,
    ]:
        self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            default_page_size=default_page_size,
            retrieval_request=retrieval_request,
        )
        return self._paginate_retrieval_results(
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=page,
            page_size=page_size,
            default_page_size=default_page_size,
            total_override=total_override,
        )

    async def _fetch_artifact_docs_for_image_hits(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_museum_id: str | None,
        image_hits: list[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        if not image_hits:
            return []

        # Join artifact_id-first: os image hits trazem sempre artifact_id, e o
        # fetch por ids é exato (terms em filtro) e preserva a ordem dos hits.
        # O join por inventário perde artefactos quando os inventários partilham
        # tokens ("260 Cer MNSR" vs "260 Cer CMP/ MNSR"): a query de inventários
        # pontua por multi_match e corta em size, deixando artefactos pedidos de
        # fora. Inventário fica apenas como fallback para hits sem artifact_id.
        artifact_ids = [
            str(hit.get("artifact_id") or "").strip()
            for hit in image_hits
            if str(hit.get("artifact_id") or "").strip()
        ]
        if artifact_ids:
            artifact_docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
                museum_slug=museum_slug,
                museum_id=museum_id,
                artifact_ids=artifact_ids,
                top_k=max(top_k, 1),
            )
            if artifact_docs:
                return artifact_docs

        inventory_numbers = [
            self._image_hit_inventory(hit)
            for hit in image_hits
            if self._image_hit_inventory(hit)
        ]
        if not inventory_numbers:
            return []
        return await self.opensearch_gateway.fetch_artifacts_by_inventory_numbers(
            museum_slug=museum_slug,
            museum_id=artifact_museum_id,
            inventory_numbers=inventory_numbers,
            top_k=max(top_k, 1),
        )

    def _museum_profile_for(self, museum_id: str | None) -> dict[str, object] | None:
        key = (museum_id or "").strip().lower()
        if not key:
            return None
        profile = _load_museum_profiles().get(key)
        return profile if isinstance(profile, dict) else None

    def _is_museum_overview_query(self, text: str, *, museum_name: str | None) -> bool:
        folded = self._fold_query_text(text)
        if not folded:
            return False
        museum_tokens = set(self._fold_query_text(museum_name or "").split())
        for pattern in _MUSEUM_OVERVIEW_PATTERNS:
            match = re.match(pattern, folded)
            if not match:
                continue
            tail_tokens = (match.group("tail") or "").split()
            if not tail_tokens:
                return True
            # "fala-me sobre o museu nacional do traje": a cauda so pode
            # conter palavras do nome do proprio museu.
            if museum_tokens and all(token in museum_tokens for token in tail_tokens):
                return True
        return False

    def _maybe_override_for_museum_overview(
        self,
        *,
        router_decision: dict[str, object],
        query_text: str,
        museum_id: str | None,
    ) -> dict[str, object]:
        """Pergunta geral sobre o museu -> forca o caminho rag para que o
        fast-path de overview (em _retrieve_context) responda com o perfil
        do museu, mesmo que o router LLM tenha dito llm_only."""
        profile = self._museum_profile_for(museum_id)
        if profile is None:
            return router_decision
        if not self._is_museum_overview_query(
            query_text,
            museum_name=str(profile.get("name") or ""),
        ):
            return router_decision
        updated = dict(router_decision)
        updated["mode"] = "rag"
        updated["needs_retrieval"] = True
        updated["intent"] = "overview"
        updated["reason"] = self._append_reason_tag(
            str(updated.get("reason", "")),
            "deterministic_museum_overview",
        )
        return updated

    def _maybe_override_llm_only_for_visual(
        self,
        *,
        router_decision: dict[str, object],
        query_text: str,
    ) -> dict[str, object]:
        """E7B: o router LLM pode classificar pesquisas visuais ("objetos com
        flores") como llm_only, saltando o retrieval por completo. Com o
        multimodal ligado, um probe determinístico de intenção visual (regras
        de modo intent, MESMO quando MULTIMODAL_RETRIEVAL_MODE=always — para
        não converter conversa em pesquisa) promove a rota a rag, onde o fluxo
        normal artifact_search + t2i[-only] decide os ramos. Em off devolve a
        decisão intacta."""
        mode = str(self.settings.MULTIMODAL_RETRIEVAL_MODE or "off").strip().lower()
        if mode == "off" or router_decision.get("mode") != "llm_only":
            return router_decision
        from app.services.retrieval.visual_intent import decide_visual_intent

        probe = decide_visual_intent(query_text, mode="intent")
        if not probe.use_visual:
            return router_decision
        log_event(
            logger,
            logging.INFO,
            "multimodal.router.override",
            original_mode="llm_only",
            new_mode="rag",
            rule=probe.rule,
            reason=probe.reason,
        )
        return {
            **router_decision,
            "mode": "rag",
            "intent": "visual_search",
            "needs_retrieval": True,
            "reason": f"visual override: {probe.rule}",
        }

    async def _maybe_fuse_text_to_image(
        self,
        *,
        query_text: str,
        museum_slug: str,
        museum_id: str | None,
        artifact_docs: list[dict[str, object]],
        conversation_id: str,
        query_id: str,
    ):
        """Ramo texto→imagem da Fase 3, atrás de MULTIMODAL_RETRIEVAL_MODE.

        Devolve None sempre que o ramo visual não deva/consiga correr — o
        chamador mantém a baseline intacta (artifact_search nunca é
        substituído). Em ``off`` sai antes de importar/instanciar o que quer
        que seja (equivalência funcional e nenhum load do modelo VL).
        """
        mode = str(self.settings.MULTIMODAL_RETRIEVAL_MODE or "off").strip().lower()
        if mode == "off":
            return None

        from app.services.retrieval.visual_intent import decide_visual_intent

        decision = decide_visual_intent(query_text, mode=mode)
        log_event(
            logger,
            logging.INFO,
            "multimodal.router.decision",
            conversation_id=conversation_id,
            query_id=query_id,
            mode=mode,
            **decision.as_dict(),
        )
        if not decision.use_visual:
            return None

        service = getattr(self, "_multimodal_text_retrieval", None)
        if service is None:
            from app.services.retrieval.multimodal_retrieval import MultimodalTextRetrieval

            service = MultimodalTextRetrieval(
                settings=self.settings,
                opensearch_gateway=self.opensearch_gateway,
                embedding_provider=self.embedding_provider,
            )
            self._multimodal_text_retrieval = service
        try:
            return await service.fuse_text_search(
                query_text=query_text,
                museum_slug=museum_slug,
                museum_id=museum_id,
                artifact_docs=artifact_docs,
                router_decision=decision,
                trace_id=query_id,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "multimodal.fusion.error",
                conversation_id=conversation_id,
                query_id=query_id,
                error=exc,
            )
            return None

    async def _maybe_fuse_image_text(
        self,
        *,
        message_text: str,
        museum_slug: str,
        museum_id: str | None,
        i2i_hits: list[dict[str, object]],
        conversation_id: str,
        query_id: str,
        query_image_bytes: bytes | None = None,
    ):
        """Etapa 8 — imagem+texto atrás de MULTIMODAL_RETRIEVAL_MODE.

        Só corre quando o utilizador forneceu texto real com a imagem e o modo
        multimodal está ligado. Em off (ou falha), devolve None e o endpoint
        mantém o fluxo imagem-only atual, byte-idêntico."""
        mode = str(self.settings.MULTIMODAL_RETRIEVAL_MODE or "off").strip().lower()
        if mode == "off" or not (message_text or "").strip():
            return None

        from app.services.retrieval.visual_intent import decide_visual_intent

        # O probe decide o ramo t2i; a componente documental corre sempre que
        # há texto (o texto refina a imagem, nunca é ignorado).
        probe = decide_visual_intent(message_text, mode="intent")
        temporal_query = await self._interpret_temporal_query(message_text)
        temporal_filter = self._merge_temporal_filter({}, temporal_query)
        log_event(
            logger,
            logging.INFO,
            "multimodal.image_text.plan",
            conversation_id=conversation_id,
            query_id=query_id,
            run_t2i=probe.use_visual,
            probe_rule=probe.rule,
            temporal="_temporal_interval" in temporal_filter,
        )

        service = getattr(self, "_multimodal_text_retrieval", None)
        if service is None:
            from app.services.retrieval.multimodal_retrieval import MultimodalTextRetrieval

            service = MultimodalTextRetrieval(
                settings=self.settings,
                opensearch_gateway=self.opensearch_gateway,
                embedding_provider=self.embedding_provider,
            )
            self._multimodal_text_retrieval = service
        try:
            return await service.fuse_image_text_search(
                message_text=message_text,
                museum_slug=museum_slug,
                museum_id=museum_id,
                i2i_hits=list(i2i_hits),
                run_t2i=probe.use_visual,
                temporal_filter=temporal_filter or None,
                router_decision=probe,
                trace_id=query_id,
                query_image_bytes=query_image_bytes,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "multimodal.image_text.error",
                conversation_id=conversation_id,
                query_id=query_id,
                error=exc,
            )
            return None

    def _artifact_results_as_prompt_docs(
        self,
        artifact_results: list[ArtifactResult],
    ) -> list[dict[str, object]]:
        docs: list[dict[str, object]] = []
        excluded_fields = {"images"}
        if not self.settings.CHAT_INCLUDE_ORIGIN_HISTORY_IN_LLM_CONTEXT:
            excluded_fields.add("origin_history")
        for artifact in artifact_results:
            payload = artifact.model_dump(
                mode="json",
                exclude_none=True,
                exclude=excluded_fields,
            )
            if payload:
                docs.append(payload)
        return docs

    def _image_matches_as_visible_prompt_docs(
        self,
        image_matches: list[ImageMatchResult],
    ) -> list[dict[str, object]]:
        docs: list[dict[str, object]] = []
        artifact_fields = (
            "inventory_number",
            "title",
            "museum",
            "category",
            "super_category",
            "creator",
            "creators",
            "date_or_period",
            "support_or_material",
            "technique",
            "origin_history",
            "production_center",
            "description",
            "detail_type",
            "sets",
            "exhibitions",
            "image_count",
        )
        for match in image_matches:
            artifact = match.artifact if isinstance(match.artifact, dict) else {}
            inventory = (
                str(match.inventory or "").strip()
                or str(artifact.get("inventory_number") or artifact.get("inventory") or "").strip()
            )
            title = str(match.title or "").strip() or str(artifact.get("title") or "").strip()
            payload: dict[str, object] = {"result_card_type": "image_match"}
            if title:
                payload["title"] = title
            if inventory:
                payload["inventory"] = inventory
            for field in artifact_fields:
                value = artifact.get(field)
                if value in (None, "", [], {}):
                    continue
                if field == "origin_history" and not self.settings.CHAT_INCLUDE_ORIGIN_HISTORY_IN_LLM_CONTEXT:
                    continue
                if field == "inventory_number" and inventory:
                    continue
                if field == "title" and title:
                    continue
                payload[field] = value
            docs.append(payload)
        return docs

    def _build_visible_results_retrieval_context(
        self,
        *,
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        page: int,
        page_size: int,
        total: int,
        match_section_label: str = "visible_image_matches",
        prefer_artifact_results: bool = False,
        include_image_matches_section: bool = True,
    ) -> str:
        artifact_docs = self._artifact_results_as_prompt_docs(artifact_results)
        image_docs = (
            self._image_matches_as_visible_prompt_docs(image_matches)
            if image_matches
            else []
        )
        if prefer_artifact_results and artifact_docs:
            docs_for_prompt = artifact_docs
        else:
            docs_for_prompt = image_docs or artifact_docs
        visible_count = len(docs_for_prompt)
        if visible_count <= 0:
            return ""

        sections = [
            f"visible_results_page: {max(int(page), 1)}",
            f"visible_results_page_size: {max(int(page_size), 1)}",
            f"visible_results_count: {visible_count}",
            f"visible_results_total: {max(int(total), 0)}",
            "The frontend displays exactly visible_results_count result cards from current_visible_results for this answer.",
        ]

        if self.settings.CHAT_COMPACT_ANSWER_CONTEXT:
            # Quick win PR-01: cartões compactos legíveis em vez de JSON cru
            # com todos os campos (menos tokens, menos campos internos a
            # "não mencionar"). Mesma ordem e mesmos rótulos [doc_N], pelo que
            # a sanitização por índice continua alinhada.
            docs_context = self._format_docs_compact(docs=docs_for_prompt)
        else:
            docs_context = self._format_docs_for_prompt(
                docs=docs_for_prompt,
                top_k=max(len(docs_for_prompt), 1),
            )
        if docs_context:
            sections.append("current_visible_results:")
            sections.append(docs_context)

        if (
            image_matches
            and include_image_matches_section
            # Com cartões compactos, o dump JSON das correspondências repetia a
            # mesma informação (prompt de imagem de ~9,8 mil caracteres).
            and not (self.settings.CHAT_COMPACT_ANSWER_CONTEXT and docs_context)
        ):
            image_match_payload = [
                match.model_dump(
                    exclude_none=True,
                    exclude={"artifact", "navigation_target"},
                )
                for match in image_matches
            ]
            if not docs_context:
                sections.append("current_visible_results:")
                sections.append(json.dumps(image_match_payload, ensure_ascii=True))
            sections.append(f"{match_section_label}:")
            sections.append(json.dumps(image_match_payload, ensure_ascii=True))

        return "\n".join(sections).strip()

    def _is_llm_token_limit_error(self, exc: Exception) -> bool:
        message = f"{exc} {getattr(exc, '__cause__', '')}".casefold()
        # Limite de SAÍDA (max_tokens atingido) não é contexto a mais: repetir
        # com menos cartões não resolve e custava uma geração completa por
        # tentativa.
        if "length limit was reached" in message or "lengthfinishreason" in message:
            return False
        return any(
            marker in message
            for marker in (
                "length limit",
                "context length",
                "maximum context",
                "max context",
                "token limit",
                "too many tokens",
                "prompt is too long",
                "input tokens",
                "prompt_tokens",
                "completionusage",
                "maximum number of tokens",
                "exceeded the model",
                "request too large",
            )
        )

    def _final_answer_stream_factory(
        self,
        status_cb: StatusCallback | None,
        requested_format: ResponseFormatObject,
        *,
        query_id: str,
    ) -> Callable[[], Callable[[str], Awaitable[None]]] | None:
        if (
            self.settings.CHAT_STREAM_FINAL_ANSWER
            and status_cb is not None
            and requested_format.type == "text"
        ):
            return self._make_token_emitter(status_cb, query_id=query_id)
        return None

    def _media_final_prompt(
        self,
        *,
        modality: str,
        museum_slug: str,
        museum_name: str | None,
        filters_state: dict[str, Any],
        sort_state: dict[str, Any],
        user_message: str,
        retrieval_context: str,
        language: str | None,
    ) -> str:
        """Prompt final das rotas de imagem e de modelo 3D.

        Com CHAT_COMPACT_ANSWER_CONTEXT usa o prompt v2 (curto, PT acentuado,
        com a nota de que semelhança não prova identidade); desligado, é
        exatamente a chamada histórica.
        """
        if self.settings.CHAT_COMPACT_ANSWER_CONTEXT:
            return build_final_answer_prompt_v2(
                museum_name=museum_name,
                intent=f"{modality}_search",
                mode="rag",
                rolling_summary="",
                filters_state=filters_state,
                user_message=user_message,
                rewritten_query=user_message,
                retrieval_context=retrieval_context,
                use_history_for_answer=False,
                language=language,
                input_modality=modality,
            )
        return build_final_answer_prompt(
            museum_slug=museum_slug,
            museum_name=museum_name,
            input_modality=modality,
            mode="rag",
            intent=f"{modality}_search",
            rolling_summary="",
            filters_state=filters_state,
            sort_state=sort_state,
            user_message=user_message,
            rewritten_query=user_message,
            retrieval_context=retrieval_context,
            use_history_for_answer=False,
            language=language,
        )

    def _make_token_emitter(
        self,
        status_cb: StatusCallback | None,
        *,
        query_id: str,
    ) -> Callable[[], Callable[[str], Awaitable[None]]]:
        """Fábrica de callbacks de tokens (quick win LLM-03).

        Cada tentativa de geração recebe um emissor novo com `seq` a começar
        em 0, para o cliente poder descartar texto parcial de uma tentativa
        anterior. Falhas do canal nunca interrompem a geração.
        """

        def factory() -> Callable[[str], Awaitable[None]]:
            seq = 0

            async def emit(delta: str) -> None:
                nonlocal seq
                if status_cb is None:
                    return
                event = {"type": "token", "query_id": query_id, "seq": seq, "delta": delta}
                seq += 1
                try:
                    await status_cb(event)
                except Exception:  # noqa: BLE001 — o canal SSE nunca pode travar a resposta
                    pass

            return emit

        return factory

    async def _generate_final_answer_with_context_retries(
        self,
        *,
        initial_message: str,
        response_format: ResponseFormatObject,
        system_prompt: str | None,
        model_override: str | None,
        visible_result_count: int,
        build_message_for_result_count: Callable[[int], str],
        stream_cb_factory: Callable[[], Callable[[str], Awaitable[None]]] | None = None,
    ):
        async def _call(message: str):
            if stream_cb_factory is not None:
                return await self.llm_service.generate_stream(
                    message=message,
                    response_format=response_format,
                    system_prompt=system_prompt,
                    model_override=model_override,
                    on_delta=stream_cb_factory(),
                )
            return await self.llm_service.generate(
                message=message,
                response_format=response_format,
                system_prompt=system_prompt,
                model_override=model_override,
            )

        try:
            return await _call(initial_message)
        except LLMServiceError as exc:
            if not self._is_llm_token_limit_error(exc) or visible_result_count <= 0:
                raise
            last_error: LLMServiceError = exc

        for result_count in range(visible_result_count - 1, -1, -1):
            retry_message = build_message_for_result_count(result_count)
            if not retry_message.strip():
                continue
            try:
                return await _call(retry_message)
            except LLMServiceError as exc:
                if not self._is_llm_token_limit_error(exc):
                    raise
                last_error = exc

        raise last_error

    def _results_page_query_text(self, request: dict[str, Any]) -> str:
        for key in ("query_text", "lexical_query", "original_query", "user_message"):
            value = str(request.get(key) or "").strip()
            if value:
                return value
        plan = request.get("plan")
        if isinstance(plan, dict):
            for key in ("query_text", "semantic_query"):
                value = str(plan.get(key) or "").strip()
                if value:
                    return value
        kind = str(request.get("kind") or "").strip()
        return kind

    def _build_results_page_reply_prompt(
        self,
        *,
        museum_slug: str,
        language: str | None,
        query_text: str,
        page: int,
        total: int,
        docs_context: str,
    ) -> str:
        resolved_language = normalize_language(language)
        if resolved_language == "en":
            return "\n".join(
                [
                    "You are a virtual assistant for a 360 museum tour.",
                    "Final answer language: English.",
                    "The user clicked a UI control to view the next page of search results.",
                    "Write a concise answer about ONLY the visible results in current_page_results.",
                    "Do not summarize, reuse, or mention previous result pages.",
                    "Do not use conversation history as a source of facts.",
                    "Never expose artifact_id or internal doc markers such as [doc_1].",
                    "When referring to an object, prefer its title and optionally inventory reference.",
                    "If using an ordered list, use consecutive markers (1., 2., 3.) without blank lines between items.",
                    f"museum_slug: {museum_slug}",
                    f"search_query: {query_text or 'unknown'}",
                    f"results_page: {page}",
                    f"reported_total: {total}",
                    "current_page_results:",
                    docs_context,
                ]
            )

        return "\n".join(
            [
                "Es um assistente virtual para uma visita 360 ao museu.",
                "Idioma final da resposta: portugues.",
                "O utilizador clicou num controlo de UI para ver a proxima pagina de resultados.",
                "Escreve uma resposta curta sobre APENAS os resultados visiveis em current_page_results.",
                "Nao resumas, reutilizes ou menciones paginas de resultados anteriores.",
                "Nao uses historico da conversa como fonte de factos.",
                "Nunca exponhas artifact_id nem marcadores internos como [doc_1].",
                "Ao referires um objeto, privilegia o titulo e opcionalmente a referencia de inventario.",
                "Se usares lista numerada, usa marcadores consecutivos (1., 2., 3.) sem linhas em branco entre itens.",
                f"museum_slug: {museum_slug}",
                f"search_query: {query_text or 'desconhecida'}",
                f"results_page: {page}",
                f"reported_total: {total}",
                "current_page_results:",
                docs_context,
            ]
        )

    async def _generate_results_page_reply(
        self,
        *,
        payload: ChatResultsPageRequest,
        request: dict[str, Any],
        docs: list[dict[str, object]],
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult] | None = None,
        page: int,
        total: int,
    ) -> str:
        language = normalize_language(payload.language)
        docs_for_prompt = (
            self._image_matches_as_visible_prompt_docs(image_matches or [])
            if image_matches
            else docs or self._artifact_results_as_prompt_docs(artifact_results)
        )
        docs_context = self._format_docs_for_prompt(
            docs=docs_for_prompt,
            top_k=max(len(docs_for_prompt), 1),
        )
        if not docs_context:
            return translate("message.results_page_fallback", language)

        # Quick win LLM-01: «ver mais resultados» é um clique de UI; uma frase
        # de template substitui a chamada LLM (os cartões carregam a informação).
        if self.settings.CHAT_TEMPLATE_RESULTS_PAGE_REPLY:
            return translate(
                "message.results_page_template",
                language,
                page=max(int(page), 1),
                count=len(docs_for_prompt),
                total=max(int(total), len(docs_for_prompt)),
            )

        prompt = self._build_results_page_reply_prompt(
            museum_slug=self._request_museum_slug(payload, request),
            language=language,
            query_text=self._results_page_query_text(request),
            page=page,
            total=total,
            docs_context=docs_context,
        )
        try:
            llm_response = await self.llm_service.generate(
                message=prompt,
                response_format=ResponseFormatObject(type="text"),
                system_prompt=self._final_system_prompt(None, language),
                model_override=None,
            )
            reply = self._sanitize_assistant_reply(
                str(getattr(llm_response, "text", "") or ""),
                docs=docs_for_prompt,
                language=language,
            )
            if reply:
                return reply
        except Exception as exc:
            pass
        return translate("message.results_page_fallback", language)

    async def _materialize_text_results_page(
        self,
        *,
        state: ChatSessionState,
        payload: ChatResultsPageRequest,
        request: dict[str, Any],
        page: int,
        page_size: int,
    ) -> ChatResultsPageResponse:
        from_offset = (page - 1) * page_size
        request_museum_slug = self._request_museum_slug(payload, request)
        request_museum_id = self._request_museum_id(payload, request)
        search_scope = self._search_scope_from_request(
            request,
            fallback_museum_slug=request_museum_slug,
            fallback_museum_id=request_museum_id,
        )
        # Com fichas iguais juntas, a página N não pode ser um `from` no
        # OpenSearch (as fichas juntas na página 1 voltariam na 2): pede-se
        # desde o início até ao fim da página pedida, junta-se e corta-se.
        collapse = self._collapse_enabled()
        fetch_from = 0 if collapse else from_offset
        fetch_size = self._collapse_fetch_size(from_offset + page_size) if collapse else page_size
        window = self._retrieval_window_from_request(request)
        page_result = await self.opensearch_gateway.search_relevant_context_page(
            museum_slug=request_museum_slug,
            museum_id=request_museum_id,
            query_text=str(request.get("query_text") or ""),
            lexical_query=str(request.get("lexical_query") or "") or None,
            query_embedding=list(request.get("query_embedding") or []),
            from_offset=fetch_from,
            page_size=fetch_size,
            filters=dict(request.get("filters") or {}),
            sort=dict(request.get("sort") or {}),
            retrieval_window_size=max(window, fetch_size) if window is not None else None,
        )
        docs = list(page_result.results)
        if collapse:
            docs = self._collapse_identical_set_members(docs)[from_offset: from_offset + page_size]
        image_matches: list[ImageMatchResult] = []
        image_hits: list[dict[str, Any]] = []
        artifact_ids = [
            str(doc.get("artifact_id") or "").strip()
            for doc in docs
            if str(doc.get("artifact_id") or "").strip()
        ]
        if artifact_ids:
            try:
                image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                    museum_slug=request_museum_slug,
                    museum_id=request_museum_id,
                    artifact_ids=artifact_ids,
                    per_artifact=1,
                    max_total=max(len(artifact_ids), 1),
                )
            except Exception as exc:
                image_hits = []
            image_matches = self._build_image_matches(
                image_hits=image_hits,
                artifact_docs=docs,
            )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=request_museum_slug,
            museum_id=request_museum_id,
            docs=docs,
        )
        # Build the full artifact records (all images) so the detail modal matches
        # the image/model search behaviour. The 1-image `image_hits` above only feed
        # the result-card thumbnails, not the modal gallery.
        artifact_results = await self._build_artifact_results(
            museum_slug=request_museum_slug,
            museum_id=request_museum_id,
            artifact_docs=docs,
        )
        image_matches = self._enrich_image_matches(
            context="text_page",
            conversation_id=payload.conversation_id,
            museum_slug=request_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        reported_total = self._reported_results_total(request, page_result.total)
        reply = await self._generate_results_page_reply(
            payload=payload,
            request=request,
            docs=docs,
            artifact_results=artifact_results,
            image_matches=image_matches,
            page=page,
            total=reported_total,
        )
        results_request_id = self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            default_page_size=page_size,
            retrieval_request=request,
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            reply=reply,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            results_page=page,
            results_page_size=page_size,
            results_total=reported_total,
            results_has_more=(from_offset + page_size) < reported_total,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    async def _materialize_media_results_page(
        self,
        *,
        state: ChatSessionState,
        payload: ChatResultsPageRequest,
        request: dict[str, Any],
        page: int,
        page_size: int,
    ) -> ChatResultsPageResponse:
        from_offset = (page - 1) * page_size
        kind = str(request.get("kind") or "")
        request_museum_slug = self._request_museum_slug(payload, request)
        museum_id = self._request_museum_id(payload, request)
        search_scope = self._search_scope_from_request(
            request,
            fallback_museum_slug=request_museum_slug,
            fallback_museum_id=museum_id,
        )
        # Fichas iguais juntas: como na página de texto, desde o início.
        collapse = self._collapse_enabled()
        fetch_from = 0 if collapse else from_offset
        fetch_size = self._collapse_fetch_size(from_offset + page_size) if collapse else page_size
        window = self._retrieval_window_from_request(request)
        window = max(window, fetch_size) if window is not None else None
        if kind == "model":
            page_result = await self.opensearch_gateway.search_similar_images_multi_page(
                museum_slug=request_museum_slug,
                museum_id=museum_id,
                image_embeddings=list(request.get("image_embeddings") or []),
                from_offset=fetch_from,
                page_size=fetch_size,
                retrieval_window_size=window,
            )
        else:
            page_result = await self.opensearch_gateway.search_similar_images_page(
                museum_slug=request_museum_slug,
                museum_id=museum_id,
                image_embedding=list(request.get("image_embedding") or []),
                from_offset=fetch_from,
                page_size=fetch_size,
                retrieval_window_size=window,
            )

        image_hits = list(page_result.results)
        artifact_docs = await self._fetch_artifact_docs_for_image_hits(
            museum_slug=request_museum_slug,
            museum_id=museum_id,
            artifact_museum_id=str(request.get("artifact_museum_id") or "").strip() or None,
            image_hits=image_hits,
            top_k=fetch_size,
        )
        if collapse:
            artifact_docs = self._collapse_identical_set_members(artifact_docs)[from_offset: from_offset + page_size]
            kept = {str(doc.get("artifact_id") or "").strip() for doc in artifact_docs}
            image_hits = [hit for hit in image_hits if str(hit.get("artifact_id") or "").strip() in kept]
        image_matches = self._build_image_matches(
            image_hits=image_hits,
            artifact_docs=artifact_docs,
        )
        artifact_results = await self._build_artifact_results(
            museum_slug=request_museum_slug,
            museum_id=museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=request_museum_slug,
            museum_id=museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context=f"{kind}_page",
            conversation_id=payload.conversation_id,
            museum_slug=request_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        reported_total = self._reported_results_total(request, page_result.total)
        reply = await self._generate_results_page_reply(
            payload=payload,
            request=request,
            docs=artifact_docs,
            artifact_results=artifact_results,
            image_matches=image_matches,
            page=page,
            total=reported_total,
        )
        results_request_id = self._cache_last_retrieval_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            default_page_size=page_size,
            retrieval_request=request,
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            reply=reply,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            results_page=page,
            results_page_size=page_size,
            results_total=reported_total,
            results_has_more=(from_offset + page_size) < reported_total,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    async def get_results_page(self, payload: ChatResultsPageRequest) -> ChatResultsPageResponse:
        state = self.session_store.get(payload.conversation_id)
        if state is None or state.museum_slug != payload.museum_slug:
            return ChatResultsPageResponse(
                conversation_id=payload.conversation_id,
                results_page=max(payload.results_page, 1),
                results_page_size=self._resolve_results_page_size(
                    payload.results_page_size,
                    default_size=self._text_results_default_page_size(),
                ),
                results_total=0,
                results_has_more=False,
                results_request_id=payload.results_request_id,
            )
        self._sync_state_language(state, payload.language)

        requested_results_request_id = str(payload.results_request_id or "").strip()
        paged_results_by_request_id = getattr(state, "paged_results_by_request_id", None)
        if not isinstance(paged_results_by_request_id, dict):
            paged_results_by_request_id = {}
            state.paged_results_by_request_id = paged_results_by_request_id
        snapshot = (
            paged_results_by_request_id.get(requested_results_request_id)
            if requested_results_request_id
            else None
        )
        latest_results_request_id = str(
            (state.last_paged_retrieval_request or {}).get("results_request_id") or ""
        ).strip()
        if (
            requested_results_request_id
            and snapshot is None
            and requested_results_request_id != latest_results_request_id
        ):
            return ChatResultsPageResponse(
                conversation_id=payload.conversation_id,
                results_page=max(payload.results_page, 1),
                results_page_size=self._resolve_results_page_size(
                    payload.results_page_size,
                    default_size=self._text_results_default_page_size(),
                ),
                results_total=0,
                results_has_more=False,
                results_request_id=requested_results_request_id,
            )
        retrieval_request = dict(
            (snapshot or {}).get("retrieval_request")
            or state.last_paged_retrieval_request
            or {}
        )
        retrieval_kind = str(retrieval_request.get("kind") or "")
        default_page_size_floor = (
            self._text_results_default_page_size()
            if retrieval_kind == "text"
            else max(self.settings.CHAT_RETRIEVAL_TOP_K, 1)
        )
        default_page_size = max(
            int((snapshot or {}).get("default_page_size") or state.last_paged_results_default_page_size or 0),
            default_page_size_floor,
        )
        results_page = max(payload.results_page, 1)
        results_page_size = self._resolve_results_page_size(
            payload.results_page_size,
            default_size=default_page_size,
        )
        if retrieval_kind == "text":
            response = await self._materialize_text_results_page(
                state=state,
                payload=payload,
                request=retrieval_request,
                page=results_page,
                page_size=results_page_size,
            )
            self.session_store.save(state)
            return response
        if retrieval_kind in {"image", "model"}:
            response = await self._materialize_media_results_page(
                state=state,
                payload=payload,
                request=retrieval_request,
                page=results_page,
                page_size=results_page_size,
            )
            self.session_store.save(state)
            return response
        artifact_results = [
            ArtifactResult(**item)
            for item in (
                (snapshot or {}).get("artifact_results")
                or state.last_paged_artifact_results
            )
        ]
        image_matches = [
            ImageMatchResult(**item)
            for item in (
                (snapshot or {}).get("image_matches")
                or state.last_paged_image_matches
            )
        ]
        navigation_targets = [
            TourNavigationTarget(**item)
            for item in (
                (snapshot or {}).get("navigation_targets")
                or state.last_paged_navigation_targets
            )
        ]
        (
            paged_artifacts,
            paged_image_matches,
            paged_navigation_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._paginate_retrieval_results(
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=default_page_size,
        )
        reply = await self._generate_results_page_reply(
            payload=payload,
            request=retrieval_request,
            docs=[],
            artifact_results=paged_artifacts,
            image_matches=paged_image_matches,
            page=results_page,
            total=results_total,
        )
        search_scope = self._search_scope_from_request(
            retrieval_request,
            fallback_museum_slug=self._request_museum_slug(payload, retrieval_request),
            fallback_museum_id=self._request_museum_id(payload, retrieval_request),
        )
        return ChatResultsPageResponse(
            conversation_id=payload.conversation_id,
            reply=reply,
            artifact_results=paged_artifacts,
            image_matches=paged_image_matches,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=(
                str(retrieval_request.get("results_request_id") or requested_results_request_id).strip()
                or None
            ),
            search_scope=search_scope,
        )

    def _log_correlation_ids(
        self,
        *,
        metadata: dict[str, Any] | None,
        museum_slug: str | None,
    ) -> dict[str, str | None]:
        """Resolve the IDs required by the backend_query log event.

        `session_id`, `participant_id` and `task_id` have no first-class field
        on the chat request schemas today; they are read from the optional
        `metadata` payload (`ChatMessageRequest.metadata` and friends) so a
        future frontend/evaluation harness can populate them without a schema
        change. `tour_id` falls back to `museum_slug`, which is the identifier
        actually used to resolve a virtual tour in this codebase (see
        `poi_tours/panorama-overlays-inventory-<museum_slug>.json`).
        """
        meta = metadata or {}

        def _meta_str(key: str) -> str | None:
            value = meta.get(key)
            text = str(value).strip() if value is not None else ""
            return text or None

        resolved_tour_id = _meta_str("tour_id") or (str(museum_slug).strip() or None if museum_slug else None)
        return {
            "session_id": _meta_str("session_id"),
            "participant_id": _meta_str("participant_id"),
            "task_id": _meta_str("task_id"),
            "tour_id": resolved_tour_id,
        }

    def _selected_artifact_from_metadata(
        self,
        metadata: dict[str, Any] | None,
    ) -> dict[str, str | None] | None:
        if not isinstance(metadata, dict):
            return None

        selected = metadata.get("selected_artifact")
        selected_payload = selected if isinstance(selected, dict) else {}

        def _meta_str(*keys: str) -> str | None:
            for key in keys:
                value = selected_payload.get(key)
                if value is None:
                    value = metadata.get(key)
                text = str(value).strip() if value is not None else ""
                if text:
                    return text
            return None

        artifact_id = _meta_str("artifact_id", "selected_artifact_id")
        if not artifact_id:
            return None

        return {
            "artifact_id": artifact_id,
            "inventory_number": _meta_str("inventory_number", "selected_inventory_number"),
            "title": _meta_str("title", "selected_artifact_title"),
            "source_query_id": _meta_str("source_query_id", "selected_artifact_query_id"),
            "source": _meta_str("source", "selected_artifact_source"),
            "museum_id": _meta_str("museum_id", "selected_museum_id"),
            "museum_slug": _meta_str("museum_slug", "selected_museum_slug"),
            "museum_name": _meta_str("museum_name", "selected_museum_name"),
        }

    def _selected_artifact_context_mode(self, message: str) -> str:
        folded = f" {self._fold_query_text(message)} "
        anchored_patterns = (
            r"\b(outro|outros|outra|outras)\b",
            r"\bparecid[oa]s?\b",
            r"\bsemelhantes?\b",
            r"\bsimilares?\b",
            r"\brelacionad[oa]s?\b",
            r"\bmais como (este|esta|isto|isso)\b",
            r"\bmore like (this|it)\b",
            r"\bother\b",
            r"\bsimilar\b",
            r"\blike this\b",
            r"\brelated\b",
        )
        if any(re.search(pattern, folded) for pattern in anchored_patterns):
            return "anchored_similarity"
        return "selected_artifact"

    def _is_targeted_artifact_search_request(self, message: str) -> bool:
        folded = f" {self._fold_query_text(message)} "
        targeted_patterns = (
            r"\b(encontra|encontrar|procura|procurar|pesquisa|pesquisar)\b",
            r"\b(existe|existem|ha|haver|tem|tens)\b",
            r"\b(find|search|look for|are there|is there|exists?)\b",
        )
        return any(re.search(pattern, folded) for pattern in targeted_patterns)

    def _resolve_requested_search_museum(self, query: str) -> SearchMuseum | None:
        folded_query = f" {self._fold_query_text(query)} "
        if not folded_query.strip():
            return None

        candidates: list[tuple[int, SearchMuseum]] = []
        for museum in _load_search_museums():
            alias_matches = False
            for alias in museum.aliases:
                folded_alias = self._fold_query_text(alias)
                if not folded_alias:
                    continue
                phrase = r"\s+".join(re.escape(token) for token in folded_alias.split())
                if re.search(rf"\b{phrase}\b", folded_query):
                    alias_matches = True
                    candidates.append((len(folded_alias), museum))
                    break
            if alias_matches:
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _strip_search_museum_expression_from_query(
        self,
        text: str,
        target_museum: SearchMuseum | None,
    ) -> str:
        raw_text = (text or "").strip()
        if not raw_text or target_museum is None:
            return raw_text

        folded_text, source_indices = self._fold_query_text_with_index(raw_text)
        if not folded_text or not source_indices:
            return raw_text

        remove_mask = [False] * len(raw_text)
        patterns: list[str] = []
        aliases = sorted(
            {self._fold_query_text(alias) for alias in target_museum.aliases},
            key=len,
            reverse=True,
        )
        for folded_alias in aliases:
            tokens = folded_alias.split()
            if not tokens:
                continue
            phrase = r"\s+".join(re.escape(token) for token in tokens)
            patterns.append(
                rf"\b(?:(?:no|na|nos|nas|em|do|da|dos|das|de|in|at|the)\s+)?"
                rf"(?:(?:museu|museu\s+nacional|mosteiro)\s+)?{phrase}\b"
            )

        for pattern in patterns:
            for match in re.finditer(pattern, folded_text):
                start, end = match.span()
                if start >= len(source_indices) or end <= 0:
                    continue
                source_start = source_indices[max(start, 0)]
                source_end = source_indices[min(end - 1, len(source_indices) - 1)] + 1
                for index in range(max(source_start, 0), min(source_end, len(remove_mask))):
                    remove_mask[index] = True

        cleaned = "".join(
            char for index, char in enumerate(raw_text) if not remove_mask[index]
        )
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,.;:!?") or raw_text

    def _build_search_scope(
        self,
        *,
        current_museum_slug: str,
        current_museum_id: str | None,
        current_museum_name: str | None,
        search_museum_slug: str,
        search_museum_id: str | None,
        search_museum_name: str | None,
    ) -> ChatSearchScope:
        current_key = (current_museum_id or current_museum_slug or "").strip()
        search_key = (search_museum_id or search_museum_slug or "").strip()
        return ChatSearchScope(
            museum_id=search_museum_id,
            museum_slug=search_museum_slug,
            museum_name=search_museum_name or current_museum_name,
            is_cross_museum=bool(current_key and search_key and current_key != search_key),
        )

    def _search_scope_from_request(
        self,
        request: dict[str, Any],
        *,
        fallback_museum_slug: str,
        fallback_museum_id: str | None,
    ) -> ChatSearchScope | None:
        raw_scope = request.get("search_scope")
        if isinstance(raw_scope, ChatSearchScope):
            return raw_scope
        if isinstance(raw_scope, dict):
            museum_slug = str(raw_scope.get("museum_slug") or "").strip()
            if museum_slug:
                return ChatSearchScope(
                    museum_id=str(raw_scope.get("museum_id") or "").strip() or None,
                    museum_slug=museum_slug,
                    museum_name=str(raw_scope.get("museum_name") or "").strip() or None,
                    is_cross_museum=bool(raw_scope.get("is_cross_museum")),
                )
        if not fallback_museum_slug:
            return None
        return ChatSearchScope(
            museum_id=fallback_museum_id,
            museum_slug=fallback_museum_slug,
            museum_name=None,
            is_cross_museum=False,
        )

    def _request_museum_slug(self, payload: Any, request: dict[str, Any]) -> str:
        return str(request.get("museum_slug") or getattr(payload, "museum_slug", "") or "").strip()

    def _request_museum_id(self, payload: Any, request: dict[str, Any]) -> str | None:
        return str(request.get("museum_id") or getattr(payload, "museum_id", "") or "").strip() or None

    def _selected_artifact_reference_terms(
        self,
        doc: dict[str, object],
        selected_artifact: dict[str, str | None],
    ) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        def _append(value: object) -> None:
            if isinstance(value, list):
                for item in value:
                    _append(item)
                return
            text = str(value or "").strip()
            if not text:
                return
            folded = self._fold_query_text(text)
            if not folded or folded in seen:
                return
            seen.add(folded)
            terms.append(text)

        for key in (
            "title",
            "category",
            "super_category",
            "support_or_material",
            "technique",
            "date_or_period",
            "creator",
            "creators",
        ):
            _append(doc.get(key))
        _append(selected_artifact.get("title"))
        return terms[:12]

    def _build_anchored_similarity_query(
        self,
        *,
        query: str,
        reference_doc: dict[str, object],
        selected_artifact: dict[str, str | None],
    ) -> str:
        reference_terms = self._selected_artifact_reference_terms(
            reference_doc,
            selected_artifact,
        )
        parts = [str(query or "").strip(), *reference_terms]
        return " ".join(part for part in parts if part).strip() or "objetos semelhantes"

    async def _retrieve_selected_artifact_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        selected_artifact: dict[str, str | None],
        query: str,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]]:
        artifact_id = str(selected_artifact.get("artifact_id") or "").strip()
        retrieval_request: dict[str, Any] = {
            "kind": "selected_artifact",
            "museum_id": museum_id,
            "museum_slug": museum_slug,
            "original_query": query,
            "search_query": query,
            "query_text": query,
            "lexical_query": query,
            "selected_artifact": dict(selected_artifact),
            "selected_context_mode": "selected_artifact",
            "filters": {"artifact_id": artifact_id} if artifact_id else {},
            "sort": {},
            "results_total": 0,
        }
        if not artifact_id:
            return "", 0, [], retrieval_request

        try:
            docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
                museum_slug=museum_slug,
                museum_id=museum_id,
                artifact_ids=[artifact_id],
                top_k=1,
            )
        except Exception:
            docs = []

        retrieval_request["results_total"] = len(docs)
        if not docs:
            context = "\n".join(
                [
                    "selected_result_mode: true",
                    "selected_artifact_found: false",
                    f"selected_artifact_id: {artifact_id}",
                    "Instruction: The user selected one result card, but the artifact was not found in retrieval. Do not answer from unrelated collection results.",
                ]
            )
            return context, 0, [], retrieval_request

        docs_context = self._format_docs_for_prompt(docs=docs, top_k=1)
        context = "\n".join(
            [
                "selected_result_mode: true",
                "selected_artifact_found: true",
                "Instruction: The user selected exactly one result card in the UI. Answer ONLY about selected_artifact. If the requested fact is not present in selected_artifact, say that the available record does not contain that information. Do not use other search results as factual context.",
                "selected_artifact:",
                docs_context,
            ]
        ).strip()
        return context, len(docs), docs, retrieval_request

    async def _retrieve_anchored_similarity_context(
        self,
        *,
        selected_artifact: dict[str, str | None],
        query: str,
        filters: dict[str, object],
        sort: dict[str, object],
        result_window_size: int | None,
        reference_museum_slug: str | None = None,
        reference_museum_id: str | None = None,
        search_museum_slug: str | None = None,
        search_museum_id: str | None = None,
        museum_slug: str | None = None,
        museum_id: str | None = None,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]]:
        reference_museum_slug = (reference_museum_slug or museum_slug or "").strip()
        reference_museum_id = (reference_museum_id or museum_id or "").strip() or None
        search_museum_slug = (search_museum_slug or museum_slug or reference_museum_slug).strip()
        search_museum_id = (search_museum_id or museum_id or reference_museum_id or "").strip() or None
        artifact_id = str(selected_artifact.get("artifact_id") or "").strip()
        retrieval_request: dict[str, Any] = {
            "kind": "anchored_similarity",
            "museum_id": search_museum_id,
            "museum_slug": search_museum_slug,
            "reference_museum_id": reference_museum_id,
            "reference_museum_slug": reference_museum_slug,
            "original_query": query,
            "search_query": query,
            "query_text": query,
            "lexical_query": query,
            "selected_artifact": dict(selected_artifact),
            "selected_context_mode": "anchored_similarity",
            "filters": dict(filters),
            "sort": dict(sort),
            "results_total": 0,
        }
        if not artifact_id:
            return "", 0, [], retrieval_request

        try:
            reference_docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
                museum_slug=reference_museum_slug,
                museum_id=reference_museum_id,
                artifact_ids=[artifact_id],
                top_k=1,
            )
        except Exception:
            reference_docs = []

        if not reference_docs:
            context = "\n".join(
                [
                    "selected_result_mode: true",
                    "selected_context_mode: anchored_similarity",
                    "reference_artifact_found: false",
                    f"selected_artifact_id: {artifact_id}",
                    "Instruction: The user selected one result card as a reference for finding similar objects, but the reference artifact was not found. Do not answer from unrelated collection results.",
                ]
            )
            return context, 0, [], retrieval_request

        reference_doc = reference_docs[0]
        anchored_query = self._build_anchored_similarity_query(
            query=query,
            reference_doc=reference_doc,
            selected_artifact=selected_artifact,
        )
        retrieval_window = (
            max(int(result_window_size), 1) + 1
            if result_window_size is not None
            else None
        )
        (
            _candidate_context,
            _candidate_total,
            candidate_docs,
            candidate_request,
        ) = await self._retrieve_context(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            query=anchored_query,
            filters=filters,
            sort=sort,
            result_window_size=retrieval_window,
        )

        filtered_docs = [
            doc
            for doc in candidate_docs
            if str(doc.get("artifact_id") or "").strip() != artifact_id
        ]
        if result_window_size is not None:
            filtered_docs = filtered_docs[: max(int(result_window_size), 1)]
        retrieval_request = {
            **candidate_request,
            "kind": "anchored_similarity",
            "original_query": query,
            "anchor_query": anchored_query,
            "selected_artifact": dict(selected_artifact),
            "selected_context_mode": "anchored_similarity",
            "reference_artifact_found": True,
            "reference_artifact_id": artifact_id,
            "results_total": len(filtered_docs),
        }

        reference_context = self._format_docs_for_prompt(
            docs=reference_docs,
            top_k=1,
        )
        candidate_context = self._format_docs_for_prompt(
            docs=filtered_docs,
            top_k=max(len(filtered_docs), 1),
        )
        context_parts = [
            "selected_result_mode: true",
            "selected_context_mode: anchored_similarity",
            "Instruction: The user selected reference_artifact in the UI and is asking for other/similar objects. Use reference_artifact only as the comparison anchor. Answer about similar_candidate_results only, and do not include the reference artifact as a candidate.",
            "reference_artifact:",
            reference_context,
            "similar_candidate_results:",
            candidate_context or "No similar candidate results were found.",
        ]
        return "\n".join(context_parts).strip(), len(filtered_docs), filtered_docs, retrieval_request

    def _log_retrieved_artifact_entries(
        self,
        docs: list[dict[str, Any]],
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for doc in docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            entry = {
                "artifact_id": artifact_id,
                "inventory_number": doc.get("inventory_number") or None,
                "title": doc.get("title") or None,
                "score": doc.get("score"),
                "source": source,
            }
            group_size = doc.get("group_size")
            if isinstance(group_size, int) and group_size > 1:
                entry["group_size"] = group_size
            entries.append(entry)
        return entries

    def _log_shown_artifact_entries(
        self,
        artifact_results: list[ArtifactResult],
    ) -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": artifact.artifact_id,
                "inventory_number": artifact.inventory_number,
                "title": artifact.title,
            }
            for artifact in artifact_results
        ]

    def _log_navigation_target_entries(
        self,
        *,
        navigation_targets: list[TourNavigationTarget],
        artifact_results: list[ArtifactResult],
    ) -> list[dict[str, Any]]:
        artifact_id_by_inventory = {
            str(artifact.inventory_number or "").strip().casefold(): artifact.artifact_id
            for artifact in artifact_results
            if str(artifact.inventory_number or "").strip()
        }
        entries: list[dict[str, Any]] = []
        for target in navigation_targets:
            inventory_key = str(target.inventory_id or "").strip().casefold()
            entries.append(
                {
                    "artifact_id": artifact_id_by_inventory.get(inventory_key),
                    "inventory_number": target.inventory_id,
                    "has_navigation_target": True,
                    "panorama_id": target.panorama_key,
                    "overlay_id": target.overlay_id,
                }
            )
        return entries

    async def _emit_query_log(
        self,
        *,
        payload: Any,
        conversation_id: str,
        query_id: str,
        language: str | None,
        raw_query: str | None,
        resolved_query: str | None,
        lexical_query: str | None,
        embedding_query: str | None,
        route: str | None,
        retrieval_mode: str,
        filters_applied: dict[str, Any] | None,
        boosts_applied: dict[str, Any] | None,
        retrieved_artifacts: list[dict[str, Any]],
        shown_artifacts: list[dict[str, Any]],
        navigation_targets: list[dict[str, Any]],
        latency_retrieval_ms: float | None,
        latency_llm_ms: float | None,
        latency_total_ms: float | None,
        status: str,
        error: str | None,
        opensearch_query: dict[str, Any] | None = None,
        system_config: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        metadata = getattr(payload, "metadata", None)
        museum_slug = getattr(payload, "museum_slug", None)
        correlation = self._log_correlation_ids(metadata=metadata, museum_slug=museum_slug)
        event: dict[str, Any] = {
            "event_type": "backend_query",
            "timestamp": utc_timestamp(),
            "session_id": correlation["session_id"],
            "conversation_id": conversation_id,
            "query_id": query_id,
            "participant_id": correlation["participant_id"],
            "task_id": correlation["task_id"],
            "tour_id": correlation["tour_id"],
            "museum_slug": museum_slug,
            "language": language,
            "raw_query": raw_query,
            "resolved_query": resolved_query,
            "lexical_query": lexical_query,
            "embedding_query": embedding_query,
            "route": route,
            "retrieval_mode": retrieval_mode,
            "filters_applied": filters_applied or {},
            "boosts_applied": boosts_applied or {},
            "retrieved_artifacts": retrieved_artifacts,
            "shown_artifacts": shown_artifacts,
            "navigation_targets": navigation_targets,
            "latency_retrieval_ms": latency_retrieval_ms,
            "latency_llm_ms": latency_llm_ms,
            "latency_total_ms": latency_total_ms,
            "status": status,
            "error": error,
            "opensearch_query": opensearch_query,
        }
        if system_config:
            # Reprodutibilidade por linha (exigido pelo estudo de utilizador):
            # modelos+revisões, configuração de fusão/rerank e índices usados.
            event["system_config"] = system_config
        if extra:
            # Quick wins: campos adicionais (fast_path, llm_finish_reason,
            # llm_usage, grounding). Só aparecem quando há algo a registar,
            # pelo que os eventos históricos mantêm exatamente o mesmo esquema.
            for key, value in extra.items():
                if value is not None and key not in event:
                    event[key] = value
        await self.query_logger.log_event(event)

    def _image_system_config(self, *, modality: str,
                             image_bytes: bytes | None = None,
                             outcome: Any = None) -> dict[str, Any]:
        """Configuração efetiva da rota de imagem para o registo estruturado.

        Nunca inclui a imagem nem caminhos locais — apenas um hash não
        identificativo dos bytes carregados."""
        import hashlib

        config: dict[str, Any] = {
            "modality": modality,
            "visual_embedding_model": self.settings.multimodal_embedding_model_resolved,
            "visual_embedding_revision": getattr(
                self.settings, "QWEN_MULTIMODAL_EMBEDDING_MODEL_REVISION", "") or None,
            "index_image": self.settings.OPENSEARCH_INDEX_IMAGE,
            "index_artifact": self.settings.OPENSEARCH_INDEX_ARTIFACT,
            "vl_rerank_stage": self._resolve_vl_rerank_stage(),
            "vl_reranker_model": self.settings.vl_reranker_model_resolved,
            "vl_reranker_revision": self.settings.vl_reranker_model_revision_resolved,
        }
        if image_bytes:
            config["source_image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        if modality == "image_text":
            config.update({
                "joint_branch_enabled": bool(getattr(
                    self.settings, "MULTIMODAL_JOINT_IMAGE_TEXT_ENABLED", False)),
                "joint_top_k": int(getattr(
                    self.settings, "MULTIMODAL_JOINT_IMAGE_TEXT_TOP_K", 0)),
                "fusion": {
                    "rrf_k": int(self.settings.MULTIMODAL_RRF_K),
                    "w_i2i": float(self.settings.MULTIMODAL_I2I_WEIGHT),
                    "w_joint": float(getattr(
                        self.settings, "MULTIMODAL_JOINT_IMAGE_TEXT_WEIGHT", 0.0)),
                    "w_t2i": float(self.settings.MULTIMODAL_IMAGE_WEIGHT),
                    "w_artifact": float(
                        self.settings.MULTIMODAL_IMAGE_TEXT_ARTIFACT_WEIGHT),
                },
                "global_rerank_top_n": int(getattr(
                    self.settings, "MULTIMODAL_GLOBAL_VL_RERANK_TOP_N", 0)),
            })
            diagnostics = getattr(outcome, "diagnostics", None) or {}
            fusion_diag = diagnostics.get("fusion") or {}
            config["branch_sizes"] = fusion_diag.get("branch_sizes")
            config["structured_filters"] = diagnostics.get("constraints")
            grr = diagnostics.get("global_vl_rerank") or {}
            if grr:
                config["global_rerank"] = {
                    "candidates": grr.get("candidates"),
                    "positions_changed": grr.get("positions_changed"),
                    "fallback": grr.get("fallback"),
                }
        return config

    def _matched_retrieval_boosts_for_log(
        self,
        *,
        query_text: str | None,
        lexical_query: str | None,
    ) -> list[dict[str, Any]]:
        if self.settings.CHAT_RETRIEVAL_EMBEDDING_ONLY:
            return []
        boost_resolver = getattr(self.opensearch_gateway, "matched_retrieval_boosts", None)
        if not callable(boost_resolver):
            return []
        try:
            boosts = boost_resolver(
                query_text=str(query_text or ""),
                lexical_query=lexical_query,
            )
        except Exception:
            return []
        return boosts if isinstance(boosts, list) else []

    def _matched_lay_vocabulary_for_log(
        self,
        *,
        museum_slug: str | None,
        museum_id: str | None,
        query_text: str | None,
        lexical_query: str | None,
    ) -> list[dict[str, Any]]:
        """Regras de vocabulário leigo aplicadas (mesma decisão do query body)."""
        if self.settings.CHAT_RETRIEVAL_EMBEDDING_ONLY:
            return []
        resolver = getattr(self.opensearch_gateway, "matched_lay_vocabulary", None)
        if not callable(resolver):
            return []
        try:
            matches = resolver(
                museum_slug=museum_slug,
                museum_id=museum_id,
                query_text=str(query_text or ""),
                lexical_query=(lexical_query or query_text or "").strip(),
            )
        except Exception:
            return []
        return matches if isinstance(matches, list) else []

    def _text_boosts_applied_for_log(
        self,
        *,
        router_decision: dict[str, object],
        retrieval_request: dict[str, Any],
    ) -> dict[str, Any]:
        if router_decision.get("mode") != "rag":
            return {}
        if retrieval_request.get("kind") == "selected_artifact":
            return {}

        boosts: dict[str, Any] = {"in_tour_boost": self.settings.CHAT_IN_TOUR_BOOST}
        retrieval_boosts = retrieval_request.get("retrieval_boosts")
        if isinstance(retrieval_boosts, list) and retrieval_boosts:
            boosts["retrieval_boosts"] = retrieval_boosts
        return boosts

    def _rag_debug_json(self, event: str, payload: dict[str, Any]) -> None:
        if not self.settings.BACKEND_RAG_DEBUG_ENABLED:
            return
        log_json_event(
            logger,
            logging.INFO,
            event,
            payload,
            max_chars=max(int(self.settings.BACKEND_RAG_DEBUG_MAX_CHARS), 1000),
        )

    def _rag_debug_docs(self, docs: list[dict[str, object]], *, limit: int = 8) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for index, doc in enumerate(docs[: max(limit, 0)], start=1):
            entries.append(
                {
                    "rank": index,
                    "artifact_id": doc.get("artifact_id"),
                    "inventory_number": doc.get("inventory_number") or doc.get("inventory"),
                    "title": doc.get("title"),
                    "score": doc.get("score"),
                    "rerank_score": doc.get("rerank_score"),
                    "museum_id": doc.get("museum_id"),
                    "category": doc.get("category"),
                    "in_tour": doc.get("in_tour"),
                    "snippet": doc.get("snippet"),
                }
            )
        return entries

    async def handle_message(
        self,
        payload: ChatMessageRequest,
        *,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        """Decide primeiro o destino da peça selecionada (se o pedido a traz):
        um pedido independente segue sem ela e a resposta avisa o widget."""
        metadata = payload.metadata if isinstance(payload.metadata, dict) else None
        selected = self._selected_artifact_from_metadata(metadata)
        # «Ver na tour»: o frontend diz a intenção e a peça; não há nada a
        # decidir nem a pesquisar. Antes da libertação automática, que veria
        # «Sapatos (par)/Feminino (4270)» como uma pesquisa nova.
        if (
            selected
            and bool(getattr(self.settings, "CHAT_DESCRIBE_ARTIFACT_INTENT", False))
            and self._metadata_intent(metadata) == "describe_artifact"
        ):
            return await self._answer_describe_artifact(payload, selected=selected, status_cb=status_cb)
        status: str | None = None
        subject: dict[str, Any] | None = None
        if not selected and bool(getattr(self.settings, "CHAT_IMPLICIT_ARTIFACT_CONTEXT", False)):
            subject = self._resolve_implicit_artifact(
                payload.message, metadata, museum_slug=payload.museum_slug, museum_id=payload.museum_id
            )
            log_event(
                logger, logging.INFO, "chat.implicit_artifact.decision",
                resolved_by=(subject or {}).get("resolved_by"),
                artifact_id=(subject or {}).get("artifact_id"),
                visible=len((metadata or {}).get("visible_artifacts") or []),
                has_context=bool((metadata or {}).get("context_artifact")),
            )
            if subject:
                kept = dict(metadata or {})
                kept["selected_artifact_id"] = subject["artifact_id"]
                kept["selected_artifact"] = {
                    "artifact_id": subject["artifact_id"],
                    "inventory_number": subject.get("inventory_number"),
                    "title": subject.get("title"),
                    "source": f"implicit_{subject['resolved_by']}",
                }
                update: dict[str, Any] = {"metadata": kept}
                if subject["resolved_by"] in {"ordinal", "inventory", "title"}:
                    # «E o segundo?» chega ao LLM à letra e ele não sabe que «o
                    # segundo» é esta peça: diz-se qual é, na própria pergunta.
                    label = " ".join(
                        part for part in (
                            f"«{subject['title']}»" if subject.get("title") else None,
                            f"({subject['inventory_number']})" if subject.get("inventory_number") else None,
                        ) if part
                    )
                    if label:
                        english = normalize_language(payload.language) == "en"
                        note = f"(I mean the object {label})" if english else f"(refiro-me à peça {label})"
                        update["message"] = f"{(payload.message or '').rstrip()} {note}"
                payload = payload.model_copy(update=update)
        elif selected and bool(getattr(self.settings, "CHAT_SELECTED_ARTIFACT_AUTO_RELEASE", False)):
            decision = self._selected_artifact_decision(
                payload.message, museum_slug=payload.museum_slug, museum_id=payload.museum_id
            )
            status = "released" if decision == "release" else "used"
            log_event(
                logger, logging.INFO, "chat.selected_artifact.decision",
                decision=decision, selected_artifact_id=selected.get("artifact_id"),
            )
            if decision == "release":
                kept = {
                    k: v for k, v in (metadata or {}).items()
                    if k not in {"selected_artifact", "selected_artifact_id", "selected_context_mode"}
                }
                kept["released_selected_artifact_id"] = selected.get("artifact_id")
                payload = payload.model_copy(update={"metadata": kept})
        response = await self._handle_message_with_selection(payload, status_cb=status_cb)
        if status is not None:
            response.selected_artifact_status = status  # type: ignore[assignment]
        if subject:
            response.subject_artifact = ChatSubjectArtifact(
                artifact_id=subject["artifact_id"],
                inventory_number=subject.get("inventory_number"),
                title=subject.get("title"),
                resolved_by=subject["resolved_by"],
                context_source=subject.get("context_source"),
            )
        return response

    _OPENING_GREETING = re.compile(
        r"^\s*(?:ol[áa]|boas|bom dia|boa tarde|boa noite|hello|hi)"
        r"(?:[,!.:]\s*(?:(?:veja|repare|observe|olhe|look at|look|see)\s+)?|\s+(?:veja|repare|observe|olhe|look at|look|see)\s+)",
        re.IGNORECASE,
    )

    @classmethod
    def _strip_opening_greeting(cls, text: str) -> str:
        """«Olá, veja a Fuga…» → «A Fuga…»: a apresentação de uma peça começa
        pela peça, mesmo quando o modelo cumprimenta por hábito."""
        stripped = cls._OPENING_GREETING.sub("", text or "", count=1)
        if stripped == (text or "") or not stripped:
            return text or ""
        return stripped[0].upper() + stripped[1:]

    @staticmethod
    def _metadata_intent(metadata: dict[str, Any] | None) -> str | None:
        """Intenção explícita enviada pelo frontend (metadata.intent), se houver."""
        if not isinstance(metadata, dict):
            return None
        intent = str(metadata.get("intent") or "").strip().lower()
        return intent or None

    async def _answer_describe_artifact(
        self,
        payload: ChatMessageRequest,
        *,
        selected: dict[str, str | None],
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        """Intenção «describe_artifact» (CHAT_DESCRIBE_ARTIFACT_INTENT).

        O visitante carregou em «Ver na tour» e está a olhar para a peça: a
        ficha vai buscar-se pelo artifact_id (pelo n.º de inventário se o id
        já não existir) e o LLM apresenta-a com um prompt próprio. Só essa
        peça sai nos cartões; nunca há pesquisa.
        """
        conversation_id = payload.conversation_id or str(uuid4())
        requested_format = payload.response_format or ResponseFormatObject(type="text")
        query_id = str(uuid4())
        request_started_at = time.monotonic()
        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        self.session_store.append_turn(state, role="user", text=payload.message)
        await self._emit_status(status_cb, "status.analyzing_request", language=language)

        artifact_id = str(selected.get("artifact_id") or "").strip()
        inventory_number = str(selected.get("inventory_number") or "").strip()
        museum_slug = str(selected.get("museum_slug") or payload.museum_slug).strip() or payload.museum_slug
        museum_id = str(selected.get("museum_id") or self._resolve_museum_id(payload) or "").strip() or None
        museum_name = str(selected.get("museum_name") or payload.museum_name or "").strip() or payload.museum_name
        log_event(
            logger,
            logging.INFO,
            "chat.message.route",
            conversation_id=conversation_id,
            query_id=query_id,
            mode="rag",
            intent="describe_artifact",
            fast_path="describe_artifact",
            selected_artifact_id=artifact_id,
        )

        retrieval_started_at = time.monotonic()
        docs: list[dict[str, object]] = []
        try:
            docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
                museum_slug=museum_slug,
                museum_id=museum_id,
                artifact_ids=[artifact_id],
                top_k=1,
            )
        except Exception as exc:
            log_event(logger, logging.WARNING, "chat.describe_artifact.fetch.error", error=exc)
        if not docs and inventory_number:
            try:
                docs = await self.opensearch_gateway.fetch_artifacts_by_inventory_numbers(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    inventory_numbers=[inventory_number],
                    top_k=1,
                )
            except Exception as exc:
                log_event(logger, logging.WARNING, "chat.describe_artifact.fetch.error", error=exc)
        docs = docs[:1]
        retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000

        reply = ""
        status = "ok"
        error: str | None = None
        llm_response = None
        llm_latency_ms: float | None = None
        grounding_report = None
        if not docs:
            reply = translate("message.describe_artifact_not_found", language)
        else:
            prompt = build_describe_artifact_prompt(
                museum_name=museum_name,
                user_message=payload.message,
                card_context=self._format_docs_compact(docs=docs, description_chars=900),
                language=language,
            )
            self._rag_debug_json(
                "rag.final_prompt",
                {
                    "conversation_id": conversation_id,
                    "query_id": query_id,
                    "intent": "describe_artifact",
                    "final_prompt_chars": len(prompt),
                },
            )
            await self._emit_status(status_cb, "status.generating_final_answer", language=language)
            llm_started_at = time.monotonic()
            try:
                llm_response = await self._generate_final_answer_with_context_retries(
                    initial_message=prompt,
                    response_format=requested_format,
                    system_prompt=self._final_system_prompt(payload.system_prompt, language),
                    model_override=payload.model_override,
                    visible_result_count=0,
                    build_message_for_result_count=lambda _n: prompt,
                    stream_cb_factory=self._final_answer_stream_factory(
                        status_cb, requested_format, query_id=query_id
                    ),
                )
                reply = self._strip_opening_greeting(
                    self._sanitize_assistant_reply(llm_response.text, docs=docs, language=language)
                )
            except LLMServiceError as exc:
                log_event(
                    logger, logging.ERROR, "chat.message.llm.error",
                    conversation_id=conversation_id, query_id=query_id, error=exc,
                )
                reply = translate("error.llm_unavailable", language, error=str(exc))
                status, error = "error", str(exc)
            llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
            if llm_response is not None:
                grounding_report = self._grounding_report(
                    reply=reply, docs=docs, conversation_id=conversation_id, query_id=query_id,
                    user_message=payload.message,
                )

        artifact_results = await self._build_artifact_results(
            museum_slug=museum_slug, museum_id=museum_id, artifact_docs=docs,
        ) if docs else []
        image_matches: list[ImageMatchResult] = []
        navigation_targets: list[TourNavigationTarget] = []
        if docs:
            try:
                image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    artifact_ids=[str(docs[0].get("artifact_id") or artifact_id)],
                    per_artifact=1,
                    max_total=1,
                )
            except Exception as exc:
                log_event(logger, logging.WARNING, "chat.message.thumbnail_fetch.error", error=exc)
                image_hits = []
            image_matches = self._build_image_matches(image_hits=image_hits, artifact_docs=docs)
            navigation_targets = self._resolve_navigation_targets(
                museum_slug=museum_slug, museum_id=museum_id, docs=docs,
            )
            image_matches = self._enrich_image_matches(
                context="text",
                conversation_id=conversation_id,
                museum_slug=museum_slug,
                image_matches=image_matches,
                artifact_results=artifact_results,
                navigation_targets=navigation_targets,
            )

        router_decision: dict[str, object] = {
            "mode": "rag",
            "intent": "describe_artifact",
            "rewritten_query": payload.message,
            "needs_retrieval": False,
            "reason": "frontend_intent_describe_artifact",
            "is_follow_up": False,
            "use_history_for_query": False,
            "use_history_for_answer": False,
            "carry_filters": False,
            "carry_sort": False,
            "filters_delta": {},
            "sort_delta": {},
        }
        state.filters = {}
        state.sort = {}
        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(state=state, artifact_docs=docs, preserve_existing_when_empty=False)
        self.session_store.append_turn(state, role="assistant", text=reply)
        self._update_rolling_summary(
            state=state,
            latest_user_message=payload.message,
            latest_assistant_message=reply,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        total_ms = (time.monotonic() - request_started_at) * 1000
        search_scope = self._build_search_scope(
            current_museum_slug=payload.museum_slug,
            current_museum_id=self._resolve_museum_id(payload),
            current_museum_name=payload.museum_name,
            search_museum_slug=museum_slug,
            search_museum_id=museum_id,
            search_museum_name=museum_name,
        )
        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=(payload.message or "").strip(),
            resolved_query=(payload.message or "").strip(),
            lexical_query=None,
            embedding_query=None,
            route="rag",
            retrieval_mode="describe_artifact",
            filters_applied={
                "filters": {"artifact_id": artifact_id} if artifact_id else {},
                "sort": {},
                "selected_artifact": dict(selected),
                "selected_context_mode": "describe_artifact",
                "target_museum": search_scope.model_dump(mode="json"),
            },
            boosts_applied={},
            retrieved_artifacts=self._log_retrieved_artifact_entries(docs, source="describe_artifact"),
            shown_artifacts=self._log_shown_artifact_entries(artifact_results),
            navigation_targets=self._log_navigation_target_entries(
                navigation_targets=navigation_targets,
                artifact_results=artifact_results,
            ),
            latency_retrieval_ms=retrieval_latency_ms,
            latency_llm_ms=llm_latency_ms,
            latency_total_ms=total_ms,
            status=status,
            error=error,
            opensearch_query=None,
            extra={
                "fast_path": "describe_artifact",
                "artifact_found": bool(docs),
                "llm_finish_reason": getattr(llm_response, "finish_reason", None),
                "llm_usage": getattr(llm_response, "usage", None),
                "llm_endpoint": getattr(llm_response, "endpoint", None),
                "grounding": grounding_report,
            },
        )
        log_event(
            logger,
            logging.INFO,
            "chat.message.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status=status,
            fast_path="describe_artifact",
            shown_artifacts=len(artifact_results),
            duration_ms=round(total_ms, 1),
        )
        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=requested_format,
            reply=reply,
            model_hint=(
                getattr(llm_response, "model", None) or "deterministic:describe_artifact"
            ),
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
            selected_artifact_status="used",
            results_page=1,
            results_page_size=max(len(artifact_results), 1),
            results_total=len(artifact_results),
            results_has_more=False,
            results_request_id=None,
            search_scope=search_scope,
        )

    @staticmethod
    def _artifact_refs(raw: Any, limit: int = 60) -> list[dict[str, str | None]]:
        refs: list[dict[str, str | None]] = []
        for item in (raw if isinstance(raw, list) else [raw])[:limit]:
            if not isinstance(item, dict):
                continue
            artifact_id = str(item.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            refs.append({
                "artifact_id": artifact_id,
                "inventory_number": str(item.get("inventory_number") or "").strip() or None,
                "title": str(item.get("title") or "").strip() or None,
                "source": str(item.get("source") or "").strip() or None,
            })
        return refs

    def _resolve_implicit_artifact(
        self,
        message: str,
        metadata: dict[str, Any] | None,
        *,
        museum_slug: str,
        museum_id: str | None,
    ) -> dict[str, Any] | None:
        """A peça de que a mensagem fala, sem o visitante a ter escolhido.

        1. referência explícita a um cartão visível: ordinal («o segundo»), n.º
           de inventário («e o 5622?») ou nome com determinante («o retrato de
           família é de quando?»);
        2. sem referência, uma pergunta sem assunto próprio («de que é feito?»,
           «quem o usava?», «peças parecidas») vai para o contexto que o widget
           manda: o hotspot aberto na visita ou o primeiro resultado;
        3. tudo o resto (nova pesquisa, «onde estou?», zonas, saudação) segue
           sem peça. Tudo por regras.
        """
        metadata = metadata if isinstance(metadata, dict) else {}
        visible = self._artifact_refs(metadata.get("visible_artifacts"))
        contexts = self._artifact_refs(metadata.get("context_artifact"), limit=1)
        context = contexts[0] if contexts else None
        if not visible and not context:
            return None
        folded = self._fold_query_text(message)
        tokens = folded.split()
        if not tokens:
            return None
        try:
            if (
                self._is_tour_location_query(message)
                or self._match_tour_room_query(message, museum_slug=museum_slug, museum_id=museum_id) is not None
                or (self._is_obvious_greeting_or_smalltalk(message)
                    and not any(t in _SELECTED_INTERROGATIVES for t in tokens))
            ):
                return None
        except Exception:  # pragma: no cover - deteção auxiliar; nunca bloqueia
            return None

        def pick(ref: dict[str, str | None], how: str) -> dict[str, Any]:
            return {**ref, "resolved_by": how, "context_source": ref.get("source") if how == "context" else None}

        if visible:
            # 1a. ordinal
            for index, token in enumerate(tokens):
                position = _ORDINAL_WORDS.get(token)
                if position is None and token.isdigit() and 0 < int(token) <= 20 and index + 1 < len(tokens) and tokens[index + 1] in {"o", "a"} and index > 0 and tokens[index - 1] in _REFERENCE_DETERMINERS:
                    position = int(token) - 1  # «o 2.º» -> «o 2 o»
                if position is None:
                    continue
                if -len(visible) <= position < len(visible):
                    return pick(visible[position], "ordinal")
            # 1b. n.º de inventário de um cartão visível
            for ref in visible:
                inventory = self._fold_query_text(ref.get("inventory_number") or "")
                if not inventory:
                    continue
                inv_tokens = inventory.split()
                for start in range(len(tokens) - len(inv_tokens) + 1):
                    if tokens[start:start + len(inv_tokens)] != inv_tokens:
                        continue
                    before = tokens[start - 1] if start > 0 else ""
                    alone = len(tokens) <= len(inv_tokens) + 1
                    if alone or before in _REFERENCE_DETERMINERS or before in {"inventario", "numero", "n", "peca", "objeto", "e"}:
                        return pick(ref, "inventory")
            # 1c. nome do cartão a seguir a um determinante («o retrato de família»)
            if not any(t in _REFERENCE_SEARCH_VERBS for t in tokens):
                # o nome principal tem de vir a seguir a um determinante; entre os
                # cartões que batem, ganha o que partilha mais palavras do título
                # («o vestido de criança» -> Vestido/Criança, não o primeiro vestido)
                message_tokens = set(tokens)
                best: tuple[int, int] | None = None
                for position, ref in enumerate(visible):
                    title_tokens = self._fold_query_text(ref.get("title") or "").split()
                    if not title_tokens:
                        continue
                    head = title_tokens[0]
                    if not any(tokens[i] == head and tokens[i - 1] in _REFERENCE_DETERMINERS for i in range(1, len(tokens))):
                        continue
                    score = len({t for t in title_tokens if len(t) > 2} & message_tokens)
                    if best is None or score > best[0]:
                        best = (score, position)
                if best is not None:
                    return pick(visible[best[1]], "title")

        # 2. pergunta sem assunto próprio -> contexto (hotspot aberto / primeiro resultado).
        # Mais exigente do que com uma peça escolhida à mão: qualquer palavra de
        # conteúdo que não seja do título dessa peça faz dela uma pergunta com
        # assunto próprio («quem foi o marquês de Angeja?»). Exceção: sim/não
        # sobre a peça («é de seda?», «foi usado por homens?»).
        if not context:
            return None
        decision = self._selected_artifact_decision(message, museum_slug=museum_slug, museum_id=museum_id)
        if decision == "similar":
            return pick(context, "context")
        if decision != "about":
            return None
        # Hotspot aberto na visita: o visitante está a olhar para a peça, e ela
        # tem prioridade. Vale a regra da peça escolhida à mão: só um pedido
        # independente (nova pesquisa, «onde estou?», zonas, saudação) a larga.
        if (context.get("source") or "") == "tour_hotspot":
            return pick(context, "context")
        first_raw = (message or "").strip().split(" ", 1)[0].strip("¿¡\"'«").casefold()
        if first_raw in {"é", "era", "foi", "será", "está", "estava", "is", "was", "does", "did"}:
            return pick(context, "context")
        context_title = set(self._fold_query_text(context.get("title") or "").split())
        is_attribute = lambda t: any(t.startswith(stem) for stem in _SELECTED_ATTRIBUTE_STEMS)  # noqa: E731
        content = [
            t for t in tokens
            if t not in _SELECTED_FILLER_WORDS and t not in _SELECTED_SEARCH_VERBS
            and t not in _SELECTED_INTERROGATIVES and t not in _SELECTED_REFERENCE_WORDS
            and t not in _REFERENCE_DETERMINERS and t not in context_title and not is_attribute(t)
            and t not in _SELECTED_GENERIC_NOUNS
        ]
        return None if content else pick(context, "context")

    def _selected_artifact_decision(self, message: str, *, museum_slug: str, museum_id: str | None) -> str:
        """"about" (pergunta sobre a peça), "similar" (parte dela: «outras
        parecidas») ou "release" (pedido independente: nova pesquisa, «onde
        estou?», zonas da visita, saudação)."""
        folded = self._fold_query_text(message)
        tokens = folded.split()
        if not tokens:
            return "about"
        if self._selected_artifact_context_mode(message) == "anchored_similarity":
            return "similar"
        if any(t in _SELECTED_REFERENCE_WORDS for t in tokens):
            return "about"
        try:
            if (
                self._is_tour_location_query(message)
                # «O que é?» com uma peça em foco é sobre ela, não small talk:
                # só conta como conversa se não houver interrogativo
                or (self._is_obvious_greeting_or_smalltalk(message)
                    and not any(t in _SELECTED_INTERROGATIVES for t in tokens))
                or any(re.fullmatch(p, _SMALLTALK_PREFIX.sub("", folded)) for p in _SMALLTALK_PATTERNS)
                or self._match_tour_room_query(message, museum_slug=museum_slug, museum_id=museum_id) is not None
            ):
                return "release"
        except Exception:  # pragma: no cover - deteção auxiliar; nunca bloqueia
            pass
        is_attribute = lambda t: any(t.startswith(stem) for stem in _SELECTED_ATTRIBUTE_STEMS)  # noqa: E731
        content = [
            t for t in tokens
            if t not in _SELECTED_FILLER_WORDS and t not in _SELECTED_SEARCH_VERBS
            and t not in _SELECTED_INTERROGATIVES and not is_attribute(t)
            and t not in _SELECTED_GENERIC_NOUNS
        ]
        has_attribute = any(is_attribute(t) for t in tokens)
        has_search_verb = any(t in _SELECTED_SEARCH_VERBS for t in tokens)
        if has_search_verb:
            # «mostra-me mais detalhes» é sobre a peça; «encontra chapéus» não
            return "release" if content else "about"
        # «É de seda?», «era usado por homens?»: pergunta de sim/não sobre a peça
        # (o «é» perde o acento ao normalizar e confunde-se com «e chapéus?»)
        first_raw = (message or "").strip().split(" ", 1)[0].strip("¿¡\"'«").casefold()
        if first_raw in {"é", "era", "foi", "será", "está", "estava", "is", "was", "does", "did", "has"}:
            return "about"
        if tokens[0] in _SELECTED_INTERROGATIVES or (len(tokens) > 1 and tokens[1] in _SELECTED_INTERROGATIVES):
            # «de que é feito?», «quem o usava?»; «que vestidos de noiva?» traz assunto próprio
            return "release" if (content and not has_attribute) else "about"
        # frase solta: «conta-me a história» é da peça; «peças do século XIX» é pesquisa
        return "release" if content else "about"

    async def _handle_message_with_selection(
        self,
        payload: ChatMessageRequest,
        *,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        conversation_id = payload.conversation_id or str(uuid4())
        requested_format = payload.response_format or ResponseFormatObject(type="text")
        query_id = str(uuid4())
        request_started_at = time.monotonic()
        retrieval_latency_ms: float | None = None
        llm_latency_ms: float | None = None

        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        self.session_store.append_turn(state, role="user", text=payload.message)
        log_event(
            logger,
            logging.INFO,
            "chat.message.start",
            conversation_id=conversation_id,
            query_id=query_id,
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            language=language,
            response_format=requested_format.type,
            message_chars=len((payload.message or "").strip()),
            has_metadata=bool(payload.metadata),
        )
        await self._emit_status(status_cb, "status.analyzing_request", language=language)

        context_policy = self._derive_context_policy(
            message=payload.message,
            state=state,
        )

        # Quick win LLM-01: saudação/small talk pura não precisa de router nem
        # de LLM (custava 2 chamadas remotas e inventava peças). A deteção é a
        # mesma do fallback do router: igualdade exata após fold, pelo que
        # «Olá, encontra vestidos» continua a ir ao router.
        if self.settings.CHAT_TOUR_LOCATION_ANSWERS and requested_format.type == "text":
            tour_kwargs = dict(
                payload=payload,
                state=state,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                requested_format=requested_format,
                request_started_at=request_started_at,
                status_cb=status_cb,
            )
            if self._is_tour_location_query(payload.message):
                return await self._answer_tour_location_fast_path(**tour_kwargs)
            room_query = self._match_tour_room_query(
                payload.message,
                museum_slug=payload.museum_slug,
                museum_id=self._resolve_museum_id(payload),
            )
            if room_query is not None:
                return await self._answer_tour_room_fast_path(room_query=room_query, **tour_kwargs)
            if self._is_tour_pieces_list_query(payload.message):
                return await self._answer_tour_pieces_fast_path(**tour_kwargs)
            if bool(getattr(self.settings, "CHAT_CATALOGUE_ANALYTICS", False)):
                analytics = parse_catalogue_analytics(payload.message)
                if analytics is not None:
                    return await self._answer_catalogue_analytics_fast_path(analytics=analytics, **tour_kwargs)

        if (
            self.settings.CHAT_DETERMINISTIC_SMALLTALK
            and requested_format.type == "text"
            and self._is_obvious_greeting_or_smalltalk(payload.message)
        ):
            return await self._answer_smalltalk_fast_path(
                payload=payload,
                state=state,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                requested_format=requested_format,
                request_started_at=request_started_at,
                status_cb=status_cb,
            )

        try:
            router_decision = await self._route_message(
                payload=payload,
                state=state,
                context_policy=context_policy,
            )
        except LLMServiceError:
            router_decision = self._fallback_router_decision(
                payload.message,
                context_policy=context_policy,
            )
        router_decision = self._apply_context_policy_guardrails(
            router_decision=router_decision,
            context_policy=context_policy,
            user_message=payload.message,
        )
        router_decision = self._maybe_override_llm_only_for_visual(
            router_decision=router_decision,
            query_text=payload.message,
        )
        router_decision = self._maybe_override_for_museum_overview(
            router_decision=router_decision,
            query_text=payload.message,
            museum_id=self._resolve_museum_id(payload),
        )
        rewritten_query = str(router_decision.get("rewritten_query", payload.message)).strip()
        original_message = (payload.message or "").strip()
        current_museum_id = self._resolve_museum_id(payload)
        requested_museum = self._resolve_requested_search_museum(original_message)
        search_museum_slug = requested_museum.museum_slug if requested_museum else payload.museum_slug
        search_museum_id = requested_museum.museum_id if requested_museum else current_museum_id
        search_museum_name = (
            requested_museum.museum_name
            if requested_museum
            else payload.museum_name
        )
        search_query_message = self._strip_search_museum_expression_from_query(
            original_message,
            requested_museum,
        )
        search_scope = self._build_search_scope(
            current_museum_slug=payload.museum_slug,
            current_museum_id=current_museum_id,
            current_museum_name=payload.museum_name,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            search_museum_name=search_museum_name,
        )
        selected_artifact = self._selected_artifact_from_metadata(
            payload.metadata if isinstance(payload.metadata, dict) else None
        )
        selected_context_mode = (
            self._selected_artifact_context_mode(payload.message)
            if selected_artifact
            else None
        )
        reference_museum_slug = payload.museum_slug
        reference_museum_id = current_museum_id
        reference_museum_name = payload.museum_name
        if selected_artifact:
            reference_museum_slug = str(selected_artifact.get("museum_slug") or payload.museum_slug).strip() or payload.museum_slug
            reference_museum_id = str(selected_artifact.get("museum_id") or current_museum_id or "").strip() or None
            reference_museum_name = str(selected_artifact.get("museum_name") or payload.museum_name or "").strip() or payload.museum_name
        if (
            selected_artifact
            and requested_museum is not None
            and selected_context_mode == "selected_artifact"
            and self._is_targeted_artifact_search_request(payload.message)
        ):
            selected_context_mode = "anchored_similarity"
        if selected_artifact and selected_context_mode == "selected_artifact":
            search_museum_slug = reference_museum_slug
            search_museum_id = reference_museum_id
            search_museum_name = reference_museum_name
        elif (
            selected_artifact
            and selected_context_mode == "anchored_similarity"
            and requested_museum is None
        ):
            search_museum_slug = reference_museum_slug
            search_museum_id = reference_museum_id
            search_museum_name = reference_museum_name
        if selected_artifact:
            search_scope = self._build_search_scope(
                current_museum_slug=payload.museum_slug,
                current_museum_id=current_museum_id,
                current_museum_name=payload.museum_name,
                search_museum_slug=search_museum_slug,
                search_museum_id=search_museum_id,
                search_museum_name=search_museum_name,
            )
        if selected_artifact:
            router_decision = {
                **router_decision,
                "mode": "rag",
                "intent": (
                    "anchored_similarity_search"
                    if selected_context_mode == "anchored_similarity"
                    else "selected_artifact_question"
                ),
                "use_history_for_query": False,
                "use_history_for_answer": False,
                "carry_filters": False,
                "carry_sort": False,
            }

        log_event(
            logger,
            logging.INFO,
            "chat.message.route",
            conversation_id=conversation_id,
            query_id=query_id,
            mode=router_decision.get("mode"),
            intent=router_decision.get("intent"),
            rewritten_query_chars=len(rewritten_query),
            selected_artifact=bool(selected_artifact),
            selected_context_mode=selected_context_mode,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
        )
        self._rag_debug_json(
            "rag.router_decision",
            {
                "conversation_id": conversation_id,
                "query_id": query_id,
                "input": {
                    "museum_slug": payload.museum_slug,
                    "museum_id": payload.museum_id,
                    "language": language,
                    "message": original_message,
                    "message_chars": len(original_message),
                    "has_metadata": bool(payload.metadata),
                },
                "router_decision": router_decision,
                "context_policy": context_policy,
                "selected_artifact": selected_artifact,
                "selected_context_mode": selected_context_mode,
                "search_scope": search_scope.model_dump(mode="json"),
            },
        )

        carry_filters = bool(router_decision.get("carry_filters", False))
        carry_sort = bool(router_decision.get("carry_sort", False))
        use_history_for_answer = bool(router_decision.get("use_history_for_answer", False))

        base_filters = state.filters if carry_filters else {}
        base_sort = state.sort if carry_sort else {}
        effective_filters = self._merge_state_with_delta(
            base=base_filters,
            delta=router_decision.get("filters_delta"),
        )
        effective_sort = self._merge_state_with_delta(
            base=base_sort,
            delta=router_decision.get("sort_delta"),
        )

        retrieval_context = ""
        retrieved_docs_count = 0
        retrieved_docs: list[dict[str, object]] = []
        retrieval_request: dict[str, Any] = {}
        image_matches: list[ImageMatchResult] = []
        artifact_results: list[ArtifactResult] = []
        artifact_image_hits: list[dict[str, Any]] = []
        text_results_default_page_size = self._text_results_default_page_size()
        text_results_page = max(payload.results_page, 1)
        text_results_page_size = self._resolve_results_page_size(
            payload.results_page_size,
            default_size=text_results_default_page_size,
        )
        text_results_window_size = min(text_results_page * text_results_page_size, 50)
        if (
            selected_artifact
            and selected_context_mode == "selected_artifact"
            and self.settings.CHAT_ENABLE_RAG
        ):
            await self._emit_status(status_cb, "status.searching_collection", language=language)
            retrieval_started_at = time.monotonic()
            (
                retrieval_context,
                retrieved_docs_count,
                retrieved_docs,
                retrieval_request,
            ) = await self._retrieve_selected_artifact_context(
                museum_slug=reference_museum_slug,
                museum_id=reference_museum_id,
                selected_artifact=selected_artifact,
                query=payload.message,
            )
            retrieval_request["search_scope"] = search_scope.model_dump(mode="json")
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=retrieved_docs_count,
            )
            if retrieved_docs:
                artifact_ids = [
                    str(doc.get("artifact_id") or "").strip()
                    for doc in retrieved_docs
                    if str(doc.get("artifact_id") or "").strip()
                ]
                if artifact_ids:
                    try:
                        artifact_image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                            museum_slug=reference_museum_slug,
                            museum_id=reference_museum_id,
                            artifact_ids=artifact_ids,
                            per_artifact=1,
                            max_total=max(len(artifact_ids), 1),
                        )
                    except Exception as exc:
                        log_event(
                            logger,
                            logging.WARNING,
                            "chat.message.thumbnail_fetch.error",
                            conversation_id=conversation_id,
                            query_id=query_id,
                            artifact_count=len(artifact_ids),
                            error=exc,
                        )
                        artifact_image_hits = []
                    image_matches = self._build_image_matches(
                        image_hits=artifact_image_hits,
                        artifact_docs=retrieved_docs,
                    )
        elif (
            selected_artifact
            and selected_context_mode == "anchored_similarity"
            and self.settings.CHAT_ENABLE_RAG
        ):
            await self._emit_status(status_cb, "status.searching_collection", language=language)
            retrieval_started_at = time.monotonic()
            (
                retrieval_context,
                retrieved_docs_count,
                retrieved_docs,
                retrieval_request,
            ) = await self._retrieve_anchored_similarity_context(
                reference_museum_slug=reference_museum_slug,
                reference_museum_id=reference_museum_id,
                search_museum_slug=search_museum_slug,
                search_museum_id=search_museum_id,
                selected_artifact=selected_artifact,
                query=search_query_message,
                filters=effective_filters,
                sort=effective_sort,
                result_window_size=text_results_window_size,
            )
            retrieval_request["search_scope"] = search_scope.model_dump(mode="json")
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=retrieved_docs_count,
            )
            if retrieved_docs:
                artifact_ids = [
                    str(doc.get("artifact_id") or "").strip()
                    for doc in retrieved_docs
                    if str(doc.get("artifact_id") or "").strip()
                ]
                if artifact_ids:
                    try:
                        artifact_image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                            museum_slug=search_museum_slug,
                            museum_id=search_museum_id,
                            artifact_ids=artifact_ids,
                            per_artifact=1,
                            max_total=max(len(artifact_ids), 1),
                        )
                    except Exception as exc:
                        log_event(
                            logger,
                            logging.WARNING,
                            "chat.message.thumbnail_fetch.error",
                            conversation_id=conversation_id,
                            query_id=query_id,
                            artifact_count=len(artifact_ids),
                            error=exc,
                        )
                        artifact_image_hits = []
                    image_matches = self._build_image_matches(
                        image_hits=artifact_image_hits,
                        artifact_docs=retrieved_docs,
                    )
        elif router_decision["mode"] == "rag" and self.settings.CHAT_ENABLE_RAG:
            # A query de retrieval e a rewritten do router (resolve follow-ups
            # com o contexto: "e de linho?" -> "vestidos de linho") quando passa
            # os guardrails; senao a mensagem do utilizador. Alimenta embedding,
            # extrator lexical, reranker e ramo t2i de forma consistente.
            retrieval_query, retrieval_query_source = self._resolve_contextual_retrieval_query(
                original_query=search_query_message,
                rewritten_query=rewritten_query,
                state=state,
                requested_museum=requested_museum,
                museum_slug=search_museum_slug,
                museum_id=search_museum_id,
                museum_name=search_museum_name,
            )
            # O router e erratico a resolver elipses ("e de prata?"): quando a
            # politica de contexto marca follow-up e a rewritten do router nao
            # sobreviveu aos guardrails, um resolver LLM dedicado tenta a
            # resolucao — validado pelos mesmos guardrails.
            if (
                retrieval_query_source == "user_message"
                and bool(context_policy.get("use_history_for_query", False))
            ):
                follow_up_rewrite = await self._resolve_follow_up_query_with_llm(
                    current_message=search_query_message,
                    state=state,
                )
                if follow_up_rewrite:
                    resolved_query, resolved_source = self._resolve_contextual_retrieval_query(
                        original_query=search_query_message,
                        rewritten_query=follow_up_rewrite,
                        state=state,
                        requested_museum=requested_museum,
                        museum_slug=search_museum_slug,
                        museum_id=search_museum_id,
                        museum_name=search_museum_name,
                    )
                    if resolved_source == "router_rewrite":
                        retrieval_query = resolved_query
                        retrieval_query_source = "follow_up_resolver"
            # A query aceite passa a ser A rewritten do sistema: alinha o
            # prompt de resposta e os logs com o que foi realmente pesquisado
            # (uma rewritten alucinada rejeitada nunca deve chegar ao LLM final).
            rewritten_query = retrieval_query
            router_decision["rewritten_query"] = retrieval_query
            await self._emit_status(status_cb, "status.searching_collection", language=language)
            retrieval_started_at = time.monotonic()
            (
                retrieval_context,
                retrieved_docs_count,
                retrieved_docs,
                retrieval_request,
            ) = await self._retrieve_context(
                museum_slug=search_museum_slug,
                museum_id=search_museum_id,
                query=retrieval_query,
                filters=effective_filters,
                sort=effective_sort,
                result_window_size=text_results_window_size,
                # Lido na mensagem do utilizador, nao na rewritten do router
                # (que descarta "nesta visita" e afins).
                tour_scope_expression=self._extract_tour_scope_expression(
                    search_query_message
                ),
                inventory_query=original_message,
            )
            retrieval_request["museum_slug"] = search_museum_slug
            retrieval_request["search_scope"] = search_scope.model_dump(mode="json")
            retrieval_request["resolved_query"] = retrieval_query
            retrieval_request["resolved_query_source"] = retrieval_query_source
            multimodal_outcome = None
            # E7B: o ramo t2i é independente do resultado de artifact_search —
            # corre também com zero docs textuais (t2i-only), desde que o
            # retrieval textual tenha sido tentado (kind=="text").
            if str(retrieval_request.get("kind") or "") == "text":
                multimodal_outcome = await self._maybe_fuse_text_to_image(
                    query_text=retrieval_query,
                    museum_slug=search_museum_slug,
                    museum_id=search_museum_id,
                    artifact_docs=retrieved_docs,
                    conversation_id=conversation_id,
                    query_id=query_id,
                )
            if multimodal_outcome is not None:
                # A fusão pode voltar a trazer membros do mesmo conjunto (pelas
                # legendas ou pelas imagens): junta-se outra vez.
                retrieved_docs = self._collapse_identical_set_members(list(multimodal_outcome.docs))
                retrieved_docs_count = multimodal_outcome.total
                context_top_k = max(self.settings.CHAT_RETRIEVAL_TOP_K, 1)
                retrieval_context = self._format_docs_for_prompt(
                    docs=retrieved_docs[:context_top_k],
                    top_k=context_top_k,
                )
                retrieval_request["kind"] = "multimodal_text"
                retrieval_request["results_total"] = multimodal_outcome.total
                retrieval_request["multimodal"] = (
                    multimodal_outcome.debug_payload()
                    if self.settings.MULTIMODAL_DEBUG
                    else {
                        "router_rule": multimodal_outcome.router.rule,
                        "t2i_kept": multimodal_outcome.diagnostics.get(
                            "text_to_image", {}
                        ).get("kept_after_floor"),
                        "fused_size": multimodal_outcome.total,
                    }
                )
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=retrieved_docs_count,
            )
            if retrieved_docs:
                matched_image_hits = (
                    list(multimodal_outcome.matched_image_hits)
                    if multimodal_outcome is not None
                    else []
                )
                matched_artifact_ids = {
                    str(hit.get("artifact_id") or "").strip()
                    for hit in matched_image_hits
                    if str(hit.get("artifact_id") or "").strip()
                }
                artifact_ids = [
                    str(doc.get("artifact_id") or "").strip()
                    for doc in retrieved_docs
                    if str(doc.get("artifact_id") or "").strip()
                    and str(doc.get("artifact_id") or "").strip() not in matched_artifact_ids
                ]
                if artifact_ids:
                    try:
                        artifact_image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                            museum_slug=search_museum_slug,
                            museum_id=search_museum_id,
                            artifact_ids=artifact_ids,
                            per_artifact=1,
                            max_total=max(len(artifact_ids), 1),
                        )
                    except Exception as exc:
                        log_event(
                            logger,
                            logging.WARNING,
                            "chat.message.thumbnail_fetch.error",
                            conversation_id=conversation_id,
                            query_id=query_id,
                            artifact_count=len(artifact_ids),
                            error=exc,
                        )
                        artifact_image_hits = []
                # A imagem que fez match no ramo visual manda na thumbnail; os
                # restantes artefactos usam a imagem principal (fallback atual).
                artifact_image_hits = matched_image_hits + artifact_image_hits
                if artifact_image_hits:
                    image_matches = self._build_image_matches(
                        image_hits=artifact_image_hits,
                        artifact_docs=retrieved_docs,
                    )
        else:
            pass
        log_event(
            logger,
            logging.INFO,
            "chat.message.retrieval.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            mode=router_decision.get("mode"),
            retrieval_kind=retrieval_request.get("kind"),
            retrieved_docs_count=retrieved_docs_count,
            image_matches=len(image_matches),
            duration_ms=round(retrieval_latency_ms, 1) if retrieval_latency_ms is not None else None,
        )
        self._rag_debug_json(
            "rag.retrieval_result",
            {
                "conversation_id": conversation_id,
                "query_id": query_id,
                "route": router_decision.get("mode"),
                "intent": router_decision.get("intent"),
                "search_scope": search_scope.model_dump(mode="json"),
                "effective_filters": effective_filters,
                "effective_sort": effective_sort,
                "latency_ms": round(retrieval_latency_ms, 1) if retrieval_latency_ms is not None else None,
                "context_chars": len(retrieval_context),
                "retrieved_docs_count": retrieved_docs_count,
                "retrieval_request": retrieval_request,
                "retrieved_docs": self._rag_debug_docs(retrieved_docs),
                "thumbnail_image_hits": self._rag_debug_docs(artifact_image_hits),
                "image_matches_count": len(image_matches),
            },
        )
        # Build the full artifact records (all images) so the detail modal matches
        # the image/model search behaviour. `artifact_image_hits` above only carries
        # one image per artifact (for the result-card thumbnails), so passing it here
        # would cap the modal gallery at a single image.
        artifact_results = await self._build_artifact_results(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            artifact_docs=retrieved_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            docs=retrieved_docs,
        )
        image_matches = self._enrich_image_matches(
            context="text",
            conversation_id=conversation_id,
            museum_slug=search_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        (
            paged_artifact_results,
            paged_image_matches,
            paged_navigation_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._build_paged_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=text_results_default_page_size,
            total_override=retrieved_docs_count,
            retrieval_request=retrieval_request,
        )
        results_request_id = str(
            state.last_paged_retrieval_request.get("results_request_id") or ""
        ).strip() or None
        visible_retrieval_context = self._build_visible_results_retrieval_context(
            artifact_results=paged_artifact_results,
            image_matches=[],
            page=results_page,
            page_size=results_page_size,
            total=results_total,
            prefer_artifact_results=True,
            include_image_matches_section=False,
        )
        retrieval_kind_for_log = str(retrieval_request.get("kind") or "")
        retrieval_mode_for_log = (
            retrieval_kind_for_log
            if retrieval_kind_for_log in {"selected_artifact", "anchored_similarity"}
            else ("hybrid_text" if router_decision.get("mode") == "rag" else "none")
        )
        if (
            retrieval_kind_for_log == "inventory"
            and bool(getattr(self.settings, "CHAT_INVENTORY_GRAMMAR_V2", False))
        ):
            # Só com o flag ligado: o rótulo histórico (hybrid_text) mantém-se
            # nos logs do estudo; o fast-path passa a ser distinguível.
            retrieval_mode_for_log = "inventory_lookup"
        retrieved_source_for_log = (
            retrieval_kind_for_log
            if retrieval_kind_for_log in {"selected_artifact", "anchored_similarity"}
            else "hybrid"
        )
        final_retrieval_context = (
            retrieval_context
            if retrieval_kind_for_log == "anchored_similarity"
            else (visible_retrieval_context or retrieval_context)
        )
        retry_context_result_count = (
            0
            if retrieval_kind_for_log == "anchored_similarity"
            else len(paged_artifact_results)
        )
        self._rag_debug_json(
            "rag.visible_context",
            {
                "conversation_id": conversation_id,
                "query_id": query_id,
                "retrieval_kind": retrieval_kind_for_log,
                "retrieval_mode": retrieval_mode_for_log,
                "results": {
                    "page": results_page,
                    "page_size": results_page_size,
                    "total": results_total,
                    "has_more": results_has_more,
                    "shown_artifacts": len(paged_artifact_results),
                    "shown_images": len(paged_image_matches),
                    "navigation_targets": len(paged_navigation_targets),
                },
                "visible_context_chars": len(visible_retrieval_context),
                "final_retrieval_context_chars": len(final_retrieval_context),
                "artifact_results": [
                    result.model_dump(mode="json")
                    for result in paged_artifact_results[:8]
                ],
                "image_matches": [
                    result.model_dump(mode="json")
                    for result in paged_image_matches[:8]
                ],
                "navigation_targets": [
                    result.model_dump(mode="json")
                    for result in paged_navigation_targets[:8]
                ],
            },
        )

        # O modelo escolhe os cartões: só com mais do que um cartão visível
        # (com um só, nada há a escolher) e nunca na semelhança ancorada,
        # cujo contexto repete os rótulos [doc_N] da peça de referência.
        select_cards = (
            bool(getattr(self.settings, "CHAT_MODEL_SELECTS_CARDS", False))
            and requested_format.type == "text"
            and retrieval_kind_for_log != "anchored_similarity"
            and len(paged_artifact_results) > 1
        )

        def build_text_final_message_for_result_count(result_count: int) -> str:
            limited_visible_retrieval_context = self._build_visible_results_retrieval_context(
                artifact_results=paged_artifact_results[: max(result_count, 0)],
                image_matches=[],
                page=results_page,
                page_size=results_page_size,
                total=results_total,
                prefer_artifact_results=True,
                include_image_matches_section=False,
            )
            limited_final_retrieval_context = (
                retrieval_context
                if retrieval_kind_for_log == "anchored_similarity"
                else limited_visible_retrieval_context
            )
            return self._build_final_prompt(
                payload=payload,
                state=state,
                router_decision=router_decision,
                retrieval_context=limited_final_retrieval_context,
                effective_filters=effective_filters,
                effective_sort=effective_sort,
                use_history_for_answer=use_history_for_answer,
                museum_slug=search_museum_slug,
                museum_name=search_museum_name,
                select_cards=select_cards,
            )

        final_message = self._build_final_prompt(
            payload=payload,
            state=state,
            router_decision=router_decision,
            retrieval_context=final_retrieval_context,
            effective_filters=effective_filters,
            effective_sort=effective_sort,
            use_history_for_answer=use_history_for_answer,
            museum_slug=search_museum_slug,
            museum_name=search_museum_name,
            select_cards=select_cards,
        )
        self._rag_debug_json(
            "rag.final_prompt",
            {
                "conversation_id": conversation_id,
                "query_id": query_id,
                "model_override": payload.model_override,
                "response_format": requested_format.model_dump(mode="json"),
                "final_prompt_chars": len(final_message),
                "retrieval_context_chars": len(final_retrieval_context),
                "retry_context_result_count": retry_context_result_count,
                "use_history_for_answer": use_history_for_answer,
                "router_decision": router_decision,
            },
        )
        await self._emit_status(status_cb, "status.generating_final_answer", language=language)

        # Quick win LLM-03: com CHAT_STREAM_FINAL_ANSWER e um canal SSE ativo,
        # os tokens da resposta final seguem para o cliente à medida que
        # chegam; o evento `result` (texto sanitizado) continua a ser o payload
        # autoritativo. Pedidos não-stream e formatos JSON ficam como estavam.
        stream_cb_factory = None
        if (
            self.settings.CHAT_STREAM_FINAL_ANSWER
            and status_cb is not None
            and requested_format.type == "text"
        ):
            stream_cb_factory = self._make_token_emitter(status_cb, query_id=query_id)
            if select_cards:
                stream_cb_factory = self._card_selection_stream_factory(stream_cb_factory)

        llm_started_at = time.monotonic()
        try:
            llm_response = await self._generate_final_answer_with_context_retries(
                initial_message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
                visible_result_count=retry_context_result_count,
                build_message_for_result_count=build_text_final_message_for_result_count,
                stream_cb_factory=stream_cb_factory,
            )
        except LLMServiceError as exc:
            llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.message.llm.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round(llm_latency_ms, 1),
                error=exc,
            )
            # Soft-fail in dev so frontend keeps moving while LLM infra is still unstable.
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=original_message,
                resolved_query=rewritten_query,
                lexical_query=retrieval_request.get("lexical_query"),
                embedding_query=retrieval_request.get("query_text"),
                route=str(router_decision.get("mode")),
                retrieval_mode=retrieval_mode_for_log,
                filters_applied={
                    "filters": effective_filters,
                    "sort": effective_sort,
                    "temporal_query": retrieval_request.get("temporal_query"),
                    "tour_scope": retrieval_request.get("tour_scope"),
                    "selected_artifact": retrieval_request.get("selected_artifact"),
                    "selected_context_mode": retrieval_request.get("selected_context_mode"),
                    "target_museum": retrieval_request.get("search_scope"),
                },
                boosts_applied=self._text_boosts_applied_for_log(
                    router_decision=router_decision,
                    retrieval_request=retrieval_request,
                ),
                retrieved_artifacts=self._log_retrieved_artifact_entries(retrieved_docs, source=retrieved_source_for_log),
                shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
                navigation_targets=self._log_navigation_target_entries(
                    navigation_targets=paged_navigation_targets,
                    artifact_results=paged_artifact_results,
                ),
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=llm_latency_ms,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
                opensearch_query=retrieval_request.get("opensearch_query"),
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override
                or self.settings.llm_model_resolved,
                image_matches=paged_image_matches,
                artifact_results=paged_artifact_results,
                navigation_targets=paged_navigation_targets,
                results_page=results_page,
                results_page_size=results_page_size,
                results_total=results_total,
                results_has_more=results_has_more,
                results_request_id=results_request_id,
                search_scope=search_scope,
            )
        llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.message.llm.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            model=llm_response.model,
            response_format=llm_response.response_format.type,
            duration_ms=round(llm_latency_ms, 1),
        )

        reply_docs = self._artifact_results_as_prompt_docs(paged_artifact_results) or retrieved_docs
        # O bloco «[[cartões: …]]» sai antes da sanitização, que trocaria os
        # rótulos [doc_N] pelos títulos.
        llm_text, model_cards = (
            self._split_card_selection(llm_response.text) if select_cards else (llm_response.text, None)
        )
        final_reply_text = self._sanitize_assistant_reply(
            llm_text,
            docs=reply_docs,
            language=language,
        )
        # Quick win FILT-02 (modo observação): regista títulos/inventários
        # citados que não existem nos cartões visíveis. Nunca altera a resposta.
        grounding_report = self._grounding_report(
            reply=final_reply_text,
            docs=reply_docs,
            conversation_id=conversation_id,
            query_id=query_id,
            user_message=original_message,
        )
        card_selection: dict[str, Any] | None = None
        if select_cards:
            (
                paged_artifact_results,
                paged_image_matches,
                paged_navigation_targets,
                results_total,
                results_has_more,
                card_selection,
            ) = self._select_cards(
                user_message=original_message,
                retrieval_request=retrieval_request,
                artifact_results=paged_artifact_results,
                image_matches=paged_image_matches,
                navigation_targets=paged_navigation_targets,
                model_indices=model_cards,
                results_total=results_total,
                results_has_more=results_has_more,
                reply_text=final_reply_text,
            )
            log_event(
                logger,
                logging.INFO,
                "chat.card_selection",
                conversation_id=conversation_id,
                query_id=query_id,
                **card_selection,
            )

        if not carry_filters:
            state.filters = {}
        if not carry_sort:
            state.sort = {}

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(
            state=state,
            artifact_docs=retrieved_docs,
            preserve_existing_when_empty=bool(router_decision.get("use_history_for_query", False)),
        )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=payload.message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=original_message,
            resolved_query=rewritten_query,
            lexical_query=retrieval_request.get("lexical_query"),
            embedding_query=retrieval_request.get("query_text"),
            route=str(router_decision.get("mode")),
            retrieval_mode=retrieval_mode_for_log,
            filters_applied={
                "filters": effective_filters,
                "sort": effective_sort,
                "temporal_query": retrieval_request.get("temporal_query"),
                "tour_scope": retrieval_request.get("tour_scope"),
                "selected_artifact": retrieval_request.get("selected_artifact"),
                "selected_context_mode": retrieval_request.get("selected_context_mode"),
                "target_museum": retrieval_request.get("search_scope"),
            },
            boosts_applied=self._text_boosts_applied_for_log(
                router_decision=router_decision,
                retrieval_request=retrieval_request,
            ),
            retrieved_artifacts=self._log_retrieved_artifact_entries(retrieved_docs, source=retrieved_source_for_log),
            shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
            navigation_targets=self._log_navigation_target_entries(
                navigation_targets=paged_navigation_targets,
                artifact_results=paged_artifact_results,
            ),
            latency_retrieval_ms=retrieval_latency_ms,
            latency_llm_ms=llm_latency_ms,
            latency_total_ms=(time.monotonic() - request_started_at) * 1000,
            status="ok",
            error=None,
            opensearch_query=retrieval_request.get("opensearch_query"),
            extra={
                "llm_finish_reason": getattr(llm_response, "finish_reason", None),
                "llm_usage": getattr(llm_response, "usage", None),
                "llm_endpoint": getattr(llm_response, "endpoint", None),
                "grounding": grounding_report,
                "card_selection": card_selection,
            },
        )

        log_event(
            logger,
            logging.INFO,
            "chat.message.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status="ok",
            results_total=results_total,
            shown_artifacts=len(paged_artifact_results),
            shown_images=len(paged_image_matches),
            duration_ms=round((time.monotonic() - request_started_at) * 1000, 1),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=paged_image_matches,
            artifact_results=paged_artifact_results,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    async def regenerate_last_reply(
        self,
        payload: ChatRegenerateRequest,
        *,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        state = self.session_store.load_or_create(
            conversation_id=payload.conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        if not state.history:
            raise ValueError(translate("error.no_history_regenerate", language))

        while state.history and state.history[-1].role == "assistant":
            state.history.pop()

        if not state.history or state.history[-1].role != "user":
            raise ValueError(translate("error.no_user_message_regenerate", language))

        last_user_message = state.history[-1].text
        state.history.pop()
        self.session_store.save(state)

        regenerated_payload = ChatMessageRequest(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            museum_name=payload.museum_name,
            language=language,
            message=last_user_message,
            conversation_id=payload.conversation_id,
            response_format=payload.response_format,
            system_prompt=payload.system_prompt,
            model_override=payload.model_override,
            metadata=payload.metadata,
        )
        return await self.handle_message(regenerated_payload, status_cb=status_cb)

    async def handle_image_message(
        self,
        payload: ChatImageMessageRequest,
        *,
        image_bytes: bytes,
        image_filename: str | None = None,
        image_content_type: str | None = None,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        conversation_id = payload.conversation_id or str(uuid4())
        requested_format = payload.response_format or ResponseFormatObject(type="text")
        query_id = str(uuid4())
        request_started_at = time.monotonic()
        retrieval_latency_ms: float | None = None
        llm_latency_ms: float | None = None
        explicit_museum_id = self._resolve_explicit_museum_id_values(
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        museum_id = self._resolve_museum_id_values(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        user_message = (payload.message or "").strip() or translate(
            "message.default_image_query",
            language,
        )
        requested_museum = self._resolve_requested_search_museum(user_message)
        if requested_museum is not None:
            museum_id = requested_museum.museum_id
            explicit_museum_id = requested_museum.museum_id
        search_museum_slug = requested_museum.museum_slug if requested_museum else payload.museum_slug
        search_museum_id = museum_id
        search_museum_name = (
            requested_museum.museum_name
            if requested_museum
            else payload.museum_name
        )
        search_scope = self._build_search_scope(
            current_museum_slug=payload.museum_slug,
            current_museum_id=self._resolve_museum_id_values(
                museum_slug=payload.museum_slug,
                museum_id=payload.museum_id,
                metadata=payload.metadata,
            ),
            current_museum_name=payload.museum_name,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            search_museum_name=search_museum_name,
        )

        self.session_store.append_turn(state, role="user", text=user_message)
        log_event(
            logger,
            logging.INFO,
            "chat.image.start",
            conversation_id=conversation_id,
            query_id=query_id,
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            language=language,
            response_format=requested_format.type,
            image_filename=image_filename,
            image_content_type=image_content_type,
            image_bytes=len(image_bytes or b""),
            message_chars=len(user_message),
        )
        await self._emit_status(status_cb, "status.analyzing_image", language=language)

        image_matches: list[ImageMatchResult] = []
        artifact_docs: list[dict[str, object]] = []
        artifact_results: list[ArtifactResult] = []
        navigation_targets: list[TourNavigationTarget] = []
        image_results_total = 0
        image_opensearch_query: dict[str, Any] | None = None
        retrieval_started_at = time.monotonic()
        try:
            image_embedding = await self.embedding_provider.embed_multimodal_image_bytes(
                image_bytes=image_bytes,
                text=None,
            )
        except Exception as exc:
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.image.embedding.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round(retrieval_latency_ms, 1),
                error=exc,
            )
            fallback = translate("error.image_processing_failed", language)
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=user_message,
                resolved_query=user_message,
                lexical_query=None,
                embedding_query=None,
                route="image_search",
                retrieval_mode="image_similarity",
                system_config=self._image_system_config(
                    modality="image", image_bytes=image_bytes),
                filters_applied={
                    "filters": dict(state.filters),
                    "sort": dict(state.sort),
                    "target_museum": search_scope.model_dump(mode="json"),
                },
                boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
                retrieved_artifacts=[],
                shown_artifacts=[],
                navigation_targets=[],
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=None,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                search_scope=search_scope,
            )

        await self._emit_status(status_cb, "status.searching_collection", language=language)
        image_candidates_k = max(
            self.settings.CHAT_IMAGE_RETRIEVAL_TOP_K,
            self.settings.CHAT_IMAGE_ARTIFACT_TOP_K,
            self.settings.CHAT_RETRIEVAL_CANDIDATES,
            1,
        )
        # Visual reranking scores a wider i2i pool and truncates back to
        # image_candidates_k afterwards (downstream shape unchanged).
        vl_stage = self._resolve_vl_rerank_stage()
        image_rerank_pool_k = image_candidates_k
        if vl_stage != "off":
            image_rerank_pool_k = max(
                image_candidates_k,
                int(self.settings.VL_RERANKER_CANDIDATES),
            )
        image_retrieval_window_size = self._image_retrieval_window_size(
            minimum=image_rerank_pool_k,
        )
        try:
            image_page = await self.opensearch_gateway.search_similar_images_page(
                museum_slug=search_museum_slug,
                museum_id=search_museum_id,
                image_embedding=image_embedding,
                from_offset=0,
                page_size=image_rerank_pool_k,
                retrieval_window_size=image_retrieval_window_size,
            )
            image_opensearch_query = image_page.query_body
            image_hits = image_page.results
            image_results_total = self._bounded_retrieval_total(
                image_page.total,
                image_retrieval_window_size,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.image.search.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round((time.monotonic() - retrieval_started_at) * 1000, 1),
                error=exc,
            )
            image_hits = []

        # Passe local (pool i2i pré-fusão): corre em local/both, e em global
        # apenas para pedidos SÓ-imagem (aí o pool i2i é o pool inteiro). Em
        # global com texto o rerank é adiado para o pool fundido pós-RRF.
        user_query_text = (payload.message or "").strip()
        has_query_text = bool(user_query_text)
        run_local_vl_pass = self._local_vl_rerank_pass_enabled(
            vl_stage, has_query_text=has_query_text
        )
        image_hits = await self._maybe_rerank_image_hits(
            query_text=user_query_text,
            query_image_bytes=image_bytes,
            image_hits=image_hits,
            top_k=image_candidates_k,
            enabled=run_local_vl_pass,
        )

        # E8: imagem+texto — quando o utilizador forneceu texto real e o modo
        # multimodal está ligado, funde i2i + joint imagem+texto + t2i +
        # pesquisa documental.
        image_text_outcome = None
        if image_hits:
            image_text_outcome = await self._maybe_fuse_image_text(
                message_text=user_query_text,
                museum_slug=search_museum_slug,
                museum_id=search_museum_id,
                i2i_hits=image_hits,
                conversation_id=conversation_id,
                query_id=query_id,
                query_image_bytes=image_bytes,
            )
        if image_text_outcome is not None and vl_stage in {"global", "both"}:
            # Rerank VL global: a query multimodal completa avalia o top-N
            # fundido — candidatos de TODAS as origens, não só do pool i2i.
            image_text_outcome = await self._maybe_global_vl_rerank_outcome(
                outcome=image_text_outcome,
                query_text=user_query_text,
                query_image_bytes=image_bytes,
                conversation_id=conversation_id,
                query_id=query_id,
            )
        elif (
            image_text_outcome is None
            and image_hits
            and has_query_text
            and vl_stage == "global"
        ):
            # Fusão indisponível (modo off ou falha isolada): o pool i2i é o
            # pool inteiro — o passe único de rerank corre agora para o stage
            # global não degradar silenciosamente face ao comportamento local.
            image_hits = await self._maybe_rerank_image_hits(
                query_text=user_query_text,
                query_image_bytes=image_bytes,
                image_hits=image_hits,
                top_k=image_candidates_k,
                enabled=True,
            )

        # Instrumentation: the two image modalities must be distinguishable in
        # the query log (user-study requirement) — plain upload logs
        # image_similarity; the fused joint image+text path logs
        # image_text_joint.
        retrieval_mode_used = ("image_text_joint" if image_text_outcome is not None
                               else "image_similarity")
        if image_text_outcome is not None:
            # Fichas iguais do mesmo conjunto: só se juntam (o pool i2i não é
            # alargado, pelo que a página pode ficar com menos cartões).
            artifact_docs = self._collapse_identical_set_members(list(image_text_outcome.docs))
            image_results_total = image_text_outcome.total
            image_matches = self._build_image_matches(
                image_hits=image_text_outcome.matched_image_hits,
                artifact_docs=artifact_docs,
            )
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=len(artifact_docs),
            )
        elif image_hits:
            try:
                artifact_docs = await self._fetch_artifact_docs_for_image_hits(
                    museum_slug=search_museum_slug,
                    museum_id=search_museum_id,
                    artifact_museum_id=explicit_museum_id,
                    image_hits=image_hits,
                    top_k=image_candidates_k,
                )
            except Exception as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "chat.image.artifact_fetch.error",
                    conversation_id=conversation_id,
                    query_id=query_id,
                    image_hits=len(image_hits),
                    error=exc,
                )
                artifact_docs = []

            artifact_docs = self._collapse_identical_set_members(artifact_docs)
            image_matches = self._build_image_matches(image_hits=image_hits, artifact_docs=artifact_docs)
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=len(artifact_docs),
            )
        else:
            await self._emit_status(
                status_cb,
                "status.artifacts_found",
                language=language,
                artifact_count=0,
            )
        retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.image.retrieval.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            image_hits=len(image_hits),
            artifact_docs=len(artifact_docs),
            results_total=image_results_total,
            duration_ms=round(retrieval_latency_ms, 1),
        )

        artifact_results = await self._build_artifact_results(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context="image",
            conversation_id=conversation_id,
            museum_slug=search_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        (
            paged_artifact_results,
            paged_image_matches,
            paged_navigation_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._build_paged_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=max(self.settings.CHAT_IMAGE_ARTIFACT_TOP_K, 1),
            total_override=image_results_total,
            retrieval_request={
                # multimodal_image: paginação pós-fusão via snapshot cacheado
                # (o materializer media re-executaria a pesquisa e desfaria a
                # fusão página a página).
                "kind": "multimodal_image" if image_text_outcome is not None else "image",
                "museum_id": search_museum_id,
                "museum_slug": search_museum_slug,
                "artifact_museum_id": explicit_museum_id,
                "search_scope": search_scope.model_dump(mode="json"),
                "image_embedding": image_embedding,
                "retrieval_window_size": image_retrieval_window_size,
                "results_total": image_results_total,
                "opensearch_query": image_opensearch_query,
                "user_message": user_query_text or None,
                "multimodal": (
                    image_text_outcome.debug_payload()
                    if self.settings.MULTIMODAL_DEBUG
                    else {
                        "vl_rerank_stage": vl_stage,
                        "branch_sizes": image_text_outcome.diagnostics.get(
                            "fusion", {}
                        ).get("branch_sizes"),
                        "global_vl_rerank": image_text_outcome.diagnostics.get(
                            "global_vl_rerank"
                        ),
                        "fused_size": image_text_outcome.total,
                    }
                )
                if image_text_outcome is not None
                else None,
            },
        )
        results_request_id = str(
            state.last_paged_retrieval_request.get("results_request_id") or ""
        ).strip() or None

        retrieval_context = self._build_visible_results_retrieval_context(
            artifact_results=paged_artifact_results,
            image_matches=paged_image_matches,
            page=results_page,
            page_size=results_page_size,
            total=results_total,
            match_section_label="visible_image_retrieval_matches",
        )
        retry_context_result_count = (
            len(paged_image_matches) if paged_image_matches else len(paged_artifact_results)
        )

        def build_image_final_message_for_result_count(result_count: int) -> str:
            limited_retrieval_context = self._build_visible_results_retrieval_context(
                artifact_results=paged_artifact_results[: max(result_count, 0)],
                image_matches=paged_image_matches[: max(result_count, 0)],
                page=results_page,
                page_size=results_page_size,
                total=results_total,
                match_section_label="visible_image_retrieval_matches",
            )
            return self._media_final_prompt(
                modality="image",
                museum_slug=search_museum_slug,
                museum_name=search_museum_name,
                filters_state=effective_filters,
                sort_state=effective_sort,
                user_message=user_message,
                retrieval_context=limited_retrieval_context,
                language=language,
            )

        router_decision: dict[str, object] = {
            "mode": "rag",
            "intent": "image_search",
            "rewritten_query": user_message,
            "needs_retrieval": True,
            "reason": "image_upload",
            "filters_delta": {},
            "sort_delta": {},
        }
        effective_filters = dict(state.filters)
        effective_sort = dict(state.sort)

        final_message = self._media_final_prompt(
            modality="image",
            museum_slug=search_museum_slug,
            museum_name=search_museum_name,
            filters_state=effective_filters,
            sort_state=effective_sort,
            user_message=user_message,
            retrieval_context=retrieval_context,
            language=language,
        )
        await self._emit_status(status_cb, "status.generating_final_answer", language=language)

        llm_started_at = time.monotonic()
        try:
            llm_response = await self._generate_final_answer_with_context_retries(
                initial_message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
                visible_result_count=retry_context_result_count,
                build_message_for_result_count=build_image_final_message_for_result_count,
                stream_cb_factory=self._final_answer_stream_factory(
                    status_cb, requested_format, query_id=query_id
                ),
            )
        except LLMServiceError as exc:
            llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.image.llm.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round(llm_latency_ms, 1),
                error=exc,
            )
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=user_message,
                resolved_query=user_message,
                lexical_query=None,
                embedding_query=None,
                route="image_search",
                retrieval_mode=retrieval_mode_used,
                system_config=self._image_system_config(
                    modality=("image_text" if image_text_outcome is not None
                              else "image"),
                    image_bytes=image_bytes, outcome=image_text_outcome),
                filters_applied={
                    "filters": effective_filters,
                    "sort": effective_sort,
                    "target_museum": search_scope.model_dump(mode="json"),
                },
                boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
                retrieved_artifacts=self._log_retrieved_artifact_entries(artifact_docs, source="image"),
                shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
                navigation_targets=self._log_navigation_target_entries(
                    navigation_targets=paged_navigation_targets,
                    artifact_results=paged_artifact_results,
                ),
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=llm_latency_ms,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
                opensearch_query=image_opensearch_query,
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                image_matches=paged_image_matches,
                artifact_results=paged_artifact_results,
                navigation_targets=paged_navigation_targets,
                results_page=results_page,
                results_page_size=results_page_size,
                results_total=results_total,
                results_has_more=results_has_more,
                results_request_id=results_request_id,
                search_scope=search_scope,
            )
        llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.image.llm.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            model=llm_response.model,
            response_format=llm_response.response_format.type,
            duration_ms=round(llm_latency_ms, 1),
        )

        reply_docs = self._artifact_results_as_prompt_docs(paged_artifact_results) or artifact_docs
        final_reply_text = self._sanitize_assistant_reply(
            llm_response.text,
            docs=reply_docs,
            language=language,
        )
        grounding_report = self._grounding_report(
            reply=final_reply_text,
            docs=reply_docs,
            conversation_id=conversation_id,
            query_id=query_id,
            user_message=user_message,
        )

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(
            state=state,
            artifact_docs=artifact_docs,
        )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=user_message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=user_message,
            resolved_query=user_message,
            lexical_query=None,
            embedding_query=None,
            route="image_search",
            retrieval_mode=retrieval_mode_used,
            system_config=self._image_system_config(
                modality=("image_text" if image_text_outcome is not None else "image"),
                image_bytes=image_bytes, outcome=image_text_outcome),
            filters_applied={
                "filters": effective_filters,
                "sort": effective_sort,
                "target_museum": search_scope.model_dump(mode="json"),
            },
            boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
            retrieved_artifacts=self._log_retrieved_artifact_entries(artifact_docs, source="image"),
            shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
            navigation_targets=self._log_navigation_target_entries(
                navigation_targets=paged_navigation_targets,
                artifact_results=paged_artifact_results,
            ),
            latency_retrieval_ms=retrieval_latency_ms,
            latency_llm_ms=llm_latency_ms,
            latency_total_ms=(time.monotonic() - request_started_at) * 1000,
            status="ok",
            error=None,
            opensearch_query=image_opensearch_query,
            extra={
                "llm_finish_reason": getattr(llm_response, "finish_reason", None),
                "llm_usage": getattr(llm_response, "usage", None),
                "llm_endpoint": getattr(llm_response, "endpoint", None),
                "grounding": grounding_report,
            },
        )

        log_event(
            logger,
            logging.INFO,
            "chat.image.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status="ok",
            results_total=results_total,
            shown_artifacts=len(paged_artifact_results),
            shown_images=len(paged_image_matches),
            duration_ms=round((time.monotonic() - request_started_at) * 1000, 1),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=paged_image_matches,
            artifact_results=paged_artifact_results,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    async def handle_model_message(
        self,
        payload: ChatModelMessageRequest,
        *,
        model_bytes: bytes,
        model_filename: str | None = None,
        model_content_type: str | None = None,
        status_cb: StatusCallback | None = None,
    ) -> ChatMessageResponse:
        conversation_id = payload.conversation_id or str(uuid4())
        requested_format = payload.response_format or ResponseFormatObject(type="text")
        query_id = str(uuid4())
        request_started_at = time.monotonic()
        retrieval_latency_ms: float | None = None
        llm_latency_ms: float | None = None
        explicit_museum_id = self._resolve_explicit_museum_id_values(
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        museum_id = self._resolve_museum_id_values(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            metadata=payload.metadata,
        )
        state = self.session_store.load_or_create(
            conversation_id=conversation_id,
            museum_slug=payload.museum_slug,
        )
        language = self._sync_state_language(state, payload.language)
        user_message = (payload.message or "").strip() or translate(
            "message.default_model_query",
            language,
        )
        requested_museum = self._resolve_requested_search_museum(user_message)
        if requested_museum is not None:
            museum_id = requested_museum.museum_id
            explicit_museum_id = requested_museum.museum_id
        search_museum_slug = requested_museum.museum_slug if requested_museum else payload.museum_slug
        search_museum_id = museum_id
        search_museum_name = (
            requested_museum.museum_name
            if requested_museum
            else payload.museum_name
        )
        search_scope = self._build_search_scope(
            current_museum_slug=payload.museum_slug,
            current_museum_id=self._resolve_museum_id_values(
                museum_slug=payload.museum_slug,
                museum_id=payload.museum_id,
                metadata=payload.metadata,
            ),
            current_museum_name=payload.museum_name,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            search_museum_name=search_museum_name,
        )

        self.session_store.append_turn(state, role="user", text=user_message)
        log_event(
            logger,
            logging.INFO,
            "chat.model.start",
            conversation_id=conversation_id,
            query_id=query_id,
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            search_museum_slug=search_museum_slug,
            search_museum_id=search_museum_id,
            language=language,
            response_format=requested_format.type,
            model_filename=model_filename,
            model_content_type=model_content_type,
            model_bytes=len(model_bytes or b""),
            message_chars=len(user_message),
        )
        await self._emit_status(status_cb, "status.preparing_model", language=language)

        image_matches: list[ImageMatchResult] = []
        artifact_docs: list[dict[str, object]] = []
        artifact_results: list[ArtifactResult] = []
        navigation_targets: list[TourNavigationTarget] = []
        file_name = (model_filename or "model.glb").strip() or "model.glb"
        model_results_total = 0
        model_image_embeddings: list[list[float]] = []
        model_opensearch_query: dict[str, Any] | None = None

        retrieval_started_at = time.monotonic()
        try:
            retrieval_result = await self.model_retrieval_service.retrieve(
                museum_slug=search_museum_slug,
                museum_id=search_museum_id,
                model_bytes=model_bytes,
                file_name=file_name,
                artifact_museum_id=explicit_museum_id,
                progress_cb=lambda message, fields: self._emit_status(
                    status_cb,
                    message,
                    language=language,
                    **fields,
                ),
            )
            artifact_docs = self._collapse_identical_set_members(list(retrieval_result.artifact_docs))
            model_results_total = retrieval_result.image_hits_total
            model_image_embeddings = retrieval_result.image_embeddings
            model_opensearch_query = retrieval_result.opensearch_query
            image_matches = self._build_image_matches(
                image_hits=retrieval_result.image_hits,
                artifact_docs=artifact_docs,
            )
        except Exception as exc:
            retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.model.retrieval.error",
                conversation_id=conversation_id,
                query_id=query_id,
                file_name=file_name,
                duration_ms=round(retrieval_latency_ms, 1),
                error=exc,
            )
            fallback = translate("error.model_processing_failed", language)
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=user_message,
                resolved_query=user_message,
                lexical_query=None,
                embedding_query=None,
                route="model_search",
                retrieval_mode="model_multiview_similarity",
                filters_applied={
                    "filters": dict(state.filters),
                    "sort": dict(state.sort),
                    "target_museum": search_scope.model_dump(mode="json"),
                },
                boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
                retrieved_artifacts=[],
                shown_artifacts=[],
                navigation_targets=[],
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=None,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                search_scope=search_scope,
            )
        retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.model.retrieval.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            file_name=file_name,
            artifact_docs=len(artifact_docs),
            image_matches=len(image_matches),
            results_total=model_results_total,
            duration_ms=round(retrieval_latency_ms, 1),
        )

        await self._emit_status(
            status_cb,
            "status.artifacts_found",
            language=language,
            artifact_count=len(artifact_docs),
        )

        artifact_results = await self._build_artifact_results(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            artifact_docs=artifact_docs,
        )
        navigation_targets = self._resolve_navigation_targets(
            museum_slug=search_museum_slug,
            museum_id=search_museum_id,
            docs=artifact_docs,
        )
        image_matches = self._enrich_image_matches(
            context="model",
            conversation_id=conversation_id,
            museum_slug=search_museum_slug,
            image_matches=image_matches,
            artifact_results=artifact_results,
            navigation_targets=navigation_targets,
        )
        (
            paged_artifact_results,
            paged_image_matches,
            paged_navigation_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._build_paged_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=image_matches,
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=max(self.settings.CHAT_IMAGE_ARTIFACT_TOP_K, 1),
            total_override=model_results_total,
            retrieval_request={
                "kind": "model",
                "museum_id": search_museum_id,
                "museum_slug": search_museum_slug,
                "artifact_museum_id": explicit_museum_id,
                "search_scope": search_scope.model_dump(mode="json"),
                "image_embeddings": model_image_embeddings,
                "retrieval_window_size": retrieval_result.retrieval_window_size,
                "results_total": model_results_total,
                "opensearch_query": model_opensearch_query,
            },
        )
        results_request_id = str(
            state.last_paged_retrieval_request.get("results_request_id") or ""
        ).strip() or None

        retrieval_context = self._build_visible_results_retrieval_context(
            artifact_results=paged_artifact_results,
            image_matches=paged_image_matches,
            page=results_page,
            page_size=results_page_size,
            total=results_total,
            match_section_label="visible_model_view_matches",
        )
        retry_context_result_count = (
            len(paged_image_matches) if paged_image_matches else len(paged_artifact_results)
        )

        def build_model_final_message_for_result_count(result_count: int) -> str:
            limited_retrieval_context = self._build_visible_results_retrieval_context(
                artifact_results=paged_artifact_results[: max(result_count, 0)],
                image_matches=paged_image_matches[: max(result_count, 0)],
                page=results_page,
                page_size=results_page_size,
                total=results_total,
                match_section_label="visible_model_view_matches",
            )
            return self._media_final_prompt(
                modality="model",
                museum_slug=search_museum_slug,
                museum_name=search_museum_name,
                filters_state=effective_filters,
                sort_state=effective_sort,
                user_message=user_message,
                retrieval_context=limited_retrieval_context,
                language=language,
            )

        router_decision: dict[str, object] = {
            "mode": "rag",
            "intent": "model_search",
            "rewritten_query": user_message,
            "needs_retrieval": True,
            "reason": "model_upload",
            "filters_delta": {},
            "sort_delta": {},
        }
        effective_filters = dict(state.filters)
        effective_sort = dict(state.sort)

        final_message = self._media_final_prompt(
            modality="model",
            museum_slug=search_museum_slug,
            museum_name=search_museum_name,
            filters_state=effective_filters,
            sort_state=effective_sort,
            user_message=user_message,
            retrieval_context=retrieval_context,
            language=language,
        )
        await self._emit_status(status_cb, "status.generating_final_answer", language=language)

        llm_started_at = time.monotonic()
        try:
            llm_response = await self._generate_final_answer_with_context_retries(
                initial_message=final_message,
                response_format=requested_format,
                system_prompt=self._final_system_prompt(payload.system_prompt, language),
                model_override=payload.model_override,
                visible_result_count=retry_context_result_count,
                build_message_for_result_count=build_model_final_message_for_result_count,
                stream_cb_factory=self._final_answer_stream_factory(
                    status_cb, requested_format, query_id=query_id
                ),
            )
        except LLMServiceError as exc:
            llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
            log_event(
                logger,
                logging.ERROR,
                "chat.model.llm.error",
                conversation_id=conversation_id,
                query_id=query_id,
                duration_ms=round(llm_latency_ms, 1),
                error=exc,
            )
            fallback = translate("error.llm_unavailable", language, error=str(exc))
            await self._emit_query_log(
                payload=payload,
                conversation_id=conversation_id,
                query_id=query_id,
                language=language,
                raw_query=user_message,
                resolved_query=user_message,
                lexical_query=None,
                embedding_query=None,
                route="model_search",
                retrieval_mode="model_multiview_similarity",
                filters_applied={
                    "filters": effective_filters,
                    "sort": effective_sort,
                    "target_museum": search_scope.model_dump(mode="json"),
                },
                boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
                retrieved_artifacts=self._log_retrieved_artifact_entries(artifact_docs, source="model"),
                shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
                navigation_targets=self._log_navigation_target_entries(
                    navigation_targets=paged_navigation_targets,
                    artifact_results=paged_artifact_results,
                ),
                latency_retrieval_ms=retrieval_latency_ms,
                latency_llm_ms=llm_latency_ms,
                latency_total_ms=(time.monotonic() - request_started_at) * 1000,
                status="error",
                error=str(exc),
                opensearch_query=model_opensearch_query,
            )
            return ChatMessageResponse(
                conversation_id=conversation_id,
                query_id=query_id,
                response_format=requested_format,
                reply=fallback,
                model_hint=payload.model_override or self.settings.llm_model_resolved,
                image_matches=paged_image_matches,
                artifact_results=paged_artifact_results,
                navigation_targets=paged_navigation_targets,
                results_page=results_page,
                results_page_size=results_page_size,
                results_total=results_total,
                results_has_more=results_has_more,
                results_request_id=results_request_id,
                search_scope=search_scope,
            )
        llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
        log_event(
            logger,
            logging.INFO,
            "chat.model.llm.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            model=llm_response.model,
            response_format=llm_response.response_format.type,
            duration_ms=round(llm_latency_ms, 1),
        )

        reply_docs = self._artifact_results_as_prompt_docs(paged_artifact_results) or artifact_docs
        final_reply_text = self._sanitize_assistant_reply(
            llm_response.text,
            docs=reply_docs,
            language=language,
        )
        grounding_report = self._grounding_report(
            reply=final_reply_text,
            docs=reply_docs,
            conversation_id=conversation_id,
            query_id=query_id,
            user_message=user_message,
        )

        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(
            state=state,
            artifact_docs=artifact_docs,
        )
        self.session_store.append_turn(state, role="assistant", text=final_reply_text)
        self._update_rolling_summary(
            state=state,
            latest_user_message=user_message,
            latest_assistant_message=final_reply_text,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)

        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=user_message,
            resolved_query=user_message,
            lexical_query=None,
            embedding_query=None,
            route="model_search",
            retrieval_mode="model_multiview_similarity",
            filters_applied={
                "filters": effective_filters,
                "sort": effective_sort,
                "target_museum": search_scope.model_dump(mode="json"),
            },
            boosts_applied={"image_in_tour_boost": self.settings.IMAGE_IN_TOUR_BOOST},
            retrieved_artifacts=self._log_retrieved_artifact_entries(artifact_docs, source="model"),
            shown_artifacts=self._log_shown_artifact_entries(paged_artifact_results),
            navigation_targets=self._log_navigation_target_entries(
                navigation_targets=paged_navigation_targets,
                artifact_results=paged_artifact_results,
            ),
            latency_retrieval_ms=retrieval_latency_ms,
            latency_llm_ms=llm_latency_ms,
            latency_total_ms=(time.monotonic() - request_started_at) * 1000,
            status="ok",
            error=None,
            opensearch_query=model_opensearch_query,
            extra={
                "llm_finish_reason": getattr(llm_response, "finish_reason", None),
                "llm_usage": getattr(llm_response, "usage", None),
                "llm_endpoint": getattr(llm_response, "endpoint", None),
                "grounding": grounding_report,
            },
        )

        log_event(
            logger,
            logging.INFO,
            "chat.model.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status="ok",
            results_total=results_total,
            shown_artifacts=len(paged_artifact_results),
            shown_images=len(paged_image_matches),
            duration_ms=round((time.monotonic() - request_started_at) * 1000, 1),
        )

        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=llm_response.response_format,
            reply=final_reply_text,
            reply_json=llm_response.parsed_json,
            model_hint=llm_response.model,
            image_matches=paged_image_matches,
            artifact_results=paged_artifact_results,
            navigation_targets=paged_navigation_targets,
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=results_request_id,
            search_scope=search_scope,
        )

    def _is_tour_pieces_list_query(self, message: str) -> bool:
        """«Que peças posso ver nesta visita?», «o que há para ver na visita?»,
        «lista as peças da visita», «what can I see in this tour?».

        Nomeia a visita, pede peças em geral (ou «o que ver»), e não traz
        mais nada: «que vestidos há nesta visita» é uma pesquisa com âmbito."""
        tokens = self._fold_query_text(message).split()
        if not tokens or not any(t in _TOUR_SCOPE_WORDS for t in tokens):
            return False
        asks_pieces = any(t in _TOUR_PIECES_WORDS for t in tokens)
        asks_what_to_see = ("que" in tokens or "what" in tokens) and any(
            t in {"ver", "visitar", "observar", "conhecer", "see", "visit"} for t in tokens
        )
        if not (asks_pieces or asks_what_to_see):
            return False
        return all(t in _TOUR_PIECES_LIST_WORDS for t in tokens)

    async def _answer_tour_pieces_fast_path(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        conversation_id: str,
        query_id: str,
        language: str,
        requested_format: ResponseFormatObject,
        request_started_at: float,
        status_cb: StatusCallback | None,
    ) -> ChatMessageResponse:
        """Todas as peças assinaladas na visita, pela ordem da visita, sem
        pesquisa nem LLM: as fichas vêm do índice numa só chamada e os cartões
        saem paginados («ver mais» pela cache da sessão)."""
        museum_slug = payload.museum_slug
        museum_id = self._resolve_museum_id(payload)
        await self._emit_status(status_cb, "status.analyzing_request", language=language)
        rooms = self.tour_navigation_service.list_rooms(museum_slug=museum_slug, museum_id=museum_id)
        inventories: list[str] = []
        targets: list[dict[str, Any]] = []
        seen: set[str] = set()
        rooms_with_pieces = 0
        for room in rooms:
            if room.get("inventories"):
                rooms_with_pieces += 1
            room_inventories = [i for i in (room.get("inventories") or []) if self._normalize_inventory_key(i) not in seen]
            for inventory in room_inventories:
                seen.add(self._normalize_inventory_key(inventory))
                inventories.append(inventory)
            targets.extend(room.get("targets") or [])
        described = {"known": bool(inventories), "scope": "tour", "room": None, "inventories": inventories, "targets": targets}

        def intro(docs: list[dict[str, object]]) -> str:
            if not docs:
                return translate("message.tour_pieces_none", language)
            return translate("message.tour_pieces_list", language, count=len(docs), rooms=rooms_with_pieces)

        return await self._finish_tour_context_answer(
            payload=payload,
            state=state,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            requested_format=requested_format,
            request_started_at=request_started_at,
            status_cb=status_cb,
            described=described,
            room_name=None,
            intro=intro,
            pieces_key=None,
            facts=None,
            fast_path="tour_pieces",
            tour_room=None,
            log_extra={"tour_pieces": {"inventories": len(inventories), "rooms": rooms_with_pieces}},
        )

    # ----- Analítica do acervo (CHAT_CATALOGUE_ANALYTICS) ----- #

    def _analytics_date_label(self, doc: dict[str, object], language: str) -> str:
        """«do século XVIII», «dos séculos XVIII a XIX», «de 1730», «de 1715–1730»."""
        en = normalize_language(language) == "en"
        start, end = doc.get("start_year"), doc.get("end_year")
        if not isinstance(start, int) or isinstance(start, bool):
            return str(doc.get("date_or_period") or "").strip()
        end = end if isinstance(end, int) and not isinstance(end, bool) else start
        a, b = century_of(start), century_of(end)

        def bounds(century: int) -> tuple[int, int]:
            return ((century - 1) * 100 + 1, century * 100) if century > 0 else (century * 100, century * 100 + 99)

        # datado só por século: o intervalo cobre os séculos inteiros
        if start == bounds(a)[0] and end == bounds(b)[1]:
            if a == b:
                return translate("message.analytics_date_century", language, century=century_label(a, english=en))
            return translate("message.analytics_date_centuries", language, start=century_label(a, english=en), end=century_label(b, english=en))
        if start == end:
            return translate("message.analytics_date_year", language, year=self._analytics_year_text(start, en))
        return translate("message.analytics_date_years", language, start=self._analytics_year_text(start, en), end=self._analytics_year_text(end, en))

    @staticmethod
    def _analytics_year_text(year: int, english: bool) -> str:
        if year < 0:
            return f"{-year} BC" if english else f"{-year} a.C."
        return str(year)

    async def _answer_catalogue_analytics_fast_path(
        self,
        *,
        analytics: AnalyticsQuery,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        conversation_id: str,
        query_id: str,
        language: str,
        requested_format: ResponseFormatObject,
        request_started_at: float,
        status_cb: StatusCallback | None,
    ) -> ChatMessageResponse:
        """«Qual é a peça mais antiga?», «quantas peças há nesta visita?», «de
        que século é a maioria?», «que autor tem mais obras?»: calculado no
        OpenSearch, resposta por template com os factos e os cartões (o LLM
        só redige se CHAT_CATALOGUE_ANALYTICS_LLM_WORDING)."""
        museum_slug = payload.museum_slug
        museum_id = self._resolve_museum_id(payload)
        en = normalize_language(language) == "en"
        await self._emit_status(status_cb, "status.searching_collection", language=language)

        # período explícito («do século XVIII», «entre 1800 e 1850») como filtro
        temporal = await self._interpret_temporal_query(payload.message)
        filters = self._merge_temporal_filter({}, temporal)
        subject = analytics.subject
        if temporal.start_year is not None and temporal.end_year is not None:
            subject = parse_catalogue_analytics(self._strip_temporal_expression_from_query(payload.message, temporal))
            subject = subject.subject if subject else analytics.subject
        cards = max(int(getattr(self.settings, "CHAT_CATALOGUE_ANALYTICS_CARDS", 24)), 1)
        kind = analytics.kind
        try:
            result = await self.opensearch_gateway.aggregate_artifacts(
                museum_slug=museum_slug,
                museum_id=museum_id,
                in_tour=analytics.in_tour,
                subject=subject,
                filters=filters,
                order=("asc" if kind == OLDEST else "desc") if kind in {OLDEST, NEWEST} else None,
                size=cards if kind in {OLDEST, NEWEST} else (8 if kind == COUNT and subject else 0),
                centuries=kind == CENTURIES,
                top_creators=5 if kind == TOP_CREATORS else 0,
            )
        except Exception as exc:
            log_event(logger, logging.WARNING, "chat.catalogue_analytics.error", conversation_id=conversation_id, query_id=query_id, error=exc)
            result = {"docs": [], "total_all": 0, "total_dated": 0, "centuries": [], "creators": []}

        docs: list[dict[str, object]] = list(result.get("docs") or [])
        total_all = int(result.get("total_all") or 0)
        total_dated = int(result.get("total_dated") or 0)
        scope = translate("message.analytics_scope_tour" if analytics.in_tour else "message.analytics_scope_museum", language)
        subject_text = translate("message.analytics_subject", language, subject=subject) if subject else ""
        period_text = ""
        if temporal.start_year is not None and temporal.end_year is not None:
            period_text = translate(
                "message.analytics_period", language,
                start=self._analytics_year_text(temporal.start_year, en), end=self._analytics_year_text(temporal.end_year, en),
            )
        pct = round(100 * total_dated / total_all) if total_all else 0
        facts: list[str] = []
        cards_docs: list[dict[str, object]] = []

        if kind in {OLDEST, NEWEST}:
            if not docs:
                intro = translate("message.analytics_none", language, scope=scope, subject=subject_text)
            else:
                first = docs[0]
                cards_docs = docs
                intro = translate(
                    f"message.analytics_{kind}", language,
                    dated=total_dated, pct=pct, total=total_all, scope=scope, subject=subject_text,
                    title=str(first.get("title") or "").strip() or "—",
                    inventory=self._doc_inventory(first) or "—",
                    date=self._analytics_date_label(first, language), shown=len(docs),
                )
                facts = [
                    (f"Objects with a date in the record {scope}{subject_text}: {total_dated} of {total_all}" if en
                     else f"Peças com data na ficha {scope}{subject_text}: {total_dated} de {total_all}"),
                    (f"The {'oldest' if kind == OLDEST else 'most recent'} is {first.get('title')} ({self._doc_inventory(first)}), {self._analytics_date_label(first, language)}" if en
                     else f"A mais {'antiga' if kind == OLDEST else 'recente'} é «{first.get('title')}» ({self._doc_inventory(first)}), {self._analytics_date_label(first, language)}"),
                    ("The cards below are in that order" if en else "Os cartões abaixo estão por essa ordem"),
                ]
        elif kind == COUNT:
            cards_docs = docs if subject else []
            dated_note = translate("message.analytics_count_dated", language, dated=total_dated)
            intro = translate(
                "message.analytics_count", language, count=total_all, scope=scope, subject=subject_text,
                period=period_text, dated_note=dated_note,
            )
            if cards_docs:
                intro += translate("message.analytics_count_cards", language)
            facts = [f"{'Objects' if en else 'Peças'} {scope}{subject_text}{period_text}: {total_all}; {dated_note}"]
        elif kind == CENTURIES:
            buckets = sorted(((century_of(int(key)), count) for key, count in result.get("centuries") or []), key=lambda b: b[0])
            if not buckets:
                intro = translate("message.analytics_none", language, scope=scope, subject=subject_text)
            else:
                top_century, top_count = max(buckets, key=lambda b: b[1])
                listing = ", ".join(f"{century_label(c, english=en)} {n}" for c, n in buckets)
                intro = translate(
                    "message.analytics_centuries", language, dated=total_dated, scope=scope, subject=subject_text,
                    top=century_label(top_century, english=en), top_count=top_count, listing=listing,
                    undated=max(total_all - total_dated, 0),
                )
                facts = [intro]
        else:  # TOP_CREATORS
            creators = list(result.get("creators") or [])
            if not creators:
                intro = translate("message.analytics_none", language, scope=scope, subject=subject_text)
            else:
                listing = ", ".join(f"{name} ({count})" for name, count in creators)
                intro = translate(
                    "message.analytics_top_creators", language, scope=scope, subject=subject_text,
                    listing=listing, total=total_all, attributed=int(result.get("total_attributed") or 0),
                )
                facts = [intro]

        targets = self.tour_navigation_service.resolve_targets(
            museum_slug=museum_slug, museum_id=museum_id,
            inventories=[self._doc_inventory(d) for d in cards_docs if self._doc_inventory(d)],
            limit=max(len(cards_docs), 1),
        ) if cards_docs else []
        described = {
            "known": bool(cards_docs) or total_all > 0, "scope": "catalogue", "room": None,
            "inventories": [self._doc_inventory(d) for d in cards_docs if self._doc_inventory(d)], "targets": targets,
        }
        log_event(
            logger, logging.INFO, "chat.catalogue_analytics",
            conversation_id=conversation_id, query_id=query_id, kind=kind, subject=subject, in_tour=analytics.in_tour,
            total_all=total_all, total_dated=total_dated, cards=len(cards_docs),
        )
        return await self._finish_tour_context_answer(
            payload=payload,
            state=state,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            requested_format=requested_format,
            request_started_at=request_started_at,
            status_cb=status_cb,
            described=described,
            room_name=None,
            intro=intro,
            pieces_key=None,
            facts=facts or None,
            fast_path="catalogue_analytics",
            tour_room=None,
            docs=cards_docs,
            use_llm_wording=bool(getattr(self.settings, "CHAT_CATALOGUE_ANALYTICS_LLM_WORDING", False)),
            log_extra={
                "catalogue_analytics": {
                    "kind": kind, "subject": subject, "in_tour": analytics.in_tour,
                    "temporal": self._temporal_query_filter_payload(temporal),
                    "total_all": total_all, "total_dated": total_dated,
                    "centuries": result.get("centuries"), "creators": result.get("creators"),
                    "opensearch_query": result.get("query_body"),
                }
            },
        )

    def _is_tour_location_query(self, message: str) -> bool:
        folded = self._fold_query_text(message)
        if not folded:
            return False
        return any(re.match(pattern, folded) for pattern in _TOUR_LOCATION_PATTERNS)

    def _tour_location_from_metadata(
        self,
        metadata: dict[str, Any] | None,
    ) -> dict[str, str | None] | None:
        """Posição na visita enviada pelo widget (`metadata.tour_location`)."""
        if not isinstance(metadata, dict):
            return None
        raw = metadata.get("tour_location")
        if not isinstance(raw, dict):
            return None

        def _text(*keys: str) -> str | None:
            for key in keys:
                value = raw.get(key)
                text = str(value).strip() if value is not None else ""
                if text:
                    return text
            return None

        panorama_key = _text("panorama_key", "panorama_id", "panoramaKey")
        room = _text("room", "location")
        if not panorama_key and not room:
            return None
        return {"panorama_key": panorama_key, "room": room, "title": _text("title")}

    async def _answer_tour_location_fast_path(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        conversation_id: str,
        query_id: str,
        language: str,
        requested_format: ResponseFormatObject,
        request_started_at: float,
        status_cb: StatusCallback | None,
    ) -> ChatMessageResponse:
        """«Onde estou?» respondido a partir da visita, não do acervo.

        Fonte de verdade: a posição que o widget envia (panorama/sala) e os
        overlays desse panorama nos ficheiros poi_tours. Sem posição, diz que
        não sabe (antes, a pesquisa híbrida inventava uma «sala»). Cartões =
        as peças assinaladas nesse ponto, na ordem da visita, com os alvos de
        navegação do próprio panorama. Sem chamadas ao LLM.
        """
        museum_slug = payload.museum_slug
        museum_id = self._resolve_museum_id(payload)
        location = self._tour_location_from_metadata(payload.metadata)
        await self._emit_status(status_cb, "status.analyzing_request", language=language)

        described: dict[str, Any] = {"known": False, "scope": "none", "room": None, "inventories": [], "targets": []}
        if location is not None:
            try:
                described = self.tour_navigation_service.describe_location(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    panorama_key=location.get("panorama_key"),
                    room=location.get("room"),
                )
            except Exception as exc:
                log_event(logger, logging.WARNING, "chat.tour_location.describe.error", error=exc)
        room_name = described.get("room") or (location or {}).get("room") or (location or {}).get("title")
        if location is None:
            intro = translate("message.tour_location_unknown", language)
            pieces_key = None
            facts: list[str] | None = None
        else:
            intro = (
                translate("message.tour_location_here", language, room=room_name)
                if room_name
                else translate("message.tour_location_here_unnamed", language)
            )
            pieces_key = (
                "message.tour_location_pieces"
                if described.get("scope") == "panorama"
                else "message.tour_location_room_pieces"
            )
            facts = [
                (f"O visitante está agora em: {room_name}" if room_name else "O visitante está num ponto da visita sem nome")
                if normalize_language(language) != "en"
                else (f"The visitor is now in: {room_name}" if room_name else "The visitor is at an unnamed point of the tour")
            ]
        return await self._finish_tour_context_answer(
            payload=payload,
            state=state,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            requested_format=requested_format,
            request_started_at=request_started_at,
            status_cb=status_cb,
            described=described,
            room_name=room_name,
            intro=intro,
            pieces_key=pieces_key,
            facts=facts,
            fast_path="tour_location",
            tour_room=None,
            log_extra={
                "tour_location": {
                    "sent": location,
                    "known": bool(described.get("known")),
                    "scope": described.get("scope"),
                    "room": room_name,
                    "inventories": list(described.get("inventories") or []),
                }
            },
        )

    @staticmethod
    def _tour_room_pieces(room: dict[str, Any]) -> list[TourRoomPiece]:
        """Peças de uma zona, uma por n.º de inventário, pela ordem da visita."""
        pieces: list[TourRoomPiece] = []
        seen: set[str] = set()
        for target in room.get("targets") or []:
            inventory = str(target.get("inventory_id") or "").strip()
            overlay = str(target.get("overlay_id") or "").strip()
            panorama = str(target.get("panorama_key") or "").strip()
            if not inventory or not overlay or not panorama or inventory.lower() in seen:
                continue
            seen.add(inventory.lower())
            pieces.append(
                TourRoomPiece(
                    inventory_id=inventory,
                    title=str(target.get("title") or "").strip() or None,
                    overlay_id=overlay,
                    panorama_key=panorama,
                )
            )
        return pieces

    @staticmethod
    def _asks_for_tour_areas_list(folded: str, tokens: list[str]) -> bool:
        """«Que salas há?», «listas as áreas do museu?», «por onde posso andar?»."""
        if any(t in _TOUR_PIECES_WORDS for t in tokens):
            return False
        if _TOUR_WHERE_TO_GO_PATTERN.search(folded):
            return True
        index = next((i for i, t in enumerate(tokens) if t in _TOUR_AREA_PLURALS), None)
        if index is None:
            return False
        # «pinturas de salas», «mobiliário das salas»: a zona é complemento de outra coisa
        if index > 0 and tokens[index - 1] in {"de", "da", "do", "das", "dos", "of", "com", "with", "em", "in"}:
            before = [t for t in tokens[:index - 1] if t not in _TOUR_LIST_TAIL_WORDS]
            if before:
                return False
        return all(t in _TOUR_LIST_TAIL_WORDS or t in _TOUR_AREA_PLURALS for t in tokens[index + 1:])

    def _match_tour_room_query(
        self,
        message: str,
        *,
        museum_slug: str,
        museum_id: str | None,
    ) -> dict[str, Any] | None:
        """Pergunta sobre um lugar da visita («e a sala cantonal?», «encontra o
        pátio», «leva-me ao claustro», «encontra as salas deste museu»).

        Devolve {"kind": "room", ...} para um lugar conhecido, {"kind":
        "choices", "candidates": [...]} quando o nome serve a vários («o
        claustro», «o túmulo»), {"kind": "list"} para a lista de lugares,
        {"kind": "unknown", "name": …} para uma sala que não existe na visita,
        ou None (segue para o router: é uma pesquisa).
        """
        folded = self._fold_query_text(message)
        if not folded:
            return None
        tokens = folded.split()
        if self._asks_for_tour_areas_list(folded, tokens):
            return {"kind": "list"}
        try:
            match = self.tour_navigation_service.match_rooms(museum_slug=museum_slug, museum_id=museum_id, text=message)
        except Exception as exc:
            log_event(logger, logging.WARNING, "chat.tour_room.find.error", error=exc)
            return None
        if match is not None:
            matched = match["matched_tokens"]
            # Iniciais do nome na visita («Túmulo de V. da Gama»): «vasco» cobre o «V».
            initials = {
                t for c in match["candidates"] for alias in c.get("aliases") or [c.get("folded") or ""]
                for t in alias.split() if len(t) == 1 and t.isalpha()
            }
            remainder = [
                t for t in tokens
                if t not in matched and t not in _TOUR_ROOM_QUESTION_WORDS and not (len(t) == 1 and t.isalpha())
                and not (len(t) >= 3 and t[0] in initials)
            ]
            # «encontra peças de D. Manuel»: pedir peças com o nome do lugar só
            # meio presente é uma pesquisa, não a «Capela dos filhos de D. Manuel I».
            asks_for_pieces = any(t in {"pecas", "peca", "objetos", "objeto", "obras", "obra", "artefactos", "objects", "pieces", "works"} for t in tokens)
            if not match.get("full") and asks_for_pieces:
                return None
            if not remainder:
                candidates = match["candidates"]
                if len(candidates) == 1:
                    return {"kind": "room", **candidates[0], "matched_tokens": matched}
                return {"kind": "choices", "candidates": candidates}
            # «vestidos na sala cantonal» é uma pesquisa (não cai para a lista)
            if any(t in matched for t in tokens):
                return None
        # Sala pedida que não existe: «e a sala das armas?» -> o nome vem logo a
        # seguir à palavra de tipo (e às ligações); «vestidos da sala» não é isto.
        type_words = {"sala", "capela", "galeria", "zona", "room", "chapel"}
        for index, token in enumerate(tokens):
            if token not in type_words:
                continue
            following = [t for t in tokens[index + 1:] if t not in {"da", "de", "do", "das", "dos", "of", "the"}]
            leftover = [t for t in following if t not in _TOUR_ROOM_QUESTION_WORDS]
            before = [t for t in tokens[:index] if t not in _TOUR_ROOM_QUESTION_WORDS]
            if leftover and len(leftover) <= 4 and not before:
                return {"kind": "unknown", "name": " ".join(leftover)}
        return None

    async def _answer_tour_room_fast_path(
        self,
        *,
        room_query: dict[str, Any],
        payload: ChatMessageRequest,
        state: ChatSessionState,
        conversation_id: str,
        query_id: str,
        language: str,
        requested_format: ResponseFormatObject,
        request_started_at: float,
        status_cb: StatusCallback | None,
    ) -> ChatMessageResponse:
        museum_slug = payload.museum_slug
        museum_id = self._resolve_museum_id(payload)
        await self._emit_status(status_cb, "status.analyzing_request", language=language)
        rooms = self.tour_navigation_service.list_rooms(museum_slug=museum_slug, museum_id=museum_id)
        room_targets = [
            TourRoomTarget(
                room=str(r["room"]),
                panorama_key=str(r["panorama_keys"][0]),
                overlay_id=None,
                piece_count=len(r.get("inventories") or []),
                description=r.get("description"),
                pieces=self._tour_room_pieces(r),
            )
            for r in rooms
            if r.get("panorama_keys")
        ]
        described: dict[str, Any] = {"known": False, "scope": "none", "room": None, "inventories": [], "targets": []}
        tour_room: TourRoomTarget | None = None
        tour_rooms: list[TourRoomTarget] = []
        facts: list[str] | None = None
        pieces_key: str | None = None
        kind = room_query.get("kind")
        if kind == "list":
            intro = translate("message.tour_rooms_list", language, count=len(room_targets))
            tour_rooms = room_targets
            room_name = None
        elif kind == "choices":
            wanted = {str(c.get("room") or "") for c in room_query.get("candidates") or []}
            tour_rooms = [t for t in room_targets if t.room in wanted]
            intro = translate("message.tour_room_choices", language, count=len(tour_rooms))
            room_name = None
        elif kind == "unknown":
            intro = translate("message.tour_room_unknown", language)
            tour_rooms = room_targets
            room_name = None
        else:
            room_name = str(room_query.get("room") or "")
            described = {
                "known": True,
                "scope": "room",
                "room": room_name,
                "inventories": list(room_query.get("inventories") or []),
                "targets": list(room_query.get("targets") or []),
            }
            panorama_keys = list(room_query.get("panorama_keys") or [])
            description = str(room_query.get("description") or "").strip() or None
            if panorama_keys:
                tour_room = TourRoomTarget(
                    room=room_name,
                    panorama_key=panorama_keys[0],
                    overlay_id=None,
                    piece_count=len(described["inventories"]),
                    description=description,
                )
            intro = translate("message.tour_room_intro", language, room=room_name)
            if description:
                intro = f"{intro} {description}"
            pieces_key = "message.tour_location_room_pieces"
            en = normalize_language(language) == "en"
            facts = [
                (f"The tour area asked about exists in the virtual tour: {room_name}" if en
                 else f"A zona da visita perguntada existe na visita virtual: {room_name}"),
                ("The visitor can go there with the button under the answer" if en
                 else "O visitante pode ir até lá com o botão que aparece por baixo da resposta"),
            ]
            if description:
                facts.append(
                    (f"Curated description of the area: {description}" if en
                     else f"Descrição curada da zona: {description}")
                )
        return await self._finish_tour_context_answer(
            payload=payload,
            state=state,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            requested_format=requested_format,
            request_started_at=request_started_at,
            status_cb=status_cb,
            described=described,
            room_name=room_name,
            intro=intro,
            pieces_key=pieces_key,
            facts=facts,
            fast_path="tour_room",
            tour_room=tour_room,
            tour_rooms=tour_rooms,
            log_extra={
                "tour_room": {
                    "kind": kind,
                    "room": room_name,
                    "asked": room_query.get("name"),
                    "inventories": list(described.get("inventories") or []),
                    "panorama_key": tour_room.panorama_key if tour_room else None,
                    "rooms_listed": len(tour_rooms),
                }
            },
        )

    async def _finish_tour_context_answer(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        conversation_id: str,
        query_id: str,
        language: str,
        requested_format: ResponseFormatObject,
        request_started_at: float,
        status_cb: StatusCallback | None,
        described: dict[str, Any],
        room_name: str | None,
        intro: str | Callable[[list[dict[str, object]]], str],
        pieces_key: str | None,
        facts: list[str] | None,
        fast_path: str,
        tour_room: TourRoomTarget | None,
        log_extra: dict[str, Any],
        tour_rooms: list[TourRoomTarget] | None = None,
        docs: list[dict[str, object]] | None = None,
        use_llm_wording: bool | None = None,
    ) -> ChatMessageResponse:
        """Parte comum das respostas de contexto da visita (posição, zona) e da
        analítica do acervo.

        Busca as fichas das peças assinaladas (ordem da visita) — ou usa as
        `docs` já vindas da pesquisa —, compõe a resposta — template, ou
        redação do LLM com os factos e os cartões trancados quando
        CHAT_TOUR_CONTEXT_LLM_WORDING (ou `use_llm_wording`) — e fecha estado,
        log e resposta como as outras rotas.
        """
        museum_slug = payload.museum_slug
        museum_id = self._resolve_museum_id(payload)
        inventories: list[str] = list(described.get("inventories") or [])

        retrieval_started_at = time.monotonic()
        if docs is not None:
            docs = list(docs)
            inventories = inventories or [self._doc_inventory(doc) for doc in docs if self._doc_inventory(doc)]
        else:
            docs = []
        if inventories and not docs:
            try:
                fetched = await self.opensearch_gateway.fetch_artifacts_by_inventory_numbers(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    inventory_numbers=inventories,
                    top_k=max(len(inventories), 1),
                )
            except Exception as exc:
                log_event(logger, logging.WARNING, "chat.tour_context.fetch.error", error=exc)
                fetched = []
            # Ordem da visita, não a do OpenSearch.
            order = {self._normalize_inventory_key(inv): index for index, inv in enumerate(inventories)}
            docs = sorted(
                fetched,
                key=lambda doc: order.get(self._normalize_inventory_key(self._doc_inventory(doc)), len(order)),
            )
        retrieval_latency_ms = (time.monotonic() - retrieval_started_at) * 1000

        # A lista de todas as peças só sabe quantas tem depois de ir ao índice.
        parts = [intro(docs) if callable(intro) else intro]
        if pieces_key is not None:
            if docs:
                pieces = "; ".join(self._tour_location_piece_label(doc) for doc in docs)
                parts.append(translate(pieces_key, language, count=len(docs), pieces=pieces))
            else:
                parts.append(translate("message.tour_location_no_pieces", language))
        if tour_room is not None:
            parts.append(translate("message.tour_room_go", language))
        template_reply = " ".join(parts)

        reply = template_reply
        llm_latency_ms = 0.0
        llm_response = None
        wording = "template"
        llm_wording = self.settings.CHAT_TOUR_CONTEXT_LLM_WORDING if use_llm_wording is None else use_llm_wording
        if facts is not None and llm_wording:
            en = normalize_language(language) == "en"
            llm_facts = list(facts)
            if docs:
                listing = "; ".join(self._tour_location_piece_label(doc) for doc in docs)
                llm_facts.append(
                    (f"Highlighted pieces in this area ({len(docs)}), in tour order: {listing}" if en
                     else f"Peças assinaladas nesta zona ({len(docs)}), pela ordem da visita: {listing}")
                )
            else:
                llm_facts.append("There are no highlighted pieces in this area" if en
                                 else "Não há peças assinaladas nesta zona")
            prompt = build_tour_context_prompt(
                museum_name=payload.museum_name,
                user_message=payload.message,
                facts=llm_facts,
                cards_context=self._format_docs_compact(docs=docs) if docs else "",
                language=language,
            )
            await self._emit_status(status_cb, "status.generating_final_answer", language=language)
            llm_started_at = time.monotonic()
            try:
                llm_response = await self._generate_final_answer_with_context_retries(
                    initial_message=prompt,
                    response_format=requested_format,
                    system_prompt=self._final_system_prompt(payload.system_prompt, language),
                    model_override=payload.model_override,
                    visible_result_count=0,
                    build_message_for_result_count=lambda _n: prompt,
                    stream_cb_factory=self._final_answer_stream_factory(
                        status_cb, requested_format, query_id=query_id
                    ),
                )
                text = self._sanitize_assistant_reply(llm_response.text, docs=docs, language=language)
                if text.strip():
                    reply = text
                    wording = "llm"
            except LLMServiceError as exc:
                log_event(
                    logger, logging.WARNING, "chat.tour_context.llm.error",
                    conversation_id=conversation_id, query_id=query_id, error=exc,
                )
            llm_latency_ms = (time.monotonic() - llm_started_at) * 1000
        # O nome da zona vem da visita, não dos cartões: citá-lo em negrito
        # («**Sala da Música**») não é inventar um título.
        grounding_report = self._grounding_report(
            reply=reply, docs=docs, conversation_id=conversation_id, query_id=query_id,
            user_message=" ".join(part for part in (payload.message, room_name) if part),
        ) if wording == "llm" else None

        artifact_results = await self._build_artifact_results(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_docs=docs,
        ) if docs else []
        wanted_inventories = {self._normalize_inventory_key(self._doc_inventory(doc)) for doc in docs}
        navigation_targets = [
            TourNavigationTarget(
                overlay_id=str(target.get("overlay_id") or ""),
                panorama_key=str(target.get("panorama_key") or ""),
                inventory_id=str(target.get("inventory_id") or ""),
                location=target.get("location"),
                title=target.get("title"),
            )
            for target in described.get("targets") or []
            if self._normalize_inventory_key(str(target.get("inventory_id") or "")) in wanted_inventories
        ]

        router_decision: dict[str, object] = {
            "mode": "rag",
            "intent": "search",
            "rewritten_query": payload.message,
            "needs_retrieval": False,
            "reason": f"deterministic_{fast_path}_fast_path",
            "is_follow_up": False,
            "use_history_for_query": False,
            "use_history_for_answer": False,
            "carry_filters": False,
            "carry_sort": False,
            "filters_delta": {},
            "sort_delta": {},
        }
        log_event(
            logger,
            logging.INFO,
            "chat.message.route",
            conversation_id=conversation_id,
            query_id=query_id,
            mode="rag",
            intent="search",
            fast_path=fast_path,
            location_known=bool(described.get("known")),
            scope=described.get("scope"),
            pieces=len(docs),
            wording=wording,
        )
        retrieval_request = {
            "kind": fast_path,
            "museum_id": museum_id,
            "museum_slug": museum_slug,
            "original_query": payload.message,
            "results_total": len(artifact_results),
        }
        (
            paged_artifacts,
            _paged_images,
            paged_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._build_paged_results(
            state=state,
            artifact_results=artifact_results,
            image_matches=[],
            navigation_targets=navigation_targets,
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=self._text_results_default_page_size(),
            total_override=len(artifact_results),
            retrieval_request=retrieval_request,
        )
        results_request_id = str(
            state.last_paged_retrieval_request.get("results_request_id") or ""
        ).strip() or None
        state.filters = {}
        state.sort = {}
        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(state=state, artifact_docs=docs, preserve_existing_when_empty=False)
        self.session_store.append_turn(state, role="assistant", text=reply)
        self._update_rolling_summary(
            state=state,
            latest_user_message=payload.message,
            latest_assistant_message=reply,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)
        total_ms = (time.monotonic() - request_started_at) * 1000
        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=(payload.message or "").strip(),
            resolved_query=(payload.message or "").strip(),
            lexical_query=None,
            embedding_query=None,
            route="rag",
            retrieval_mode=fast_path,
            filters_applied={"filters": {}, "sort": {}},
            boosts_applied={},
            retrieved_artifacts=self._log_retrieved_artifact_entries(docs, source=fast_path),
            shown_artifacts=self._log_shown_artifact_entries(paged_artifacts),
            navigation_targets=self._log_navigation_target_entries(
                navigation_targets=paged_targets,
                artifact_results=paged_artifacts,
            ),
            latency_retrieval_ms=retrieval_latency_ms if inventories else None,
            latency_llm_ms=llm_latency_ms,
            latency_total_ms=total_ms,
            status="ok",
            error=None,
            opensearch_query=None,
            extra={
                "fast_path": fast_path,
                "wording": wording,
                "llm_finish_reason": getattr(llm_response, "finish_reason", None),
                "llm_usage": getattr(llm_response, "usage", None),
                "llm_endpoint": getattr(llm_response, "endpoint", None),
                "grounding": grounding_report,
                **log_extra,
            },
        )
        log_event(
            logger,
            logging.INFO,
            "chat.message.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status="ok",
            fast_path=fast_path,
            duration_ms=round(total_ms, 1),
        )
        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=requested_format,
            reply=reply,
            model_hint=f"{'llm' if wording == 'llm' else 'deterministic'}:{fast_path}",
            artifact_results=paged_artifacts,
            navigation_targets=paged_targets,
            tour_room=tour_room,
            tour_rooms=list(tour_rooms or []),
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=results_request_id,
            search_scope=self._build_search_scope(
                current_museum_slug=museum_slug,
                current_museum_id=museum_id,
                current_museum_name=payload.museum_name,
                search_museum_slug=museum_slug,
                search_museum_id=museum_id,
                search_museum_name=payload.museum_name,
            ),
        )

    @staticmethod
    def _normalize_inventory_key(value: str | None) -> str:
        return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())

    def _tour_location_piece_label(self, doc: dict[str, object]) -> str:
        title = str(doc.get("title") or "").strip()
        inventory = self._doc_inventory(doc)
        if title and inventory:
            return f"{title} ({inventory})"
        return title or inventory or str(doc.get("artifact_id") or "")

    async def _answer_smalltalk_fast_path(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        conversation_id: str,
        query_id: str,
        language: str,
        requested_format: ResponseFormatObject,
        request_started_at: float,
        status_cb: StatusCallback | None,
    ) -> ChatMessageResponse:
        """Resposta determinística a saudações (quick win LLM-01).

        Reproduz o que a rota llm_only faz ao estado e ao log — histórico,
        rolling summary, evento backend_query com route=llm_only /
        retrieval_mode=none — mas sem qualquer chamada ao LLM. O evento leva
        `fast_path="smalltalk"` e `latency_llm_ms=0` para ser distinguível.
        """
        # «obrigado» e «adeus» não se respondem com «Olá!»
        tokens = set(self._fold_query_text(payload.message).split())
        if tokens & _SMALLTALK_THANKS_WORDS:
            reply = translate("message.smalltalk_thanks_reply", language)
        elif tokens & _SMALLTALK_BYE_WORDS:
            reply = translate("message.smalltalk_bye_reply", language)
        else:
            reply = translate("message.smalltalk_reply", language)
        router_decision: dict[str, object] = {
            "mode": "llm_only",
            "intent": "fallback",
            "rewritten_query": payload.message,
            "needs_retrieval": False,
            "reason": "deterministic_smalltalk_fast_path",
            "is_follow_up": False,
            "use_history_for_query": False,
            "use_history_for_answer": False,
            "carry_filters": False,
            "carry_sort": False,
            "filters_delta": {},
            "sort_delta": {},
        }
        log_event(
            logger,
            logging.INFO,
            "chat.message.route",
            conversation_id=conversation_id,
            query_id=query_id,
            mode="llm_only",
            intent="fallback",
            fast_path="smalltalk",
        )
        # Mesma rotação da cache de resultados paginados que a rota llm_only
        # faz (lista vazia): «ver mais» de mensagens antigas continua a
        # funcionar por results_request_id.
        (
            _paged_artifacts,
            _paged_images,
            _paged_targets,
            results_page,
            results_page_size,
            results_total,
            results_has_more,
        ) = self._build_paged_results(
            state=state,
            artifact_results=[],
            image_matches=[],
            navigation_targets=[],
            page=payload.results_page,
            page_size=payload.results_page_size,
            default_page_size=self._text_results_default_page_size(),
            total_override=0,
            retrieval_request={},
        )
        results_request_id = str(
            state.last_paged_retrieval_request.get("results_request_id") or ""
        ).strip() or None
        # Mesma limpeza de estado que a rota llm_only (carry_* = False).
        state.filters = {}
        state.sort = {}
        self._apply_router_decision_to_state(state=state, router_decision=router_decision)
        self._update_last_result_ids(
            state=state,
            artifact_docs=[],
            preserve_existing_when_empty=False,
        )
        self.session_store.append_turn(state, role="assistant", text=reply)
        self._update_rolling_summary(
            state=state,
            latest_user_message=payload.message,
            latest_assistant_message=reply,
            router_decision=router_decision,
        )
        self.session_store.save(state)
        await self._emit_status(status_cb, "status.answer_ready", language=language)
        total_ms = (time.monotonic() - request_started_at) * 1000
        await self._emit_query_log(
            payload=payload,
            conversation_id=conversation_id,
            query_id=query_id,
            language=language,
            raw_query=(payload.message or "").strip(),
            resolved_query=(payload.message or "").strip(),
            lexical_query=None,
            embedding_query=None,
            route="llm_only",
            retrieval_mode="none",
            filters_applied={"filters": {}, "sort": {}},
            boosts_applied={},
            retrieved_artifacts=[],
            shown_artifacts=[],
            navigation_targets=[],
            latency_retrieval_ms=None,
            latency_llm_ms=0.0,
            latency_total_ms=total_ms,
            status="ok",
            error=None,
            opensearch_query=None,
            extra={"fast_path": "smalltalk"},
        )
        log_event(
            logger,
            logging.INFO,
            "chat.message.finish",
            conversation_id=conversation_id,
            query_id=query_id,
            status="ok",
            fast_path="smalltalk",
            duration_ms=round(total_ms, 1),
        )
        return ChatMessageResponse(
            conversation_id=conversation_id,
            query_id=query_id,
            response_format=requested_format,
            reply=reply,
            model_hint="deterministic:smalltalk",
            results_page=results_page,
            results_page_size=results_page_size,
            results_total=results_total,
            results_has_more=results_has_more,
            results_request_id=results_request_id,
            search_scope=self._build_search_scope(
                current_museum_slug=payload.museum_slug,
                current_museum_id=self._resolve_museum_id(payload),
                current_museum_name=payload.museum_name,
                search_museum_slug=payload.museum_slug,
                search_museum_id=self._resolve_museum_id(payload),
                search_museum_name=payload.museum_name,
            ),
        )

    def _is_obvious_greeting_or_smalltalk(self, message: str) -> bool:
        """Saudação, agradecimento, despedida ou conversa sem pedido («olá
        irmão», «tudo bem contigo?», «obrigado!», «ok», «quem és tu?»).

        Por estrutura, não por lista de frases: basta que nenhuma palavra seja
        de conteúdo (tudo saudação, vocativo, cortesia ou enchimento). Assim
        «olá, encontra vestidos» continua a ser pesquisa. Medido a 2026-09-17:
        «olá irmão» ia para a pesquisa e o LLM inventava três peças.
        """
        folded = self._fold_query_text(message)
        if not folded:
            return False
        tokens = folded.split()
        if len(tokens) > 12:
            return False
        if all(t in _SMALLTALK_WORDS for t in tokens):
            return True
        # «quem és tu?», «o que sabes fazer?», «podes ajudar-me?»: pedido de
        # ajuda ou apresentação, sem assunto do acervo
        core = _SMALLTALK_PREFIX.sub("", folded)
        return any(re.fullmatch(pattern, core) for pattern in _SMALLTALK_PATTERNS)

    def _fallback_router_decision(
        self,
        message: str,
        *,
        context_policy: dict[str, object],
    ) -> dict[str, object]:
        if self._is_obvious_greeting_or_smalltalk(message):
            return {
                "mode": "llm_only",
                "intent": "fallback",
                "rewritten_query": message,
                "needs_retrieval": False,
                "reason": "router_unavailable_deterministic_fallback_llm_only",
                "is_follow_up": bool(context_policy.get("is_follow_up", False)),
                "use_history_for_query": False,
                "use_history_for_answer": bool(
                    context_policy.get("use_history_for_answer", False)
                ),
                "carry_filters": False,
                "carry_sort": False,
                "filters_delta": {},
                "sort_delta": {},
            }

        return {
            "mode": "rag",
            "intent": "search",
            "rewritten_query": message,
            "needs_retrieval": True,
            "reason": "router_unavailable_deterministic_fallback_rag",
            "is_follow_up": bool(context_policy.get("is_follow_up", False)),
            "use_history_for_query": False,
            "use_history_for_answer": bool(
                context_policy.get("use_history_for_answer", False)
            ),
            "carry_filters": False,
            "carry_sort": False,
            "filters_delta": {},
            "sort_delta": {},
        }

    def _deterministic_router_decision(
        self,
        *,
        payload: ChatMessageRequest,
        context_policy: dict[str, object],
    ) -> dict[str, object]:
        """Router sem LLM (CHAT_ROUTER_MODE=deterministic).

        Tudo o que não é small talk (já tratado antes, ou pelo fallback) vai a
        retrieval; a política de contexto decide seguimento e uso de histórico;
        a query de retrieval é a mensagem do utilizador (as elipses passam pelo
        resolver dedicado, como já acontecia quando a rewritten do router não
        sobrevivia aos guardrails); filtros temporais e de âmbito continuam a
        vir da extração determinística em _retrieve_context.
        """
        if self._is_obvious_greeting_or_smalltalk(payload.message):
            return self._fallback_router_decision(payload.message, context_policy=context_policy)
        is_follow_up = bool(context_policy.get("is_follow_up", False))
        use_history_for_query = bool(context_policy.get("use_history_for_query", False))
        use_history_for_answer = bool(context_policy.get("use_history_for_answer", False))
        return {
            "mode": "rag",
            "intent": "refine" if use_history_for_query else "search",
            "rewritten_query": (payload.message or "").strip(),
            "needs_retrieval": True,
            "reason": f"deterministic_router:{context_policy.get('reason', '')}",
            "is_follow_up": is_follow_up,
            "use_history_for_query": use_history_for_query,
            "use_history_for_answer": use_history_for_answer,
            "carry_filters": use_history_for_query,
            "carry_sort": use_history_for_query,
            "filters_delta": {},
            "sort_delta": {},
        }

    async def _route_message(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        context_policy: dict[str, object],
    ) -> dict[str, object]:
        if str(getattr(self.settings, "CHAT_ROUTER_MODE", "llm") or "llm").lower() == "deterministic":
            return self._deterministic_router_decision(payload=payload, context_policy=context_policy)
        router_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "mode": {"type": "string", "enum": ["rag", "llm_only"]},
                "intent": {"type": "string", "enum": ["overview", "search", "refine", "fallback"]},
                "is_follow_up": {"type": "boolean"},
                "use_history_for_query": {"type": "boolean"},
                "use_history_for_answer": {"type": "boolean"},
                "carry_filters": {"type": "boolean"},
                "carry_sort": {"type": "boolean"},
                "rewritten_query": {"type": "string"},
                "needs_retrieval": {"type": "boolean"},
                "reason": {"type": "string"},
                "filters_delta": {
                    "type": "object",
                    "additionalProperties": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {"type": "null"},
                        ]
                    },
                },
                "sort_delta": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": [
                "mode",
                "intent",
                "is_follow_up",
                "use_history_for_query",
                "use_history_for_answer",
                "carry_filters",
                "carry_sort",
                "rewritten_query",
                "needs_retrieval",
                "reason",
                "filters_delta",
                "sort_delta",
            ],
        }

        if bool(getattr(self.settings, "CHAT_ROUTER_COMPACT_SCHEMA", False)):
            # Quick win LLM-02 (parcial): o router só decide o que a política
            # determinística não consegue — modo, intenção, se é seguimento,
            # a query autocontida e filtros explícitos. Os restantes campos
            # (needs_retrieval, use_history_*, carry_*) já eram sobrepostos
            # pelos guardrails abaixo e passam a vir da política de contexto.
            router_schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": router_schema["properties"]["mode"],
                    "intent": router_schema["properties"]["intent"],
                    "is_follow_up": router_schema["properties"]["is_follow_up"],
                    "rewritten_query": router_schema["properties"]["rewritten_query"],
                    "filters_delta": router_schema["properties"]["filters_delta"],
                },
                "required": ["mode", "intent", "is_follow_up", "rewritten_query", "filters_delta"],
            }

        use_history_for_query_hint = bool(context_policy.get("use_history_for_query", False))
        use_history_for_answer_hint = bool(context_policy.get("use_history_for_answer", False))
        include_aux_context = use_history_for_query_hint or use_history_for_answer_hint
        history_lines = (
            [
                f"- {turn.role}: {turn.text}"
                for turn in state.history[-min(self.settings.CHAT_HISTORY_WINDOW, 4) :]
            ]
            if include_aux_context
            else []
        )
        conversation_state_aux: dict[str, object] = {
            "intent_prev": state.intent,
            "history_size": len(state.history),
        }
        router_prompt = build_router_user_prompt(
            museum_slug=payload.museum_slug,
            museum_name=payload.museum_name,
            rolling_summary=state.rolling_summary if include_aux_context else "",
            filters_state={
                key: value
                for key, value in state.filters.items()
                if key != "artifact_id"
            },
            sort_state=state.sort,
            history_lines=history_lines,
            current_user_message=payload.message,
            router_schema=router_schema,
            context_policy_hint=context_policy,
            conversation_state_aux=conversation_state_aux,
            language=state.language,
        )

        router_response = await self.llm_service.generate(
            message=router_prompt,
            response_format=ResponseFormatObject(type="json_object"),
            system_prompt=get_router_system_prompt(language=state.language),
            model_override=payload.model_override,
        )

        parsed = router_response.parsed_json
        if not isinstance(parsed, dict):
            raise LLMServiceError("Router did not return an object.")

        mode = str(parsed.get("mode", "llm_only"))
        intent = str(parsed.get("intent", "fallback"))
        rewritten_query = str(parsed.get("rewritten_query", payload.message))
        needs_retrieval = bool(parsed.get("needs_retrieval", mode == "rag"))
        reason = str(parsed.get("reason", ""))
        if "reason" not in parsed and bool(getattr(self.settings, "CHAT_ROUTER_COMPACT_SCHEMA", False)):
            reason = "compact_schema"
        is_follow_up = bool(parsed.get("is_follow_up", context_policy.get("is_follow_up", False)))
        use_history_for_query = bool(
            parsed.get(
                "use_history_for_query",
                context_policy.get("use_history_for_query", False),
            )
        )
        use_history_for_answer = bool(
            parsed.get(
                "use_history_for_answer",
                context_policy.get("use_history_for_answer", False),
            )
        )
        carry_filters = bool(parsed.get("carry_filters", context_policy.get("carry_filters", False)))
        carry_sort = bool(parsed.get("carry_sort", context_policy.get("carry_sort", False)))
        filters_delta = parsed.get("filters_delta") or {}
        sort_delta = parsed.get("sort_delta") or {}

        if not isinstance(filters_delta, dict):
            filters_delta = {}
        if not isinstance(sort_delta, dict):
            sort_delta = {}

        if self._extract_tour_scope_expression(payload.message):
            is_follow_up = False
            use_history_for_query = False
            carry_filters = False
            carry_sort = False
            filters_delta.pop("artifact_id", None)
            reason = self._append_reason_tag(reason, "guardrail_tour_scope_query")

        if mode not in {"rag", "llm_only"}:
            mode = "llm_only"
        if intent not in {"overview", "search", "refine", "fallback"}:
            intent = "fallback"
        if not is_follow_up:
            use_history_for_query = False
            carry_filters = False
            carry_sort = False
        if not use_history_for_query:
            carry_filters = False
            carry_sort = False
        if intent in {"search", "refine"}:
            if mode != "rag" or not needs_retrieval:
                reason = self._append_reason_tag(
                    reason,
                    "guardrail_search_requires_retrieval",
                )
            mode = "rag"
            needs_retrieval = True
        elif needs_retrieval:
            if mode != "rag":
                reason = self._append_reason_tag(
                    reason,
                    "guardrail_needs_retrieval_requires_rag",
                )
            mode = "rag"
            needs_retrieval = True
        elif mode == "rag":
            needs_retrieval = True
        else:
            needs_retrieval = False

        return {
            "mode": mode,
            "intent": intent,
            "is_follow_up": is_follow_up,
            "use_history_for_query": use_history_for_query,
            "use_history_for_answer": use_history_for_answer,
            "carry_filters": carry_filters,
            "carry_sort": carry_sort,
            "rewritten_query": rewritten_query,
            "needs_retrieval": needs_retrieval,
            "reason": reason,
            "filters_delta": filters_delta,
            "sort_delta": sort_delta,
        }

    @staticmethod
    def _append_reason_tag(reason: str, tag: str) -> str:
        cleaned_reason = str(reason or "").strip()
        cleaned_tag = str(tag or "").strip()
        if not cleaned_tag:
            return cleaned_reason
        if not cleaned_reason:
            return cleaned_tag
        if cleaned_tag in cleaned_reason:
            return cleaned_reason
        return f"{cleaned_reason} | {cleaned_tag}"

    def _apply_context_policy_guardrails(
        self,
        *,
        router_decision: dict[str, object],
        context_policy: dict[str, object],
        user_message: str,
    ) -> dict[str, object]:
        guarded = dict(router_decision)
        if not bool(context_policy.get("use_history_for_query", False)):
            return guarded

        guarded["is_follow_up"] = True
        guarded["use_history_for_query"] = True
        guarded["carry_filters"] = True
        guarded["carry_sort"] = True

        # Follow-up queries can still use non-artifact state filters.
        guarded["mode"] = "rag"
        guarded["needs_retrieval"] = True

        rewritten_query = str(guarded.get("rewritten_query", "")).strip()
        if not rewritten_query:
            guarded["rewritten_query"] = user_message

        reason = str(guarded.get("reason", "")).strip()
        guarded["reason"] = self._append_reason_tag(
            reason,
            "guardrail_context_policy_follow_up",
        )
        return guarded

    def _inventory_candidate_variants(self, value: str) -> list[str]:
        folded = self._fold_query_text(value)
        compact = re.sub(r"[^a-z0-9]+", "", folded, flags=re.UNICODE)

        variants: list[str] = []

        def add(candidate: str) -> None:
            cleaned = candidate.strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        add(folded)
        for prefix in sorted(_INVENTORY_PREFIXES, key=len, reverse=True):
            if not compact.startswith(prefix):
                continue
            suffix = compact[len(prefix):]
            if suffix and re.search(r"\d", suffix):
                add(f"{prefix} {suffix}")
                break
        add(compact)
        return variants

    def _looks_like_inventory_suffix(self, token: str) -> bool:
        cleaned = str(token or "").strip()
        return bool(cleaned and len(cleaned) <= 32 and re.search(r"\d", cleaned))

    def _looks_like_common_year(self, token: str) -> bool:
        cleaned = str(token or "").strip()
        if not re.fullmatch(r"\d{4}", cleaned):
            return False
        year = int(cleaned)
        return 1500 <= year <= 2099

    def _add_inventory_candidate(
        self,
        candidates: list[str],
        seen: set[str],
        value: str,
    ) -> None:
        for candidate in self._inventory_candidate_variants(value):
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    def _extract_structured_inventory_candidates(self, folded: str) -> list[str]:
        """Candidatos canónicos a partir da gramática real dos inventários
        (quick win CHAT_INVENTORY_GRAMMAR_V2). Devolve as formas que batem no
        `term` exato de `inventory_number` (normalizador lowercase+ascii) e,
        quando há sigla de coleção, também a forma com «cmp/ mnsr».
        """
        found: list[str] = []

        def add(value: str) -> None:
            cleaned = value.strip()
            if cleaned and cleaned not in found:
                found.append(cleaned)

        for match in _INVENTORY_V2_MNSR_NUMBER_FIRST.finditer(folded):
            num = match.group("num")
            code = match.group("type")
            coll = match.group("coll")
            if coll:
                add(f"{num} {code} {coll}/ mnsr")
            add(f"{num} {code} mnsr")
        for match in _INVENTORY_V2_MNSR_PREFIX_FIRST.finditer(folded):
            add(f"{match.group('num')} {match.group('type')} mnsr")
        for match in _INVENTORY_V2_MNAZ.finditer(folded):
            add(f"mnaz {match.group('num')} {match.group('type')}")
        for match in _INVENTORY_V2_MJ.finditer(folded):
            if match.group("sn"):
                add(f"mj_sn_{match.group('sn')}")
                continue
            base = f"mj_{match.group('kind')}{match.group('num')}"
            if match.group("sub"):
                add(f"{base}/{match.group('sub')}")
            add(base)
        return found

    def _extract_inventory_candidates(self, query: str) -> list[str]:
        folded = self._fold_query_text(query)
        if not folded:
            return []

        tokens = folded.split()
        prefix_set = set(_INVENTORY_PREFIXES)
        candidates: list[str] = []
        seen: set[str] = set()

        if bool(getattr(self.settings, "CHAT_INVENTORY_GRAMMAR_V2", False)):
            # Formas canónicas primeiro: são as que batem no term exato; o
            # resto da função continua a correr e acrescenta as variantes
            # históricas (superconjunto quando ligado, idêntico quando desligado).
            for candidate in self._extract_structured_inventory_candidates(folded):
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)

        for index, token in enumerate(tokens):
            if token in prefix_set and index + 1 < len(tokens):
                suffix = tokens[index + 1]
                if self._looks_like_inventory_suffix(suffix):
                    self._add_inventory_candidate(candidates, seen, f"{token} {suffix}")
                continue

            for prefix in sorted(_INVENTORY_PREFIXES, key=len, reverse=True):
                if not token.startswith(prefix) or token == prefix:
                    continue
                suffix = token[len(prefix):]
                if self._looks_like_inventory_suffix(suffix):
                    self._add_inventory_candidate(candidates, seen, token)
                break

        for index, token in enumerate(tokens):
            if token not in _ARTIFACT_REFERENCE_MARKERS:
                continue
            value_index = index + 1
            if (
                value_index < len(tokens)
                and tokens[value_index] in _ARTIFACT_REFERENCE_NUMBER_CONNECTORS
            ):
                value_index += 1
            if value_index >= len(tokens):
                continue
            value = tokens[value_index]
            if not self._looks_like_inventory_suffix(value):
                continue
            if self._looks_like_common_year(value):
                continue

            parts = [value]
            next_index = value_index + 1
            if (
                next_index < len(tokens)
                and re.fullmatch(r"[a-z0-9]{1,6}", tokens[next_index])
                and re.search(r"\d", tokens[next_index])
            ):
                parts.append(tokens[next_index])
            self._add_inventory_candidate(candidates, seen, " ".join(parts))

        for index in range(len(tokens)):
            for marker in _EXPLICIT_INVENTORY_MARKERS:
                marker_size = len(marker)
                if tuple(tokens[index : index + marker_size]) != marker:
                    continue
                window_start = index + marker_size
                window = tokens[window_start : window_start + 4]
                for offset, token in enumerate(window):
                    if token in prefix_set and offset + 1 < len(window):
                        suffix = window[offset + 1]
                        if self._looks_like_inventory_suffix(suffix):
                            self._add_inventory_candidate(
                                candidates,
                                seen,
                                f"{token} {suffix}",
                            )
                    elif self._looks_like_inventory_suffix(token):
                        self._add_inventory_candidate(candidates, seen, token)
                break

        return candidates

    def _retrieve_museum_overview_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        raw_query: str,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]] | None:
        """Fast-path deterministico para "fala-me do museu": responde com o
        perfil estatico do museu (app/data/museum_profiles.json) em vez de
        pesquisar artefactos — sem cartoes espurios, sem ramo visual (o hook
        multimodal exige kind=="text")."""
        profile = self._museum_profile_for(museum_id or museum_slug)
        if profile is None:
            return None
        if not self._is_museum_overview_query(
            raw_query,
            museum_name=str(profile.get("name") or ""),
        ):
            return None

        lines = [f"nome: {profile.get('name')}"]
        description = str(profile.get("description") or "").strip()
        if description:
            lines.append(f"descricao: {description}")
        address = str(profile.get("address") or "").strip()
        if address:
            lines.append(f"morada: {address}")
        year = profile.get("inauguration_year")
        if year:
            lines.append(f"ano_de_inauguracao: {year}")
        context = "[museu]\n" + "\n".join(lines)

        log_event(
            logger,
            logging.INFO,
            "chat.retrieval.museum_overview",
            museum_slug=museum_slug,
            museum_id=museum_id,
            query_chars=len(raw_query),
        )
        retrieval_request: dict[str, Any] = {
            "kind": "museum_overview",
            "museum_id": museum_id,
            "museum_slug": museum_slug,
            "original_query": raw_query,
            "results_total": 0,
        }
        return context, 0, [], retrieval_request

    async def _retrieve_inventory_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        raw_query: str,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]] | None:
        inventory_candidates = self._extract_inventory_candidates(raw_query)
        if not inventory_candidates:
            return None

        top_k = min(max(self.settings.CHAT_RETRIEVAL_TOP_K, 1), 5)
        try:
            docs = await self.opensearch_gateway.search_artifacts_by_inventory_candidates(
                museum_slug=museum_slug,
                museum_id=museum_id,
                inventory_numbers=inventory_candidates,
                top_k=top_k,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.retrieval.inventory.error",
                museum_slug=museum_slug,
                museum_id=museum_id,
                inventory_count=len(inventory_candidates),
                error=exc,
            )
            return None

        if not docs:
            return None

        if bool(getattr(self.settings, "CHAT_INVENTORY_GRAMMAR_V2", False)):
            # Com uma referência exata na pergunta, mostrar só as fichas cujo
            # inventário coincide com um candidato: a lista de 5 «parecidos»
            # (793 Cer, 406 Mob, ...) levava o LLM a descrever a peça errada
            # (medido 2026-09-15: «o cartão confirma apenas a arca 770 Mob»).
            wanted = {re.sub(r"[^a-z0-9]+", "", self._fold_query_text(c)) for c in inventory_candidates}
            exact = [
                doc for doc in docs
                if re.sub(r"[^a-z0-9]+", "", self._fold_query_text(str(doc.get("inventory_number") or ""))) in wanted
            ]
            if exact:
                docs = exact

        docs_for_context = docs[:top_k]
        context = self._format_docs_for_prompt(docs=docs_for_context, top_k=top_k)
        retrieval_request = {
            "kind": "inventory",
            "museum_id": museum_id,
            "museum_slug": museum_slug,
            "original_query": raw_query,
            "search_query": raw_query,
            "query_text": raw_query,
            "inventory_candidates": inventory_candidates,
            "filters": {},
            "sort": {},
            "results_total": len(docs),
        }
        return context, len(docs), docs, retrieval_request

    async def _retrieve_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        query: str,
        filters: dict[str, object],
        sort: dict[str, object],
        result_window_size: int | None = None,
        tour_scope_expression: str | None = None,
        inventory_query: str | None = None,
    ) -> tuple[str, int, list[dict[str, object]], dict[str, Any]]:
        raw_query = (query or "").strip()
        overview_result = self._retrieve_museum_overview_context(
            museum_slug=museum_slug,
            museum_id=museum_id,
            raw_query=raw_query,
        )
        if overview_result is not None:
            return overview_result
        # Com CHAT_INVENTORY_GRAMMAR_V2 o número de inventário é lido na
        # MENSAGEM ORIGINAL: a query contextual já vem sem o código do museu
        # («793 Mob MNSR» -> «793 Mob», por _strip_unrequested_museum_terms),
        # o que impedia o fast-path de inventário (medido em 2026-09-15).
        inventory_source = raw_query
        if bool(getattr(self.settings, "CHAT_INVENTORY_GRAMMAR_V2", False)) and (inventory_query or "").strip():
            inventory_source = str(inventory_query).strip()
        inventory_result = await self._retrieve_inventory_context(
            museum_slug=museum_slug,
            museum_id=museum_id,
            raw_query=inventory_source,
        )
        if inventory_result is not None:
            return inventory_result

        temporal_query = await self._interpret_temporal_query(raw_query)
        # O ambito da visita ("nesta visita", "no tour 360") tem de ser lido na
        # MENSAGEM ORIGINAL: a rewritten do router remove estas expressoes ao
        # extrair a expressao nominal pesquisavel, pelo que ler daqui perderia
        # o filtro in_tour. O chamador passa-a explicitamente; o fallback
        # mantem o comportamento para chamadas que nao a fornecem.
        tour_scope_expression = (
            tour_scope_expression
            or self._extract_tour_scope_expression(raw_query)
        )
        retrieval_filters = self._merge_temporal_filter(filters, temporal_query)
        retrieval_filters = self._merge_tour_scope_filter(
            retrieval_filters,
            tour_scope_expression,
        )
        search_query = self._strip_temporal_expression_from_query(raw_query, temporal_query)
        search_query = self._strip_tour_scope_expression_from_query(
            search_query,
            tour_scope_expression,
        )
        if not search_query:
            search_query = "objetos"
        original_search_query = search_query
        lexical_query_fallback = self._build_lexical_query(
            query=search_query,
            museum_slug=museum_slug,
            museum_id=museum_id,
        )
        # lexical_query is intentionally short and optimized for BM25.
        lexical_query = lexical_query_fallback
        # embedding_query keeps the original/resolved natural-language request for semantic retrieval.
        embedding_query = original_search_query
        query_rewrite_source = "heuristic"

        # CHAT_ENABLE_LLM_LEXICAL_QUERY estava declarado e documentado mas nunca
        # era lido (auditoria 2026-09-14, achado §6). Passa a ser honrado:
        # desligado, a query lexical é só a heurística (poupa ~0,35–0,46 s de
        # ida ao LLM remoto por pesquisa). Default True = comportamento histórico.
        if self.settings.CHAT_ENABLE_LLM_LEXICAL_QUERY:
            llm_lexical_query, _ = await self._rewrite_retrieval_query_with_llm(
                query=search_query,
                museum_slug=museum_slug,
                museum_id=museum_id,
                filters=filters,
                sort=sort,
            )
        else:
            llm_lexical_query = ""
        if llm_lexical_query:
            llm_lexical_query = self._strip_temporal_expression_from_query(
                llm_lexical_query,
                temporal_query,
            )
            llm_lexical_query = self._strip_tour_scope_expression_from_query(
                llm_lexical_query,
                tour_scope_expression,
            )
        if llm_lexical_query:
            lexical_query = llm_lexical_query
            query_rewrite_source = "llm"

        temporal_filter_payload = self._temporal_query_filter_payload(temporal_query)

        tour_scope_filter_payload = self._tour_scope_filter_payload(tour_scope_expression)

        if self.settings.CHAT_USE_QUERY_EMBEDDINGS:
            try:
                query_embedding = await self.embedding_provider.embed_text(embedding_query)
            except NotImplementedError:
                log_event(
                    logger,
                    logging.WARNING,
                    "chat.retrieval.embedding.unsupported",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    query_chars=len(embedding_query),
                )
                return "", 0, [], {}
            except Exception as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "chat.retrieval.embedding.error",
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    query_chars=len(embedding_query),
                    error=exc,
                )
                return "", 0, [], {}
        else:
            log_event(
                logger,
                logging.INFO,
                "chat.retrieval.embedding.disabled",
                museum_slug=museum_slug,
                museum_id=museum_id,
            )
            return "", 0, [], {}

        final_top_k = max(self.settings.CHAT_RETRIEVAL_TOP_K, 1)
        if result_window_size is None:
            retrieval_page_size = max(
                self.settings.CHAT_RETRIEVAL_CANDIDATES,
                final_top_k,
                1,
            )
        else:
            retrieval_page_size = max(int(result_window_size), final_top_k, 1)
        # Second-stage reranking scores a wider first-stage pool and truncates
        # back to retrieval_page_size afterwards (downstream shape unchanged).
        rerank_pool_size = retrieval_page_size
        if self.settings.CHAT_ENABLE_RERANKING:
            rerank_pool_size = max(
                retrieval_page_size,
                int(self.settings.RERANKER_CANDIDATES),
            )
        retrieval_window_size = self._text_retrieval_window_size(
            minimum=rerank_pool_size,
        )
        retrieval_boosts = self._matched_retrieval_boosts_for_log(
            query_text=embedding_query,
            lexical_query=lexical_query,
        )
        retrieval_boosts = retrieval_boosts + self._matched_lay_vocabulary_for_log(
            museum_slug=museum_slug,
            museum_id=museum_id,
            query_text=embedding_query,
            lexical_query=lexical_query,
        )
        self._rag_debug_json(
            "rag.text_query_plan",
            {
                "museum_slug": museum_slug,
                "museum_id": museum_id,
                "raw_query": raw_query,
                "search_query": search_query,
                "embedding_query": embedding_query,
                "lexical_query": lexical_query,
                "query_rewrite_source": query_rewrite_source,
                "filters": dict(retrieval_filters),
                "sort": dict(sort),
                "temporal_query": self._temporal_query_filter_payload(temporal_query),
                "tour_scope": self._tour_scope_filter_payload(tour_scope_expression),
                "retrieval_page_size": retrieval_page_size,
                "retrieval_window_size": retrieval_window_size,
                "retrieval_boosts": retrieval_boosts,
                "query_embedding": query_embedding,
            },
        )

        # Fichas iguais do mesmo conjunto: pede-se mais para, depois de as
        # juntar, a página continuar cheia (o corte a retrieval_page_size é
        # feito em _maybe_rerank_text_docs).
        fetch_size = self._collapse_fetch_size(rerank_pool_size)
        try:
            page_result = await self.opensearch_gateway.search_relevant_context_page(
                museum_slug=museum_slug,
                museum_id=museum_id,
                query_text=embedding_query,
                lexical_query=lexical_query,
                query_embedding=query_embedding,
                from_offset=0,
                page_size=fetch_size,
                filters=retrieval_filters,
                sort=sort,
                retrieval_window_size=max(retrieval_window_size, fetch_size),
            )
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "chat.retrieval.opensearch.error",
                museum_slug=museum_slug,
                museum_id=museum_id,
                page_size=retrieval_page_size,
                retrieval_window_size=retrieval_window_size,
                error=exc,
            )
            return "", 0, [], {}

        docs = self._collapse_identical_set_members(list(page_result.results))
        results_total = self._bounded_retrieval_total(page_result.total, retrieval_window_size)
        retrieval_request = {
            "kind": "text",
            "museum_id": museum_id,
            "museum_slug": museum_slug,
            "original_query": raw_query,
            "search_query": search_query,
            "query_text": embedding_query,
            "lexical_query": lexical_query,
            "query_rewrite_source": query_rewrite_source,
            "query_embedding": query_embedding,
            "filters": dict(retrieval_filters),
            "sort": dict(sort),
            "retrieval_boosts": retrieval_boosts,
            "retrieval_window_size": retrieval_window_size,
            "results_total": results_total,
            "temporal_query": temporal_filter_payload,
            "tour_scope": tour_scope_filter_payload,
            "opensearch_query": getattr(page_result, "query_body", None),
        }

        if not docs:
            return "", results_total, [], retrieval_request

        docs = await self._maybe_rerank_text_docs(
            query_text=embedding_query,
            docs=docs,
            top_k=retrieval_page_size,
        )

        context_top_k = final_top_k
        if result_window_size is not None:
            try:
                context_top_k = max(
                    context_top_k,
                    min(max(int(result_window_size), 1), len(docs)),
                )
            except (TypeError, ValueError):
                pass
        docs_for_context = docs[:context_top_k]
        context = self._format_docs_for_prompt(docs=docs_for_context, top_k=context_top_k)
        return context, results_total, docs, retrieval_request

    def _tokenize_for_language_guardrail(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKD", (text or "").casefold())
        folded = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        return re.findall(r"[a-z0-9]+", folded, flags=re.UNICODE)

    def _is_probably_portuguese_query(self, text: str) -> bool:
        tokens = self._tokenize_for_language_guardrail(text)
        if not tokens:
            return False
        pt_hits = sum(token in _PT_QUERY_LANGUAGE_HINTS for token in tokens)
        en_hits = sum(token in _EN_QUERY_LANGUAGE_HINTS for token in tokens)
        if re.search(r"[ãõáàâéêíóôúç]", (text or "").casefold()):
            pt_hits += 1
        return pt_hits > en_hits and pt_hits > 0

    def _is_probably_english_query(self, text: str) -> bool:
        tokens = self._tokenize_for_language_guardrail(text)
        if not tokens:
            return False
        en_hits = sum(token in _EN_QUERY_LANGUAGE_HINTS for token in tokens)
        pt_hits = sum(token in _PT_QUERY_LANGUAGE_HINTS for token in tokens)
        return en_hits > pt_hits and en_hits > 0

    def _query_terms_overlap_source(self, source: str, candidate: str) -> bool:
        source_tokens = {
            token for token in self._tokenize_for_language_guardrail(source) if len(token) > 1
        }
        candidate_tokens = [
            token for token in self._tokenize_for_language_guardrail(candidate) if len(token) > 1
        ]
        if not source_tokens or not candidate_tokens:
            return False
        return any(token in source_tokens for token in candidate_tokens)

    def _fold_query_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", (text or "").casefold())
        folded = "".join(
            char for char in normalized if not unicodedata.combining(char)
        )
        return re.sub(r"[^a-z0-9]+", " ", folded, flags=re.UNICODE).strip()

    def _fold_query_text_with_index(self, text: str) -> tuple[str, list[int]]:
        folded_chars: list[str] = []
        source_indices: list[int] = []
        for index, char in enumerate(text or ""):
            normalized = unicodedata.normalize("NFKD", char.casefold())
            for normalized_char in normalized:
                if unicodedata.combining(normalized_char):
                    continue
                folded_chars.append(normalized_char if normalized_char.isalnum() else " ")
                source_indices.append(index)
        return "".join(folded_chars), source_indices

    def _temporal_alias_expressions(self, temporal_query: TemporalQuery) -> list[str]:
        expressions: list[str] = []
        if temporal_query.expression:
            expressions.append(temporal_query.expression)
        for aliases, candidate in _HARDCODED_HISTORICAL_PERIODS:
            if (
                candidate.start_year == temporal_query.start_year
                and candidate.end_year == temporal_query.end_year
                and candidate.expression == temporal_query.expression
            ):
                expressions.extend(aliases)
        seen: set[str] = set()
        deduped: list[str] = []
        for expression in expressions:
            folded = self._fold_query_text(expression)
            if not folded or folded in seen:
                continue
            seen.add(folded)
            deduped.append(expression)
        return deduped

    def _strip_temporal_expression_from_query(
        self,
        text: str,
        temporal_query: TemporalQuery,
    ) -> str:
        raw_text = (text or "").strip()
        if (
            not raw_text
            or temporal_query.start_year is None
            or temporal_query.end_year is None
        ):
            return raw_text

        folded_text, source_indices = self._fold_query_text_with_index(raw_text)
        if not folded_text or not source_indices:
            return raw_text

        year = r"(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})"
        patterns = [
            rf"\b(?:entre|de)\s+{year}\s+(?:e|a|ate)\s+{year}\b",
            rf"\b(?:no\s+periodo|do\s+periodo|periodo)\s+{year}\s+(?:(?:e|a|ate)\s+)?{year}\b",
            rf"\b{year}\s+(?:e|a|ate)\s+{year}\b",
        ]

        for expression in self._temporal_alias_expressions(temporal_query):
            tokens = self._fold_query_text(expression).split()
            if not tokens:
                continue
            phrase = r"\s+".join(re.escape(token) for token in tokens)
            patterns.append(
                rf"\b(?:(?:de|do|da|dos|das|no|na|nos|nas|em)\s+)?"
                rf"{phrase}(?:\s+de\s+d\s+joao\s+(?:v|vi))?\b"
            )

        remove_mask = [False] * len(raw_text)
        for pattern in patterns:
            for match in re.finditer(pattern, folded_text):
                if match.end() <= match.start():
                    continue
                start = source_indices[match.start()]
                end = source_indices[match.end() - 1] + 1
                for index in range(start, min(end, len(remove_mask))):
                    remove_mask[index] = True

        cleaned = "".join(
            char for index, char in enumerate(raw_text) if not remove_mask[index]
        )
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,.;:!?")

    def _extract_tour_scope_expression(self, text: str) -> str | None:
        raw_text = (text or "").strip()
        if not raw_text:
            return None

        folded_text, source_indices = self._fold_query_text_with_index(raw_text)
        if not folded_text or not source_indices:
            return None

        for pattern in _TOUR_SCOPE_PATTERNS:
            match = re.search(pattern, folded_text)
            if not match:
                continue
            start = source_indices[match.start()]
            end = source_indices[match.end() - 1] + 1
            expression = raw_text[start:end].strip(" ,.;:!?")
            return expression or None

        return None

    def _strip_tour_scope_expression_from_query(
        self,
        text: str,
        tour_scope_expression: str | None,
    ) -> str:
        raw_text = (text or "").strip()
        if not raw_text or not tour_scope_expression:
            return raw_text

        folded_text, source_indices = self._fold_query_text_with_index(raw_text)
        if not folded_text or not source_indices:
            return raw_text

        remove_mask = [False] * len(raw_text)
        for pattern in _TOUR_SCOPE_PATTERNS:
            for match in re.finditer(pattern, folded_text):
                if match.end() <= match.start():
                    continue
                start = source_indices[match.start()]
                end = source_indices[match.end() - 1] + 1
                for index in range(start, min(end, len(remove_mask))):
                    remove_mask[index] = True

        cleaned = "".join(
            char for index, char in enumerate(raw_text) if not remove_mask[index]
        )
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,.;:!?")

    def _extract_explicit_temporal_query(self, text: str) -> TemporalQuery:
        folded = self._fold_query_text(text)
        if not folded:
            return TemporalQuery(None, None, None, None)

        year = r"(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})"
        explicit_patterns = (
            rf"\b(?:entre|de)\s+({year})\s+(?:e|a|ate)\s+({year})\b",
            rf"\b(?:no\s+periodo|do\s+periodo|periodo)\s+({year})\s+(?:(?:e|a|ate)\s+)?({year})\b",
            rf"\b({year})\s+(?:e|a|ate)\s+({year})\b",
        )
        for pattern in explicit_patterns:
            match = re.search(pattern, folded)
            if not match:
                continue
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            if start_year > end_year:
                start_year, end_year = end_year, start_year
            return TemporalQuery(
                start_year=start_year,
                end_year=end_year,
                expression=match.group(0),
                confidence=1.0,
            )

        return TemporalQuery(None, None, None, None)

    def _resolve_hardcoded_historical_period(self, text: str) -> TemporalQuery:
        folded = self._fold_query_text(text)
        if not folded:
            return TemporalQuery(None, None, None, None)

        for aliases, temporal_query in _HARDCODED_HISTORICAL_PERIODS:
            for alias in aliases:
                folded_alias = self._fold_query_text(alias)
                if not folded_alias:
                    continue
                if (
                    temporal_query.expression == "periodo joanino"
                    and re.search(r"\b(?:d\s*)?joao\s+vi\b", folded)
                ):
                    continue
                if re.search(rf"\b{re.escape(folded_alias)}\b", folded):
                    return temporal_query

        return TemporalQuery(None, None, None, None)

    def _parse_supported_century(self, token: str) -> int | None:
        normalized = (token or "").casefold().strip()
        if not normalized:
            return None
        if normalized in _ROMAN_CENTURY_NUMERALS:
            return _ROMAN_CENTURY_NUMERALS[normalized]

        match = re.match(r"^([0-9]{1,2})", normalized)
        if not match:
            return None
        century = int(match.group(1))
        if not 15 <= century <= 30:
            return None
        return century

    def _extract_century_temporal_query(self, text: str) -> TemporalQuery:
        folded = self._fold_query_text(text)
        if not folded:
            return TemporalQuery(None, None, None, None)

        century_token = r"([ivxlcdm]+|[0-9]{1,2}(?:o|a|st|nd|rd|th)?)"
        century_patterns = (
            rf"\b(?:seculo|seculos|sec|secs|century|centuries)\s+{century_token}\b",
            rf"\b{century_token}\s+(?:seculo|seculos|sec|secs|century|centuries)\b",
        )
        for pattern in century_patterns:
            match = re.search(pattern, folded)
            if not match:
                continue
            century = self._parse_supported_century(match.group(1))
            if century is None:
                continue
            start_year = (century - 1) * 100
            return TemporalQuery(
                start_year=start_year,
                end_year=start_year + 99,
                expression=match.group(0),
                confidence=1.0,
            )

        return TemporalQuery(None, None, None, None)

    def _build_temporal_interpretation_prompt(self, text: str) -> str:
        return "\n".join(
            [
                "Interpreta referencias temporais numa pergunta sobre patrimonio/museus.",
                "A tarefa NAO e responder ao utilizador.",
                "Extrai um intervalo de anos apenas se a pergunta mencionar anos, datas, seculos, decadas ou um periodo historico reconhecivel.",
                "Para periodos historicos implicitos, resolve para o intervalo historico mais usado no contexto portugues/europeu.",
                "Nao inventes datas se nao houver referencia temporal.",
                "Responde apenas JSON neste formato:",
                '{"start_year":1900,"end_year":1910,"expression":"expressao temporal mencionada","confidence":0.9}',
                "Se nao houver referencia temporal, responde:",
                '{"start_year":null,"end_year":null,"expression":null,"confidence":0.0}',
                "Regras:",
                "- start_year e end_year devem ser inteiros ou null.",
                "- expression deve ser a expressao temporal presente na pergunta, sem traduzir.",
                "- confidence deve estar entre 0 e 1.",
                "- Se tiveres duvida, usa null/null com confidence baixa.",
                f"pergunta: {text}",
            ]
        )

    async def _interpret_temporal_query(self, text: str) -> TemporalQuery:
        explicit = self._extract_explicit_temporal_query(text)
        if explicit.start_year is not None and explicit.end_year is not None:
            return explicit

        hardcoded_period = self._resolve_hardcoded_historical_period(text)
        if hardcoded_period.start_year is not None and hardcoded_period.end_year is not None:
            return hardcoded_period

        century = self._extract_century_temporal_query(text)
        if century.start_year is not None and century.end_year is not None:
            return century

        folded = self._fold_query_text(text)
        temporal_hints = (
            "periodo",
            "epoca",
            "era",
            "seculo",
            "seculos",
            "sec",
            "secs",
            "century",
            "centuries",
            "decada",
            "anos",
        )
        if not folded or not any(re.search(rf"\b{hint}\b", folded) for hint in temporal_hints):
            return TemporalQuery(None, None, None, None)

        try:
            response = await self.llm_service.generate(
                message=self._build_temporal_interpretation_prompt(text),
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=(
                    "Es um interpretador temporal para retrieval museologico. "
                    "Responde sempre JSON valido e nunca texto livre."
                ),
                model_override=None,
            )
        except Exception:
            return TemporalQuery(None, None, None, None)

        payload = response.parsed_json
        if not isinstance(payload, dict):
            return TemporalQuery(None, None, None, None)

        start_year = payload.get("start_year")
        end_year = payload.get("end_year")
        confidence = payload.get("confidence")
        if not isinstance(start_year, int) or not isinstance(end_year, int):
            return TemporalQuery(None, None, None, None)
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        resolved_confidence = (
            float(confidence)
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else 0.0
        )
        if resolved_confidence < 0.55:
            return TemporalQuery(None, None, None, None)

        expression = str(payload.get("expression") or "").strip() or None
        return TemporalQuery(
            start_year=start_year,
            end_year=end_year,
            expression=expression,
            confidence=max(0.0, min(resolved_confidence, 1.0)),
        )

    def _temporal_query_filter_payload(
        self,
        temporal_query: TemporalQuery,
    ) -> dict[str, object] | None:
        if temporal_query.start_year is None or temporal_query.end_year is None:
            return None
        return {
            "start_year": temporal_query.start_year,
            "end_year": temporal_query.end_year,
            "expression": temporal_query.expression,
            "confidence": temporal_query.confidence,
            "include_unknown": False,
        }

    def _merge_temporal_filter(
        self,
        filters: dict[str, object],
        temporal_query: TemporalQuery,
    ) -> dict[str, object]:
        temporal_filter = self._temporal_query_filter_payload(temporal_query)
        if not temporal_filter:
            return dict(filters)
        merged = dict(filters)
        merged["_temporal_interval"] = temporal_filter
        return merged

    def _tour_scope_filter_payload(
        self,
        tour_scope_expression: str | None,
    ) -> dict[str, object] | None:
        expression = str(tour_scope_expression or "").strip()
        if not expression:
            return None
        return {
            "in_tour": True,
            "expression": expression,
        }

    def _merge_tour_scope_filter(
        self,
        filters: dict[str, object],
        tour_scope_expression: str | None,
    ) -> dict[str, object]:
        if not tour_scope_expression:
            return dict(filters)
        merged = dict(filters)
        merged["in_tour"] = True
        return merged

    def _has_query_language_mismatch(self, source: str, candidate: str) -> bool:
        source_pt = self._is_probably_portuguese_query(source)
        source_en = self._is_probably_english_query(source)
        candidate_pt = self._is_probably_portuguese_query(candidate)
        candidate_en = self._is_probably_english_query(candidate)
        if source_pt:
            if candidate_en and not candidate_pt:
                return True
            if candidate_pt or self._query_terms_overlap_source(source, candidate):
                return False
            return True
        if source_en and candidate_pt and not candidate_en:
            return True
        return False

    def _strip_unrequested_museum_terms(
        self,
        *,
        text: str,
        original: str,
        museum_slug: str | None,
        museum_id: str | None,
        museum_name: str | None,
    ) -> str:
        """Remove nome/slug/id do museu de ``text`` quando o utilizador NAO os
        escreveu em ``original`` (a pesquisa ja esta limitada ao museu; termos
        do museu injetados pelo LLM so poluem BM25/embeddings)."""
        original_folded = self._fold_query_text(original)
        banned_sequences: list[list[str]] = []
        for term in (museum_slug, museum_id, museum_name):
            term_folded = self._fold_query_text(str(term or ""))
            if not term_folded or term_folded in original_folded:
                continue
            banned_sequences.append(term_folded.split())
        if not banned_sequences:
            return text

        raw_tokens = text.split()
        folded_tokens = [self._fold_query_text(token) for token in raw_tokens]
        keep = [True] * len(raw_tokens)
        for sequence in banned_sequences:
            length = len(sequence)
            if length == 0:
                continue
            for start in range(0, len(folded_tokens) - length + 1):
                if folded_tokens[start : start + length] == sequence and all(
                    keep[start : start + length]
                ):
                    for index in range(start, start + length):
                        keep[index] = False

        kept_tokens = [token for token, flag in zip(raw_tokens, keep) if flag]
        # preposicoes orfas no fim ("vestidos pretos no" -> "vestidos pretos")
        dangling = {"no", "na", "nos", "nas", "do", "da", "dos", "das", "de", "em", "ao", "a", "o"}
        while kept_tokens and self._fold_query_text(kept_tokens[-1]) in dangling:
            kept_tokens.pop()
        return " ".join(kept_tokens).strip()

    def _filter_lexical_to_source(self, candidate: str, source: str) -> str:
        """Garantia dura de que o extrator lexical so EXTRAI: qualquer token do
        candidato que nao exista na fonte e descartado (o prompt proibe
        sinonimos/traducoes, logo subset e o contrato)."""
        source_tokens = set(self._fold_query_text(source).split())
        kept: list[str] = []
        for token in candidate.split():
            folded = self._fold_query_text(token)
            if folded and folded in source_tokens:
                kept.append(token)
        return " ".join(kept).strip()

    def _resolve_contextual_retrieval_query(
        self,
        *,
        original_query: str,
        rewritten_query: str,
        state: ChatSessionState,
        requested_museum: Any | None,
        museum_slug: str | None,
        museum_id: str | None,
        museum_name: str | None,
    ) -> tuple[str, str]:
        """Escolhe a query de retrieval: a rewritten do router (resolvida com o
        contexto conversacional, ex. "e de linho?" -> "vestidos de linho") quando
        passa os guardrails, senao a mensagem do utilizador.

        Guardrails deterministicos, por ordem: strip de museu pedido/injetado,
        lingua igual, tamanho, e anti-invencao (a rewritten tem de partilhar
        pelo menos um token significativo com a mensagem atual ou com as
        ultimas mensagens do utilizador — mata reescritas alucinadas tipo
        "Ola" -> "Qual e o nome do museu?").
        """
        original = (original_query or "").strip()
        candidate = (rewritten_query or "").strip()
        if not candidate or not original:
            return original, "user_message"

        candidate = self._strip_search_museum_expression_from_query(
            candidate,
            requested_museum,
        )
        candidate = self._strip_unrequested_museum_terms(
            text=candidate,
            original=original,
            museum_slug=museum_slug,
            museum_id=museum_id,
            museum_name=museum_name,
        ).strip()
        if not candidate:
            return original, "user_message"
        if self._fold_query_text(candidate) == self._fold_query_text(original):
            return original, "user_message"
        if len(candidate) > 220:
            return original, "user_message"
        if self._has_query_language_mismatch(original, candidate):
            return original, "user_message"

        candidate_tokens = {
            token
            for token in self._fold_query_text(candidate).split()
            if len(token) >= 3
        }
        if not candidate_tokens:
            return original, "user_message"
        source_tokens: set[str] = set()
        source_texts = [original]
        user_turns = [
            turn.text
            for turn in state.history
            if str(getattr(turn, "role", "")) == "user"
        ]
        source_texts.extend(user_turns[-3:])
        for source_text in source_texts:
            source_tokens.update(
                self._fold_query_text(source_text).split()
            )
        # Anti-invencao estrito: a rewritten so RESOLVE referencias — TODOS os
        # seus tokens significativos tem de existir na mensagem atual ou nas
        # ultimas mensagens do utilizador. Um follow-up legitimo ("e de linho?"
        # -> "vestidos de linho") passa porque "vestidos" vem do historico;
        # substituicoes alucinadas ("vestidos de seda" -> "vestidos de linho"
        # na primeira mensagem) sao rejeitadas.
        if not candidate_tokens.issubset(source_tokens):
            return original, "user_message"

        return candidate, "router_rewrite"

    async def _resolve_follow_up_query_with_llm(
        self,
        *,
        current_message: str,
        state: ChatSessionState,
    ) -> str:
        """Resolver dedicado de follow-ups elipticos ("e de prata?").

        Chamada LLM pequena e de tarefa unica, usada apenas quando o router
        nao resolveu a referencia. O output passa pelos MESMOS guardrails
        deterministicos antes de ser aceite.
        """
        cleaned = (current_message or "").strip()
        if not cleaned:
            return ""
        user_turns = [
            str(turn.text)
            for turn in state.history
            if str(getattr(turn, "role", "")) == "user"
        ]
        previous_user_messages = [text for text in user_turns[:-1] if text.strip()][-3:]
        if not previous_user_messages:
            return ""

        prompt = build_follow_up_resolution_prompt(
            current_message=cleaned,
            previous_user_messages=previous_user_messages,
        )
        try:
            response = await self.llm_service.generate(
                message=prompt,
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=FOLLOW_UP_RESOLUTION_SYSTEM_PROMPT,
                model_override=None,
            )
        except Exception:
            return ""
        payload = response.parsed_json
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("resolved_query") or "").strip()

    async def _rewrite_retrieval_query_with_llm(
        self,
        *,
        query: str,
        museum_slug: str,
        museum_id: str | None,
        filters: dict[str, object],
        sort: dict[str, object],
    ) -> tuple[str, str]:
        raw_query = (query or "").strip()
        if not raw_query:
            return "", ""

        prompt = build_retrieval_query_rewrite_prompt(
            user_query=raw_query,
            filters=dict(filters),
            sort=dict(sort),
        )
        try:
            response = await self.llm_service.generate(
                message=prompt,
                response_format=ResponseFormatObject(type="json_object"),
                system_prompt=RETRIEVAL_QUERY_REWRITE_SYSTEM_PROMPT,
                model_override=None,
            )
        except Exception:
            return "", ""

        payload = response.parsed_json
        if not isinstance(payload, dict):
            return "", ""

        lexical_query = str(payload.get("lexical_query") or "").strip()
        if lexical_query and self._has_query_language_mismatch(raw_query, lexical_query):
            lexical_query = ""
        # O extrator so pode EXTRAIR: tokens fora da fonte (museus injetados,
        # pesquisas inventadas para saudacoes, etc.) sao descartados; se nada
        # sobrar, o chamador cai na heuristica deterministica.
        if lexical_query:
            lexical_query = self._filter_lexical_to_source(lexical_query, raw_query)
        return lexical_query, ""

    def _build_lexical_query(
        self,
        *,
        query: str,
        museum_slug: str,
        museum_id: str | None,
    ) -> str:
        def _fold_token(value: str) -> str:
            normalized_value = unicodedata.normalize("NFKD", value.casefold())
            return "".join(
                char for char in normalized_value if not unicodedata.combining(char)
            )

        text = (query or "").strip().lower()
        if not text:
            return ""

        for pattern in (
            r"\b(no|na|do|da|em)\s+museu\s+[a-z0-9_-]+\b",
            r"\bmuseu\s+[a-z0-9_-]+\b",
        ):
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        normalized = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        tokens = [token for token in normalized.split() if token]

        drop_tokens = {
            "no",
            "na",
            "nos",
            "nas",
            "do",
            "da",
            "dos",
            "das",
            "de",
            "a",
            "o",
            "as",
            "os",
            "um",
            "uma",
            "uns",
            "umas",
            "e",
            "ou",
            "que",
            "para",
            "por",
            "com",
            "sem",
            "sobre",
            "em",
            "ao",
            "aos",
            "encontra",
            "encontrar",
            "procura",
            "procurar",
            "procuro",
            "mostrar",
            "mostra",
            "dizer",
            "diz",
            "fala",
            "falar",
            "ver",
            "podes",
            "pode",
            "podem",
            "conseguir",
            "consegues",
            "eu",
            "me",
            "mim",
            "tu",
            "te",
            "ti",
            "voce",
            "voces",
            "isto",
            "isso",
            "este",
            "esta",
            "estes",
            "estas",
            "esse",
            "essa",
            "esses",
            "essas",
            "desse",
            "dessa",
            "desses",
            "dessas",
            "neste",
            "nesta",
            "nestes",
            "nestas",
            "nesse",
            "nessa",
            "nesses",
            "nessas",
            "quero",
            "queria",
            "queremos",
            "preciso",
            "precisava",
            "favor",
            "porfavor",
            "ok",
            "okay",
            "nope",
            "nao",
            "sim",
            "entao",
            "bom",
            "ora",
            "tipo",
            "agora",
            "antes",
            "obra",
            "peca",
            "pecas",
            "museu",
            "colecao",
            "colecoes",
            "acervo",
            "artefacto",
            "artefactos",
        }
        drop_tokens = {_fold_token(token) for token in drop_tokens}
        drop_tokens.add(_fold_token((museum_slug or "").strip().lower()))
        if museum_id:
            drop_tokens.add(_fold_token(museum_id.strip().lower()))

        filtered: list[str] = []
        fallback: list[str] = []
        seen_filtered: set[str] = set()
        seen_fallback: set[str] = set()
        for token in tokens:
            folded = _fold_token(token)
            if len(folded) <= 1:
                continue
            if folded not in seen_fallback:
                seen_fallback.add(folded)
                fallback.append(token)
            if folded in drop_tokens:
                continue
            if folded in seen_filtered:
                continue
            seen_filtered.add(folded)
            filtered.append(token)

        if filtered:
            return " ".join(filtered[:12])

        if fallback:
            return " ".join(fallback[:12])

        return ""

    def _derive_context_policy(
        self,
        *,
        message: str,
        state: ChatSessionState,
    ) -> dict[str, object]:
        def _policy(
            *,
            is_follow_up: bool,
            use_history_for_query: bool,
            use_history_for_answer: bool,
            carry_filters: bool,
            carry_sort: bool,
            reason: str,
        ) -> dict[str, object]:
            return {
                "is_follow_up": is_follow_up,
                "use_history_for_query": use_history_for_query,
                "use_history_for_answer": use_history_for_answer,
                "carry_filters": carry_filters,
                "carry_sort": carry_sort,
                "reason": reason,
            }

        text = (message or "").strip().lower()
        if not text:
            return _policy(
                is_follow_up=False,
                use_history_for_query=False,
                use_history_for_answer=False,
                carry_filters=False,
                carry_sort=False,
                reason="empty_message",
            )

        has_previous_context = bool(state.history[:-1] or state.rolling_summary.strip())
        if not has_previous_context:
            return _policy(
                is_follow_up=False,
                use_history_for_query=False,
                use_history_for_answer=False,
                carry_filters=False,
                carry_sort=False,
                reason="no_previous_context",
            )

        standalone_patterns = (
            r"\bnova\s+(pesquisa|busca|consulta|quest[ãa]o|pergunta)\b",
            r"\b(do zero|novo tema|mudar de tema|outro tema|novo pedido)\b",
            r"\bsem rela[cç][ãa]o com (a )?(anterior|ultima)\b",
            r"\bmudando de assunto\b",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in standalone_patterns):
            return _policy(
                is_follow_up=False,
                use_history_for_query=False,
                use_history_for_answer=False,
                carry_filters=False,
                carry_sort=False,
                reason="explicit_new_query",
            )

        if self._extract_tour_scope_expression(text):
            return _policy(
                is_follow_up=False,
                use_history_for_query=False,
                use_history_for_answer=False,
                carry_filters=False,
                carry_sort=False,
                reason="tour_scope_query",
            )

        referential_query_patterns = (
            r"^\s*(e|ent[aã]o|agora)\b",
            r"^\s*(h[aá]|tem|existe[m]?|mostra)\s+mais\b",
            r"\b(deste|desta|desses|dessas|disso|nisso|daquilo)\b",
            r"\b(neste|nesta|nestes|nestas|nesse|nessa|nesses|nessas)\b",
            r"\b(esse|essa|esses|essas)\b",
            r"\b(nele|nela|dele|dela|disto|nisto)\b",
            r"\b(anterior|ultima|acima|mesmo|mesma)\b",
            r"\b(mais recente|mais antigo|mais antiga)\b",
            r"\b(do s[eé]culo|desse s[eé]culo)\b",
            r"\b(s[oó] os|s[oó] as|apenas os|apenas as)\b",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in referential_query_patterns):
            return _policy(
                is_follow_up=True,
                use_history_for_query=True,
                use_history_for_answer=True,
                carry_filters=True,
                carry_sort=True,
                reason="referential_follow_up_query",
            )

        referential_answer_patterns = (
            r"\b(explica melhor|detalha|resume|reformula|desenvolve|continua)\b",
            r"\b(com mais detalhe|mais detalhe)\b",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in referential_answer_patterns):
            return _policy(
                is_follow_up=True,
                use_history_for_query=False,
                use_history_for_answer=True,
                carry_filters=False,
                carry_sort=False,
                reason="referential_follow_up_answer_only",
            )

        return _policy(
            is_follow_up=False,
            use_history_for_query=False,
            use_history_for_answer=False,
            carry_filters=False,
            carry_sort=False,
            reason="standalone_query_default",
        )

    def _doc_inventory(self, doc: dict[str, object]) -> str:
        return str(doc.get("inventory_number") or doc.get("inventory") or "").strip()

    def _doc_search_text(self, doc: dict[str, object]) -> str:
        return str(doc.get("search_text") or doc.get("full_text") or "").strip()

    def _image_hit_inventory(self, hit: dict[str, object]) -> str:
        return str(hit.get("inventory_number") or hit.get("inventory") or "").strip()

    def _image_match_name(self, hit: dict[str, object]) -> str:
        local_path = str(hit.get("local_path") or "").strip()
        if local_path:
            return local_path.replace("\\", "/").strip("/")

        original_image_name = str(hit.get("original_image_name") or "").strip()
        if original_image_name:
            return original_image_name

        image_id = str(hit.get("image_id") or hit.get("id") or "").strip()
        if image_id:
            return f"{image_id.replace(':', '_')}.jpg"
        return ""

    def _format_docs_for_prompt(self, *, docs: list[dict[str, object]], top_k: int) -> str:
        snippets: list[str] = []
        excluded_keys = {
            "artifact_id",
            "search_text",
            "full_text",
            "creator_ids",
            "set_ids",
            "set_numbers",
            "exhibition_ids",
            "exhibition_types",
            "bibliography",
            "group_artifact_ids",
        }
        if not self.settings.CHAT_INCLUDE_ORIGIN_HISTORY_IN_LLM_CONTEXT:
            excluded_keys.update({"origin_history", "historical_origin"})
        for index, doc in enumerate(docs[: max(top_k, 1)], start=1):
            prompt_doc = {
                key: value
                for key, value in doc.items()
                if key not in excluded_keys and value not in (None, "")
            }
            if not isinstance(prompt_doc.get("group_size"), int) or prompt_doc["group_size"] <= 1:
                prompt_doc.pop("group_size", None)
                prompt_doc.pop("group_inventory_numbers", None)
            snippet = json.dumps(prompt_doc, ensure_ascii=True)
            snippets.append(f"[doc_{index}] {snippet}")
        return "\n".join(snippets)

    _COMPACT_CARD_DESCRIPTION_CHARS = 280

    def _format_docs_compact(
        self,
        *,
        docs: list[dict[str, object]],
        description_chars: int | None = None,
    ) -> str:
        """Cartões compactos para o prompt final (quick win PR-01).

        Um bloco por cartão, rotulado [doc_N] como o formato histórico, com os
        campos que um visitante pergunta: título, inventário, categoria, data,
        materiais/técnica, autor, descrição truncada e se está na visita.
        Nada de artifact_id, image_paths, sets/exhibitions ids ou search_text.
        `description_chars` alarga a descrição quando o cartão é um só
        (apresentação de uma peça).
        """
        limit = max(int(description_chars), 1) if description_chars else self._COMPACT_CARD_DESCRIPTION_CHARS

        def text(value: object) -> str:
            if isinstance(value, list):
                return ", ".join(str(item).strip() for item in value if str(item).strip())
            return re.sub(r"\s+", " ", str(value or "")).strip()

        def truncate(value: str) -> str:
            if len(value) <= limit:
                return value
            cut = value[: limit + 1]
            cut = cut[: cut.rfind(" ")] if " " in cut else cut[:limit]
            return cut.rstrip(" ,;:") + "…"

        cards: list[str] = []
        for index, doc in enumerate(docs, start=1):
            title = text(doc.get("title")) or "sem título"
            inventory = text(doc.get("inventory_number")) or text(doc.get("inventory"))
            category = text(doc.get("category"))
            super_category = text(doc.get("super_category"))
            if super_category and super_category.casefold() != category.casefold():
                category = f"{category} ({super_category})" if category else super_category
            creators = text(doc.get("creators")) or text(doc.get("creator"))
            materials = text(doc.get("support_or_material"))
            technique = text(doc.get("technique"))
            date = text(doc.get("date_or_period"))
            if not date:
                start, end = doc.get("start_year"), doc.get("end_year")
                if isinstance(start, int) and isinstance(end, int) and start != end:
                    date = f"{start}–{end}"
                elif isinstance(start, int):
                    date = str(start)
            description = truncate(text(doc.get("description")))
            production = text(doc.get("production_center"))
            in_tour = doc.get("in_tour")
            museum = text(doc.get("museum"))

            lines = [f"[doc_{index}] Título: {title} | Inventário: {inventory or 'não indicado'}"]
            facts = [
                f"Categoria: {category or 'não indicada'}",
                f"Data: {date or 'não indicada'}",
            ]
            if materials:
                facts.append(f"Materiais: {materials}")
            if technique:
                facts.append(f"Técnica: {technique}")
            if creators:
                facts.append(f"Autor: {creators}")
            if production:
                facts.append(f"Centro de fabrico: {production}")
            if museum:
                facts.append(f"Museu: {museum}")
            lines.append("    " + " | ".join(facts))
            if description:
                lines.append(f"    Descrição: {description}")
            if isinstance(in_tour, bool):
                lines.append(f"    Na visita virtual: {'sim' if in_tour else 'não'}")
            group_size = doc.get("group_size")
            if isinstance(group_size, int) and group_size > 1:
                numbers = self._as_string_list(doc.get("group_inventory_numbers"))
                span = f" ({numbers[0]} a {numbers[-1]})" if len(numbers) >= 2 else ""
                lines.append(
                    f"    Conjunto de {group_size} peças iguais, uma ficha cada{span}; este cartão representa todas"
                )
            cards.append("\n".join(lines))
        return "\n".join(cards)

    def _as_string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                items.append(text)
        return items

    async def _build_artifact_results(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_docs: list[dict[str, object]],
        max_images_per_artifact: int | None = None,
        artifact_image_hits: list[dict[str, Any]] | None = None,
    ) -> list[ArtifactResult]:
        if not artifact_docs:
            return []

        artifact_ids = self._extract_artifact_ids_from_docs(artifact_docs)
        image_hits: list[dict[str, Any]] = list(artifact_image_hits or [])
        image_limit = (
            max(int(max_images_per_artifact), 0)
            if max_images_per_artifact is not None
            else None
        )
        if artifact_ids and image_limit != 0 and artifact_image_hits is None:
            if image_limit is None:
                per_artifact = 1
                total_expected = 0
                for doc in artifact_docs:
                    raw_count = doc.get("image_count")
                    if isinstance(raw_count, str) and raw_count.strip().isdigit():
                        count = int(raw_count.strip())
                    elif isinstance(raw_count, (int, float)):
                        count = int(raw_count)
                    else:
                        count = 0
                    count = max(count, 1)
                    per_artifact = max(per_artifact, count)
                    total_expected += count

                per_artifact = min(per_artifact, 120)
                max_total = min(max(total_expected, len(artifact_ids)), 1000)
            else:
                per_artifact = max(image_limit, 1)
                max_total = max(len(artifact_ids) * per_artifact, 1)

            try:
                image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    artifact_ids=artifact_ids,
                    per_artifact=per_artifact,
                    max_total=max_total,
                )
            except Exception as exc:
                image_hits = []

        images_by_artifact: dict[str, list[dict[str, Any]]] = {}
        for hit in image_hits:
            artifact_id = str(hit.get("artifact_id") or "").strip()
            if not artifact_id:
                continue
            images_by_artifact.setdefault(artifact_id, []).append(hit)

        results: list[ArtifactResult] = []
        for doc in artifact_docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id:
                continue

            images: list[ArtifactImageResult] = []
            seen_keys: set[str] = set()

            def _append_image(
                *,
                image_id: str | None = None,
                image_order: int | None = None,
                local_path: str | None = None,
                source_url: str | None = None,
                caption: str | None = None,
                alt_text: str | None = None,
                original_image_name: str | None = None,
            ) -> None:
                normalized_image_id = str(image_id or "").strip() or None
                normalized_local_path = str(local_path or "").strip().replace("\\", "/") or None
                normalized_source_url = str(source_url or "").strip() or None
                normalized_original_name = str(original_image_name or "").strip() or None
                if normalized_original_name is None and normalized_local_path:
                    normalized_original_name = normalized_local_path.strip("/")
                if normalized_original_name is None and normalized_source_url:
                    normalized_original_name = Path(
                        urlparse(normalized_source_url).path
                    ).name.strip() or None
                if normalized_original_name is None and normalized_image_id:
                    normalized_original_name = f"{normalized_image_id.replace(':', '_')}.jpg"

                dedupe_key = (
                    normalized_image_id
                    or normalized_local_path
                    or normalized_source_url
                    or normalized_original_name
                )
                if image_limit is not None and len(images) >= image_limit:
                    return
                if not dedupe_key:
                    return
                if dedupe_key in seen_keys:
                    return
                seen_keys.add(dedupe_key)

                images.append(
                    ArtifactImageResult(
                        original_image_name=normalized_original_name,
                        image_id=normalized_image_id,
                        image_order=image_order,
                        local_path=normalized_local_path,
                        source_url=normalized_source_url,
                        caption=str(caption or "").strip() or None,
                        alt_text=str(alt_text or "").strip() or None,
                    )
                )

            for hit in images_by_artifact.get(artifact_id, []):
                _append_image(
                    image_id=str(hit.get("image_id") or hit.get("id") or "").strip() or None,
                    image_order=(
                        hit.get("image_order")
                        if isinstance(hit.get("image_order"), int)
                        else None
                    ),
                    local_path=str(hit.get("local_path") or "").strip() or None,
                    source_url=str(hit.get("source_url") or "").strip() or None,
                    caption=str(hit.get("caption") or "").strip() or None,
                    alt_text=str(hit.get("alt_text") or "").strip() or None,
                    original_image_name=str(hit.get("original_image_name") or "").strip() or None,
                )

            image_paths = self._as_string_list(doc.get("image_paths"))
            image_urls = self._as_string_list(doc.get("image_urls"))
            image_ids = self._as_string_list(doc.get("image_ids"))
            max_len = max(len(image_paths), len(image_urls), len(image_ids))
            for index in range(max_len):
                _append_image(
                    image_id=image_ids[index] if index < len(image_ids) else None,
                    local_path=image_paths[index] if index < len(image_paths) else None,
                    source_url=image_urls[index] if index < len(image_urls) else None,
                )

            raw_image_count = doc.get("image_count")
            image_count_value: int | None = None
            if isinstance(raw_image_count, str) and raw_image_count.strip().isdigit():
                image_count_value = int(raw_image_count.strip())
            elif isinstance(raw_image_count, (int, float)):
                image_count_value = int(raw_image_count)
            if image_count_value is not None and image_count_value < 0:
                image_count_value = None

            # --- Campos relacionais (export relacional do RAIZ) ---
            def _as_str_list(value: Any) -> list[str]:
                if not isinstance(value, list):
                    return []
                out: list[str] = []
                for item in value:
                    if item is None:
                        continue
                    text = str(item).strip()
                    if text:
                        out.append(text)
                return out

            def _as_int(value: Any) -> int | None:
                if isinstance(value, bool):
                    return None
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
                if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                    return int(value.strip())
                return None

            creators_list = _as_str_list(doc.get("creators"))
            legacy_creator = str(doc.get("creator") or "").strip()
            if not creators_list and legacy_creator:
                creators_list = [legacy_creator]
            creator_value = legacy_creator or (creators_list[0] if creators_list else None)

            raw_in_tour = doc.get("in_tour")
            in_tour_bool = bool(raw_in_tour) if not isinstance(raw_in_tour, str) else raw_in_tour.strip().lower() in {"true", "1", "yes"}

            results.append(
                ArtifactResult(
                    artifact_id=artifact_id,
                    tipo_inventario=str(doc.get("tipo_inventario") or "").strip() or None,
                    inventory_number=self._doc_inventory(doc) or None,
                    title=str(doc.get("title") or "").strip() or None,
                    museum_id=str(doc.get("museum_id") or "").strip() or None,
                    museum=str(doc.get("museum") or doc.get("museum_name") or "").strip() or None,
                    category=str(doc.get("category") or "").strip() or None,
                    super_category=str(doc.get("super_category") or "").strip() or None,
                    creator=creator_value,
                    creators=creators_list,
                    creator_ids=_as_str_list(doc.get("creator_ids")),
                    date_or_period=str(doc.get("date_or_period") or "").strip() or None,
                    start_year=_as_int(doc.get("start_year")),
                    end_year=_as_int(doc.get("end_year")),
                    support_or_material=str(
                        doc.get("support_or_material") or doc.get("support") or ""
                    ).strip()
                    or None,
                    technique=str(doc.get("technique") or "").strip() or None,
                    origin_history=str(
                        doc.get("origin_history") or doc.get("historical_origin") or ""
                    ).strip()
                    or None,
                    incorporation=str(doc.get("incorporation") or "").strip() or None,
                    production_center=str(
                        doc.get("production_center")
                        or doc.get("manufacturer_location")
                        or doc.get("location")
                        or ""
                    ).strip()
                    or None,
                    description=str(doc.get("description") or "").strip() or None,
                    search_text=self._doc_search_text(doc) or None,
                    detail_type=str(doc.get("detail_type") or "").strip() or None,
                    detail_url=str(doc.get("detail_url") or "").strip() or None,
                    in_tour=in_tour_bool,
                    sets=_as_str_list(doc.get("sets")),
                    set_ids=_as_str_list(doc.get("set_ids")),
                    set_numbers=_as_str_list(doc.get("set_numbers")),
                    exhibitions=_as_str_list(doc.get("exhibitions")),
                    exhibition_ids=_as_str_list(doc.get("exhibition_ids")),
                    exhibition_types=_as_str_list(doc.get("exhibition_types")),
                    exhibition_count=_as_int(doc.get("exhibition_count")),
                    bibliography=str(doc.get("bibliography") or "").strip() or None,
                    bibliography_count=_as_int(doc.get("bibliography_count")),
                    image_count=image_count_value,
                    images=images,
                    group_size=max(_as_int(doc.get("group_size")) or 1, 1),
                    group_inventory_numbers=_as_str_list(doc.get("group_inventory_numbers")),
                    group_artifact_ids=_as_str_list(doc.get("group_artifact_ids")),
                )
            )

        return results

    async def get_artifact_full(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_id: str,
    ) -> ArtifactResult | None:
        """Devolve um ArtifactResult completo (com imagens) por id, ou None."""
        artifact_id = (artifact_id or "").strip()
        if not artifact_id:
            return None
        docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_ids=[artifact_id],
            top_k=1,
        )
        if not docs:
            return None
        results = await self._build_artifact_results(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_docs=docs,
        )
        return results[0] if results else None

    async def get_artifact_full_by_inventory(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        inventory_number: str,
    ) -> ArtifactResult | None:
        """Devolve um ArtifactResult completo por numero de inventario, ou None."""
        inventory_number = (inventory_number or "").strip()
        if not inventory_number:
            return None
        # A visita anuncia o número que lá está escrito, que pode não ser o do
        # catálogo («MNAz 6991 Az» na visita, «MNAz 6991/2 Az» no catálogo). Os
        # ficheiros da visita não se podem corrigir (o player passa a mostrar a
        # marca de água do 3DVista), por isso a correspondência está no mapa
        # dos POIs (`inventoryIdsOriginal` -> `inventoryIds`) e aplica-se aqui.
        try:
            corrected = self.tour_navigation_service.corrected_inventories(
                museum_slug=museum_slug, museum_id=museum_id, inventory=inventory_number
            )
        except Exception:  # pragma: no cover - tradução auxiliar; nunca bloqueia
            corrected = []
        if corrected:
            inventory_number = corrected[0]
        docs = await self.opensearch_gateway.fetch_artifacts_by_inventory_numbers(
            museum_slug=museum_slug,
            museum_id=museum_id,
            inventory_numbers=[inventory_number],
            top_k=1,
        )
        if not docs:
            docs = await self.opensearch_gateway.search_artifacts_by_inventory_candidates(
                museum_slug=museum_slug,
                museum_id=museum_id,
                inventory_numbers=[inventory_number],
                top_k=1,
            )
        if not docs:
            return None
        results = await self._build_artifact_results(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_docs=docs,
        )
        return results[0] if results else None

    def _build_related_artifact_images(self, art: dict[str, Any]) -> list[ArtifactImageResult]:
        related_images: list[ArtifactImageResult] = []
        image_paths = self._as_string_list(art.get("image_paths"))
        image_urls = self._as_string_list(art.get("image_urls"))
        image_ids = self._as_string_list(art.get("image_ids") or art.get("image_file_ids"))
        max_len = min(max(len(image_paths), len(image_urls), len(image_ids)), 1)
        for index in range(max_len):
            image_id = image_ids[index] if index < len(image_ids) else None
            local_path = image_paths[index] if index < len(image_paths) else None
            source_url = image_urls[index] if index < len(image_urls) else None
            original_name = local_path
            if not original_name and image_id:
                original_name = f"{image_id.replace(':', '_')}.jpg"
            if original_name or image_id or local_path or source_url:
                related_images.append(
                    ArtifactImageResult(
                        original_image_name=original_name,
                        image_id=image_id,
                        local_path=local_path,
                        source_url=source_url,
                    )
                )
        return related_images

    async def _build_related_images_lookup(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        related_docs: list[dict[str, Any]],
    ) -> dict[str, list[ArtifactImageResult]]:
        if not related_docs:
            return {}

        artifact_ids = self._extract_artifact_ids_from_docs(related_docs)
        image_hits: list[dict[str, Any]] = []
        if artifact_ids:
            try:
                image_hits = await self.opensearch_gateway.fetch_images_by_artifact_ids(
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    artifact_ids=artifact_ids,
                    per_artifact=1,
                    max_total=max(len(artifact_ids), 1),
                )
            except Exception as exc:
                image_hits = []

        artifact_results = await self._build_artifact_results(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_docs=related_docs,
            max_images_per_artifact=1,
            artifact_image_hits=image_hits,
        )
        return {
            result.artifact_id: result.images
            for result in artifact_results
            if result.artifact_id
        }

    def _build_navigation_lookup_for_related(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        related_docs: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        inventories: list[str] = []
        for art in related_docs:
            inv = str(art.get("inventory_number") or "").strip()
            if inv:
                inventories.append(inv)

        navigation_lookup: dict[str, dict[str, Any]] = {}
        if not inventories:
            return navigation_lookup

        try:
            navigation_targets_raw = self.tour_navigation_service.resolve_targets(
                museum_slug=museum_slug,
                museum_id=museum_id,
                inventories=inventories,
                limit=max(len(inventories), 1),
            )
        except Exception:
            return navigation_lookup

        for tgt in navigation_targets_raw:
            inv_id = str(tgt.get("inventory_id") or "").strip()
            if inv_id and inv_id not in navigation_lookup:
                navigation_lookup[inv_id] = tgt
        return navigation_lookup

    def _build_related_item(
        self,
        art: dict[str, Any],
        *,
        navigation_lookup: dict[str, dict[str, Any]],
        images_by_artifact: dict[str, list[ArtifactImageResult]] | None = None,
    ) -> RelatedArtifactItem:
        artifact_id = str(art.get("artifact_id") or "")
        inv = str(art.get("inventory_number") or "").strip() or None
        nav = navigation_lookup.get(inv or "") if inv else None
        nav_target = None
        if nav:
            nav_target = TourNavigationTarget(
                overlay_id=str(nav.get("overlay_id") or ""),
                panorama_key=str(nav.get("panorama_key") or ""),
                inventory_id=str(nav.get("inventory_id") or ""),
                location=nav.get("location"),
                title=nav.get("title"),
            )
        related_images = (
            (images_by_artifact or {}).get(artifact_id)
            or self._build_related_artifact_images(art)
        )
        return RelatedArtifactItem(
            artifact_id=artifact_id,
            inventory_number=inv,
            title=str(art.get("title") or "").strip() or None,
            museum_id=str(art.get("museum_id") or "").strip() or None,
            museum=str(art.get("museum") or "").strip() or None,
            category=str(art.get("category") or "").strip() or None,
            creators=list(art.get("creators") or []),
            date_or_period=str(art.get("date_or_period") or "").strip() or None,
            detail_type=str(art.get("detail_type") or "").strip() or None,
            detail_url=str(art.get("detail_url") or "").strip() or None,
            in_tour=bool(art.get("in_tour")),
            image_count=art.get("image_count") if isinstance(art.get("image_count"), int) else None,
            images=related_images,
            navigation_target=nav_target,
        )

    async def get_related_artifacts_page(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_id: str,
        tipo: str,
        entity_id: str,
        offset: int,
        limit: int,
    ) -> RelatedArtifactsPageResponse:
        cleaned_tipo = (tipo or "").strip()
        if cleaned_tipo not in {"conjunto", "exposicao"}:
            raise ValueError("tipo invalido.")
        cleaned_entity_id = (entity_id or "").strip()
        cleaned_artifact_id = (artifact_id or "").strip()
        resolved_offset = max(int(offset), 0)
        resolved_limit = max(1, min(int(limit), 50))

        related, total = await self.opensearch_gateway.fetch_artifacts_by_entity(
            tipo=cleaned_tipo,
            entity_id=cleaned_entity_id,
            museum_slug=museum_slug,
            museum_id=museum_id,
            top_k=resolved_limit,
            from_offset=resolved_offset,
            exclude_artifact_id=cleaned_artifact_id or None,
        )
        navigation_lookup = await asyncio.to_thread(
            self._build_navigation_lookup_for_related,
            museum_slug=museum_slug,
            museum_id=museum_id,
            related_docs=related,
        )
        images_by_artifact = await self._build_related_images_lookup(
            museum_slug=museum_slug,
            museum_id=museum_id,
            related_docs=related,
        )
        artifacts = [
            self._build_related_item(
                art,
                navigation_lookup=navigation_lookup,
                images_by_artifact=images_by_artifact,
            )
            for art in related
        ]
        return RelatedArtifactsPageResponse(
            artifact_id=cleaned_artifact_id,
            tipo=cleaned_tipo,  # type: ignore[arg-type]
            entity_id=cleaned_entity_id,
            artifacts=artifacts,
            artifacts_offset=resolved_offset,
            artifacts_limit=resolved_limit,
            artifacts_total=max(int(total), 0),
            artifacts_has_more=resolved_offset + len(artifacts) < max(int(total), 0),
        )

    # ------------------------------------------------------------------ #
    # Modal de detalhe: autores + conjuntos + exposicoes do artefacto.
    # ------------------------------------------------------------------ #
    async def get_artifact_detail_context(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        artifact_id: str,
    ) -> ArtifactDetailContextResponse:
        """Devolve, para um artefacto, info expandida de autores/conjuntos/exposicoes
        + listas de outros artefactos ligados a cada conjunto/exposicao."""
        artifact_id = (artifact_id or "").strip()
        if not artifact_id:
            return ArtifactDetailContextResponse(artifact_id="")

        # 1) Obter o artifact_doc para saber os ids das entidades a buscar.
        artifact_docs = await self.opensearch_gateway.fetch_artifacts_by_ids(
            museum_slug=museum_slug,
            museum_id=museum_id,
            artifact_ids=[artifact_id],
            top_k=1,
        )
        if not artifact_docs:
            return ArtifactDetailContextResponse(artifact_id=artifact_id)
        doc = artifact_docs[0]

        def _as_list(value: Any) -> list[str]:
            if isinstance(value, str):
                return [
                    item.strip()
                    for item in re.split(r"[;,\n|]+", value)
                    if item.strip()
                ]
            if not isinstance(value, list):
                return []
            return [str(v).strip() for v in value if str(v or "").strip()]

        def _entity_lookup_ids(tipo: str, values: list[str]) -> list[str]:
            prefix = f"{tipo}:"
            candidates: list[str] = []
            seen: set[str] = set()
            for value in values:
                cleaned = str(value or "").strip()
                if not cleaned:
                    continue
                variants = (
                    [cleaned, cleaned[len(prefix) :]]
                    if cleaned.startswith(prefix)
                    else [f"{prefix}{cleaned}", cleaned]
                )
                for variant in variants:
                    normalized = variant.strip()
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    candidates.append(normalized)
            return candidates

        creator_ids = _as_list(doc.get("creator_ids"))
        set_ids = _as_list(doc.get("set_ids"))
        exhibition_ids = _as_list(doc.get("exhibition_ids"))

        # Conjuntos/exposicoes usam entity_id prefixado no indice relacional. Autores
        # sao mais simples: creator_ids ja contem o _id do indice cultural_heritage_authors.
        prefixed_sets = _entity_lookup_ids("conjunto", set_ids)
        prefixed_exhibitions = _entity_lookup_ids("exposicao", exhibition_ids)

        top_k_related = int(getattr(self.settings, "CHAT_RELATED_ARTIFACTS_PAGE_SIZE", 10) or 10)

        # 2) Buscar metainfo de cada entidade em paralelo.
        authors_task = self.opensearch_gateway.fetch_authors_by_ids(
            author_ids=creator_ids,
        ) if creator_ids else asyncio.sleep(0, result=[])
        sets_task = self.opensearch_gateway.fetch_entities_by_ids(
            tipo="conjunto", entity_ids=prefixed_sets,
        ) if prefixed_sets else asyncio.sleep(0, result=[])
        exhibitions_task = self.opensearch_gateway.fetch_entities_by_ids(
            tipo="exposicao", entity_ids=prefixed_exhibitions,
        ) if prefixed_exhibitions else asyncio.sleep(0, result=[])

        authors_docs, sets_docs, exhibitions_docs = await asyncio.gather(
            authors_task, sets_task, exhibitions_task,
            return_exceptions=False,
        )

        # 3) Para cada conjunto/exposicao, buscar artefactos relacionados em paralelo.
        set_related_tasks = [
            self.opensearch_gateway.fetch_artifacts_by_entity(
                tipo="conjunto",
                entity_id=str(s.get("entity_id") or ""),
                museum_slug=museum_slug,
                museum_id=museum_id,
                top_k=top_k_related,
                exclude_artifact_id=artifact_id,
            )
            for s in sets_docs
        ]
        exhibition_related_tasks = [
            self.opensearch_gateway.fetch_artifacts_by_entity(
                tipo="exposicao",
                entity_id=str(e.get("entity_id") or ""),
                museum_slug=museum_slug,
                museum_id=museum_id,
                top_k=top_k_related,
                exclude_artifact_id=artifact_id,
            )
            for e in exhibitions_docs
        ]
        all_related = await asyncio.gather(
            *(set_related_tasks + exhibition_related_tasks), return_exceptions=False
        )
        set_related = all_related[: len(set_related_tasks)]
        exhibition_related = all_related[len(set_related_tasks) :]

        # 4) Resolver imagens e navigation_targets dos artefactos relacionados (em massa).
        all_related_docs: list[dict[str, Any]] = []
        all_inventories: list[str] = []
        for related, _total in list(set_related) + list(exhibition_related):
            all_related_docs.extend(related)
            for art in related:
                inv = str(art.get("inventory_number") or "").strip()
                if inv:
                    all_inventories.append(inv)
        images_by_artifact = await self._build_related_images_lookup(
            museum_slug=museum_slug,
            museum_id=museum_id,
            related_docs=all_related_docs,
        )
        navigation_lookup: dict[str, dict[str, Any]] = {}
        if all_inventories:
            try:
                navigation_targets_raw = await asyncio.to_thread(
                    self.tour_navigation_service.resolve_targets,
                    museum_slug=museum_slug,
                    museum_id=museum_id,
                    inventories=all_inventories,
                    limit=max(len(all_inventories), 1),
                )
            except Exception:
                navigation_targets_raw = []
            for tgt in navigation_targets_raw:
                inv_id = str(tgt.get("inventory_id") or "").strip()
                if inv_id and inv_id not in navigation_lookup:
                    navigation_lookup[inv_id] = tgt

        # 5) Construir resposta.
        authors_response: list[AuthorEntity] = []
        for a in authors_docs:
            biography = str(a.get("biografia") or a.get("biography") or "").strip() or None
            authors_response.append(
                AuthorEntity(
                    entity_id=str(a.get("entity_id") or ""),
                    name=str(a.get("name") or "").strip() or None,
                    atividade=str(a.get("atividade") or "").strip() or None,
                    data_nascimento=str(a.get("data_nascimento") or "").strip() or None,
                    data_obito=str(a.get("data_obito") or "").strip() or None,
                    local_nascimento=str(a.get("local_nascimento") or "").strip() or None,
                    local_obito=str(a.get("local_obito") or "").strip() or None,
                    biografia=biography,
                    biography=biography,
                    url=str(a.get("url") or "").strip() or None,
                    n_objetos=a.get("n_objetos") if isinstance(a.get("n_objetos"), int) else None,
                )
            )

        sets_response: list[SetEntityWithArtifacts] = []
        for s, (related, total) in zip(sets_docs, set_related):
            sets_response.append(
                SetEntityWithArtifacts(
                    entity_id=str(s.get("entity_id") or ""),
                    name=str(s.get("name") or "").strip() or None,
                    num_conjunto=str(s.get("num_conjunto") or "").strip() or None,
                    historial=str(s.get("historial") or "").strip() or None,
                    descricao=str(s.get("descricao") or "").strip() or None,
                    url=str(s.get("url") or "").strip() or None,
                    n_objetos=total,
                    artifacts=[
                        self._build_related_item(
                            a,
                            navigation_lookup=navigation_lookup,
                            images_by_artifact=images_by_artifact,
                        )
                        for a in related
                    ],
                    artifacts_returned=len(related),
                )
            )

        exhibitions_response: list[ExhibitionEntityWithArtifacts] = []
        for e, (related, total) in zip(exhibitions_docs, exhibition_related):
            exhibitions_response.append(
                ExhibitionEntityWithArtifacts(
                    entity_id=str(e.get("entity_id") or ""),
                    name=str(e.get("name") or "").strip() or None,
                    tipo_exposicao=str(e.get("tipo_exposicao") or "").strip() or None,
                    local=str(e.get("local") or "").strip() or None,
                    ano_inicial=e.get("ano_inicial") if isinstance(e.get("ano_inicial"), int) else None,
                    ano_final=e.get("ano_final") if isinstance(e.get("ano_final"), int) else None,
                    texto=str(e.get("texto") or "").strip() or None,
                    ficha_tecnica=str(e.get("ficha_tecnica") or "").strip() or None,
                    url=str(e.get("url") or "").strip() or None,
                    n_objetos=total,
                    artifacts=[
                        self._build_related_item(
                            a,
                            navigation_lookup=navigation_lookup,
                            images_by_artifact=images_by_artifact,
                        )
                        for a in related
                    ],
                    artifacts_returned=len(related),
                )
            )

        return ArtifactDetailContextResponse(
            artifact_id=artifact_id,
            authors=authors_response,
            sets=sets_response,
            exhibitions=exhibitions_response,
        )

    def _build_image_matches(
        self,
        *,
        image_hits: list[dict[str, object]],
        artifact_docs: list[dict[str, object]],
    ) -> list[ImageMatchResult]:
        doc_by_inventory: dict[str, dict[str, object]] = {}
        ordered_inventories: list[str] = []
        doc_by_id: dict[str, dict[str, object]] = {}
        ordered_artifact_ids: list[str] = []
        for doc in artifact_docs:
            inventory = self._doc_inventory(doc)
            if inventory:
                inventory_key = inventory.casefold()
                doc_by_inventory[inventory_key] = doc
                if inventory_key not in ordered_inventories:
                    ordered_inventories.append(inventory_key)

            artifact_id = str(doc.get("artifact_id") or "").strip()
            if artifact_id:
                doc_by_id[artifact_id] = doc
                if artifact_id not in ordered_artifact_ids:
                    ordered_artifact_ids.append(artifact_id)

        best_hit_by_inventory: dict[str, dict[str, object]] = {}
        best_hit_by_artifact: dict[str, dict[str, object]] = {}

        def register_best(
            bucket: dict[str, dict[str, object]],
            key: str,
            hit: dict[str, object],
            score_value: float,
            original_image_name: str,
        ) -> None:
            current_best = bucket.get(key)
            if current_best is None:
                bucket[key] = {
                    "hit": hit,
                    "score_value": score_value,
                    "original_image_name": original_image_name,
                }
                return

            current_score = current_best.get("score_value")
            current_score_value = (
                float(current_score)
                if isinstance(current_score, (int, float))
                else float("-inf")
            )
            if score_value > current_score_value:
                bucket[key] = {
                    "hit": hit,
                    "score_value": score_value,
                    "original_image_name": original_image_name,
                }

        for hit in image_hits:
            original_image_name = self._image_match_name(hit)
            if not original_image_name:
                continue
            inventory = self._image_hit_inventory(hit)
            artifact_id = str(hit.get("artifact_id") or "").strip()
            score_raw = hit.get("score")
            score_value = float(score_raw) if isinstance(score_raw, (int, float)) else float("-inf")
            if inventory:
                register_best(
                    best_hit_by_inventory,
                    inventory.casefold(),
                    hit,
                    score_value,
                    original_image_name,
                )
            if artifact_id:
                register_best(
                    best_hit_by_artifact,
                    artifact_id,
                    hit,
                    score_value,
                    original_image_name,
                )

        def build_match(
            *,
            best_entry: dict[str, object],
            doc: dict[str, object],
        ) -> ImageMatchResult | None:
            hit_obj = best_entry.get("hit")
            if not isinstance(hit_obj, dict):
                return None
            hit: dict[str, object] = hit_obj
            original_image_name = str(best_entry.get("original_image_name") or "").strip()
            if not original_image_name:
                original_image_name = self._image_match_name(hit)
            if not original_image_name:
                return None

            score_raw = hit.get("score")
            score: float | None = None
            if isinstance(score_raw, (int, float)):
                score = float(score_raw)

            return ImageMatchResult(
                original_image_name=original_image_name,
                artifact_id=str(doc.get("artifact_id") or hit.get("artifact_id") or "").strip()
                or None,
                score=score,
                title=str(doc.get("title") or hit.get("artifact_title") or "").strip() or None,
                inventory=self._doc_inventory(doc) or self._image_hit_inventory(hit) or None,
                image_id=str(hit.get("image_id") or hit.get("id") or "").strip() or None,
                local_path=str(hit.get("local_path") or "").strip() or None,
                source_url=str(hit.get("source_url") or "").strip() or None,
            )

        matches: list[ImageMatchResult] = []
        for inventory_key in ordered_inventories:
            best_entry = best_hit_by_inventory.get(inventory_key)
            if not isinstance(best_entry, dict):
                continue
            doc = doc_by_inventory.get(inventory_key, {})
            match = build_match(best_entry=best_entry, doc=doc)
            if match is not None:
                matches.append(match)

        if matches:
            return matches

        for artifact_id in ordered_artifact_ids:
            best_entry = best_hit_by_artifact.get(artifact_id)
            if not isinstance(best_entry, dict):
                continue
            doc = doc_by_id.get(artifact_id, {})
            match = build_match(best_entry=best_entry, doc=doc)
            if match is not None:
                matches.append(match)
        return matches

    def _enrich_image_matches(
        self,
        *,
        context: str,
        conversation_id: str,
        museum_slug: str,
        image_matches: list[ImageMatchResult],
        artifact_results: list[ArtifactResult],
        navigation_targets: list[TourNavigationTarget],
    ) -> list[ImageMatchResult]:
        if not image_matches:
            return []

        artifact_by_id = {
            artifact.artifact_id: artifact
            for artifact in artifact_results
            if artifact.artifact_id
        }
        artifact_by_inventory = {
            str(artifact.inventory_number or "").strip().casefold(): artifact
            for artifact in artifact_results
            if str(artifact.inventory_number or "").strip()
        }
        navigation_by_inventory = {
            str(target.inventory_id or "").strip().casefold(): target
            for target in navigation_targets
            if str(target.inventory_id or "").strip()
        }

        enriched: list[ImageMatchResult] = []
        for match in image_matches:
            artifact_id = str(match.artifact_id or "").strip()
            inventory = str(match.inventory or "").strip()
            inventory_key = inventory.casefold()
            artifact = artifact_by_inventory.get(inventory_key) if inventory_key else None
            if artifact is None and artifact_id:
                artifact = artifact_by_id.get(artifact_id)

            target: TourNavigationTarget | None = None
            if inventory_key:
                target = navigation_by_inventory.get(inventory_key)
            if target is None and artifact is not None:
                artifact_inventory = str(artifact.inventory_number or "").strip()
                if artifact_inventory:
                    target = navigation_by_inventory.get(artifact_inventory.casefold())

            enriched.append(
                match.model_copy(
                    update={
                        "artifact": artifact.model_dump(exclude_none=True) if artifact else None,
                        "navigation_target": target.model_dump(exclude_none=True) if target else None,
                    }
                )
            )

        return enriched

    def _resolve_navigation_targets(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        docs: list[dict[str, object]],
    ) -> list[TourNavigationTarget]:
        inventories: list[str] = []
        for doc in docs:
            inventory = self._doc_inventory(doc)
            if inventory:
                inventories.append(inventory)

        if not inventories:
            return []

        resolved = self.tour_navigation_service.resolve_targets(
            museum_slug=museum_slug,
            museum_id=museum_id,
            inventories=inventories,
            limit=max(len(inventories), 1),
        )

        targets: list[TourNavigationTarget] = []
        for item in resolved:
            overlay_id = str(item.get("overlay_id") or "").strip()
            panorama_key = str(item.get("panorama_key") or "").strip()
            inventory_id = str(item.get("inventory_id") or "").strip()
            if not overlay_id or not panorama_key or not inventory_id:
                continue
            targets.append(
                TourNavigationTarget(
                    overlay_id=overlay_id,
                    panorama_key=panorama_key,
                    inventory_id=inventory_id,
                    location=str(item.get("location") or "").strip() or None,
                    title=str(item.get("title") or "").strip() or None,
                )
            )
        return targets

    # ----- O modelo escolhe os cartões (CHAT_MODEL_SELECTS_CARDS) ----- #

    @staticmethod
    def _split_card_selection(text: str) -> tuple[str, list[int] | None]:
        """Retira o bloco «[[cartões: …]]» da resposta.

        Devolve o texto sem o bloco e os rótulos dos cartões (1-based, pela
        ordem do modelo, sem repetidos); `None` quando não há bloco e `[]`
        quando o modelo diz «nenhum». Um bloco cortado pelo limite de tokens
        («[[cartões: doc_1, doc_» no fim) também conta.
        """
        raw = text or ""
        bodies = [match.group(1) for match in _CARD_SELECTION_BLOCK.finditer(raw)]
        cleaned = _CARD_SELECTION_BLOCK.sub("", raw)
        tail = _CARD_SELECTION_OPEN_TAIL.search(cleaned)
        if tail is not None:
            bodies.append(tail.group(1))
            cleaned = cleaned[: tail.start()]
        if not bodies:
            return raw, None
        indices: list[int] = []
        body = " ".join(bodies)
        # Só os rótulos «doc_N»; números soltos apenas se não houver rótulos
        # (um inventário citado no bloco, «MNAz 12 Az», não é um cartão).
        labelled = re.findall(r"doc[_\s-]*(\d{1,3})", body, flags=re.IGNORECASE)
        for number in labelled or re.findall(r"(?<![\w/.-])(\d{1,3})(?![\w/.-])", body):
            index = int(number)
            if index > 0 and index not in indices:
                indices.append(index)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned).rstrip()
        return cleaned, indices

    @staticmethod
    def _card_selection_stream_factory(
        factory: Callable[[], Callable[[str], Awaitable[None]]],
    ) -> Callable[[], Callable[[str], Awaitable[None]]]:
        """Envolve o emissor de tokens para o bloco «[[cartões: …]]» não chegar
        ao visitante a meio da resposta: a partir de «[[» tudo fica retido (o
        `result` final traz o texto já limpo). Se o que ficou retido crescer
        sem ser o bloco, é libertado."""

        def wrapped() -> Callable[[str], Awaitable[None]]:
            emit = factory()
            held = ""

            async def filtered(delta: str) -> None:
                nonlocal held
                buffer = held + (delta or "")
                start = buffer.find("[[")
                if start == -1:
                    if buffer.endswith("["):
                        out, held = buffer[:-1], "["
                    else:
                        out, held = buffer, ""
                else:
                    out, held = buffer[:start], buffer[start:]
                    if len(held) > 200 and not re.match(r"\[\[\s*(?:cart|card)", held, flags=re.IGNORECASE):
                        out, held = out + held, ""
                if out:
                    await emit(out)

            return filtered

        return wrapped

    def _name_tokens(self, text: str, *, stopwords: frozenset[str]) -> list[str]:
        return [token for token in self._fold_query_text(text).split() if token not in stopwords]

    @staticmethod
    def _tokens_alike(left: str, right: str) -> bool:
        """«egito» ≈ «egipto», «cenas» ≈ «cena»; nunca palavras curtas diferentes."""
        if left == right:
            return True
        if len(left) < 4 or len(right) < 4:
            return False
        return difflib.SequenceMatcher(None, left, right).ratio() >= 0.8

    def _artifacts_named_in_query(
        self,
        *,
        user_message: str,
        artifact_results: list[ArtifactResult],
        retrieval_request: dict[str, Any],
    ) -> list[str]:
        """Peças que o pedido nomeia, pelo n.º de inventário ou pelo título.

        Pelo n.º: os candidatos do fast-path de inventário, ou um inventário
        de um cartão escrito na pergunta. Pelo título: o que sobra da pergunta
        sem verbos de pesquisa, âmbito da visita e palavras de ligação começa
        por um determinante no singular e cobre o título do cartão (e o
        título cobre-o), com tolerância a gralhas. Lista de artifact_id pela
        ordem dos cartões; vazia quando não é um pedido de uma peça concreta.
        """
        raw = (user_message or "").strip()
        if not raw or not artifact_results:
            return []

        def alnum(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", self._fold_query_text(value))

        wanted = {alnum(str(c)) for c in (retrieval_request.get("inventory_candidates") or []) if str(c).strip()}
        wanted.discard("")
        query_alnum = alnum(raw)
        by_inventory: list[str] = []
        for artifact in artifact_results:
            inventory = alnum(str(artifact.inventory_number or ""))
            if not inventory:
                continue
            if inventory in wanted or (
                len(inventory) >= 3 and any(ch.isdigit() for ch in inventory) and inventory in query_alnum
            ):
                by_inventory.append(artifact.artifact_id)
        if by_inventory:
            return by_inventory

        scope = self._extract_tour_scope_expression(raw)
        without_scope = self._strip_tour_scope_expression_from_query(raw, scope) if scope else raw
        folded = self._fold_query_text(without_scope).split()
        # descarta a cabeça («encontra», «mostra-me», «onde está») até ao determinante
        head = 0
        while head < len(folded) and folded[head] not in _NAME_QUERY_SINGULAR_DETERMINERS:
            if folded[head] not in _NAME_QUERY_STOPWORDS:
                return []
            head += 1
        if head >= len(folded):
            return []
        query_tokens = [token for token in folded[head:] if token not in _NAME_QUERY_STOPWORDS]
        if len(query_tokens) < 2:
            return []

        by_title: list[str] = []
        for artifact in artifact_results:
            title = re.sub(r"\([^)]*\)", " ", str(artifact.title or "")).split("/", 1)[0]
            title_tokens = self._name_tokens(title, stopwords=_TITLE_STOPWORDS)
            if not title_tokens:
                continue
            query_covered = all(any(self._tokens_alike(q, t) for t in title_tokens) for q in query_tokens)
            title_covered = all(any(self._tokens_alike(q, t) for q in query_tokens) for t in title_tokens)
            if query_covered and title_covered:
                by_title.append(artifact.artifact_id)
        return by_title

    def _cards_cited_in_reply(self, reply: str, artifact_results: list[ArtifactResult]) -> list[str]:
        """Cartões cujo n.º de inventário ou título aparece na resposta, pela
        ordem da primeira menção. Títulos só com duas ou mais palavras: «Botão»
        aparece em qualquer resposta sobre botões."""
        folded_reply = f" {self._fold_query_text(reply)} "
        if not folded_reply.strip():
            return []
        positions: dict[str, int] = {}
        for artifact in artifact_results:
            candidates: list[str] = []
            inventory = self._fold_query_text(str(artifact.inventory_number or ""))
            if inventory and any(ch.isdigit() for ch in inventory):
                candidates.append(inventory)
            title = self._fold_query_text(str(artifact.title or ""))
            if len(title.split()) >= 2:
                candidates.append(title)
            found = [folded_reply.find(f" {c} ") for c in candidates]
            found = [p for p in found if p >= 0]
            if found and artifact.artifact_id not in positions:
                positions[artifact.artifact_id] = min(found)
        return sorted(positions, key=positions.get)

    def _select_cards(
        self,
        *,
        user_message: str,
        retrieval_request: dict[str, Any],
        artifact_results: list[ArtifactResult],
        image_matches: list[ImageMatchResult],
        navigation_targets: list[TourNavigationTarget],
        model_indices: list[int] | None,
        results_total: int,
        results_has_more: bool,
        reply_text: str = "",
    ) -> tuple[list[ArtifactResult], list[ImageMatchResult], list[TourNavigationTarget], int, bool, dict[str, Any]]:
        """Reduz os cartões aos que o modelo usou ou recomendou.

        As peças citadas no texto (n.º de inventário ou título) mandam, pela
        ordem em que aparecem: o 9B engana-se nos rótulos do bloco («doc_5»
        quando descreveu o doc_6, medido em produção a 2026-09-30). O bloco
        só decide quando o texto não cita nenhuma peça. Um pedido de uma peça
        concreta (nome ou n.º) mostra só as peças que correspondem, as citadas
        primeiro, e fecha o «ver mais». Sem citações, sem bloco e sem peça
        nomeada, fica tudo como estava.
        """
        chosen: list[str] = []
        for index in model_indices or []:
            if 1 <= index <= len(artifact_results):
                artifact_id = artifact_results[index - 1].artifact_id
                if artifact_id not in chosen:
                    chosen.append(artifact_id)
        cited = self._cards_cited_in_reply(reply_text, artifact_results)
        named = self._artifacts_named_in_query(
            user_message=user_message, artifact_results=artifact_results, retrieval_request=retrieval_request,
        )
        report: dict[str, Any] = {
            "model_cards": list(model_indices) if model_indices is not None else None,
            "model_artifacts": len(chosen),
            "cited_artifacts": len(cited),
            "block_only_dropped": len([a for a in chosen if a not in cited]) if cited else 0,
            "named_artifacts": len(named),
            "shown_before": len(artifact_results),
        }
        used = cited or chosen
        if named:
            ordered = [a for a in used if a in named] + [a for a in named if a not in used]
            total, has_more = len(ordered), False
            report["rule"] = "named"
        elif used:
            ordered, total, has_more = used, results_total, results_has_more
            report["rule"] = "cited" if cited else "model"
        else:
            report["rule"] = "unchanged"
            report["shown_after"] = len(artifact_results)
            return artifact_results, image_matches, navigation_targets, results_total, results_has_more, report

        def norm(value: str | None) -> str:
            return str(value or "").strip().casefold()

        by_id = {artifact.artifact_id: artifact for artifact in artifact_results}
        artifacts = [by_id[a] for a in ordered if a in by_id]
        rank = {artifact.artifact_id: position for position, artifact in enumerate(artifacts)}
        rank_by_inventory = {
            norm(artifact.inventory_number): position
            for position, artifact in enumerate(artifacts)
            if norm(artifact.inventory_number)
        }

        def image_rank(match: ImageMatchResult) -> int | None:
            position = rank.get(str(match.artifact_id or "").strip())
            return position if position is not None else rank_by_inventory.get(norm(match.inventory))

        images = sorted(
            (match for match in image_matches if image_rank(match) is not None),
            key=lambda match: image_rank(match) or 0,
        )
        targets = sorted(
            (target for target in navigation_targets if norm(target.inventory_id) in rank_by_inventory),
            key=lambda target: rank_by_inventory[norm(target.inventory_id)],
        )
        report["shown_after"] = len(artifacts)
        return artifacts, images, targets, total, has_more, report

    def _sanitize_assistant_reply(
        self,
        text: str,
        *,
        docs: list[dict[str, object]] | None = None,
        language: str | None = None,
    ) -> str:
        cleaned = text or ""
        doc_label_map = self._build_doc_label_map(docs or [], language=language)
        if doc_label_map:
            cleaned = self._replace_doc_refs_with_labels(
                text=cleaned,
                doc_label_map=doc_label_map,
                language=language,
            )

        artifact_label = translate("sanitizer.artifact_label", language)
        context_label = translate("sanitizer.context_label", language)
        cleaned = re.sub(
            r"`?\bartifact[_-]?\d+\b`?",
            artifact_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(?:retrieval_context|current_message|explicit_state|recent_history_aux|rolling_summary_aux|current_page_results|current_visible_results|visible_results_count|visible_results_page|visible_results_page_size|visible_results_total|search_query|reported_total|results_page)\b",
            context_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(?i)\bcom base na informac[aã]o fornecida pelo contexto\b[:,]?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"(?i)\bbased on the information provided by the context\b[:,]?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            rf"\b(?:documento|document)\s+{re.escape(artifact_label)}\b",
            artifact_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\bdoc(?:umento|ument)?[_\s-]*\d+\b",
            artifact_label,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([,.;:!?])", r"\1", cleaned)
        cleaned = self._renumber_ordered_list_markers(cleaned)
        if docs and str(getattr(self.settings, "CHAT_GROUNDING_CHECK", "off") or "off").strip().lower() == "repair":
            cleaned = self._repair_inventory_mentions(cleaned, docs=docs)
        return cleaned.strip()

    @staticmethod
    def _inventory_variant_pattern(inventory: str) -> re.Pattern[str] | None:
        """Padrão das formas abreviadas de um inventário com código de tipo.

        Só para inventários «<número> <Tipo> [CMP] <MUSEU>» (ex.: «71 Cer CMP/
        MNSR»): número e museu obrigatórios, códigos intermédios opcionais,
        separadores livres. Inventários curtos («38090», «15069/1») ficam de
        fora — uma forma abreviada deles não se distingue de outro número.
        """
        tokens = re.findall(r"[A-Za-z0-9]+", inventory or "")
        if len(tokens) < 3 or not tokens[0].isdigit() or not tokens[-1].isalpha():
            return None
        if not all(token.isalpha() for token in tokens[1:]):
            return None
        separator = r"[\s/._-]*"
        middle = "".join(f"(?:{separator}{re.escape(token)})?" for token in tokens[1:-1])
        return re.compile(
            rf"(?<![\w/.-]){re.escape(tokens[0])}{middle}{separator}{re.escape(tokens[-1])}(?![\w])",
            flags=re.IGNORECASE,
        )

    def _repair_inventory_mentions(self, text: str, *, docs: list[dict[str, object]]) -> str:
        """Modo repair do grounding: repõe o inventário exato do cartão.

        Erro real observado em produção (2026-09-16): o LLM escreveu «71 CMP/MNSR»
        para o cartão «71 Cer CMP/ MNSR». Só se corrige quando a forma abreviada
        corresponde a um único cartão visível; ambíguas ficam como estão e o
        relatório de grounding continua a registá-las.
        """
        inventories: list[str] = []
        for doc in docs:
            for key in ("inventory_number", "inventory"):
                value = str(doc.get(key) or "").strip()
                if value and value not in inventories:
                    inventories.append(value)
        candidates: dict[tuple[int, int], set[str]] = {}
        for inventory in inventories:
            pattern = self._inventory_variant_pattern(inventory)
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                candidates.setdefault(match.span(), set()).add(inventory)
        repairs: list[tuple[str, str]] = []
        repaired = text
        for (start, end), options in sorted(candidates.items(), reverse=True):
            if len(options) != 1:
                continue
            canonical = next(iter(options))
            found = repaired[start:end]
            if found == canonical:
                continue
            repaired = f"{repaired[:start]}{canonical}{repaired[end:]}"
            repairs.append((found, canonical))

        # Número solto depois de um rótulo de inventário («Inventário: 71»,
        # «Inv. 88»), variante também vista em produção: completa-se quando um
        # único cartão tem esse número como primeiro elemento do inventário.
        by_number: dict[str, set[str]] = {}
        for inventory in inventories:
            tokens = re.findall(r"[A-Za-z0-9]+", inventory)
            if len(tokens) > 1 and tokens[0].isdigit():
                by_number.setdefault(tokens[0], set()).add(inventory)

        def already_complete(position: int, canonical: str) -> bool:
            compact = re.sub(r"[^a-z0-9]+", "", canonical.lower())
            window = re.sub(r"[^a-z0-9]+", "", repaired[position:position + len(canonical) + 8].lower())
            return window.startswith(compact)

        label_pattern = re.compile(
            r"\b(?:invent[aá]rio|inv\.?)\s*(?:n\.?\s*[ºo°]\s*)?:?\s*(\d{1,6})(?![\w/-]|\.\w)",
            flags=re.IGNORECASE,
        )
        for match in reversed(list(label_pattern.finditer(repaired))):
            options = by_number.get(match.group(1), set())
            if len(options) != 1:
                continue
            canonical = next(iter(options))
            if already_complete(match.start(1), canonical):
                continue
            repaired = f"{repaired[:match.start(1)]}{canonical}{repaired[match.end(1):]}"
            repairs.append((match.group(1), canonical))
        if repairs:
            log_event(
                logger,
                logging.INFO,
                "chat.grounding.inventory_repaired",
                count=len(repairs),
                repairs=[f"{found} -> {canonical}" for found, canonical in reversed(repairs)][:10],
            )
        return repaired

    _QUOTED_TITLE_PATTERNS: tuple[str, ...] = (
        r"«([^»]{3,120})»",
        r"“([^”]{3,120})”",
        r"\"([^\"\n]{3,120})\"",
        r"\*\*([^*\n]{3,120})\*\*",
    )
    # «71 Cer CMP/ MNSR» tem até dois códigos entre o número e o museu.
    _INVENTORY_MENTION_PATTERN = (
        r"\b(?:(?:mnt|mnaz|mnsr|mj|mnaa|mnac|mnaj|mnar|mnap|mnmc)\s*[a-z]{0,3}\s*\d[\w./-]*"
        r"|\d{2,6}(?:\s+[a-z]{2,4}){1,2}\s+(?:mnt|mnaz|mnsr|mj))\b"
    )
    # Um inventário nunca atravessa estes caracteres; sem este corte, o texto
    # dobrado juntava «… MNSR)**\n3. **Jarra» em «mnsr 3» (falso inventário).
    _INVENTORY_SEGMENT_BREAKS = r"[\n\r()*,;:|\[\]«»\"“”–—]+"

    def _grounding_report(
        self,
        *,
        reply: str,
        docs: list[dict[str, object]] | None,
        conversation_id: str,
        query_id: str,
        user_message: str | None = None,
    ) -> dict[str, Any] | None:
        """Verificação de grounding em modo observação (quick win FILT-02).

        Compara os títulos citados (entre aspas/«»/negrito) e os números de
        inventário mencionados na resposta com os cartões visíveis. Não altera
        a resposta: devolve um relatório para o evento backend_query e emite
        um aviso quando há citações sem correspondência. CHAT_GROUNDING_CHECK
        = off desliga tudo (comportamento histórico).
        """
        mode = str(getattr(self.settings, "CHAT_GROUNDING_CHECK", "off") or "off").strip().lower()
        if mode == "off":
            return None
        text = reply or ""
        cards = list(docs or [])
        known_titles = [
            self._fold_query_text(str(doc.get("title") or ""))
            for doc in cards
            if str(doc.get("title") or "").strip()
        ]
        known_inventories = {
            re.sub(r"[^a-z0-9]+", "", self._fold_query_text(str(doc.get(key) or "")))
            for doc in cards
            for key in ("inventory_number", "inventory")
            if str(doc.get(key) or "").strip()
        }
        # Valores categóricos dos cartões (categoria, material, técnica, autor):
        # citá-los em negrito («**Mobiliário**») não é inventar uma peça.
        known_values: set[str] = set()
        for doc in cards:
            for key in ("category", "super_category", "support_or_material", "technique",
                        "creators", "creator", "date_or_period", "production_center", "museum"):
                value = doc.get(key)
                for item in (value if isinstance(value, list) else [value]):
                    folded_value = self._fold_query_text(str(item or ""))
                    if folded_value:
                        known_values.add(folded_value)

        def tokens_of(value: str) -> set[str]:
            # plural simples sobre texto folded («calcoes» ~ «calcao» -> «calc»,
            # «vestidos» ~ «vestido»); só tokens com 3+ letras
            return {re.sub(r"(oes|aes|ao|s)$", "", t) for t in value.split() if len(t) > 2}

        # Todo o texto dos cartões: uma citação de uma inscrição ou de um
        # trecho da descrição («Caprai, Rua da Misericórdia, 51-53, Lisboa»,
        # falso positivo real de 2026-09-16) está ancorada mesmo não sendo um
        # título. Tolera uma palavra mal escrita pelo LLM («Mesericórdia»).
        card_text_tokens: set[str] = set()
        for doc in cards:
            for value in doc.values():
                items = value if isinstance(value, list) else [value]
                for item in items:
                    if isinstance(item, str) and item.strip():
                        card_text_tokens |= tokens_of(self._fold_query_text(item))

        folded_user = self._fold_query_text(user_message or "")
        quoted: list[str] = []
        for pattern in self._QUOTED_TITLE_PATTERNS:
            for match in re.finditer(pattern, text):
                candidate = match.group(1).strip()
                # Negrito/aspas também servem para ênfase de frases inteiras e
                # para ecoar o pedido do utilizador; só contam como citação
                # de peça as expressões curtas, capitalizadas como um título,
                # que não vêm da pergunta.
                if len(candidate.split()) > 8 or not candidate[:1].isupper():
                    continue
                # Rótulos de formatação («**Nota:**», «**Localização:**»,
                # falso positivo real de 2026-09-16) não citam nenhuma peça.
                if candidate.rstrip().endswith(":"):
                    continue
                if candidate and candidate not in quoted:
                    quoted.append(candidate)
        unmatched_titles: list[str] = []
        for candidate in quoted:
            folded = self._fold_query_text(candidate)
            if len(folded) < 4:
                continue
            if folded_user and folded in folded_user:
                continue
            if folded in known_values:
                continue
            if any(folded in title or title in folded for title in known_titles if title):
                continue
            candidate_tokens = tokens_of(folded)
            if candidate_tokens and any(
                len(candidate_tokens & tokens_of(title)) / len(candidate_tokens | tokens_of(title)) >= 0.6
                for title in known_titles if title
            ):
                continue
            if candidate_tokens and len(candidate_tokens & card_text_tokens) / len(candidate_tokens) >= 0.75:
                continue
            unmatched_titles.append(candidate)

        mentioned_inventories: list[str] = []
        for segment in re.split(self._INVENTORY_SEGMENT_BREAKS, text):
            for match in re.finditer(self._INVENTORY_MENTION_PATTERN, self._fold_query_text(segment)):
                value = match.group(0).strip()
                if value and value not in mentioned_inventories:
                    mentioned_inventories.append(value)
        unmatched_inventories = [
            value
            for value in mentioned_inventories
            if re.sub(r"[^a-z0-9]+", "", value) not in known_inventories
        ]

        report: dict[str, Any] = {
            "mode": mode,
            "cards": len(cards),
            "quoted_titles": len(quoted),
            "unmatched_titles": unmatched_titles[:10],
            "unmatched_inventories": unmatched_inventories[:10],
            "violations": len(unmatched_titles) + len(unmatched_inventories),
        }
        if report["violations"]:
            log_event(
                logger,
                logging.WARNING,
                "chat.grounding.violation",
                conversation_id=conversation_id,
                query_id=query_id,
                cards=len(cards),
                unmatched_titles=unmatched_titles[:5],
                unmatched_inventories=unmatched_inventories[:5],
            )
        return report

    def _renumber_ordered_list_markers(self, text: str) -> str:
        item_pattern = re.compile(r"^(\s*)(\d{1,2})([.)])(\s+)(.+)$")
        lines = text.splitlines()
        item_indexes: list[int] = []

        def flush_item_indexes() -> None:
            if len(item_indexes) < 2:
                item_indexes.clear()
                return
            for list_number, line_index in enumerate(item_indexes, start=1):
                match = item_pattern.match(lines[line_index])
                if not match:
                    continue
                lines[line_index] = (
                    f"{match.group(1)}{list_number}{match.group(3)}"
                    f"{match.group(4)}{match.group(5)}"
                )
            item_indexes.clear()

        for index, line in enumerate(lines):
            if item_pattern.match(line):
                item_indexes.append(index)
                continue
            if not line.strip():
                continue
            flush_item_indexes()

        flush_item_indexes()
        return "\n".join(lines)

    def _build_doc_label_map(
        self,
        docs: list[dict[str, object]],
        *,
        language: str | None = None,
    ) -> dict[int, str]:
        labels: dict[int, str] = {}
        for index, doc in enumerate(docs, start=1):
            title = str(doc.get("title") or "").strip()
            inventory = self._doc_inventory(doc)
            if title and inventory:
                labels[index] = translate(
                    "sanitizer.titled_inventory_artifact",
                    language,
                    title=title,
                    inventory=inventory,
                )
            elif title:
                labels[index] = translate(
                    "sanitizer.titled_artifact",
                    language,
                    title=title,
                )
            elif inventory:
                labels[index] = translate(
                    "sanitizer.inventory_artifact",
                    language,
                    inventory=inventory,
                )
            else:
                labels[index] = translate("sanitizer.collection_artifact", language)
        return labels

    def _compact_for_match(self, value: str) -> str:
        lowered = (value or "").casefold()
        compact = re.sub(r"[\W_]+", "", lowered, flags=re.UNICODE)
        return compact

    def _replace_doc_refs_with_labels(
        self,
        *,
        text: str,
        doc_label_map: dict[int, str],
        language: str | None = None,
    ) -> str:
        fallback_label = translate("sanitizer.collection_artifact", language)

        def _replace(match: re.Match[str]) -> str:
            idx_text = match.group(1)
            try:
                idx = int(idx_text)
            except Exception:
                return fallback_label
            return doc_label_map.get(idx, fallback_label)

        replaced = re.sub(r"\[doc_(\d+)\]", _replace, text, flags=re.IGNORECASE)
        replaced = re.sub(r"\bdoc[_-](\d+)\b", _replace, replaced, flags=re.IGNORECASE)
        replaced = re.sub(
            r"\b(?:documento|document)\s+\[?doc[_-]?(\d+)\]?\b",
            _replace,
            replaced,
            flags=re.IGNORECASE,
        )
        return replaced

    def _build_final_prompt(
        self,
        *,
        payload: ChatMessageRequest,
        state: ChatSessionState,
        router_decision: dict[str, object],
        retrieval_context: str,
        effective_filters: dict[str, object],
        effective_sort: dict[str, object],
        use_history_for_answer: bool,
        museum_slug: str | None = None,
        museum_name: str | None = None,
        select_cards: bool = False,
    ) -> str:
        mode = str(router_decision.get("mode", "llm_only"))
        rewritten_query = str(router_decision.get("rewritten_query", payload.message))
        if self.settings.CHAT_COMPACT_ANSWER_CONTEXT:
            # Quick win PR-01: prompt final curto em PT acentuado, com os
            # cartões compactos como única fonte de factos.
            return build_final_answer_prompt_v2(
                museum_name=museum_name if museum_name is not None else payload.museum_name,
                intent=str(router_decision.get("intent", "fallback")),
                mode=mode,
                rolling_summary=state.rolling_summary if use_history_for_answer else "",
                filters_state=effective_filters,
                user_message=payload.message,
                rewritten_query=rewritten_query,
                retrieval_context=retrieval_context,
                use_history_for_answer=use_history_for_answer,
                language=state.language,
                select_cards=select_cards,
            )
        if select_cards:
            return "\n".join(
                [
                    self._build_final_prompt(
                        payload=payload, state=state, router_decision=router_decision,
                        retrieval_context=retrieval_context, effective_filters=effective_filters,
                        effective_sort=effective_sort, use_history_for_answer=use_history_for_answer,
                        museum_slug=museum_slug, museum_name=museum_name,
                    ),
                    card_selection_instruction(state.language),
                ]
            )
        return build_final_answer_prompt(
            museum_slug=museum_slug or payload.museum_slug,
            museum_name=museum_name if museum_name is not None else payload.museum_name,
            input_modality="text",
            mode=mode,
            intent=str(router_decision.get("intent", "fallback")),
            rolling_summary=state.rolling_summary if use_history_for_answer else "",
            filters_state=effective_filters,
            sort_state=effective_sort,
            user_message=payload.message,
            rewritten_query=rewritten_query,
            retrieval_context=retrieval_context,
            use_history_for_answer=use_history_for_answer,
            language=state.language,
        )

    def _resolve_museum_id_values(
        self,
        *,
        museum_slug: str,
        museum_id: str | None,
        metadata: dict[str, object] | None = None,
    ) -> str | None:
        if museum_id:
            return museum_id.strip() or None

        if metadata and isinstance(metadata, dict):
            metadata_museum_id = metadata.get("museum_id")
            if isinstance(metadata_museum_id, str) and metadata_museum_id.strip():
                return metadata_museum_id.strip()

        return museum_slug.strip() or None

    def _resolve_explicit_museum_id_values(
        self,
        *,
        museum_id: str | None,
        metadata: dict[str, object] | None = None,
    ) -> str | None:
        if museum_id:
            return museum_id.strip() or None

        if metadata and isinstance(metadata, dict):
            metadata_museum_id = metadata.get("museum_id")
            if isinstance(metadata_museum_id, str) and metadata_museum_id.strip():
                return metadata_museum_id.strip()

        return None

    def _resolve_museum_id(self, payload: ChatMessageRequest) -> str | None:
        return self._resolve_museum_id_values(
            museum_slug=payload.museum_slug,
            museum_id=payload.museum_id,
            metadata=payload.metadata if isinstance(payload.metadata, dict) else None,
        )

    def _extract_artifact_ids_from_docs(self, docs: list[dict[str, object]]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            artifact_id = str(doc.get("artifact_id") or "").strip()
            if not artifact_id or artifact_id in seen:
                continue
            seen.add(artifact_id)
            ids.append(artifact_id)
        return ids

    # ----- Fichas iguais do mesmo conjunto (CHAT_COLLAPSE_SET_DUPLICATES) ----- #

    def _collapse_enabled(self) -> bool:
        return bool(getattr(self.settings, "CHAT_COLLAPSE_SET_DUPLICATES", False))

    def _collapse_fetch_size(self, size: int, *, cap: int = 200) -> int:
        """Quantos candidatos pedir para, depois de juntar, sobrarem `size`."""
        if not self._collapse_enabled():
            return max(int(size), 1)
        factor = max(int(getattr(self.settings, "CHAT_COLLAPSE_FETCH_FACTOR", 3) or 1), 1)
        return min(max(int(size), 1) * factor, max(cap, int(size)))

    @staticmethod
    def _inventory_sort_key(inventory: str) -> tuple[str, int, str]:
        """«38195/2» antes de «38195/10»."""
        root, sep, suffix = str(inventory or "").partition("/")
        digits = re.match(r"\s*(\d+)", suffix) if sep else None
        return (root.strip().casefold(), int(digits.group(1)) if digits else -1, suffix.strip().casefold())

    def _collapse_identical_set_members(self, docs: list[dict[str, object]]) -> list[dict[str, object]]:
        """Junta num só cartão as fichas do mesmo conjunto com título e descrição
        iguais («Botões (dez)», 38195/1 a 38195/10). Conjuntos com peças
        diferentes (casaco + saia + colete) ficam como estão: o texto difere.

        O cartão é a ficha mais bem posicionada, ou a primeira que está na
        visita (para o «Ver na tour» ter alvo), com `group_size`,
        `group_inventory_numbers` e `group_artifact_ids` de todas. Ocupa o
        lugar do primeiro membro; a ordem dos restantes cartões não muda.
        Sem conjunto e sem raiz de inventário («N/M»), nunca junta.
        """
        if not self._collapse_enabled() or len(docs) < 2:
            return docs

        def key_of(doc: dict[str, object]) -> tuple[str, str] | None:
            set_ids = doc.get("set_ids")
            set_key = ""
            if isinstance(set_ids, list) and set_ids:
                set_key = f"set:{str(set_ids[0] or '').strip()}"
            elif isinstance(set_ids, str) and set_ids.strip():
                set_key = f"set:{set_ids.strip()}"
            if not set_key:
                inventory = self._doc_inventory(doc)
                root, sep, _ = inventory.partition("/")
                if not sep or not root.strip():
                    return None
                set_key = f"inv:{self._fold_query_text(root)}"
            text_key = self._fold_query_text(str(doc.get("title") or "")) + "|" + self._fold_query_text(
                str(doc.get("description") or "")
            )
            return set_key, text_key

        groups: dict[tuple[str, str], list[dict[str, object]]] = {}
        order: list[tuple[str, str] | int] = []
        for position, doc in enumerate(docs):
            key = key_of(doc)
            if key is None:
                order.append(position)
                continue
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(doc)

        def is_in_tour(doc: dict[str, object]) -> bool:
            raw = doc.get("in_tour")
            return raw.strip().lower() in {"true", "1", "yes"} if isinstance(raw, str) else bool(raw)

        collapsed: list[dict[str, object]] = []
        for item in order:
            if isinstance(item, int):
                collapsed.append(docs[item])
                continue
            members = groups[item]
            if len(members) == 1:
                collapsed.append(members[0])
                continue
            representative = next((m for m in members if is_in_tour(m)), members[0])
            inventories = sorted(
                {self._doc_inventory(m) for m in members if self._doc_inventory(m)}, key=self._inventory_sort_key
            )
            merged = dict(representative)
            merged["group_size"] = len(members)
            merged["group_inventory_numbers"] = inventories
            merged["group_artifact_ids"] = self._extract_artifact_ids_from_docs(members)
            collapsed.append(merged)
        return collapsed

    def _update_last_result_ids(
        self,
        *,
        state: ChatSessionState,
        artifact_docs: list[dict[str, object]],
        preserve_existing_when_empty: bool = False,
    ) -> None:
        artifact_ids = self._extract_artifact_ids_from_docs(artifact_docs)
        if artifact_ids:
            state.last_result_ids = artifact_ids
        elif not preserve_existing_when_empty:
            state.last_result_ids = []

    def _merge_state_with_delta(
        self,
        *,
        base: dict[str, object],
        delta: object,
    ) -> dict[str, object]:
        merged: dict[str, object] = dict(base)
        if not isinstance(delta, dict):
            return merged

        for key, value in delta.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value

        return merged

    def _apply_router_decision_to_state(
        self,
        *,
        state: ChatSessionState,
        router_decision: dict[str, object],
    ) -> None:
        intent = router_decision.get("intent")
        if isinstance(intent, str):
            state.intent = intent

        filters_delta = router_decision.get("filters_delta")
        if isinstance(filters_delta, dict):
            for key, value in filters_delta.items():
                if key == "artifact_id":
                    continue
                if value is None:
                    state.filters.pop(key, None)
                else:
                    state.filters[key] = value

        sort_delta = router_decision.get("sort_delta")
        if isinstance(sort_delta, dict):
            for key, value in sort_delta.items():
                if value is None:
                    state.sort.pop(key, None)
                else:
                    state.sort[key] = value

    def _update_rolling_summary(
        self,
        *,
        state: ChatSessionState,
        latest_user_message: str,
        latest_assistant_message: str,
        router_decision: dict[str, object],
    ) -> None:
        line = (
            f"intent={router_decision.get('intent', 'fallback')} | "
            f"user={latest_user_message[:120]} | "
            f"assistant={latest_assistant_message[:140]}"
        )

        previous = state.rolling_summary.strip()
        candidate = f"{previous}\n{line}".strip() if previous else line
        max_chars = max(self.settings.CHAT_ROLLING_SUMMARY_MAX_CHARS, 120)
        if len(candidate) > max_chars:
            candidate = candidate[-max_chars:]
        state.rolling_summary = candidate



@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        settings=settings,
        opensearch_gateway=get_opensearch_gateway(),
        embedding_provider=get_embedding_provider(),
        model_retrieval_service=get_model_retrieval_service(),
        tour_navigation_service=get_tour_navigation_service(),
        llm_service=get_llm_service(),
        session_store=get_chat_session_store(),
        query_logger=get_query_logger(),
    )
