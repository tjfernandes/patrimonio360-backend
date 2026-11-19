"""Mede a resposta da AMÁLIA a perguntas pelo aspeto, com e sem CHAT_VISUAL_ANSWER_RULE.

  python scripts/medir_resposta_visual.py [x5] [--so-com|--so-sem] [--ver]
Corre da raiz do repositório, com o .env de produção (OpenSearch, AMÁLIA e modelo de pré-processamento).
Usa o código do serviço: para cada caso de tests/visual_answer_cases.py vai buscar as fichas e as legendas
reais dos cartões, fixa esses cartões (a pesquisa e a escolha não correm) e pede a resposta à AMÁLIA real.
Conta as respostas em que se diz de uma peça uma coisa do pedido que o cartão dessa peça não diz.
Não escreve nos registos de produção. --ver mostra as frases contadas.
"""
import asyncio, os, re, sys, unicodedata
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
import logging  # noqa: E402

logging.disable(logging.CRITICAL)
from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ChatMessageRequest  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.chat_session_store import ChatSessionStore  # noqa: E402
from app.services.llm_service import get_llm_service, get_preprocess_llm_service  # noqa: E402
from app.services.opensearch_client import OpenSearchGateway  # noqa: E402
from app.services.retrieval.fusion import BranchProvenance, FusedResult  # noqa: E402
from app.services.tour_navigation import TourNavigationService  # noqa: E402
from tests import visual_answer_cases as casos  # noqa: E402

VEZES = next((int(a[1:]) for a in sys.argv[1:] if a.startswith("x") and a[1:].isdigit()), 1)
VER = "--ver" in sys.argv


def chao(texto):
    """Sem acentos nem maiúsculas, para comparar palavras."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(texto or "")) if not unicodedata.combining(c)).casefold()


class _Registo:
    """Fica com os eventos em memória: nada vai para os registos de produção."""

    def __init__(self):
        self.events = []

    async def log_event(self, event):
        self.events.append(event)


class _Escuta:
    """O LLM final: guarda o prompt e deixa passar."""

    def __init__(self, llm):
        self.llm, self.prompts = llm, []

    async def generate(self, **kwargs):
        self.prompts.append(kwargs.get("message"))
        return await self.llm.generate(**kwargs)

    async def generate_stream(self, **kwargs):
        self.prompts.append(kwargs.get("message"))
        return await self.llm.generate_stream(**kwargs)

    def __getattr__(self, nome):
        return getattr(self.llm, nome)


def servico(regra: bool) -> ChatService:
    settings = Settings(CHAT_VISUAL_ANSWER_RULE=regra, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False,
                        CHAT_STREAM_FINAL_ANSWER=False, CHAT_AUTHOR_CARD=False)
    s = object.__new__(ChatService)
    s.settings = settings
    s.session_store = ChatSessionStore(settings)
    s.llm_service = _Escuta(get_llm_service())
    s.preprocess_llm_service = get_preprocess_llm_service()
    s.query_logger = _Registo()
    s.opensearch_gateway = OpenSearchGateway(settings)
    s.embedding_provider = None
    s.model_retrieval_service = None
    s.tour_navigation_service = TourNavigationService(settings)
    s._reranker_service = None
    s._vl_reranker_service = None
    return s


def fichas(gateway, inventarios):
    cliente = gateway._ensure_client()
    r = cliente.search(index=gateway.settings.OPENSEARCH_INDEX_ARTIFACT, body={
        "size": len(inventarios), "_source": {"excludes": ["text_embedding", "search_text", "embedding_meta"]},
        "query": {"bool": {"filter": [{"term": {"museum_id": "mnaa"}}, {"terms": {"inventory_number": inventarios}}]}}})
    por_inv = {h["_source"]["inventory_number"]: h["_source"] for h in r["hits"]["hits"]}
    em_falta = [i for i in inventarios if i not in por_inv]
    if em_falta:
        raise SystemExit(f"fichas em falta no índice: {em_falta}")
    return [por_inv[i] for i in inventarios]


def fixar(s, docs):
    """A pesquisa devolve estes cartões, todos vindos também das legendas, e a escolha fica com todos."""
    async def _retrieve_context(**_kw):
        return "ctx", len(docs), [dict(d) for d in docs], {"kind": "text", "results_total": len(docs), "filters": {}, "sort": {}}

    async def _maybe_fuse_text_to_image(**_kw):
        fused = [FusedResult(artifact_id=d["artifact_id"], fusion_score=1.0 / n,
                             sources={"text_to_caption": BranchProvenance(rank=n)}) for n, d in enumerate(docs, 1)]
        return SimpleNamespace(docs=[dict(d) for d in docs], total=len(docs), fused=fused, matched_image_hits=[],
                               router=SimpleNamespace(rule="medicao"), diagnostics={}, debug_payload=lambda: {})

    async def _preselect_cards(*, artifact_results, image_matches, navigation_targets, **_kw):
        return artifact_results, image_matches, navigation_targets, {"rule": "preselect", "shown_before": len(artifact_results),
                                                                     "model_cards": list(range(1, len(artifact_results) + 1))}

    s._retrieve_context, s._maybe_fuse_text_to_image, s._preselect_cards = _retrieve_context, _maybe_fuse_text_to_image, _preselect_cards


def cartoes_do_prompt(prompt, docs):
    """O bloco de cada cartão, tal como a AMÁLIA o recebeu."""
    blocos = re.split(r"\n(?=\[doc_\d+\])", prompt.split("current_visible_results:")[-1])
    blocos = [b for b in blocos if b.startswith("[doc_")]
    return {d["inventory_number"]: b for d, b in zip(docs, blocos)} if len(blocos) == len(docs) else {}


def trechos(resposta, docs):
    """(inventários, frase) das frases que falam de uma peça. Dentro de um parágrafo ou item de lista, uma
    frase sem nome de peça continua a falar da última nomeada; as frases de resumo, noutro parágrafo, não contam."""
    pecas = [(d["inventory_number"], chao(d.get("title") or "")) for d in docs]
    saida = []
    for bloco in re.split(r"\n+", resposta):
        atual = None
        # não corta em «Inv. 12», «D. Leonor» nem «n.º 3»
        for frase in re.split(r"(?<=[.!?])(?<!Inv\.)(?<!\s[A-Z]\.)(?<!\(n\.)\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕ*\d\"«])", bloco.strip()):
            base = chao(frase)
            citadas = [inv for inv, titulo in pecas if chao(inv) in base or (len(titulo) >= 6 and titulo in base)]
            if citadas:
                atual = tuple(dict.fromkeys(citadas))
            if atual:
                saida.append((atual, frase))
    return saida


async def medir(regra: bool):
    s = servico(regra)
    total = com_invencao = pecas_inventadas = rotulos = 0
    por_caso = []
    for caso in casos.CASOS:
        docs = fichas(s.opensearch_gateway, caso["inventarios"])
        fixar(s, docs)
        mas = 0
        for _ in range(VEZES):
            pedido = ChatMessageRequest(museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga",
                                        language=caso.get("lingua", "pt"), message=caso["pedido"])
            s.llm_service.prompts.clear()
            resposta = await s.handle_message(pedido)
            texto = resposta.reply or ""
            if [a.inventory_number for a in resposta.artifact_results] != caso["inventarios"]:
                raise SystemExit(f"os cartões da resposta não são os do caso ({caso['pedido']})")
            prompt = s.llm_service.prompts[-1] if s.llm_service.prompts else ""
            cartoes = cartoes_do_prompt(prompt, docs)
            if regra != ("Não vês o que ele escreveu" in prompt or "You cannot see what they wrote" in prompt) or regra == (caso["pedido"] in prompt):
                raise SystemExit(f"o prompt não corresponde à flag ({caso['pedido']})")
            achados = []
            for invs, frase in trechos(texto, docs):
                for nome, padrao in caso["grupos"].items():
                    if re.search(padrao, chao(frase)) and cartoes and not any(re.search(padrao, chao(cartoes.get(i, ""))) for i in invs):
                        achados.append((invs, nome, frase.strip()))
            total += 1
            if re.search(r"aspeto na fotografia", chao(texto)):
                rotulos += 1
                if VER:
                    print(f"     [{'com' if regra else 'sem'}] {caso['pedido']} · rótulo do cartão na resposta · {texto[:160]!r}")
            if achados:
                mas += 1
                com_invencao += 1
                pecas_inventadas += len({a[0] for a in achados})
                if VER:
                    for invs, nome, frase in achados:
                        print(f"     [{'com' if regra else 'sem'}] {caso['pedido']} · {', '.join(invs)} · «{nome}» · {frase[:230]}")
        por_caso.append((caso["pedido"], mas))
    return total, com_invencao, pecas_inventadas, por_caso, rotulos


async def principal():
    resultados = {}
    for regra in (False, True):
        if (regra and "--so-sem" in sys.argv) or (not regra and "--so-com" in sys.argv):
            continue
        resultados[regra] = await medir(regra)
    print()
    for regra, (total, mas, pecas, por_caso, rotulos) in resultados.items():
        print(f"{'COM' if regra else 'SEM'} a regra: {mas} de {total} respostas atribuem a uma peça o que o cartão não diz "
              f"({pecas} peças) · {rotulos} copiam o rótulo «Aspeto na fotografia»")
    nomes = [c["pedido"] for c in casos.CASOS]
    for n, nome in enumerate(nomes):
        linha = " · ".join(f"{'com' if r else 'sem'} {resultados[r][3][n][1]}/{VEZES}" for r in resultados)
        print(f"   {linha} · {nome}")


asyncio.run(principal())
