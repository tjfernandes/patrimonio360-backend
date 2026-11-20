"""Mede se a resposta traz cada cartão na sua linha da lista, com e sem a lista decidida no prompt.

  python scripts/medir_lista_da_resposta.py [x4] [--ver] [--mudadas] [--so-autor] [--variantes=base,lista]

Corre da raiz do repositório, com o .env de produção (OpenSearch, AMÁLIA e modelo de pré-processamento).
Como scripts/medir_resposta_visual.py: fixa os cartões de cada caso (a pesquisa e a escolha não correm) e
pede a resposta à AMÁLIA real. Casos: «obras de Machado de Castro» (7 cartões, a 584 Esc sem título), os
gerais e os do aspeto de tests/visual_answer_cases.py. Variantes:

  base      CHAT_ANSWER_ALL_CARDS como em produção a 2026-11-20
  nome      + CHAT_UNTITLED_CARD_BY_INVENTORY
  lista     + CHAT_ANSWER_LIST_PLAN (a lista decidida no prompt e garantida na resposta)
  rede      nome + CHAT_MISSING_CARDS_IN_LIST, sem a lista no prompt (o que faltar entra na lista)

Conta, por resposta: cartões com linha própria na lista, cartões ditos só fora da lista, cartões por dizer
antes da rede, respostas cortadas, tamanho e tempo. Não escreve nos registos de produção.
"""
import asyncio, os, re, sys, time
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
import logging  # noqa: E402

logging.disable(logging.CRITICAL)
from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ChatMessageRequest  # noqa: E402
from app.services.author_cards import fold  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.chat_session_store import ChatSessionStore  # noqa: E402
from app.services.llm_service import get_llm_service, get_preprocess_llm_service  # noqa: E402
from app.services.opensearch_client import OpenSearchGateway  # noqa: E402
from app.services.retrieval.fusion import BranchProvenance, FusedResult  # noqa: E402
from app.services.tour_navigation import TourNavigationService  # noqa: E402
from tests import visual_answer_cases as casos  # noqa: E402

VEZES = next((int(a[1:]) for a in sys.argv[1:] if a.startswith("x") and a[1:].isdigit()), 2)
VER = "--ver" in sys.argv
AUTOR = {"pedido": "Encontra obras de machado de castro",
         "inventarios": ["90 Esc", "60 Esc", "57 Esc", "62 Esc", "59 Esc", "63 Esc", "584 Esc"], "visual": False}
VARIANTES = {
    "base": {},
    "nome": {"CHAT_UNTITLED_CARD_BY_INVENTORY": True},
    "lista": {"CHAT_UNTITLED_CARD_BY_INVENTORY": True, "CHAT_ANSWER_LIST_PLAN": True},
    "rede": {"CHAT_UNTITLED_CARD_BY_INVENTORY": True, "CHAT_MISSING_CARDS_IN_LIST": True},
}
ITEM = re.compile(r"^[ \t]*(?:\d+[.)]|[-•*])[ \t]+(?P<body>\S.*)$")


class _Registo:
    def __init__(self):
        self.events = []

    async def log_event(self, event):
        self.events.append(event)


class _Escuta:
    """O LLM final: guarda o prompt e a resposta e deixa passar."""

    def __init__(self, llm):
        self.llm, self.prompts, self.respostas = llm, [], []

    async def generate(self, **kwargs):
        self.prompts.append(kwargs.get("message"))
        resposta = await self.llm.generate(**kwargs)
        self.respostas.append(resposta)
        return resposta

    def __getattr__(self, nome):
        return getattr(self.llm, nome)


def servico(extra) -> ChatService:
    settings = Settings(CHAT_ANSWER_ALL_CARDS=True, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False,
                        CHAT_STREAM_FINAL_ANSWER=False, CHAT_AUTHOR_CARD=False, **extra)
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


def fixar(s, docs, visual):
    async def _retrieve_context(**_kw):
        return "ctx", len(docs), [dict(d) for d in docs], {"kind": "text", "results_total": len(docs), "filters": {}, "sort": {}}

    async def _maybe_fuse_text_to_image(**_kw):
        if not visual:
            return None
        fused = [FusedResult(artifact_id=d["artifact_id"], fusion_score=1.0 / n,
                             sources={"text_to_caption": BranchProvenance(rank=n)}) for n, d in enumerate(docs, 1)]
        return SimpleNamespace(docs=[dict(d) for d in docs], total=len(docs), fused=fused, matched_image_hits=[],
                               router=SimpleNamespace(rule="medicao"), diagnostics={}, debug_payload=lambda: {})

    async def _preselect_cards(*, artifact_results, image_matches, navigation_targets, **_kw):
        return artifact_results, image_matches, navigation_targets, {"rule": "preselect", "shown_before": len(artifact_results),
                                                                     "model_cards": list(range(1, len(artifact_results) + 1))}

    s._retrieve_context, s._maybe_fuse_text_to_image, s._preselect_cards = _retrieve_context, _maybe_fuse_text_to_image, _preselect_cards


def onde(texto, docs):
    """Por cartão: 'lista' (abre uma linha da lista), 'fora' (dito noutro sítio) ou 'falta'. Cartões com o
    mesmo título contam pelas linhas que abrem com ele."""
    linhas = texto.split("\n")
    itens = [fold(m["body"].lstrip("*_ ")) for m in map(ITEM.match, linhas) if m]
    resto = fold("\n".join(l for l in linhas if not ITEM.match(l)))
    tudo = fold(texto)
    usados, saida = set(), []
    for d in docs:
        nomes = [fold(n) for n in (d.get("title"), d["inventory_number"]) if n and len(fold(n)) >= 2]
        linha = next((i for i, corpo in enumerate(itens) if i not in usados and any(corpo.startswith(n) for n in nomes)), None)
        if linha is not None:
            usados.add(linha)
            saida.append("lista")
        elif any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", resto if itens else tudo) for n in nomes):
            saida.append("fora")
        else:
            saida.append("falta")
    return saida


async def medir(nome, extra, lista):
    s = servico(extra)
    tot = dict(respostas=0, cartoes=0, lista=0, fora=0, falta=0, completas=0, cortadas=0, rede=0, juntas=0)
    letras, tempos = [], []
    for caso in lista:
        docs = fichas(s.opensearch_gateway, caso["inventarios"])
        fixar(s, docs, caso["visual"])
        for _ in range(VEZES):
            pedido = ChatMessageRequest(museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga",
                                        language=caso.get("lingua", "pt"), message=caso["pedido"])
            s.llm_service.respostas.clear()
            t0 = time.monotonic()
            resposta = await s.handle_message(pedido)
            tempos.append(time.monotonic() - t0)
            texto = resposta.reply or ""
            bruto = str(getattr(s.llm_service.respostas[-1], "text", "") or "") if s.llm_service.respostas else ""
            antes, depois = onde(bruto, docs), onde(texto, docs)
            tot["respostas"] += 1
            tot["cartoes"] += len(docs)
            for chave in ("lista", "fora", "falta"):
                tot[chave] += antes.count(chave)
            tot["completas"] += all(x == "lista" for x in depois)
            tot["rede"] += depois != antes
            tot["cortadas"] += str(getattr(s.llm_service.respostas[-1], "finish_reason", "")) == "length" if s.llm_service.respostas else 0
            letras.append(len(texto))
            if VER and (caso is AUTOR or any(x != "lista" for x in depois)):
                print(f"\n--- [{nome}] {caso['pedido']} · antes {antes} · depois {depois}\n{texto}")
            elif "--mudadas" in sys.argv and bruto.strip() != texto.strip():
                # o que a garantia mudou: o texto da AMÁLIA e o que o visitante lê
                print(f"\n--- [{nome}] {caso['pedido']} · antes {antes}\n··· AMÁLIA:\n{bruto}\n··· FINAL:\n{texto}")
    letras.sort(); tempos.sort()
    n = tot["respostas"]
    print(f"{nome:7} respostas {n:3} · AMÁLIA sozinha: {tot['lista']}/{tot['cartoes']} cartões com linha na lista, "
          f"{tot['fora']} só fora da lista, {tot['falta']} por dizer · resposta final com todos na lista: {tot['completas']}/{n} "
          f"· rede usada {tot['rede']} · cortadas {tot['cortadas']} · mediana {letras[n // 2]} letras, {tempos[n // 2]:.1f} s (máx {tempos[-1]:.1f} s)")


async def principal():
    gerais = [] if "--so-autor" in sys.argv else (
        [dict(c, visual=False) for c in casos.CASOS_GERAIS] + [dict(c, visual=True) for c in casos.CASOS])
    so = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--variantes=")), list(VARIANTES))
    for nome, extra in VARIANTES.items():
        if nome not in so:
            continue
        print(f"== {nome}: {extra or 'como em produção'}")
        await medir(nome + "/a", extra, [AUTOR])
        if gerais:
            await medir(nome + "/g", extra, gerais)


asyncio.run(principal())
