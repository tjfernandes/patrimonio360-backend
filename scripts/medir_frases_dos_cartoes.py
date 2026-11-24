"""Mede as respostas com a frase de cada peça no cartão dela (CHAT_CARD_SUMMARIES), ao lado das de hoje.

  python scripts/medir_frases_dos_cartoes.py [x2] [--variantes=rascunho,cartoes,cartoes_amalia] [--guardar=ficheiro.json] [--ver]

Corre da raiz do repositório, com o .env de produção (OpenSearch, AMÁLIA e modelo de pré-processamento).
Como scripts/medir_lista_da_resposta.py: fixa os cartões de cada caso (a pesquisa e a escolha não correm) e
pede a resposta ao serviço, com o streaming ligado. Casos: os 20 de tests/visual_answer_cases.py (os mesmos
do docs/rascunho-e-reescrita.md) e, à parte, cinco com perguntas diretas e cartões de tipos misturados.
Variantes:

  rascunho        como em produção a 2026-11-24 (CHAT_DRAFT_THEN_REWRITE): a lista no texto
  cartoes         + CHAT_CARD_SUMMARIES: o texto é a abertura do pré-processamento; a AMÁLIA não é chamada
  cartoes_amalia  + CHAT_CARD_SUMMARIES_REWRITE_OPENING: a AMÁLIA reescreve a abertura

Conta, por variante: tempo de resposta, texto enviado por streaming igual ao final, texto com lista ou
negrito, cartões com frase (do modelo ou da ficha), tamanho das frases, frases que repetem o título ou dizem
o n.º de inventário, frases com gerúndio que a ficha não tem (como ficam e como o modelo as escreveu), e
frases com palavras ou números que a ficha que o modelo viu não tem (candidatas a invenção, para ver à mão: --guardar escreve-as todas). Não escreve nos
registos de produção.
"""
import asyncio, json, os, re, sys, time
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
GUARDAR = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--guardar=")), None)
VARIANTES = {
    "rascunho": {},
    "cartoes": {"CHAT_CARD_SUMMARIES": True},
    "cartoes_amalia": {"CHAT_CARD_SUMMARIES": True, "CHAT_CARD_SUMMARIES_REWRITE_OPENING": True},
}
# perguntas diretas e cartões de tipos misturados (a frase deve dizer o que a peça é, mesmo não sendo o pedido)
EXTRA = [
    {"pedido": "Há pinturas de Domingos Sequeira?", "visual": False,
     "inventarios": ["1086 Pint", "825 Pint", "118 Pint", "3047 Des", "2406 Des", "1392 Des"]},
    {"pedido": "Tens tapeçarias?", "visual": False, "inventarios": ["69 Tap", "68 Tap", "2 tap", "4 tap", "3 Tap", "8 Tap"]},
    {"pedido": "De que século são os biombos?", "visual": False, "inventarios": ["1639 Mov", "1638 Mov", "1641 Mov", "1560 Mov"]},
    {"pedido": "Mostra-me esculturas em madeira.", "visual": False,
     "inventarios": ["350 Esc", "2509 Esc", "1266 Esc", "460 Mol", "584 Esc", "742 Esc"]},
    {"pedido": "Há pinturas com cães?", "visual": True,
     "inventarios": ["1391 Pint", "1742 Pint", "13/4v. Ilum", "13/98v. Ilum", "1165 Pint", "1853 Pint"]},
]
ITEM = re.compile(r"^[ \t]*(?:\d+[.)]|[-•*])[ \t]+(?P<body>\S.*)$")
VISITA = re.compile(r"(?i)visita\s+virtual|\b[nd]esta\s+visita\b|\b[nd]a\s+visita\b")
GERUNDIO = re.compile(r"\b[a-z]{2,}(?:ando|endo|indo)\b")
# acabam em -ando, -endo, -indo e não são gerúndio
NAO_GERUNDIO = {"quando", "bando", "comando", "fernando", "orlando", "armando", "brando", "lindo", "infindo", "reverendo",
                "tremendo", "horrendo", "estupendo", "memorando", "contrabando", "bem-vindo", "vindo", "provindo", "rolando",
                "fundo", "mundo", "segundo", "educando", "doutorando", "venerando", "nefando", "dividendo", "adendo", "rosalindo"}
# palavras do guia, que não dizem nada da peça
GERAIS = set("""peca pecas obra obras objeto objetos esta este estas estes essa esse uma umas pelo pela pelos pelas para
com sem sobre entre como mais muito muitos varias varios tambem ainda onde quando cujo cuja seus suas dele dela deles
delas foi sao era esta estao tem ser sido feito feita feitos feitas mostra mostram representa representam representado
representada apresenta apresentam trata figura figuras cena cenas imagem que qual quais nao sim ha existe existem
visitante pedido museu colecao acervo trabalho exemplar conjunto parte partes tipo forma formas data datado datada
seculo seculos anos cerca meados inicio finais final primeira segunda metade autor autoria atribuido atribuida
executado executada realizado realizada produzido produzida criado criada decorado decorada composto composta
encontrei estas alguns algumas outras outros todas todos cada aqui eis podem pode ver veem""".split())


class _Registo:
    def __init__(self):
        self.events = []

    async def log_event(self, event):
        self.events.append(event)


class _Escuta:
    """Um LLM: guarda os pedidos e as respostas e deixa passar."""

    def __init__(self, llm):
        self.llm, self.pedidos = llm, []

    async def generate(self, **kwargs):
        resposta = await self.llm.generate(**kwargs)
        self.pedidos.append((kwargs.get("message"), resposta))
        return resposta

    async def generate_stream(self, **kwargs):
        resposta = await self.llm.generate_stream(**kwargs)
        self.pedidos.append((kwargs.get("message"), resposta))
        return resposta

    def __getattr__(self, nome):
        return getattr(self.llm, nome)


def servico(extra) -> ChatService:
    settings = Settings(**{**dict(QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False, CHAT_AUTHOR_CARD=False), **extra})
    s = object.__new__(ChatService)
    s.settings = settings
    s.session_store = ChatSessionStore(settings)
    s.llm_service = _Escuta(get_llm_service())
    s.preprocess_llm_service = _Escuta(get_preprocess_llm_service())
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


def palavras_novas(frase: str, visto: str) -> list[str]:
    """As palavras (e os números) da frase que o texto que o modelo viu não tem, pelo princípio da palavra."""
    conhecidas = set(re.findall(r"[a-z0-9]+", fold(visto)))
    prefixos = {p[:5] for p in conhecidas}
    novas = []
    for palavra in re.findall(r"[a-z0-9]+", fold(frase)):
        if palavra.isdigit():
            if palavra not in conhecidas:
                novas.append(palavra)
        elif len(palavra) >= 4 and palavra not in GERAIS and palavra not in conhecidas and palavra[:5] not in prefixos:
            novas.append(palavra)
    return list(dict.fromkeys(novas))


def gerundios(frase: str, ficha: str) -> list[str]:
    na_ficha = set(re.findall(r"[a-z-]+", fold(ficha)))
    return [g for g in GERUNDIO.findall(fold(frase)) if g not in NAO_GERUNDIO and g not in na_ficha]


async def medir(nome, extra, lista, grupo, registo):
    s = servico(extra)
    tempos = []
    for caso in lista:
        docs = fichas(s.opensearch_gateway, caso["inventarios"])
        fixar(s, docs, caso["visual"])
        for _ in range(VEZES):
            pedido = ChatMessageRequest(museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga",
                                        language=caso.get("lingua", "pt"), message=caso["pedido"])
            s.llm_service.pedidos.clear()
            s.preprocess_llm_service.pedidos.clear()
            enviado = []

            async def status_cb(evento):
                if evento.get("type") == "token":
                    if evento.get("seq") == 0:
                        enviado.clear()
                    enviado.append(evento["delta"])

            t0 = time.monotonic()
            resposta = await s.handle_message(pedido, status_cb=status_cb)
            tempos.append(time.monotonic() - t0)
            texto = resposta.reply or ""
            # o que o modelo de pré-processamento viu e escreveu, por ficha
            vistos = {}
            abertura_vista = None
            for prompt, feito in s.preprocess_llm_service.pedidos:
                prompt = str(prompt or "")
                marca = "FICHA:\n" if "FICHA:\n" in prompt else ("RECORD:\n" if "RECORD:\n" in prompt else None)
                if marca:
                    ficha = prompt.split(marca, 1)[1]
                    com_pedido = "«" in prompt.split(marca, 1)[0]
                    escrito = (feito.parsed_json or {}).get("frase") if isinstance(feito.parsed_json, dict) else None
                    vistos[ficha] = {"com_pedido": com_pedido, "escrito": escrito}
                elif "FICHAS:\n" in prompt or "RECORDS:\n" in prompt:
                    abertura_vista = (feito.parsed_json or {}).get("abertura") if isinstance(feito.parsed_json, dict) else None
            linhas = [m["body"] for m in map(ITEM.match, texto.split("\n")) if m]
            cartoes = []
            for n, cartao in enumerate(resposta.artifact_results):
                inventario = str(cartao.inventory_number or "")
                ficha = next((f for f in vistos if f"Inventário: {inventario}\n" in f or f"Inventário: {inventario} " in f.split("\n", 1)[0] + " "), "")
                if extra.get("CHAT_CARD_SUMMARIES"):
                    frase = cartao.card_summary
                else:   # a frase da linha N da lista
                    frase = re.split(r"\s[–—-]\s|:\s", linhas[n], maxsplit=1)[-1].replace("*", "") if n < len(linhas) else None
                da_ficha = bool(frase) and frase == s._first_sentence(cartao.description)
                visto = ficha + ("\n" + caso["pedido"] if vistos.get(ficha, {}).get("com_pedido") else "")
                cartoes.append({
                    "inventario": inventario, "titulo": cartao.title, "categoria": cartao.category, "frase": frase,
                    "da_ficha": da_ficha, "ficha": ficha, "com_pedido": vistos.get(ficha, {}).get("com_pedido"),
                    "novas": palavras_novas(frase or "", visto) if frase and not da_ficha else [],
                    "gerundios": gerundios(frase or "", ficha) if frase else [],
                    # o que o modelo escreveu, antes de o serviço trocar o gerúndio pelo infinitivo
                    "escrito": vistos.get(ficha, {}).get("escrito"),
                    "gerundios_do_modelo": gerundios(str(vistos.get(ficha, {}).get("escrito") or ""), ficha),
                    "repete_titulo": bool(frase and cartao.title and len(fold(cartao.title)) >= 6 and fold(frase).startswith(fold(cartao.title))),
                    "diz_inventario": bool(frase and inventario and fold(inventario) in fold(frase)),
                    "visita": bool(frase and VISITA.search(frase)),
                })
            visto_stream = "".join(enviado)
            registo.append({
                "variante": nome, "grupo": grupo, "pedido": caso["pedido"], "visual": caso["visual"], "segundos": round(tempos[-1], 2),
                "texto": texto, "abertura_do_pre": abertura_vista, "enviado_igual": visto_stream.rstrip() == texto.rstrip(),
                "enviou": bool(enviado), "chamou_amalia": len(s.llm_service.pedidos), "modelo": resposta.model_hint, "cartoes": cartoes,
            })
            if "--ver" in sys.argv:
                print(f"\n--- [{nome}] {caso['pedido']} ({tempos[-1]:.1f} s)\n{texto}")
                for c in cartoes:
                    print(f"   · {c['inventario']:14} {c['frase']}" + (f"   ‹novas: {', '.join(c['novas'])}›" if c["novas"] else ""))


def resumo(nome, grupo, registo):
    linhas = [r for r in registo if r["variante"] == nome and r["grupo"] == grupo]
    if not linhas:
        return
    cartoes = [c for r in linhas for c in r["cartoes"]]
    frases = [c for c in cartoes if c["frase"]]
    tempos = sorted(r["segundos"] for r in linhas)
    letras = sorted(len(r["texto"]) for r in linhas)
    n = len(linhas)
    comprimento = sorted(len(c["frase"]) for c in frases) or [0]
    palavras = sorted(len(c["frase"].split()) for c in frases) or [0]
    print(f"{nome:15} [{grupo}] respostas {n} · tempo mediano {tempos[n // 2]:.1f} s (máx {tempos[-1]:.1f} s) · texto mediano {letras[n // 2]} letras"
          f" · texto com lista {sum(1 for r in linhas if any(ITEM.match(l) for l in r['texto'].split(chr(10))))}, com negrito {sum('**' in r['texto'] for r in linhas)}"
          f" · streaming igual ao final {sum(r['enviado_igual'] for r in linhas if r['enviou'])}/{sum(r['enviou'] for r in linhas)}"
          f" · AMÁLIA chamada em {sum(1 for r in linhas if r['chamou_amalia'])}")
    print(f"{'':15} cartões {len(cartoes)} · com frase {len(frases)} (da ficha {sum(c['da_ficha'] for c in frases)})"
          f" · palavras: mediana {palavras[len(palavras) // 2]}, máx {palavras[-1]}, mais de 25: {sum(p > 25 for p in palavras)}"
          f" · letras: mediana {comprimento[len(comprimento) // 2]}, máx {comprimento[-1]}, mais de 220: {sum(x > 220 for x in comprimento)}"
          f" · repetem o título {sum(c['repete_titulo'] for c in frases)} · dizem o inventário {sum(c['diz_inventario'] for c in frases)}"
          f" · falam da visita {sum(c['visita'] for c in frases)} · com gerúndio que a ficha não tem {sum(bool(c['gerundios']) for c in frases)}"
          f" (como o modelo as escreveu: {sum(bool(c['gerundios_do_modelo']) for c in frases)})"
          f" · com palavras que a ficha não tem {sum(bool(c['novas']) for c in frases)}")


async def principal():
    gerais = [dict(c, visual=False) for c in casos.CASOS_GERAIS] + [dict(c, visual=True) for c in casos.CASOS]
    so = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--variantes=")), list(VARIANTES))
    registo = []
    for nome, extra in VARIANTES.items():
        if nome not in so:
            continue
        if "--so-extra" not in sys.argv:
            await medir(nome, extra, gerais, "20 casos", registo)
        await medir(nome, extra, EXTRA, "diretas", registo)
        resumo(nome, "20 casos", registo)
        resumo(nome, "diretas", registo)
    if GUARDAR:
        with open(GUARDAR, "w", encoding="utf-8") as saida:
            json.dump(registo, saida, ensure_ascii=False, indent=1)
        print(f"guardado em {GUARDAR}")


asyncio.run(principal())
