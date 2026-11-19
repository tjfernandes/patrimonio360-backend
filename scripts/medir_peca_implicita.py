"""Mede a peça implícita pelo código do serviço: regras de hoje contra CHAT_IMPLICIT_ARTIFACT_MODEL, com o Qwen real.

  python scripts/medir_peca_implicita.py afinar|validar|confirmar|aberta [x3]
Corre da raiz do repositório, com o .env do modelo de pré-processamento:  python scripts/medir_peca_implicita.py validar x3
"""
import asyncio, os, sys, time

W = os.getcwd()
sys.path.insert(0, W)
import logging  # noqa: E402

logging.disable(logging.CRITICAL)
from app.schemas.chat import ArtifactResult  # noqa: E402
from app.services.llm_service import get_preprocess_llm_service  # noqa: E402
from tests.test_tour_location import _Base  # noqa: E402
from tests import implicit_artifact_cases as casos  # noqa: E402

base = _Base("setUp"); base.setUp()
REFS = ("artifact_id", "inventory_number", "title")


class _Conjuntos:
    """As peças de cada conjunto, como o índice as devolve (o cartão incluído)."""

    def __init__(self):
        self.por_id, self.chamadas = {}, 0

    async def fetch_artifacts_by_entity(self, *, entity_id, **_kwargs):
        self.chamadas += 1
        return [dict(m) for m in self.por_id.get(entity_id, [])], len(self.por_id.get(entity_id, []))


class _Cronometro:
    def __init__(self, llm):
        self.llm, self.ms = llm, []

    async def generate(self, **kwargs):
        t = time.perf_counter()
        try:
            return await self.llm.generate(**kwargs)
        finally:
            self.ms.append((time.perf_counter() - t) * 1000)


def servico(flag):
    s = base._service(base._settings(CHAT_IMPLICIT_ARTIFACT_CONTEXT=True, CHAT_IMPLICIT_ARTIFACT_MODEL=flag))
    s._preprocess_llm_service = _Cronometro(get_preprocess_llm_service())
    s.opensearch_gateway = _Conjuntos()
    return s


def preparar(s, nome, cenario):
    """Guarda na conversa os cartões do cenário, como depois de uma resposta, e dá ao índice de teste as peças dos conjuntos."""
    cartoes = []
    for n, c in enumerate(cenario["visiveis"]):
        conjunto = f"conj-{nome}-{n}" if c.get("membros") else None
        if conjunto:
            s.opensearch_gateway.por_id[conjunto] = [{k: c.get(k) for k in REFS}] + [dict(m) for m in c["membros"]]
        cartoes.append(ArtifactResult(artifact_id=c["artifact_id"], inventory_number=c.get("inventory_number"), title=c.get("title"),
                                      set_ids=[conjunto] if conjunto else [], sets=[c["conjunto"]] if conjunto else []))
    state = s.session_store.load_or_create(conversation_id=nome, museum_slug=cenario["museu"])
    s._cache_last_retrieval_results(state=state, artifact_results=cartoes, image_matches=[], navigation_targets=[], default_page_size=8,
                                    retrieval_request={"kind": "text"})
    s.session_store.save(state)


async def resolver(s, flag, nome, mensagem):
    cenario = casos.CENARIOS[nome]
    metadata = {"visible_artifacts": [{k: c.get(k) for k in REFS} for c in cenario["visiveis"]], "context_artifact": cenario.get("foco")}
    if flag:
        achado = await s._resolve_implicit_artifact_with_model(mensagem, metadata, museum_slug=cenario["museu"], museum_id=cenario["museu"], conversation_id=nome)
    else:
        achado = s._resolve_implicit_artifact(mensagem, metadata, museum_slug=cenario["museu"], museum_id=cenario["museu"])
    return (achado or {}).get("artifact_id"), (achado or {}).get("resolved_by")


async def main():
    grupo = {"afinar": casos.AFINAR, "validar": casos.VALIDAR, "confirmar": casos.CONFIRMAR, "aberta": casos.ABERTA_COM_ASSUNTO}[sys.argv[1]]
    vezes = 3 if "x3" in sys.argv else 1
    print(f"casos: {len(grupo)}")
    for flag in (False, True):
        for _ in range(1 if not flag else vezes):
            s = servico(flag)
            for nome, cenario in casos.CENARIOS.items():
                preparar(s, nome, cenario)
            saidas = []
            for nome, mensagem, _esperado in grupo:
                saidas.append(await resolver(s, flag, nome, mensagem))
            erros = [(c, m, e, r, via) for (c, m, e), (r, via) in zip(grupo, saidas) if r != e]
            ms = sorted(s._preprocess_llm_service.ms)
            vias = {}
            for _r, via in saidas:
                vias[via] = vias.get(via, 0) + 1
            extra = f" | perguntas ao modelo {len(ms)}, mediana {ms[len(ms) // 2]:.0f} ms, máximo {ms[-1]:.0f} ms | por: {vias}" if ms else f" | por: {vias}"
            print(f"\n== {'com o modelo' if flag else 'regras de hoje'}: {len(grupo) - len(erros)} de {len(grupo)}{extra}")
            for c, m, e, r, via in erros:
                print(f"      {c:14} {m:45} esperado {e} | deu {r} ({via})")


asyncio.run(main())
