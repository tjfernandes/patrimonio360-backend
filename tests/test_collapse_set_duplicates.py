"""Fichas iguais do mesmo conjunto juntas num cartão (CHAT_COLLAPSE_SET_DUPLICATES):
«Botões (dez)», 38195/1 a 38195/10, com título e descrição iguais, ocupavam a
página inteira; passam a ser um cartão, e a pesquisa pede mais candidatos para
a página continuar cheia.

    python -m unittest tests.test_collapse_set_duplicates
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ChatResultsPageRequest  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.chat_session_store import ChatSessionStore  # noqa: E402

BORBOLETA = "Botão de madeira de cor natural pintada representando borboleta a preto e branco."


def botao(n: int, **extra) -> dict:
    doc = {
        "artifact_id": f"raiz:movel:10634{n:02d}", "inventory_number": f"38195/{n}", "title": "Botão",
        "description": BORBOLETA, "sets": ["Botões (dez)"], "set_ids": ["raiz:conjunto:30964"],
        "set_numbers": ["38195"], "museum_id": "mnt", "in_tour": False, "image_count": 1, "score": 1.0 - n / 100,
    }
    doc.update(extra)
    return doc


def outro(inv: str, title: str, **extra) -> dict:
    doc = {"artifact_id": f"a-{inv}", "inventory_number": inv, "title": title, "description": f"Descrição de {inv}.",
           "museum_id": "mnt", "in_tour": False, "image_count": 1}
    doc.update(extra)
    return doc


# A ordem da produção para «encontra botões com insetos» (janela alargada a 24):
# uma caixa, os dez botões iguais, e depois as fichas que estavam em 12.º a 24.º.
PRODUCAO = (
    [outro("1704", "Caixa/Masculino")]
    + [botao(n) for n in (1, 7, 4, 9, 5, 8, 6, 10, 3, 2)]
    + [outro(inv, t) for inv, t in [("14988", "Botões (três)"), ("13913", "Botões (oito)"), ("8250", "Botão"),
                                    ("21429", "Botão"), ("13682", "Cartela com botões"), ("11472", "Botões (dois)"),
                                    ("4658", "Botão"), ("4659", "Botão"), ("10288", "Botão"), ("8252", "Botão"),
                                    ("37253", "Botão"), ("18957", "Botão"), ("21443", "Botão")]]
)


def _service(**flags) -> ChatService:
    service = object.__new__(ChatService)
    base = dict(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False, CHAT_COLLAPSE_SET_DUPLICATES=True,
                CHAT_COMPACT_ANSWER_CONTEXT=True)
    base.update(flags)
    service.settings = Settings(**base)
    service.session_store = ChatSessionStore(service.settings)
    return service


class TestCollapse(unittest.TestCase):
    def test_ten_identical_members_become_one_card_in_place(self) -> None:
        docs = _service()._collapse_identical_set_members(PRODUCAO)
        self.assertEqual(len(docs), 1 + 1 + 13)
        self.assertEqual([d["inventory_number"] for d in docs[:3]], ["1704", "38195/1", "14988"])
        card = docs[1]
        self.assertEqual(card["group_size"], 10)
        self.assertEqual(card["group_inventory_numbers"][:3], ["38195/1", "38195/2", "38195/3"])
        self.assertEqual(card["group_inventory_numbers"][-1], "38195/10")
        self.assertEqual(len(card["group_artifact_ids"]), 10)
        self.assertEqual(card["group_artifact_ids"][0], "raiz:movel:1063401")
        # os originais não são alterados
        self.assertNotIn("group_size", PRODUCAO[1])

    def test_member_in_the_tour_represents_the_group(self) -> None:
        docs = _service()._collapse_identical_set_members([botao(1), botao(2), botao(3, in_tour=True), botao(4)])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["inventory_number"], "38195/3")
        self.assertTrue(docs[0]["in_tour"])
        self.assertEqual(docs[0]["group_size"], 4)

    def test_different_text_in_the_same_set_is_not_collapsed(self) -> None:
        conjunto = [
            outro("100/1", "Casaco/Feminino", set_ids=["raiz:conjunto:1"]),
            outro("100/2", "Saia/Feminino", set_ids=["raiz:conjunto:1"]),
            outro("100/3", "Colete/Feminino", set_ids=["raiz:conjunto:1"]),
        ]
        self.assertEqual(len(_service()._collapse_identical_set_members(conjunto)), 3)
        variante = [botao(1), botao(2, description=BORBOLETA + " Falta uma pata.")]
        self.assertEqual(len(_service()._collapse_identical_set_members(variante)), 2)

    def test_inventory_root_replaces_a_missing_set(self) -> None:
        sem_conjunto = [botao(1, set_ids=[], sets=[]), botao(2, set_ids=[], sets=[]), outro("7", "Botão", description=BORBOLETA)]
        docs = _service()._collapse_identical_set_members(sem_conjunto)
        self.assertEqual([d["inventory_number"] for d in docs], ["38195/1", "7"])
        self.assertEqual(docs[0]["group_size"], 2)
        # duas fichas «Botão» com a mesma descrição mas sem conjunto nem «N/M» ficam separadas
        soltas = [outro("7", "Botão", description=BORBOLETA), outro("8", "Botão", description=BORBOLETA)]
        self.assertEqual(len(_service()._collapse_identical_set_members(soltas)), 2)

    def test_flag_off_returns_the_same_list(self) -> None:
        service = _service(CHAT_COLLAPSE_SET_DUPLICATES=False)
        self.assertIs(service._collapse_identical_set_members(PRODUCAO), PRODUCAO)
        self.assertEqual(service._collapse_fetch_size(8), 8)

    def test_fetch_size_is_the_page_times_the_factor(self) -> None:
        self.assertEqual(_service()._collapse_fetch_size(8), 24)
        self.assertEqual(_service(CHAT_COLLAPSE_FETCH_FACTOR=2)._collapse_fetch_size(8), 16)
        self.assertEqual(_service()._collapse_fetch_size(100), 200)


class TestCardsAndPrompt(unittest.TestCase):
    def test_compact_card_says_it_is_a_set(self) -> None:
        service = _service()
        docs = service._collapse_identical_set_members([botao(1), botao(2), botao(10)])
        card = service._format_docs_compact(docs=docs)
        self.assertIn("[doc_1] Título: Botão | Inventário: 38195/1", card)
        self.assertIn("Conjunto de 3 peças iguais, uma ficha cada (38195/1 a 38195/10); este cartão representa todas", card)
        self.assertNotIn("Conjunto de", service._format_docs_compact(docs=[botao(1)]))

    def test_legacy_prompt_only_carries_group_fields_when_grouped(self) -> None:
        service = _service()
        plain = service._format_docs_for_prompt(docs=[botao(1)], top_k=1)
        self.assertNotIn("group_", plain)
        grouped = service._format_docs_for_prompt(docs=service._collapse_identical_set_members([botao(1), botao(2)]), top_k=1)
        self.assertIn('"group_size": 2', grouped)
        self.assertNotIn("group_artifact_ids", grouped)

    def test_artifact_result_carries_the_group(self) -> None:
        service = _service()
        service.opensearch_gateway = SimpleNamespace(fetch_images_by_artifact_ids=_no_images)
        docs = service._collapse_identical_set_members([botao(1), botao(2), outro("7", "Leque")])
        results = asyncio.run(service._build_artifact_results(museum_slug="mnt", museum_id="mnt", artifact_docs=docs))
        self.assertEqual([(r.inventory_number, r.group_size) for r in results], [("38195/1", 2), ("7", 1)])
        self.assertEqual(results[0].group_inventory_numbers, ["38195/1", "38195/2"])
        self.assertEqual(results[1].group_inventory_numbers, [])
        self.assertEqual(service._log_retrieved_artifact_entries(docs, source="hybrid")[0]["group_size"], 2)
        self.assertNotIn("group_size", service._log_retrieved_artifact_entries(docs, source="hybrid")[1])


async def _no_images(**_kwargs):
    return []


class _Gateway:
    """Devolve a lista da produção por fatias, como o OpenSearch faz com from/size."""

    def __init__(self, docs=PRODUCAO) -> None:
        self.docs, self.calls = docs, []

    async def search_relevant_context_page(self, *, from_offset, page_size, **kwargs):
        self.calls.append({"from_offset": from_offset, "page_size": page_size, **kwargs})
        return SimpleNamespace(results=[dict(d) for d in self.docs[from_offset:from_offset + page_size]],
                               total=len(self.docs), query_body={})

    async def fetch_images_by_artifact_ids(self, **_kwargs):
        return []

    async def search_artifacts_by_inventory_candidates(self, **_kwargs):
        return []


class _Embedder:
    async def embed_text(self, text):
        return [0.1, 0.2]


def _retrieval_service(**flags) -> ChatService:
    service = _service(CHAT_ENABLE_LLM_LEXICAL_QUERY=False, CHAT_USE_QUERY_EMBEDDINGS=True, CHAT_RETRIEVAL_TOP_K=8,
                       CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8, CHAT_ENABLE_RERANKING=False, CHAT_LAY_VOCABULARY_EXPANSION=False,
                       **flags)
    service.opensearch_gateway = _Gateway()
    service.embedding_provider = _Embedder()
    service.llm_service = None
    service.tour_navigation_service = SimpleNamespace(resolve_targets=lambda **_k: [])
    service.query_logger = None
    return service


class TestRetrieval(unittest.TestCase):
    def _retrieve(self, service):
        return asyncio.run(service._retrieve_context(
            museum_slug="mnt", museum_id="mnt", query="botões com insetos", filters={}, sort={}, result_window_size=8,
        ))

    def test_first_page_fetches_wider_and_comes_back_full(self) -> None:
        service = _retrieval_service()
        _context, total, docs, request = self._retrieve(service)
        self.assertEqual(service.opensearch_gateway.calls[0]["page_size"], 24)
        self.assertEqual(service.opensearch_gateway.calls[0]["from_offset"], 0)
        self.assertEqual(len(docs), 8)
        self.assertEqual([d["inventory_number"] for d in docs], ["1704", "38195/1", "14988", "13913", "8250", "21429", "13682", "11472"])
        self.assertEqual(docs[1]["group_size"], 10)
        self.assertEqual(total, len(PRODUCAO))

    def test_flag_off_keeps_the_old_shape(self) -> None:
        service = _retrieval_service(CHAT_COLLAPSE_SET_DUPLICATES=False)
        _context, _total, docs, _request = self._retrieve(service)
        self.assertEqual(service.opensearch_gateway.calls[0]["page_size"], 8)
        self.assertEqual([d["inventory_number"] for d in docs], ["1704", "38195/1", "38195/7", "38195/4", "38195/9", "38195/5", "38195/8", "38195/6"])
        self.assertNotIn("group_size", docs[1])

    def test_next_page_starts_from_zero_and_slices_after_collapsing(self) -> None:
        service = _retrieval_service()
        state = service.session_store.load_or_create(conversation_id="c1", museum_slug="mnt")
        payload = ChatResultsPageRequest(museum_slug="mnt", museum_id="mnt", conversation_id="c1", results_page=2, results_page_size=8)
        request = {"kind": "text", "museum_slug": "mnt", "museum_id": "mnt", "query_text": "botões com insetos",
                   "lexical_query": "botões com insetos", "query_embedding": [0.1, 0.2], "filters": {}, "sort": {},
                   "results_total": len(PRODUCAO), "retrieval_window_size": 1000}
        service._generate_results_page_reply = _no_reply
        response = asyncio.run(service._materialize_text_results_page(state=state, payload=payload, request=request, page=2, page_size=8))
        call = service.opensearch_gateway.calls[0]
        self.assertEqual((call["from_offset"], call["page_size"]), (0, 48))
        # página 2 = cartões 9 a 16 da lista já sem os nove botões repetidos
        self.assertEqual([a.inventory_number for a in response.artifact_results], ["4658", "4659", "10288", "8252", "37253", "18957", "21443"])
        self.assertEqual(response.results_page, 2)

    def test_next_page_flag_off_uses_the_offset(self) -> None:
        service = _retrieval_service(CHAT_COLLAPSE_SET_DUPLICATES=False)
        state = service.session_store.load_or_create(conversation_id="c2", museum_slug="mnt")
        payload = ChatResultsPageRequest(museum_slug="mnt", museum_id="mnt", conversation_id="c2", results_page=2, results_page_size=8)
        request = {"kind": "text", "museum_slug": "mnt", "museum_id": "mnt", "query_text": "q", "lexical_query": "q",
                   "query_embedding": [0.1, 0.2], "filters": {}, "sort": {}, "results_total": len(PRODUCAO)}
        service._generate_results_page_reply = _no_reply
        response = asyncio.run(service._materialize_text_results_page(state=state, payload=payload, request=request, page=2, page_size=8))
        call = service.opensearch_gateway.calls[0]
        self.assertEqual((call["from_offset"], call["page_size"]), (8, 8))
        self.assertEqual(len(response.artifact_results), 8)


async def _no_reply(**_kwargs):
    return None


if __name__ == "__main__":
    unittest.main()
