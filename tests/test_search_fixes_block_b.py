"""Pesquisa do chat, Bloco B: instrução de pergunta no embedding, campos da ficha
no ramo lexical e filtros dentro do kNN. Tudo atrás de flags desligadas por omissão.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.embeddings import EmbeddingProvider, QwenTextEmbedder  # noqa: E402
from app.services.opensearch_client import LEXICAL_METADATA_FIELDS, OpenSearchGateway  # noqa: E402

FLAGS = ("CHAT_RETRIEVAL_QUERY_INSTRUCTION", "CHAT_RETRIEVAL_LEXICAL_METADATA_FIELDS", "CHAT_RETRIEVAL_KNN_PREFILTER")


def _walk(node, key):
    found = []

    def visit(value):
        if isinstance(value, dict):
            if key in value and isinstance(value[key], dict):
                found.append(value[key])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(node)
    return found


def _body(filters=None, **flags):
    gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_ARTIFACT="arts", CHAT_LAY_VOCABULARY_EXPANSION=False, **flags))
    return gateway._build_query_body(museum_slug="mnaa", museum_id="mnaa", query_text="obras de machado de castro",
                                     lexical_query="machado castro", query_embedding=[0.1, 0.2], top_k=8, filters=filters or {}, sort={})


class TestFlagsDefaultOff(unittest.TestCase):
    def test_all_off(self) -> None:
        settings = Settings(_env_file=None)
        for flag in FLAGS:
            self.assertIs(getattr(settings, flag), False, flag)


class TestKnnPrefilter(unittest.TestCase):
    def test_off_the_knn_has_no_filter(self) -> None:
        (spec,) = [s["text_embedding"] for s in _walk(_body(), "knn")]
        self.assertNotIn("filter", spec)

    def test_on_the_museum_filter_goes_inside_the_knn(self) -> None:
        body = _body(CHAT_RETRIEVAL_KNN_PREFILTER=True)
        (spec,) = [s["text_embedding"] for s in _walk(body, "knn")]
        self.assertIn({"term": {"museum_id": "mnaa"}}, spec["filter"]["bool"]["filter"])
        # o filtro de fora mantém-se: o ramo lexical e a fusão continuam a contar com ele
        knn_branch = body["query"]["hybrid"]["queries"][0]["bool"]
        self.assertIn({"term": {"museum_id": "mnaa"}}, knn_branch["filter"])
        self.assertEqual(spec["k"], 1000)

    def test_on_the_era_filter_goes_inside_too(self) -> None:
        body = _body(filters={"_temporal_interval": {"start_year": 1400, "end_year": 1499}}, CHAT_RETRIEVAL_KNN_PREFILTER=True)
        (spec,) = [s["text_embedding"] for s in _walk(body, "knn")]
        outer = body["query"]["hybrid"]["queries"][0]["bool"]["filter"]
        self.assertEqual(spec["filter"]["bool"]["filter"], outer)
        self.assertGreater(len(outer), 1)


class TestLexicalMetadataFields(unittest.TestCase):
    @staticmethod
    def _clauses(body):
        return [m for m in _walk(body, "multi_match") if m["query"] == "machado castro"]

    def test_off_only_title_and_description(self) -> None:
        clauses = self._clauses(_body())
        self.assertEqual(len(clauses), 3)
        for clause in clauses:
            self.assertTrue(all(f.startswith(("title.", "description.")) for f in clause["fields"]), clause["fields"])
        self.assertEqual([c["type"] for c in clauses], ["best_fields", "phrase", "best_fields"])

    def test_on_author_category_material_and_production_center(self) -> None:
        clauses = self._clauses(_body(CHAT_RETRIEVAL_LEXICAL_METADATA_FIELDS=True))
        self.assertEqual(len(clauses), 3)
        for clause in clauses:
            for field in LEXICAL_METADATA_FIELDS:
                self.assertIn(field, clause["fields"])
            self.assertTrue(any(f.startswith("title.no_stem") for f in clause["fields"]))
        self.assertTrue(any(f.startswith("creators.text") for f in LEXICAL_METADATA_FIELDS))

    def test_on_the_all_words_clause_may_spread_over_fields(self) -> None:
        strict = [c for c in self._clauses(_body(CHAT_RETRIEVAL_LEXICAL_METADATA_FIELDS=True)) if c.get("operator") == "and"]
        self.assertEqual([c["type"] for c in strict], ["cross_fields"])
        self.assertEqual(strict[0]["boost"], 5)

    def test_constants_are_not_mutated_between_calls(self) -> None:
        _body(CHAT_RETRIEVAL_LEXICAL_METADATA_FIELDS=True)
        clauses = self._clauses(_body())
        self.assertTrue(all(len(c["fields"]) == 2 for c in clauses))


class _FakeModel:
    def __init__(self, prompts) -> None:
        self.prompts = prompts
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return SimpleNamespace(tolist=lambda: [[0.5, 0.5] for _ in texts])


def _embedder(prompts) -> QwenTextEmbedder:
    embedder = object.__new__(QwenTextEmbedder)
    embedder.model = _FakeModel(prompts)
    embedder.batch_size = 4
    return embedder


class TestQueryInstruction(unittest.TestCase):
    def test_embed_query_uses_the_models_query_prompt(self) -> None:
        embedder = _embedder({"query": "Instruct: …\nQuery:", "document": ""})
        self.assertEqual(embedder.embed_query("  Há obras de Machado de Castro?  "), [0.5, 0.5])
        (call,) = embedder.model.calls
        self.assertEqual(call["prompt_name"], "query")
        self.assertEqual(call["texts"], ["Há obras de Machado de Castro?"])
        self.assertTrue(call["normalize_embeddings"])

    def test_embed_document_never_uses_it(self) -> None:
        embedder = _embedder({"query": "Instruct: …\nQuery:"})
        embedder.embed_document("ficha")
        self.assertNotIn("prompt_name", embedder.model.calls[0])

    def test_model_without_query_prompt_embeds_as_document(self) -> None:
        embedder = _embedder({})
        embedder.embed_query("pergunta")
        self.assertNotIn("prompt_name", embedder.model.calls[0])

    def _provider(self, embedder) -> EmbeddingProvider:
        provider = object.__new__(EmbeddingProvider)
        provider.settings = SimpleNamespace(CHAT_BOUNDED_GPU_EXECUTION=False)
        provider.text_model_id = "fake"
        provider._get_text_embedder = lambda: embedder
        return provider

    def test_provider_routes_by_as_query(self) -> None:
        embedder = _embedder({"query": "Instruct: …\nQuery:"})
        provider = self._provider(embedder)
        asyncio.run(provider.embed_text("a"))
        asyncio.run(provider.embed_text("b", as_query=True))
        self.assertEqual(["prompt_name" in c for c in embedder.model.calls], [False, True])

    def test_provider_falls_back_for_embedders_without_embed_query(self) -> None:
        other = SimpleNamespace(embed_document=lambda text: [1.0])
        self.assertEqual(asyncio.run(self._provider(other).embed_text("a", as_query=True)), [1.0])


class _Gateway:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []

    async def fetch_images_by_artifact_ids(self, **_kwargs):
        return []

    async def fetch_set_object_counts(self, *, set_ids):
        return {s: 0 for s in set_ids}

    async def search_relevant_context_page(self, **kwargs):
        self.search_calls.append(kwargs)
        return SimpleNamespace(results=[], total=0, query_body={})

    async def search_artifacts_by_inventory_candidates(self, **_kwargs):
        return []


class _OldEmbedder:
    """Um embedder como os dos outros testes, sem o parâmetro novo."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed_text(self, text):
        self.calls.append(text)
        return [0.1, 0.2]


class _NewEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    async def embed_text(self, text, *, as_query=False):
        self.calls.append((text, as_query))
        return [0.1, 0.2]


def _service(embedder, **flags) -> ChatService:
    service = object.__new__(ChatService)
    base = dict(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False, CHAT_ENABLE_LLM_LEXICAL_QUERY=False,
                CHAT_LAY_VOCABULARY_EXPANSION=False, CHAT_RETRIEVAL_TOP_K=8, CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8)
    base.update(flags)
    service.settings = Settings(**base)
    service.opensearch_gateway = _Gateway()
    service.embedding_provider = embedder
    service.llm_service = None
    service.tour_navigation_service = SimpleNamespace(resolve_targets=lambda **_k: [], corrected_inventories=lambda **_k: [])
    return service


class TestChatSearchUsesTheInstruction(unittest.TestCase):
    @staticmethod
    def _retrieve(service, message="Há obras de Machado de Castro?"):
        return asyncio.run(service._retrieve_context(museum_slug="mnaa", museum_id="mnaa", query=message, filters={}, sort={},
                                                     result_window_size=8, inventory_query=message))

    def test_off_the_call_is_the_old_one(self) -> None:
        embedder = _OldEmbedder()
        self._retrieve(_service(embedder))
        self.assertEqual(embedder.calls, ["Há obras de Machado de Castro?"])

    def test_on_the_question_is_embedded_as_a_query(self) -> None:
        embedder = _NewEmbedder()
        self._retrieve(_service(embedder, CHAT_RETRIEVAL_QUERY_INSTRUCTION=True))
        self.assertEqual(embedder.calls, [("Há obras de Machado de Castro?", True)])


if __name__ == "__main__":
    unittest.main()
