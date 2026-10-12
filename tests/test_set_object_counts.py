"""set_object_counts nos cartões: total de peças de cada conjunto, pela ordem
de set_ids, numa só ida ao índice.

    python -m unittest tests.test_set_object_counts
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
from app.services.opensearch_client import OpenSearchGateway  # noqa: E402


def doc(aid, set_ids=(), sets=()):
    return {"artifact_id": aid, "inventory_number": aid, "title": aid, "museum_id": "mnt", "image_count": 0,
            "set_ids": list(set_ids), "sets": list(sets), "set_numbers": [s.split(":")[-1] for s in set_ids]}


class _Gateway:
    def __init__(self, counts=None, error=None):
        self.counts, self.error, self.calls = counts or {}, error, []

    async def fetch_images_by_artifact_ids(self, **_kwargs):
        return []

    async def fetch_set_object_counts(self, *, set_ids):
        self.calls.append(list(set_ids))
        if self.error:
            raise self.error
        return {sid: self.counts.get(sid, 0) for sid in set_ids}


def _service(gateway) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = Settings(_env_file=None, QUERY_LOG_ENABLED=False, BACKEND_LOG_ENABLED=False)
    service.opensearch_gateway = gateway
    return service


def build(service, docs):
    return asyncio.run(service._build_artifact_results(museum_slug="mnt", museum_id="mnt", artifact_docs=docs))


class TestAttach(unittest.TestCase):
    def test_counts_follow_set_ids_order(self) -> None:
        gateway = _Gateway({"raiz:conjunto:30964": 10, "raiz:conjunto:3983": 2})
        results = build(_service(gateway), [
            doc("a", ["raiz:conjunto:30964"], ["Botões (dez)"]),
            doc("b", ["raiz:conjunto:3983", "raiz:conjunto:30964"], ["Fivelas (par)", "Botões (dez)"]),
            doc("c"),
        ])
        self.assertEqual([r.set_object_counts for r in results], [[10], [2, 10], []])
        self.assertEqual([len(r.set_ids) for r in results], [len(r.set_object_counts) for r in results])
        self.assertEqual(len(gateway.calls), 1)   # uma ida ao índice para todos os cartões

    def test_unknown_set_is_zero(self) -> None:
        results = build(_service(_Gateway({})), [doc("a", ["raiz:conjunto:999"])])
        self.assertEqual(results[0].set_object_counts, [0])

    def test_no_sets_no_call(self) -> None:
        gateway = _Gateway()
        results = build(_service(gateway), [doc("a"), doc("b")])
        self.assertEqual([r.set_object_counts for r in results], [[], []])
        self.assertEqual(gateway.calls, [])

    def test_index_failure_gives_zeros(self) -> None:
        results = build(_service(_Gateway(error=RuntimeError("em baixo"))), [doc("a", ["s1", "s2"])])
        self.assertEqual(results[0].set_object_counts, [0, 0])

    def test_field_in_json(self) -> None:
        results = build(_service(_Gateway({"s1": 4})), [doc("a", ["s1"])])
        payload = results[0].model_dump(mode="json")
        self.assertEqual((payload["set_ids"], payload["set_object_counts"]), (["s1"], [4]))


class _FakeClient:
    def __init__(self):
        self.searches = []

    def mget(self, *, index, body, _source):
        self.mget_args = (index, body, _source)
        found = {"raiz:conjunto:30964": 10, "raiz:conjunto:3983": 2}
        return {"docs": [{"_id": i, "found": i in found, **({"_source": {"artifact_count": found[i]}} if i in found else {})} for i in body["ids"]]}

    def search(self, *, index, body):
        self.searches.append((index, body))
        return {"aggregations": {"g": {"buckets": [{"key": "9", "doc_count": 3}]}}}


class TestGateway(unittest.TestCase):
    def test_mget_then_fallback_count(self) -> None:
        gateway = OpenSearchGateway(Settings(_env_file=None, OPENSEARCH_INDEX_CONJUNTO="sets", OPENSEARCH_INDEX_ARTIFACT="arts"))
        fake = _FakeClient()
        gateway._client = fake
        counts = asyncio.run(gateway.fetch_set_object_counts(set_ids=["raiz:conjunto:30964", "raiz:conjunto:3983", "9", "x", "raiz:conjunto:30964"]))
        self.assertEqual(counts, {"raiz:conjunto:30964": 10, "raiz:conjunto:3983": 2, "9": 3, "x": 0})
        self.assertEqual(fake.mget_args[0], "sets")
        self.assertEqual(sorted(fake.mget_args[1]["ids"]), ["9", "raiz:conjunto:30964", "raiz:conjunto:3983", "x"])
        self.assertEqual(fake.searches[0][0], "arts")
        self.assertEqual(sorted(fake.searches[0][1]["query"]["terms"]["set_ids"]), ["9", "x"])

    def test_empty(self) -> None:
        gateway = OpenSearchGateway(Settings(_env_file=None))
        gateway._client = SimpleNamespace()   # não pode ser usado
        self.assertEqual(asyncio.run(gateway.fetch_set_object_counts(set_ids=[])), {})


if __name__ == "__main__":
    unittest.main()
