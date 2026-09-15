import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import Settings
from app.services.inference_runtime import GPUExecutor, InferenceBusyError
from app.services.query_logger import QueryLogger
from app.services.retrieval_profile import apply_profile, verify_profile_indices


class QueryLogCompactionTests(unittest.TestCase):
    def test_large_nested_vector_is_replaced(self):
        result = QueryLogger._compact_vectors({"query": {"embedding": [0.5] * 256}, "scores": [0.1, 0.2]})
        self.assertTrue(result["query"]["embedding"]["omitted"])
        self.assertEqual(result["query"]["embedding"]["dimension"], 256)
        self.assertEqual(result["scores"], [0.1, 0.2])

    def test_non_finite_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            QueryLogger._compact_vectors([float("nan")] * 128)


class RetrievalProfileTests(unittest.TestCase):
    def profile(self):
        return {"schema_version": 1, "approval": "validated", "snapshot_id": "snap",
                "text": {"model": "text/model", "revision": "t1", "dimension": 2560,
                         "normalization": "l2", "index": "exp-text"},
                "vl": {"model": "vl/model", "revision": "v1", "dimension": 2048,
                       "normalization": "l2", "index": "exp-vl"}}

    def test_unvalidated_profile_is_rejected(self):
        profile = self.profile(); profile["approval"] = "candidate"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"; path.write_text(json.dumps(profile))
            with self.assertRaises(ValueError): apply_profile(Settings(RETRIEVAL_PROFILE_PATH=str(path)))

    def test_atomic_bundle_applies_models_dimensions_and_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"; path.write_text(json.dumps(self.profile()))
            settings = apply_profile(Settings(RETRIEVAL_PROFILE_PATH=str(path)))
        self.assertEqual(settings.QWEN_TEXT_EMBEDDING_MODEL_ID, "text/model")
        self.assertEqual(settings.ARTIFACT_TEXT_EMBEDDING_DIMENSION, 2560)
        self.assertEqual(settings.OPENSEARCH_INDEX_IMAGE, "exp-vl")

    def test_index_model_space_mismatch_is_rejected(self):
        profile = self.profile()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"; path.write_text(json.dumps(profile))
            settings = Settings(RETRIEVAL_PROFILE_PATH=str(path))
            mapping = {"exp-text": {"mappings": {"_meta": {"snapshot_id": "snap", "space": {
                "model_id": "wrong", "revision": "t1", "normalization": "l2"}},
                "properties": {"text_embedding": {"dimension": 2560}}}}}
            client = SimpleNamespace(indices=SimpleNamespace(get_mapping=lambda index: mapping))
            with self.assertRaises(ValueError): verify_profile_indices(settings, client)


class GPUExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_worker_serializes_two_jobs(self):
        gate = __import__("threading").Event(); order = []
        executor = GPUExecutor(capacity=2)
        def first(): order.append("start1"); gate.wait(1); order.append("end1")
        def second(): order.append("start2"); order.append("end2")
        one = asyncio.create_task(executor.run(first)); await asyncio.sleep(0.02)
        two = asyncio.create_task(executor.run(second)); await asyncio.sleep(0.02)
        self.assertEqual(order, ["start1"]); gate.set(); await asyncio.gather(one, two); executor.close()
        self.assertEqual(order, ["start1", "end1", "start2", "end2"])

    async def test_queue_capacity_fails_fast(self):
        gate = __import__("threading").Event(); executor = GPUExecutor(capacity=1)
        running = asyncio.create_task(executor.run(lambda: gate.wait(1))); await asyncio.sleep(0.02)
        with self.assertRaises(InferenceBusyError): await executor.run(lambda: None)
        gate.set(); await running; executor.close()


if __name__ == "__main__": unittest.main()
