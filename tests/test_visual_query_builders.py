"""E7A — as queries kNN visuais não transportam preferência in_tour.

A preferência in_tour visual é pós-fusão (Etapa 10). Estes testes fixam o
contrato dos 4 builders visuais: apenas kNN + filtros obrigatórios; o campo
``in_tour`` continua no ``_source`` (pós-processamento), nunca na query.

    python -m unittest tests.test_visual_query_builders
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.opensearch_client import OpenSearchGateway  # noqa: E402


class RecorderClient:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"hits": {"hits": [], "total": {"value": 0}}}


def make_gateway():
    gateway = OpenSearchGateway.__new__(OpenSearchGateway)
    gateway.settings = SimpleNamespace(
        OPENSEARCH_INDEX_IMAGE="cultural_heritage_images_v4",
        CHAT_IN_TOUR_BOOST=0.0,
        IMAGE_IN_TOUR_BOOST=0.05,  # legacy: mesmo definido, NUNCA entra na query
    )
    client = RecorderClient()
    gateway._ensure_client = lambda: client
    return gateway, client


def query_section(call) -> str:
    return json.dumps(call["body"]["query"])


class TestVisualBuildersHaveNoInTour(unittest.TestCase):
    def _assert_clean(self, call):
        query = query_section(call)
        self.assertNotIn("in_tour", query, f"query contem in_tour: {query[:300]}")
        self.assertNotIn("constant_score", query)
        self.assertNotIn("function_score", query)
        # filtro de museu obrigatorio presente
        self.assertIn('{"term": {"museum_id": "mnt"}}', query)
        # o campo continua disponivel para pos-processamento
        self.assertIn("in_tour", call["body"]["_source"])
        self.assertEqual(call["index"], "cultural_heritage_images_v4")

    def test_t2i_i2i_page_query_is_clean(self) -> None:
        gateway, client = make_gateway()
        gateway._search_similar_images_page_sync(
            museum_slug="museu_nacional_do_traje",
            museum_id="mnt",
            image_embedding=[0.1, 0.2],
            from_offset=0,
            page_size=8,
            retrieval_window_size=150,
        )
        self._assert_clean(client.calls[0])
        self.assertIn('"knn"', query_section(client.calls[0]))

    def test_plain_image_query_is_clean(self) -> None:
        gateway, client = make_gateway()
        gateway._search_similar_images_sync(
            museum_slug="museu_nacional_do_traje",
            museum_id="mnt",
            image_embedding=[0.1, 0.2],
            top_k=6,
        )
        self._assert_clean(client.calls[0])

    def test_multi_view_query_is_clean(self) -> None:
        gateway, client = make_gateway()
        gateway._search_similar_images_multi_sync(
            museum_slug="museu_nacional_do_traje",
            museum_id="mnt",
            image_embeddings=[[0.1, 0.2], [0.3, 0.4]],
            top_k=5,
        )
        self._assert_clean(client.calls[0])

    def test_multi_view_page_query_is_clean(self) -> None:
        gateway, client = make_gateway()
        gateway._search_similar_images_multi_page_sync(
            museum_slug="museu_nacional_do_traje",
            museum_id="mnt",
            image_embeddings=[[0.1, 0.2]],
            from_offset=0,
            page_size=5,
            retrieval_window_size=150,
        )
        self._assert_clean(client.calls[0])


if __name__ == "__main__":
    unittest.main()
