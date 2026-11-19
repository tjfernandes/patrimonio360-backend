"""Ramo texto→legendas (experiência das legendas) — offline com stubs.

    python -m unittest tests.test_caption_text_branch
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # os stubs do teste vizinho

from app.core.config import Settings  # noqa: E402
from app.services.retrieval import multimodal_retrieval  # noqa: E402
from app.services.retrieval.multimodal_retrieval import MultimodalTextRetrieval  # noqa: E402
from test_multimodal_retrieval import (  # noqa: E402
    DECISION,
    StubGateway,
    StubProvider,
    art_doc,
    img_hit,
    make_settings,
    run,
)


class CaptionProvider(StubProvider):
    def __init__(self, *args, text_error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_error = text_error
        self.text_calls = 0

    async def embed_text(self, text, **kwargs):
        self.text_calls += 1
        self.text_kwargs = kwargs
        if self.text_error:
            raise self.text_error
        return [0.3, 0.4]


class CaptionGateway(StubGateway):
    def __init__(self, *args, caption_hits=None, caption_error=None, caption_count=1, count_error=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.caption_hits = caption_hits or []
        self.caption_error = caption_error
        self.caption_calls = []
        self.caption_count = caption_count
        self.count_error = count_error
        self.count_calls = []

    async def count_caption_docs(self, **kwargs):
        self.count_calls.append(kwargs)
        if self.count_error:
            raise self.count_error
        return self.caption_count

    async def search_caption_knn(self, **kwargs):
        self.caption_calls.append(kwargs)
        if self.caption_error:
            raise self.caption_error
        return list(self.caption_hits)


def cap_hit(aid, score=0.8, museum="mnt"):
    return {"artifact_id": aid, "museum_id": museum, "score": score}


def fuse(service, artifact_docs):
    return run(
        service.fuse_text_search(
            query_text="peças azuis",
            museum_slug="museu_nacional_do_traje",
            museum_id="mnt",
            artifact_docs=artifact_docs,
            router_decision=DECISION,
            trace_id="t1",
        )
    )


class TestCaptionBranch(unittest.TestCase):
    def build(self, *, gateway, provider=None, **settings):
        return MultimodalTextRetrieval(
            settings=make_settings(**settings),
            opensearch_gateway=gateway,
            embedding_provider=provider or CaptionProvider(),
        )

    def test_off_by_default_never_touches_captions(self) -> None:
        gateway = CaptionGateway(image_hits=[img_hit("A2", "i2", 0.9)], caption_hits=[cap_hit("A9")])
        provider = CaptionProvider()
        outcome = fuse(self.build(gateway=gateway, provider=provider), [art_doc("A1"), art_doc("A2")])
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A2", "A1"])
        self.assertEqual(gateway.caption_calls, [])
        self.assertEqual(provider.text_calls, 0)

    def test_replace_uses_captions_instead_of_text_to_image(self) -> None:
        gateway = CaptionGateway(
            image_hits=[img_hit("A7", "i7", 0.9)],
            caption_hits=[cap_hit("A2"), cap_hit("A3")],
            by_ids={"A3": art_doc("A3")},
        )
        provider = CaptionProvider()
        service = self.build(gateway=gateway, provider=provider, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_MODE="replace")
        outcome = fuse(service, [art_doc("A1"), art_doc("A2")])
        # A2: 1/62 + 0.7/61 > A1: 1/61 > A3: 0.7/62; A7 (só t2i) não entra
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A2", "A1", "A3"])
        self.assertEqual(gateway.search_calls, [])
        self.assertEqual(provider.calls, 0)
        self.assertEqual(outcome.matched_image_hits, [])
        a2 = next(r for r in outcome.fused if r.artifact_id == "A2")
        self.assertEqual(set(a2.sources), {"artifact_search", "text_to_caption"})
        call = gateway.caption_calls[0]
        self.assertEqual((call["index"], call["museum_id"], call["size"], call["knn_k"]), ("caps", "mnt", 30, 1000))

    def test_add_runs_both_visual_branches(self) -> None:
        gateway = CaptionGateway(
            image_hits=[img_hit("A7", "i7", 0.9)],
            caption_hits=[cap_hit("A3")],
            by_ids={"A3": art_doc("A3"), "A7": art_doc("A7")},
        )
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_MODE="add")
        outcome = fuse(service, [art_doc("A1")])
        self.assertEqual({d["artifact_id"] for d in outcome.docs}, {"A1", "A3", "A7"})
        self.assertEqual(
            outcome.diagnostics["fusion"]["branch_sizes"],
            {"artifact_search": 1, "text_to_image": 1, "text_to_caption": 1},
        )

    def test_caption_failure_falls_back_to_text_to_image(self) -> None:
        gateway = CaptionGateway(image_hits=[img_hit("A2", "i2", 0.9)], caption_error=RuntimeError("down"))
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_MODE="replace")
        outcome = fuse(service, [art_doc("A1"), art_doc("A2")])
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A2", "A1"])
        self.assertEqual(len(gateway.search_calls), 1)
        self.assertIn("error", outcome.diagnostics["text_to_caption"])

    def test_embed_failure_falls_back_to_text_to_image(self) -> None:
        gateway = CaptionGateway(image_hits=[img_hit("A2", "i2", 0.9)], caption_hits=[cap_hit("A3")])
        provider = CaptionProvider(text_error=RuntimeError("gpu"))
        service = self.build(gateway=gateway, provider=provider, CAPTION_TEXT_INDEX="caps")
        outcome = fuse(service, [art_doc("A1"), art_doc("A2")])
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A2", "A1"])
        self.assertEqual(gateway.caption_calls, [])

    def test_add_keeps_captions_when_text_to_image_fails(self) -> None:
        gateway = CaptionGateway(search_error=RuntimeError("down"), caption_hits=[cap_hit("A1")])
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_MODE="add")
        outcome = fuse(service, [art_doc("A1"), art_doc("A2")])
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A1", "A2"])

    def test_replace_without_caption_hits_keeps_baseline(self) -> None:
        gateway = CaptionGateway(image_hits=[img_hit("A2", "i2", 0.9)], caption_hits=[])
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps")
        self.assertIsNone(fuse(service, [art_doc("A1")]))

    def test_uncovered_museum_flag_off_never_counts(self) -> None:
        gateway = CaptionGateway(image_hits=[img_hit("A2", "i2", 0.9)], caption_hits=[], caption_count=0)
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps")
        self.assertIsNone(fuse(service, [art_doc("A1")]))
        self.assertEqual(gateway.count_calls, [])
        self.assertEqual(gateway.search_calls, [])

    def test_uncovered_museum_skips_captions_and_runs_text_to_image(self) -> None:
        gateway = CaptionGateway(image_hits=[img_hit("A2", "i2", 0.9)], caption_hits=[], caption_count=0)
        provider = CaptionProvider()
        service = self.build(
            gateway=gateway, provider=provider, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_UNCOVERED_MUSEUM_T2I=True
        )
        outcome = fuse(service, [art_doc("A1"), art_doc("A2")])
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A2", "A1"])
        self.assertEqual(gateway.count_calls, [{"index": "caps", "museum_id": "mnt"}])
        self.assertEqual(gateway.caption_calls, [])
        self.assertEqual(provider.text_calls, 0)
        self.assertEqual(len(gateway.search_calls), 1)
        self.assertEqual(outcome.diagnostics["text_to_caption"]["skipped"], "museum_not_in_index")
        self.assertEqual(outcome.diagnostics["branches"], ["artifact_search", "text_to_image"])
        # a resposta «não tem» fica em memória: a segunda pergunta não volta a contar
        fuse(service, [art_doc("A1"), art_doc("A2")])
        self.assertEqual(len(gateway.count_calls), 1)
        self.assertEqual(len(gateway.search_calls), 2)

    def test_uncovered_museum_is_rechecked_after_a_while(self) -> None:
        gateway = CaptionGateway(
            image_hits=[img_hit("A7", "i7", 0.9)], caption_hits=[cap_hit("A2")], caption_count=0,
        )
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_UNCOVERED_MUSEUM_T2I=True)
        fuse(service, [art_doc("A1"), art_doc("A2")])
        self.assertEqual(gateway.caption_calls, [])
        # as legendas do museu foram indexadas entretanto
        gateway.caption_count = 40
        service._caption_uncovered_at[("caps", "mnt")] -= multimodal_retrieval._CAPTION_COVERAGE_RECHECK_SECONDS + 1
        outcome = fuse(service, [art_doc("A1"), art_doc("A2")])
        self.assertEqual(len(gateway.count_calls), 2)
        self.assertEqual(len(gateway.caption_calls), 1)
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A2", "A1"])
        self.assertEqual(len(gateway.search_calls), 1)   # só a primeira pergunta usou o texto→imagem

    def test_covered_museum_without_caption_hits_keeps_baseline(self) -> None:
        gateway = CaptionGateway(image_hits=[img_hit("A2", "i2", 0.9)], caption_hits=[], caption_count=40)
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_UNCOVERED_MUSEUM_T2I=True)
        self.assertIsNone(fuse(service, [art_doc("A1")]))
        self.assertIsNone(fuse(service, [art_doc("A1")]))
        self.assertEqual(len(gateway.count_calls), 1)   # «tem legendas» conta-se uma vez
        self.assertEqual(len(gateway.caption_calls), 2)
        self.assertEqual(gateway.search_calls, [])

    def test_count_failure_keeps_caption_branch(self) -> None:
        gateway = CaptionGateway(
            image_hits=[img_hit("A7", "i7", 0.9)], caption_hits=[cap_hit("A2")], count_error=RuntimeError("down"),
        )
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_UNCOVERED_MUSEUM_T2I=True)
        outcome = fuse(service, [art_doc("A1"), art_doc("A2")])
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A2", "A1"])
        self.assertEqual(len(gateway.caption_calls), 1)
        self.assertEqual(gateway.search_calls, [])

    def test_query_instruction_only_with_the_flag(self) -> None:
        for flag, expected in ((False, {}), (True, {"as_query": True})):
            gateway = CaptionGateway(caption_hits=[cap_hit("A2")])
            provider = CaptionProvider()
            service = self.build(gateway=gateway, provider=provider, CAPTION_TEXT_INDEX="caps", CAPTION_TEXT_QUERY_INSTRUCTION=flag)
            fuse(service, [art_doc("A1"), art_doc("A2")])
            self.assertEqual(provider.text_kwargs, expected)

    def test_other_museums_and_duplicates_are_dropped(self) -> None:
        gateway = CaptionGateway(
            caption_hits=[cap_hit("B1", museum="mnaz"), cap_hit("A3"), cap_hit("A3")],
            by_ids={"A3": art_doc("A3"), "B1": art_doc("B1", museum="mnaz")},
        )
        service = self.build(gateway=gateway, CAPTION_TEXT_INDEX="caps")
        outcome = fuse(service, [art_doc("A1")])
        self.assertEqual([d["artifact_id"] for d in outcome.docs], ["A1", "A3"])
        self.assertEqual(outcome.diagnostics["text_to_caption"]["hits"], 1)

    def test_settings_validate_mode(self) -> None:
        self.assertEqual(Settings(_env_file=None, CAPTION_TEXT_MODE=" ADD ").CAPTION_TEXT_MODE, "add")
        self.assertEqual(Settings(_env_file=None).CAPTION_TEXT_INDEX, "")
        self.assertFalse(Settings(_env_file=None).CAPTION_TEXT_UNCOVERED_MUSEUM_T2I)
        with self.assertRaises(ValueError):
            Settings(_env_file=None, CAPTION_TEXT_MODE="both")


if __name__ == "__main__":
    unittest.main()
