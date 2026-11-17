"""Pesquisa do chat, Bloco D: o modelo de pré-processamento escolhe os cartões
entre mais fichas do que as da página (CHAT_PRESELECT_CARDS_CANDIDATES).
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.schemas.chat import ArtifactResult, ChatResultsPageRequest  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from tests.test_collapse_set_duplicates import PRODUCAO, _no_reply, _retrieval_service  # noqa: E402


def _service(**flags) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = Settings(_env_file=None, **flags)
    return service


def _cards(n: int) -> list[ArtifactResult]:
    return [ArtifactResult(artifact_id=f"a{i}", inventory_number=f"{i} Pint", title=f"Peça {i}", museum_id="mnaa") for i in range(1, n + 1)]


class TestCandidateCount(unittest.TestCase):
    def test_off_by_default(self) -> None:
        self.assertEqual(Settings(_env_file=None).CHAT_PRESELECT_CARDS_CANDIDATES, 0)
        self.assertEqual(_service(CHAT_PRESELECT_CARDS=True)._preselect_candidate_count(page=1, page_size=8, response_type="text"), 0)

    def test_fifteen_on_the_first_text_page(self) -> None:
        on = _service(CHAT_PRESELECT_CARDS=True, CHAT_PRESELECT_CARDS_CANDIDATES=15)
        self.assertEqual(on._preselect_candidate_count(page=1, page_size=8, response_type="text"), 15)

    def test_only_with_preselection_on_page_one_for_text(self) -> None:
        on = _service(CHAT_PRESELECT_CARDS=True, CHAT_PRESELECT_CARDS_CANDIDATES=15)
        self.assertEqual(on._preselect_candidate_count(page=2, page_size=8, response_type="text"), 0)
        self.assertEqual(on._preselect_candidate_count(page=1, page_size=8, response_type="json_object"), 0)
        without = _service(CHAT_PRESELECT_CARDS=False, CHAT_PRESELECT_CARDS_CANDIDATES=15)
        self.assertEqual(without._preselect_candidate_count(page=1, page_size=8, response_type="text"), 0)

    def test_a_value_up_to_the_page_size_changes_nothing_and_fifty_is_the_ceiling(self) -> None:
        self.assertEqual(_service(CHAT_PRESELECT_CARDS=True, CHAT_PRESELECT_CARDS_CANDIDATES=8)._preselect_candidate_count(
            page=1, page_size=8, response_type="text"), 0)
        self.assertEqual(_service(CHAT_PRESELECT_CARDS=True, CHAT_PRESELECT_CARDS_CANDIDATES=200)._preselect_candidate_count(
            page=1, page_size=8, response_type="text"), 50)


class TestCap(unittest.TestCase):
    def test_keeps_the_first_chosen_with_their_images_and_targets(self) -> None:
        cards = _cards(10)
        images = [SimpleNamespace(artifact_id=f"a{i}", inventory=f"{i} Pint") for i in range(1, 11)]
        targets = [SimpleNamespace(inventory_id=f"{i} pint") for i in (2, 9, 10)]
        artifacts, kept_images, kept_targets = ChatService._cap_preselected_cards(cards, images, targets, limit=8)
        self.assertEqual([a.artifact_id for a in artifacts], [f"a{i}" for i in range(1, 9)])
        self.assertEqual([m.artifact_id for m in kept_images], [f"a{i}" for i in range(1, 9)])
        self.assertEqual([t.inventory_id for t in kept_targets], ["2 pint"])

    def test_under_the_limit_nothing_is_dropped(self) -> None:
        cards = _cards(3)
        images, targets = [SimpleNamespace(artifact_id="a1", inventory="1 Pint")], [SimpleNamespace(inventory_id="3 Pint")]
        self.assertEqual(ChatService._cap_preselected_cards(cards, images, targets, limit=8), (cards, images, targets))


class TestNextPageSkipsWhatWasShown(unittest.TestCase):
    def _page_two(self, skip):
        service = _retrieval_service(CHAT_COLLAPSE_SET_DUPLICATES=False)
        state = service.session_store.load_or_create(conversation_id="d", museum_slug="mnt")
        payload = ChatResultsPageRequest(museum_slug="mnt", museum_id="mnt", conversation_id="d", results_page=2, results_page_size=8)
        request = {"kind": "text", "museum_slug": "mnt", "museum_id": "mnt", "query_text": "q", "lexical_query": "q",
                   "query_embedding": [0.1, 0.2], "filters": {}, "sort": {}, "results_total": len(PRODUCAO)}
        if skip is not None:
            request["preselected_beyond_page_ids"] = skip
        service._generate_results_page_reply = _no_reply
        return asyncio.run(service._materialize_text_results_page(state=state, payload=payload, request=request, page=2, page_size=8))

    def test_cards_chosen_beyond_the_first_page_do_not_come_back(self) -> None:
        plain = [a.artifact_id for a in self._page_two(None).artifact_results]
        self.assertEqual(len(plain), 8)
        skipped = [a.artifact_id for a in self._page_two([plain[0], plain[3]]).artifact_results]
        self.assertEqual(skipped, [plain[1], plain[2], *plain[4:]])

    def test_an_empty_list_changes_nothing(self) -> None:
        self.assertEqual([a.artifact_id for a in self._page_two([]).artifact_results], [a.artifact_id for a in self._page_two(None).artifact_results])


if __name__ == "__main__":
    unittest.main()
