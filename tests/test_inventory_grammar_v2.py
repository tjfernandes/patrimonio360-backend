"""Quick win CHAT_INVENTORY_GRAMMAR_V2: números de inventário reais
(«793 Mob MNSR», «MNAz 1 Az», «MJ_ES032/1») reconhecidos como fast-path.

Com o flag desligado a saída tem de ser exatamente a histórica.

    python -m unittest tests.test_inventory_grammar_v2
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.chat_service import ChatService  # noqa: E402


def _service(v2: bool) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = SimpleNamespace(CHAT_INVENTORY_GRAMMAR_V2=v2)
    return service


class TestGrammarV2(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _service(True)

    def first(self, text: str) -> str | None:
        candidates = self.service._extract_inventory_candidates(text)
        return candidates[0] if candidates else None

    def test_mnsr_number_first_real_query(self) -> None:
        # Query real do estudo (linha 180 dos logs) que hoje cai em pesquisa híbrida.
        self.assertEqual(self.first("esta mesa 793 Mob MNSR, está em que lugar do museu?"), "793 mob mnsr")
        self.assertEqual(self.first("793 Mob MNSR"), "793 mob mnsr")
        self.assertEqual(self.first("qual é a peça 255 Cer MNSR?"), "255 cer mnsr")

    def test_mnsr_prefix_first(self) -> None:
        self.assertIn("793 mob mnsr", self.service._extract_inventory_candidates("MNSR 793 Mob"))

    def test_mnsr_collection_variants(self) -> None:
        candidates = self.service._extract_inventory_candidates("339 Pin CMP/ MNSR")
        self.assertEqual(candidates[:2], ["339 pin cmp/ mnsr", "339 pin mnsr"])
        candidates = self.service._extract_inventory_candidates("1 Pin CMP (DHL)/ MNSR")
        self.assertEqual(candidates[0], "1 pin cmp/ mnsr")
        candidates = self.service._extract_inventory_candidates("47 Cer CMFC/ MNSR")
        self.assertEqual(candidates[0], "47 cer cmfc/ mnsr")

    def test_mnsr_slash_number(self) -> None:
        self.assertEqual(self.first("91/ 51 Pin MNSR"), "91 51 pin mnsr")

    def test_mnaz_prefix_type(self) -> None:
        self.assertEqual(self.first("mostra-me o MNAz 1 Az"), "mnaz 1 az")
        self.assertEqual(self.first("MNAz 1151 Proj"), "mnaz 1151 proj")
        self.assertEqual(self.first("MNAz 12/3 Cer"), "mnaz 12 3 cer")

    def test_mj_underscore_forms(self) -> None:
        candidates = self.service._extract_inventory_candidates("onde está o MJ_ES032/1?")
        self.assertEqual(candidates[:2], ["mj_es032/1", "mj_es032"])
        self.assertEqual(self.first("MJ_SN_10"), "mj_sn_10")
        self.assertEqual(self.first("MJ_IM01/2"), "mj_im01/2")

    def test_years_and_plain_numbers_are_not_inventories(self) -> None:
        for text in [
            "pinturas de 1920",
            "vestidos de 1920 no mnsr",
            "quadros do século XIX",
            "onde é que no museu encontro peças de 1780 - 1800",
            "quantos azulejos há no mnaz",
            "mob 12",
        ]:
            self.assertEqual(self.service._extract_inventory_candidates(text), [], text)

    def test_year_like_number_with_full_grammar_is_accepted(self) -> None:
        self.assertEqual(self.first("1920 Pin MNSR"), "1920 pin mnsr")

    def test_marker_path_still_works(self) -> None:
        self.assertIn("12", self.service._extract_inventory_candidates("objeto 12"))


class TestGrammarV2Off(unittest.TestCase):
    def test_flag_off_keeps_historical_output(self) -> None:
        service = _service(False)
        self.assertEqual(service._extract_inventory_candidates("793 Mob MNSR"), [])
        self.assertEqual(service._extract_inventory_candidates("esta mesa 793 Mob MNSR, está em que lugar?"), [])
        self.assertEqual(service._extract_inventory_candidates("MNSR 793 Mob"), ["mnsr 793", "mnsr793"])
        self.assertEqual(service._extract_inventory_candidates("MNAz 1 Az"), ["mnaz 1", "mnaz1"])
        self.assertEqual(service._extract_inventory_candidates("objeto 12"), ["12"])

    def test_flag_on_is_superset_of_off(self) -> None:
        on, off = _service(True), _service(False)
        for text in ["MNSR 793 Mob", "MNAz 1 Az", "objeto 12", "MJ_ES032/1", "referência 1234"]:
            for candidate in off._extract_inventory_candidates(text):
                self.assertIn(candidate, on._extract_inventory_candidates(text), text)


if __name__ == "__main__":
    unittest.main()
