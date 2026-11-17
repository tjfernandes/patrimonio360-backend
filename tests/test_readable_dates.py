"""A data das fichas por extenso nos cartões e nos prompts (CHAT_READABLE_DATES).

O export do RAIZ escreve os séculos como números («16», «18–19», «-5»); era
esse texto que aparecia no cartão e que os modelos liam.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402
from app.services.catalogue_analytics import readable_catalogue_date  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402


def _service(**flags) -> ChatService:
    service = object.__new__(ChatService)
    service.settings = Settings(_env_file=None, **flags)
    return service


class TestReadableCatalogueDate(unittest.TestCase):
    def test_one_century(self) -> None:
        self.assertEqual(readable_catalogue_date("16", 1501, 1600), "século XVI")
        self.assertEqual(readable_catalogue_date("-5", -500, -401), "século V a.C.")

    def test_several_centuries(self) -> None:
        self.assertEqual(readable_catalogue_date("18–19", 1701, 1900), "séculos XVIII–XIX")
        self.assertEqual(readable_catalogue_date("-6–-5", -600, -401), "séculos VI–V a.C.")
        self.assertEqual(readable_catalogue_date("-1–1", -100, 100), "século I a.C. – século I d.C.")

    def test_years(self) -> None:
        self.assertEqual(readable_catalogue_date("1766", 1766, 1766), "1766")
        self.assertEqual(readable_catalogue_date("1505–1506", 1505, 1506), "1505–1506")
        self.assertEqual(readable_catalogue_date("-590–-580", -590, -580), "590–580 a.C.")
        self.assertEqual(readable_catalogue_date("-27", -27, -27), "27 a.C.")
        self.assertEqual(readable_catalogue_date("-27–14", -27, 14), "27 a.C. – 14 d.C.")

    def test_the_label_comes_from_the_years_not_from_the_text(self) -> None:
        # a píxide 940 Our: o RAIZ diz «-15»; com os anos corrigidos no índice é o século XV
        self.assertEqual(readable_catalogue_date("-15", 1401, 1500), "século XV")
        self.assertEqual(readable_catalogue_date("-15", -1500, -1401), "século XV a.C.")

    def test_free_text_is_kept(self) -> None:
        self.assertEqual(readable_catalogue_date("Lisboa, c. 1715-1730", 1715, 1730), "Lisboa, c. 1715-1730")
        self.assertEqual(readable_catalogue_date("SÉC. XVI", 1501, 1600), "SÉC. XVI")
        self.assertEqual(readable_catalogue_date("Início Séc. XVI (1510-1513)", 1510, 1513), "Início Séc. XVI (1510-1513)")

    def test_without_years_the_text_is_kept(self) -> None:
        self.assertEqual(readable_catalogue_date("16", None, None), "16")
        self.assertEqual(readable_catalogue_date("", None, None), "")
        self.assertEqual(readable_catalogue_date(None, True, True), "")

    def test_years_without_text(self) -> None:
        self.assertEqual(readable_catalogue_date("", 1401, 1500), "século XV")
        self.assertEqual(readable_catalogue_date(None, 1830, None), "1830")

    def test_years_as_they_come_from_json(self) -> None:
        self.assertEqual(readable_catalogue_date("16", "1501", 1600.0), "século XVI")

    def test_applying_it_twice_changes_nothing(self) -> None:
        for text, start, end in (("16", 1501, 1600), ("-5", -500, -401), ("18–19", 1701, 1900), ("-590–-580", -590, -580)):
            once = readable_catalogue_date(text, start, end)
            self.assertEqual(readable_catalogue_date(once, start, end), once)


class TestInTheService(unittest.TestCase):
    DOC = {"title": "Píxide", "inventory_number": "940 Our", "category": "Ourivesaria", "date_or_period": "16",
           "start_year": 1501, "end_year": 1600}

    def test_off_by_default(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_READABLE_DATES)
        self.assertEqual(_service()._catalogue_date_text(self.DOC), "16")
        self.assertIn("Data: 16", _service()._format_docs_compact(docs=[self.DOC]))

    def test_on_the_prompt_reads_the_century(self) -> None:
        on = _service(CHAT_READABLE_DATES=True)
        self.assertEqual(on._catalogue_date_text(self.DOC), "século XVI")
        compact = on._format_docs_compact(docs=[self.DOC])
        self.assertIn("Data: século XVI", compact)
        self.assertNotIn("Data: 16", compact)

    def test_on_a_card_already_converted_stays(self) -> None:
        # os cartões da resposta voltam a passar pelo prompt da escolha e da resposta
        on = _service(CHAT_READABLE_DATES=True)
        card = dict(self.DOC, date_or_period="século XVI")
        self.assertIn("Data: século XVI", on._format_docs_compact(docs=[card]))

    def test_on_a_record_without_a_date_still_says_so(self) -> None:
        on = _service(CHAT_READABLE_DATES=True)
        self.assertIn("Data: não indicada", on._format_docs_compact(docs=[{"title": "Cálice", "inventory_number": "1 Our"}]))

    def test_on_the_related_item_reads_the_century(self) -> None:
        art = {"artifact_id": "a1", "inventory_number": "940 Our", "title": "Píxide", "date_or_period": "-5",
               "start_year": -500, "end_year": -401}
        on = _service(CHAT_READABLE_DATES=True)._build_related_item(art, navigation_lookup={}, images_by_artifact={"a1": []})
        off = _service()._build_related_item(art, navigation_lookup={}, images_by_artifact={"a1": []})
        self.assertEqual(on.date_or_period, "século V a.C.")
        self.assertEqual(off.date_or_period, "-5")


if __name__ == "__main__":
    unittest.main()
