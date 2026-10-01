"""Pesquisa noutro museu: a Casa-Museu Dr. Anastácio Gonçalves (cmag) e o Museu
Nacional de Arte Antiga (mnaa) reconhecidos na pergunta, como os outros quatro.

    python -m unittest tests.test_search_museums_cmag_mnaa
"""
from __future__ import annotations

import unittest

from tests.test_tour_location import _Base


class TestNewMuseums(_Base):
    CASES = [
        ("mostra-me pinturas no museu nacional de arte antiga", "mnaa"),
        ("há biombos no Museu de Arte Antiga?", "mnaa"),
        ("encontra custódias nas Janelas Verdes", "mnaa"),
        ("procura relógios na Casa-Museu Dr. Anastácio Gonçalves", "cmag"),
        ("porcelana chinesa na casa museu anastácio gonçalves", "cmag"),
        ("e no MNAA?", "mnaa"),
        ("vestidos no museu do traje", "mnt"),
        ("painéis no museu do azulejo", "mnaz"),
    ]

    def test_requested_museum(self) -> None:
        service = self._service()
        found = []
        for message, expected in self.CASES:
            museum = service._resolve_requested_search_museum(message)
            found.append((message, museum.museum_id if museum else None))
        self.assertEqual(found, [(m, e) for m, e in self.CASES])

    def test_names_and_slugs(self) -> None:
        service = self._service()
        mnaa = service._resolve_requested_search_museum("pinturas no museu nacional de arte antiga")
        cmag = service._resolve_requested_search_museum("relógios na casa-museu dr. anastácio gonçalves")
        self.assertEqual((mnaa.museum_slug, mnaa.museum_name), ("mnaa", "Museu Nacional de Arte Antiga"))
        self.assertEqual((cmag.museum_slug, cmag.museum_name), ("cmag", "Casa-Museu Dr. Anastácio Gonçalves"))

    def test_no_museum_named(self) -> None:
        service = self._service()
        self.assertIsNone(service._resolve_requested_search_museum("mostra-me pinturas antigas"))


if __name__ == "__main__":
    unittest.main()
