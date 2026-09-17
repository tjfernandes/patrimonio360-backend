"""Conversa sem pedido («olá irmão», «obrigado!», «quem és tu?») não vai à
pesquisa. Frases de afinação (TUNE_*) e de validação (VALID_*, escritas antes
de afinar a regra). Regra: `ChatService._is_obvious_greeting_or_smalltalk`.
"""
from __future__ import annotations

import unittest

from tests.test_tour_location import _Base

TUNE_SMALLTALK = [
    "olá irmão", "Olá!", "Oi", "Bom dia", "Boa tarde!", "Boa noite", "Tudo bem?", "Tudo bem contigo?",
    "Como estás?", "Como vai isso?", "Então, tudo bom?", "Olá, tudo bem?", "Obrigado", "Obrigada!", "Muito obrigado",
    "Valeu", "Ok", "Certo", "Fixe", "Adeus", "Até logo", "Tchau", "Boas", "Ei", "Olá amigo", "Olá mano, tudo bem?",
    "Quem és tu?", "O que podes fazer?", "Podes ajudar-me?", "Preciso de ajuda", "Hello", "Hi there", "Hey!",
    "Good morning", "How are you?", "Thanks!", "Thank you very much", "Bye", "Who are you?", "What can you do?",
    "haha", "hmm",
]
TUNE_SEARCH = [
    "Olá, encontra vestidos", "Bom dia, que salas existem?", "Mostra-me leques", "Vestidos de noiva",
    "Onde estou?", "Chapéus", "Obrigado, e chapéus?", "Quem fez este vestido?", "O que é um tournure?",
    "Fala-me do museu", "Boa tarde, procuro sapatos de criança", "Há brinquedos?", "Leques", "Encontra a sala da música",
    "Hi, show me fans", "Thanks, and hats?", "Que peças tens?", "O que há para ver?", "Podes ajudar-me a encontrar luvas?",
    "Quem foi o marquês de Angeja?",
]

VALID_SMALLTALK = [
    "Olá, boa tarde!", "Oi, tudo fixe?", "Bom dia, tudo bem por aí?", "Olá amiga", "Olá pessoal", "Obrigadinho",
    "Muito obrigada, adeus", "Está bem", "Sim", "Não", "Ok, obrigado", "Boa!", "Uau", "Até amanhã", "Olá Amália",
    "Quem é a Amália?", "O que sabes fazer?", "Como me podes ajudar?", "Consegues ajudar?", "Hello, how are you doing?",
    "Cheers", "Good evening!", "Could you help me?", "See you later", "Yes",
]
VALID_SEARCH = [
    "Olá, mostra-me casacos", "Bom dia. Onde estou?", "Trajes de criança", "Boa noite, há rendas?", "Sapatos",
    "Ok, e leques?", "Obrigado. Quais são as salas?", "Quem és tu para dizeres que este vestido é do século XIX?",
    "Ajuda-me a encontrar bordados", "Hi, where am I?", "Thanks. Find shoes", "Podes mostrar joias?",
    "Que vestidos há?", "Peças de prata", "Amália, encontra chapéus", "Fixe, e a sala cantonal?", "Sim, mostra mais",
    "Não, quero ver pinturas", "What rooms are there?", "Tell me about this museum",
]


class TestSmalltalkPhrasings(_Base):
    def _is(self, message: str) -> bool:
        return self._service()._is_obvious_greeting_or_smalltalk(message)

    def test_tune_smalltalk(self) -> None:
        self.assertEqual([m for m in TUNE_SMALLTALK if not self._is(m)], [])

    def test_tune_search(self) -> None:
        self.assertEqual([m for m in TUNE_SEARCH if self._is(m)], [])

    def test_valid_smalltalk(self) -> None:
        self.assertEqual([m for m in VALID_SMALLTALK if not self._is(m)], [])

    def test_valid_search(self) -> None:
        self.assertEqual([m for m in VALID_SEARCH if self._is(m)], [])


if __name__ == "__main__":
    unittest.main()
