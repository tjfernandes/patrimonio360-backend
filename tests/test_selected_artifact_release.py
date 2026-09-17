"""Peça selecionada («Perguntar», hotspot aberto): a mensagem é sobre ela, parte
dela («parecidas») ou é um pedido independente que a liberta?

Frases de afinação (TUNE_*) e de validação (VALID_*, escritas antes de afinar a
regra). Decisão por regras: `ChatService._selected_artifact_decision`.
"""
from __future__ import annotations

import asyncio
import unittest

from tests.test_tour_location import _Base

TUNE_ABOUT = [
    "De que é feito?", "De que material é?", "De quando é?", "Qual é a data?", "De que século é?", "Quem fez isto?",
    "Quem o fez?", "Quem é o autor?", "Quem usava esta peça?", "Para que servia?", "Conta-me a história",
    "Fala-me mais sobre esta peça", "Descreve-o", "Mostra-me mais detalhes", "Quero saber mais", "Mais informação",
    "Quais são as dimensões?", "Quanto mede?", "De onde veio?", "Qual é a origem?", "Que técnica foi usada?",
    "Está em bom estado?", "Tem inscrições?", "O que significa a decoração?", "Onde está exposta?", "É raro?",
    "Qual é o número de inventário?", "Como se chama?", "Que cor tem?", "E de que ano é?", "Explica-me o que é isto",
    "O que é?", "Foi restaurado?", "A quem pertenceu?", "What is it made of?", "Who made it?", "Tell me more about it",
    "When was this made?", "What was it used for?", "Describe this piece",
]
TUNE_SIMILAR = [
    "Mostra-me peças parecidas", "Há outras semelhantes?", "Encontra outros como este", "Peças relacionadas",
    "More like this", "Show me similar objects",
]
TUNE_RELEASE = [
    "Encontra chapéus de mulher", "Mostra-me leques", "Procura vestidos de noiva", "Vestidos de noiva", "E chapéus?",
    "Lista os casacos do século XVIII", "Há sapatos de criança?", "Que vestidos há na sala cantonal?",
    "Quero ver pinturas", "Onde estou?", "Em que sala me encontro?", "Que salas existem?", "Listas as áreas do museu?",
    "Leva-me ao pátio", "E a sala cantonal?", "Olá", "Obrigado", "Encontra peças de prata", "Peças do século XIX",
    "Mostra retratos de família", "Existem uniformes militares?", "Tens joias?", "Procuro bordados de Castelo Branco",
    "Azulejos com figuras", "Find women's hats", "Show me fans", "Where am I?", "What rooms are there?",
    "Are there any wedding dresses?", "Brinquedos antigos",
]

VALID_ABOUT = [
    "Qual o material?", "É de seda?", "Em que época foi usado?", "Quem o desenhou?", "Tem assinatura?",
    "Que história tem?", "Resume-me esta ficha", "Diz-me mais", "Quanto pesa?", "Qual é a altura?",
    "Foi doado por quem?", "Como foi adquirido?", "Porque é importante?", "Isto era usado por homens?",
    "Era para que ocasião?", "Que estilo é este?", "What's the story behind it?", "How old is it?",
    "Who wore this?", "Pormenores da decoração",
]
VALID_SIMILAR = ["Outros do mesmo género", "Há mais parecidos com este?", "Anything similar?"]
VALID_RELEASE = [
    "Procura luvas", "Mostra casacas masculinas", "Quero ver trajes de criança", "Há leques de marfim?",
    "Chapéus de palha", "E sapatos?", "Pesquisa rendas", "Encontra a sala da música", "Por onde posso andar?",
    "Bom dia", "Lista os xailes", "Que peças há no salão nobre?", "Existem fardas?", "Trajes regionais do Minho",
    "Show me shoes", "List the rooms", "Search for lace", "Mostra-me pinturas do século XVIII",
    "Quero encontrar coletes bordados", "Tens bonecas?",
]


class TestSelectedArtifactDecision(_Base):
    def _decide(self, message: str) -> str:
        return self._service()._selected_artifact_decision(message, museum_slug="mnt", museum_id="mnt")

    def _check(self, messages: list[str], expected: str) -> None:
        wrong = [(m, self._decide(m)) for m in messages if self._decide(m) != expected]
        self.assertEqual(wrong, [])

    def test_tune_about(self) -> None:
        self._check(TUNE_ABOUT, "about")

    def test_tune_similar(self) -> None:
        self._check(TUNE_SIMILAR, "similar")

    def test_tune_release(self) -> None:
        self._check(TUNE_RELEASE, "release")

    def test_valid_about(self) -> None:
        self._check(VALID_ABOUT, "about")

    def test_valid_similar(self) -> None:
        self._check(VALID_SIMILAR, "similar")

    def test_valid_release(self) -> None:
        self._check(VALID_RELEASE, "release")


class TestSelectedArtifactRelease(_Base):
    SELECTED = {"selected_artifact": {"artifact_id": "a-19663", "inventory_number": "19663", "title": "Barrete"}}

    def test_flag_off_keeps_old_behaviour(self) -> None:
        service = self._service(self._settings(CHAT_TOUR_LOCATION_ANSWERS=True, CHAT_SELECTED_ARTIFACT_AUTO_RELEASE=False))
        request = self._request("Onde estou?", {"panorama_key": "pan_musica"})
        request.metadata.update(self.SELECTED)
        response = asyncio.run(service.handle_message(request))
        self.assertIsNone(response.selected_artifact_status)

    def test_independent_request_releases_the_piece(self) -> None:
        service = self._service(self._settings(CHAT_TOUR_LOCATION_ANSWERS=True, CHAT_SELECTED_ARTIFACT_AUTO_RELEASE=True))
        request = self._request("Onde estou?", {"panorama_key": "pan_musica"})
        request.metadata.update(self.SELECTED)
        response = asyncio.run(service.handle_message(request))
        self.assertEqual(response.selected_artifact_status, "released")
        self.assertIn("Sala da Música", response.reply)

    def test_without_selection_there_is_no_status(self) -> None:
        service = self._service(self._settings(CHAT_TOUR_LOCATION_ANSWERS=True, CHAT_SELECTED_ARTIFACT_AUTO_RELEASE=True))
        response = asyncio.run(service.handle_message(self._request("Onde estou?", {"panorama_key": "pan_musica"})))
        self.assertIsNone(response.selected_artifact_status)


if __name__ == "__main__":
    unittest.main()
