"""Autor decidido antes da resposta (CHAT_AUTHOR_BEFORE_ANSWER): a ficha do autor e
«só peças do autor» resolvem-se com os cartões já escolhidos e antes de a AMÁLIA
escrever. O texto escreve-se uma vez, sobre os cartões que ficam, e o que o
visitante vê a ser escrito é o texto final.

    python -m unittest tests.test_author_before_answer
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from app.core.config import Settings
from app.schemas.chat import ChatAuthorCard, ResponseFormatObject
from app.services.llm_service import LLMResponse
from tests.test_author_only_results import CATALOGO, _SearchGateway
from tests.test_model_selects_cards import _FinalLLM
from tests.test_tour_location import _Base

LOPES_CARD = ChatAuthorCard(name="Gregório Lopes", artifact_count=19, in_tour_count=4)
OPENING = ("O museu tem 19 peças de Gregório Lopes. "
           "Os cartões mostram as primeiras 8 (as restantes estão em «Ver mais resultados»):")
WROTE = ("Encontrei 4 peças de Gregório Lopes nesta visita virtual:\n\n"
         "1. **Peça l1** (inv l1): pintura a óleo.\n2. **Peça l2** (inv l2): retábulo.\n3. **Peça l3** (inv l3): tábua.")
LIST = WROTE[WROTE.index("1. "):]


class _Pre:
    """O modelo de pré-processamento, por pergunta: os cartões escolhidos, se as
    palavras são o nome de um artista (`name`) e se o visitante só quer peças dele (`only`)."""

    def __init__(self, *, cards: list[str] | None, name: str = "A", only: str = "A") -> None:
        self.cards, self.name, self.only, self.prompts = cards, name, only, []

    def asked(self, fragment: str) -> int:
        return sum(1 for prompt in self.prompts if fragment in prompt)

    async def generate(self, *, message, response_format=ResponseFormatObject(type="json_object"), **_kwargs):
        self.prompts.append(message)
        if "é o nome de quê?" in message:
            reply: object = {"resposta": self.name}
        elif "[doc_1]" in message:
            if self.cards is None:
                raise RuntimeError("modelo em baixo")
            reply = {"cartoes": self.cards}
        else:
            reply = {"resposta": self.only}
        return LLMResponse(text=json.dumps(reply), model="pre", response_format=response_format, parsed_json=reply, finish_reason="stop")


class _FlowGateway(_SearchGateway):
    async def fetch_artifacts_by_inventory_numbers(self, **_kwargs):
        return []


class _Flow(_Base):
    """A pesquisa de texto de ponta a ponta, com o streaming ligado."""

    def _flow(self, pre: _Pre, text: str = WROTE, *, before: bool = True, **settings):
        base = dict(
            CHAT_AUTHOR_CARD=True, CHAT_AUTHOR_ONLY_RESULTS=True, CHAT_AUTHOR_ONLY_NEW_ANSWER=True, CHAT_AUTHOR_ANSWER_COUNTS=True,
            CHAT_AUTHOR_BEFORE_ANSWER=before, CHAT_MODEL_SELECTS_CARDS=True, CHAT_PRESELECT_CARDS=True,
            CHAT_COMPACT_ANSWER_CONTEXT=True, CHAT_ROUTER_MODE="deterministic", CHAT_STREAM_FINAL_ANSWER=True,
            CHAT_RETRIEVAL_RESULTS_PAGE_SIZE=8,
        )
        base.update(settings)
        llm = _FinalLLM(text)
        service = self._service(self._settings(**base), llm=llm)
        service.preprocess_llm_service = pre
        service.opensearch_gateway = _FlowGateway(CATALOGO)

        async def _retrieve_context(**_kwargs):
            request = {"kind": "text", "museum_slug": "mnaa", "museum_id": "mnaa", "query_text": "obras de Gregório Lopes",
                       "lexical_query": "gregório lopes", "query_embedding": [0.1, 0.2], "filters": {}, "sort": {},
                       "results_total": len(CATALOGO), "retrieval_window_size": 1000}
            return "ctx", len(CATALOGO), [dict(doc) for doc in CATALOGO[:20]], request

        service._retrieve_context = _retrieve_context
        return service, llm

    def _ask(self, service, message: str = "Mostra-me obras de Gregório Lopes."):
        events: list[dict] = []

        async def status_cb(event: dict) -> None:
            events.append(event)

        request = self._request(message, museum_slug="mnaa", museum_id="mnaa", museum_name="Museu Nacional de Arte Antiga")
        response = asyncio.run(service.handle_message(request, status_cb=status_cb))
        tokens = [event for event in events if event.get("type") == "token"]
        self.assertEqual([event["seq"] for event in tokens], list(range(len(tokens))))
        return response, "".join(event["delta"] for event in tokens)


class TestAuthorBeforeAnswer(_Flow):
    ALL = [f"doc_{n}" for n in range(1, 9)]

    def test_off_by_default(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_AUTHOR_BEFORE_ANSWER)

    def test_the_text_is_written_once_about_the_authors_cards(self) -> None:
        pre = _Pre(cards=self.ALL)
        service, llm = self._flow(pre)
        response, streamed = self._ask(service)
        # os cartões são só do autor: os 4 da página e mais 4 dele no lugar dos outros
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"])
        self.assertEqual((response.results_total, response.results_has_more), (19, True))
        # 18 fichas no catálogo dos autores, 19 peças na pesquisa só dele: a abertura diz o maior
        self.assertEqual((response.author.name, response.author.artifact_count), ("Gregório Lopes", 18))
        # a AMÁLIA escreveu uma vez, e já sobre esses cartões
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("inv l8", llm.prompts[0])
        self.assertNotIn("inv o1", llm.prompts[0])
        # o texto final é a abertura com os números e a lista dela — o mesmo que se viu a ser escrito
        self.assertEqual(response.reply, f"{OPENING}\n\n{LIST}")
        self.assertEqual(streamed, response.reply)
        self.assertNotIn("Encontrei 4 peças", streamed)
        # cada pergunta ao pré-processamento fez-se uma vez
        self.assertEqual((pre.asked("é o nome de quê?"), len(pre.prompts)), (1, 3))
        # a conversa guardada e o «ver mais» ficam com o mesmo
        state = service.session_store.get(response.conversation_id)
        self.assertEqual(state.history[-1].text, response.reply)
        stored = state.paged_results_by_request_id[response.results_request_id]["retrieval_request"]
        self.assertEqual(stored["filters"], {"creator_ids": ["a:lopes"]})
        self.assertEqual([a["artifact_id"] for a in service.query_logger.events[-1]["shown_artifacts"]],
                         ["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"])

    def test_flag_off_the_text_changes_when_the_cards_arrive(self) -> None:
        # o comportamento de antes, que a flag corrige: duas escritas, e o texto visto não é o final
        service, llm = self._flow(_Pre(cards=self.ALL), before=False)
        response, streamed = self._ask(service)
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(streamed, WROTE)
        self.assertNotEqual(streamed, response.reply)
        self.assertEqual(response.reply, f"{OPENING}\n\n{LIST}")

    def test_cards_already_only_of_the_author_get_the_opening_too(self) -> None:
        pre = _Pre(cards=["doc_1", "doc_3", "doc_5"])
        service, llm = self._flow(pre)
        response, streamed = self._ask(service)
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["l1", "l2", "l3"])
        self.assertEqual(len(llm.prompts), 1)
        opening = "O museu tem 19 peças de Gregório Lopes. Os cartões mostram as primeiras 3 (as restantes estão em «Ver mais resultados»):"
        self.assertEqual(response.reply, f"{opening}\n\n{LIST}")
        self.assertEqual(streamed, response.reply)

    def test_other_authors_wanted_too_the_card_comes_and_nothing_else_changes(self) -> None:
        pre = _Pre(cards=self.ALL, only="B")
        service, llm = self._flow(pre)
        response, streamed = self._ask(service, "Mostra-me obras de Gregório Lopes e de outros pintores.")
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["l1", "o1", "l2", "o2", "l3", "o3", "l4", "o4"])
        self.assertEqual(response.author.name, "Gregório Lopes")
        self.assertEqual((response.reply, streamed, len(llm.prompts), len(pre.prompts)), (WROTE, WROTE, 1, 3))

    def test_words_that_are_not_an_artist_are_asked_once(self) -> None:
        pre = _Pre(cards=self.ALL, name="B")
        service, llm = self._flow(pre)
        response, streamed = self._ask(service, "Mostra-me peças da rua Gregório Lopes.")
        self.assertIsNone(response.author)
        self.assertEqual(len(response.artifact_results), 8)
        self.assertEqual((response.reply, streamed, len(llm.prompts)), (WROTE, WROTE, 1))
        # o passo de depois da resposta não volta a perguntar
        self.assertEqual((pre.asked("é o nome de quê?"), len(pre.prompts)), (1, 2))
        self.assertFalse(service._author_decided_query_ids)

    def test_a_message_without_an_authors_name_asks_nothing(self) -> None:
        pre = _Pre(cards=self.ALL)
        service, llm = self._flow(pre)
        response, streamed = self._ask(service, "Mostra-me pinturas com paisagens.")
        self.assertIsNone(response.author)
        self.assertEqual((response.reply, streamed, len(llm.prompts), len(pre.prompts)), (WROTE, WROTE, 1, 1))

    def test_when_the_answer_chooses_the_cards_the_step_stays_after_it(self) -> None:
        # sem escolha do pré-processamento é a AMÁLIA que escolhe os cartões ao escrever: só depois se conhecem
        service, llm = self._flow(_Pre(cards=None), f"{WROTE}\n\n[[cartões: doc_1, doc_2, doc_3]]")
        response, streamed = self._ask(service)
        self.assertIsNone(getattr(service, "_author_decided_query_ids", None))
        self.assertEqual(response.author.name, "Gregório Lopes")
        self.assertEqual([card.artifact_id for card in response.artifact_results], ["l1", "l2", "l3"])
        # o passo de depois, como sempre: a abertura entra no fim, por cima do que se viu a ser escrito
        self.assertEqual(streamed.rstrip(), WROTE)
        self.assertTrue(response.reply.startswith("O museu tem 19 peças de Gregório Lopes."))
        self.assertEqual(len(llm.prompts), 1)

    def test_a_failing_author_search_keeps_the_cards_and_writes_once(self) -> None:
        service, llm = self._flow(_Pre(cards=self.ALL))
        service.opensearch_gateway.fail_search = True
        response, streamed = self._ask(service)
        self.assertEqual(response.author.name, "Gregório Lopes")
        self.assertEqual(len(response.artifact_results), 8)
        self.assertEqual((response.reply, streamed, len(llm.prompts)), (WROTE, WROTE, 1))

    def test_without_streaming_the_answer_is_the_same(self) -> None:
        service, llm = self._flow(_Pre(cards=self.ALL), CHAT_STREAM_FINAL_ANSWER=False)
        response, streamed = self._ask(service)
        self.assertEqual((response.reply, streamed, len(llm.prompts)), (f"{OPENING}\n\n{LIST}", "", 1))

    def test_every_card_is_still_named(self) -> None:
        service, llm = self._flow(_Pre(cards=self.ALL), CHAT_ANSWER_ALL_CARDS=True)
        response, streamed = self._ask(service)
        tail = "Nos cartões estão também: " + ", ".join(f"**Peça l{n}** (inv l{n})" for n in range(4, 9)) + "."
        self.assertEqual(response.reply, f"{OPENING}\n\n{LIST}\n\n{tail}")
        # o que se viu a ser escrito é o início do texto final: a última linha junta-se, nada muda
        self.assertTrue(response.reply.startswith(streamed))
        self.assertEqual(streamed, f"{OPENING}\n\n{LIST}")


class TestOpeningStream(_Base):
    """O streaming escreve o que _author_answer_with_counts vai dar, com qualquer corte dos tokens."""

    LONG = "Destaca-se o autorretrato de juventude, desenhado a carvão sobre papel. " * 8
    TEXTS = {
        "lista": WROTE,
        "lista sem frase antes": LIST,
        "lista com títulos a negrito": "Estas são as obras:\n\n**Fé** – modelo em barro.\n**Esperança** – modelo em barro.",
        "corrida, a primeira frase conta peças": f"Encontrei duas obras do pintor. {LONG}",
        "corrida, a primeira frase não conta": f"Sequeira foi pintor da corte. {LONG}",
        "corrida e curta": "Encontrei duas obras do pintor. Destaca-se o autorretrato de juventude.",
        "uma frase só": "Sequeira foi pintor da corte",
        "espaços à volta": f"\n\n  {WROTE}\n",
    }

    def _stream(self, text: str, size: int) -> tuple[str, str]:
        service = self._service(self._settings(CHAT_AUTHOR_ANSWER_COUNTS=True))
        opening = service._author_counts_opening(author=LOPES_CARD, total=19, shown=8, language="pt")
        sent: list[str] = []

        def factory():
            async def emit(delta: str) -> None:
                sent.append(delta)
            return emit

        async def run() -> None:
            wrapped, flush = service._author_opening_stream(factory, opening)
            emit = wrapped()
            for start in range(0, len(text), size):
                await emit(text[start:start + size])
            await flush()
            await flush()  # duas vezes não repete nada

        asyncio.run(run())
        final = service._author_answer_with_counts(text, author=LOPES_CARD, total=19, shown=8, language="pt")
        return "".join(sent), final

    def test_the_stream_is_the_final_text(self) -> None:
        for name, text in self.TEXTS.items():
            for size in (1, 3, 7, 40, 10_000):
                with self.subTest(text=name, size=size):
                    streamed, final = self._stream(text, size)
                    self.assertEqual(streamed.rstrip(), final)

    def test_the_opening_is_the_first_thing_sent_and_the_models_count_never_shows(self) -> None:
        streamed, _final = self._stream(WROTE, 5)
        self.assertTrue(streamed.startswith("O museu tem 19 peças de Gregório Lopes, 4 delas nesta visita."))
        self.assertNotIn("Encontrei 4 peças", streamed)

    def test_a_retry_starts_again_from_the_opening(self) -> None:
        service = self._service(self._settings(CHAT_AUTHOR_ANSWER_COUNTS=True))
        attempts: list[list[str]] = []

        def factory():
            sent: list[str] = []
            attempts.append(sent)

            async def emit(delta: str) -> None:
                sent.append(delta)
            return emit

        async def run() -> None:
            wrapped, flush = service._author_opening_stream(factory, "Abertura:")
            first = wrapped()
            await first("Encontrei 2 peças")
            second = wrapped()
            await second("Encontrei 3 peças:\n\n1. **A** – x.")
            await flush()

        asyncio.run(run())
        self.assertEqual(["".join(sent) for sent in attempts], ["Abertura:\n\n", "Abertura:\n\n1. **A** – x."])
