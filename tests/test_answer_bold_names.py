"""O nome de cada peça a negrito na lista da resposta (CHAT_ANSWER_BOLD_NAMES):
o título, ou o n.º de inventário quando a ficha não tem título
(docs/lista-da-resposta.md).

    python -m unittest tests.test_answer_bold_names
"""
from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.prompts.chat_prompts import answer_list_plan, build_final_answer_prompt_v2, has_answer_list_plan
from tests.test_answer_list_plan import CARDS, CONTEXT, _card
from tests.test_tour_location import _Base

MORE = [*CARDS, _card("d", "1 Mov", "Caixa", "Caixa de teca."), _card("e", "1 Pint", "São Jerónimo"), _card("f", "2 Pint", "São Jerónimo")]


class TestThePromptStaysAsMeasured(_Base):
    def test_off_by_default(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_ANSWER_BOLD_NAMES)

    def test_started_lines_have_no_bold(self) -> None:
        # o prompt é o que foi medido; o negrito é do serviço e não depende de a AMÁLIA o copiar
        self.assertEqual(answer_list_plan(CONTEXT, "pt").split("\n")[1:],
                         ["1. Fé –", "2. São Jerónimo (1 Pint) –", "3. São Jerónimo (2 Pint) –", "4. 584 Esc –"])
        service = self._service(self._settings(CHAT_ANSWER_ALL_CARDS=True, CHAT_ANSWER_LIST_PLAN=True, CHAT_ANSWER_BOLD_NAMES=True,
                                               CHAT_COMPACT_ANSWER_CONTEXT=True))
        prompt = build_final_answer_prompt_v2(
            museum_name="MNAA", intent="search", mode="rag", rolling_summary="", filters_state={}, user_message="estátuas",
            rewritten_query="estátuas", retrieval_context=CONTEXT, all_cards=True,
            list_plan=service.settings.CHAT_ANSWER_LIST_PLAN)
        self.assertTrue(has_answer_list_plan(prompt))
        self.assertNotIn("**", prompt)


class TestBoldInTheAnswer(_Base):
    def _service_with(self, **flags):
        base = dict(CHAT_ANSWER_ALL_CARDS=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True, CHAT_ANSWER_LIST_PLAN=True,
                    CHAT_MISSING_CARDS_IN_LIST=True, CHAT_ANSWER_BOLD_NAMES=True)
        base.update(flags)
        return self._service(self._settings(**base))

    def test_planned_list_has_every_name_in_bold(self) -> None:
        wrote = ("Aqui estão as peças:\n1. Fé – Modelo em barro.\n2. **Caridade** – Grupo escultórico.\n"
                 "4. A Caixa, de teca, é do século XVII.\n5. São Jerónimo (1 Pint) – Pintura.")
        text = self._service_with()._mention_every_card(wrote, MORE, language="pt", list_planned=True)
        self.assertEqual(text.split("\n"), [
            "Aqui estão as peças:",
            "1. **Fé** – Modelo em barro.",
            "2. **Caridade** – Grupo escultórico.",                      # já vinha a negrito
            "3. **584 Esc** – Grupo escultórico de Santa Ana ensinando a Virgem a ler assente sobre base quadrangular.",  # linha feita da ficha
            "4. A **Caixa**, de teca, é do século XVII.",                # com artigo
            "5. **São Jerónimo** (1 Pint) – Pintura.",                   # o n.º fica fora do negrito
            "6. **São Jerónimo** (2 Pint)",                              # sem descrição na ficha
        ])

    def test_flag_off_the_planned_list_is_as_before(self) -> None:
        wrote = "Aqui estão:\n1. Fé – Modelo em barro.\n2. Caridade – Grupo."
        text = self._service_with(CHAT_ANSWER_BOLD_NAMES=False)._mention_every_card(wrote, CARDS, language="pt", list_planned=True)
        self.assertNotIn("**", text)

    def test_a_list_she_wrote_without_the_plan_gets_the_bold_too(self) -> None:
        service = self._service_with()
        wrote = "Encontrei estas:\n\n- Fé – modelo em barro.\n- Caridade: grupo escultórico.\n\nA Fé é a mais antiga."
        text = service._bold_names_in_list(service._mention_every_card(wrote, CARDS, language="pt"), CARDS)
        self.assertEqual(text.split("\n"), [
            "Encontrei estas:", "",
            "- **Fé** – modelo em barro.",
            "- **Caridade**: grupo escultórico.",
            "- **584 Esc**: Grupo escultórico de Santa Ana ensinando a Virgem a ler assente sobre base quadrangular.",
            "", "A Fé é a mais antiga.",                                 # fora da lista nada muda
        ])

    def test_only_a_name_that_opens_the_item_and_ends_there(self) -> None:
        service = self._service_with()
        for body, expected in [
            ("Fé", "**Fé**"),
            ("Féria de verão – x", "Féria de verão – x"),                # outra palavra que começa igual
            ("Uma peça chamada Fé – x", "Uma peça chamada Fé – x"),      # o nome não abre o item
            ("584 Esc – grupo", "**584 Esc** – grupo"),
            ("__Fé__ – x", "__Fé__ – x"),
            ("**Outra** e Fé", "**Outra** e Fé"),
        ]:
            with self.subTest(body=body):
                self.assertEqual(service._bold_item_name(body, CARDS), expected)

    def test_text_without_a_list_or_without_the_flag_is_untouched(self) -> None:
        prose = "A Fé é um modelo em barro. A Caridade é um grupo."
        self.assertEqual(self._service_with()._bold_names_in_list(prose, CARDS), prose)
        listed = "Há:\n1. Fé – modelo."
        self.assertEqual(self._service_with(CHAT_ANSWER_BOLD_NAMES=False)._bold_names_in_list(listed, CARDS), listed)
        self.assertEqual(self._service_with()._bold_names_in_list("", CARDS), "")


class TestBoldInTheStream(_Base):
    """O streaming escreve os nomes a negrito como o texto final os vai ter, com qualquer corte dos tokens."""

    LONGER = [*MORE, _card("g", "3 Pint", "São Jerónimo Penitente"), _card("h", "9 Esc", "Fé e Caridade")]
    TEXTS = {
        "lista como no prompt": "Aqui estão as peças:\n1. Fé – Modelo em barro.\n2. Caridade – Grupo escultórico.\n3. 584 Esc – Grupo de Santa Ana.",
        "já a negrito": "Aqui estão:\n1. **Fé** – Modelo.\n2. **Caridade** – Grupo.",
        "marcadores e artigo": "Há estas:\n\n- A Caixa, de teca, é do século XVII.\n* Fé: modelo em barro.\n• Caridade – grupo.\n\nA Fé é a mais antiga.",
        "nomes que começam igual": "1. São Jerónimo Penitente – pintura.\n2. São Jerónimo (1 Pint) – outra.\n3. Fé e Caridade – grupo.\n4. Fé – modelo.\n5. Féria – nada.",
        "prosa": "A Fé é um modelo em barro. A Caridade é um grupo escultórico.\nSão duas peças de barro.",
        "item que não é de nenhuma peça": "1. Outra coisa qualquer – x.\n2. Fé",
        "linha comprida sem nome": "1. " + "palavra " * 60 + "\n2. Fé – modelo.",
        "espaços no fim das linhas": "Destaco:  \n1. Fé – a.  \n2. Caridade – b.  \n",
    }

    def _stream(self, text: str, size: int, cards) -> tuple[str, str]:
        service = self._service(self._settings(CHAT_ANSWER_BOLD_NAMES=True))
        sent: list[str] = []

        def factory():
            async def emit(delta: str) -> None:
                sent.append(delta)
            return emit

        async def run() -> None:
            wrapped, flush = service._bold_names_stream(factory, cards)
            emit = wrapped()
            for start in range(0, len(text), size):
                await emit(text[start:start + size])
            await flush()
            await flush()

        asyncio.run(run())
        return "".join(sent), service._bold_names_in_list(text, cards)

    def test_the_stream_is_the_final_text(self) -> None:
        for name, text in self.TEXTS.items():
            for size in (1, 2, 3, 5, 7, 40, 10_000):
                with self.subTest(text=name, size=size):
                    streamed, final = self._stream(text, size, self.LONGER)
                    self.assertEqual(streamed, final)

    def test_names_come_out_in_bold(self) -> None:
        streamed, _final = self._stream(self.TEXTS["nomes que começam igual"], 4, self.LONGER)
        self.assertEqual(streamed.split("\n"), [
            "1. **São Jerónimo Penitente** – pintura.", "2. **São Jerónimo** (1 Pint) – outra.",
            "3. **Fé e Caridade** – grupo.", "4. **Fé** – modelo.", "5. Féria – nada.",
        ])

    def test_only_the_start_of_a_line_is_held(self) -> None:
        # token a token: o que não é princípio de um item passa logo; o nome sai inteiro, já a negrito
        service = self._service(self._settings(CHAT_ANSWER_BOLD_NAMES=True))
        sent: list[str] = []

        def factory():
            async def emit(delta: str) -> None:
                sent.append(delta)
            return emit

        async def run() -> None:
            wrapped, flush = service._bold_names_stream(factory, CARDS)
            emit = wrapped()
            for piece in ["Aqui ", "estão:\n", "1", ". ", "Car", "idade", " – ", "Grupo", ".\n", "2. Fé"]:
                await emit(piece)
            await flush()

        asyncio.run(run())
        self.assertEqual(sent, ["Aqui ", "estão:\n", "1. **Caridade** – ", "Grupo", ".\n", "2. **Fé**"])

    def test_a_retry_starts_clean(self) -> None:
        service = self._service(self._settings(CHAT_ANSWER_BOLD_NAMES=True))
        attempts: list[list[str]] = []

        def factory():
            sent: list[str] = []
            attempts.append(sent)

            async def emit(delta: str) -> None:
                sent.append(delta)
            return emit

        async def run() -> None:
            wrapped, flush = service._bold_names_stream(factory, CARDS)
            first = wrapped()
            await first("1. Carid")
            second = wrapped()
            await second("1. Fé – modelo.")
            await flush()

        asyncio.run(run())
        self.assertEqual(["".join(sent) for sent in attempts], ["", "1. **Fé** – modelo."])
