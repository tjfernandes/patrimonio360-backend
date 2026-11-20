"""A lista da resposta decidida no prompt e garantida no texto.

CHAT_UNTITLED_CARD_BY_INVENTORY, CHAT_ANSWER_LIST_PLAN e CHAT_MISSING_CARDS_IN_LIST
(docs/lista-da-resposta.md).

    python -m unittest tests.test_answer_list_plan
"""
from __future__ import annotations

from app.core.config import Settings
from app.prompts.chat_prompts import answer_list_plan, build_final_answer_prompt_v2, has_answer_list_plan
from app.schemas.chat import ArtifactResult
from tests.test_tour_location import _Base

CONTEXT = (
    "[doc_1] Título: Fé | Inventário: 60 Esc\n    Categoria: Escultura\n"
    "[doc_2] Título: São Jerónimo | Inventário: 1 Pint\n"
    "[doc_3] Título: São Jerónimo | Inventário: 2 Pint\n"
    "[doc_4] Título: 584 Esc | Inventário: 584 Esc"
)


def _card(artifact_id: str, inventory: str, title: str | None, description: str | None = None) -> ArtifactResult:
    return ArtifactResult(artifact_id=artifact_id, inventory_number=inventory, title=title, description=description)


CARDS = [
    _card("a", "60 Esc", "Fé", "Modelo em barro para estátua. Tem rosto redondo."),
    _card("b", "63 Esc", "Caridade", "Grupo escultórico com uma figura feminina e duas crianças, encomendado por D. Maria I. Está descalça."),
    _card("c", "584 Esc", None, "Grupo escultórico de Santa Ana ensinando a Virgem a ler assente sobre base quadrangular."),
]


class TestListPlanPrompt(_Base):
    def _prompt(self, **kwargs) -> str:
        base = dict(museum_name="MNAA", intent="search", mode="rag", rolling_summary="", filters_state={},
                    user_message="estátuas", rewritten_query="estátuas", retrieval_context=CONTEXT)
        base.update(kwargs)
        return build_final_answer_prompt_v2(**base)

    def test_flags_are_off_by_default(self) -> None:
        settings = Settings(_env_file=None)
        self.assertFalse(settings.CHAT_UNTITLED_CARD_BY_INVENTORY)
        self.assertFalse(settings.CHAT_ANSWER_LIST_PLAN)
        self.assertFalse(settings.CHAT_MISSING_CARDS_IN_LIST)

    def test_plan_names_every_card_in_order(self) -> None:
        plan = answer_list_plan(CONTEXT, "pt")
        self.assertTrue(plan.startswith("LISTA A ESCREVER, a seguir à frase de abertura. As 4 linhas"))
        self.assertEqual(plan.split("\n")[1:], ["1. Fé –", "2. São Jerónimo (1 Pint) –", "3. São Jerónimo (2 Pint) –", "4. 584 Esc –"])
        self.assertTrue(answer_list_plan(CONTEXT, "en").startswith("LIST TO WRITE, after the opening sentence. The 4 lines"))
        self.assertEqual(answer_list_plan("[doc_1] Título: Fé | Inventário: 60 Esc", "pt"), "")

    def test_plan_goes_in_the_prompt_only_when_it_lists_every_card(self) -> None:
        prompt = self._prompt(all_cards=True, list_plan=True)
        self.assertTrue(has_answer_list_plan(prompt))
        self.assertLess(prompt.index("LISTA A ESCREVER"), prompt.index("CARTÕES CONFIRMADOS:"))
        self.assertTrue(has_answer_list_plan(self._prompt(all_cards=True, list_plan=True, language="en")))
        for kwargs in (dict(list_plan=True), dict(all_cards=True), dict(all_cards=True, list_plan=True, select_cards=True),
                       dict(all_cards=True, list_plan=True, retrieval_context="[doc_1] Título: Fé | Inventário: 60 Esc")):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(has_answer_list_plan(self._prompt(**kwargs)))
        # um cartão que fale da «lista a escrever» não conta como plano
        self.assertFalse(has_answer_list_plan(self._prompt(retrieval_context=CONTEXT + "\n    Descrição: LISTA A ESCREVER, não")))

    def test_untitled_card_goes_by_its_inventory_number(self) -> None:
        docs = [{"title": None, "inventory_number": "584 Esc", "category": "Escultura"}]
        off = self._service(self._settings())._format_docs_compact(docs=docs)
        on = self._service(self._settings(CHAT_UNTITLED_CARD_BY_INVENTORY=True))._format_docs_compact(docs=docs)
        self.assertIn("[doc_1] Título: sem título | Inventário: 584 Esc", off)
        self.assertIn("[doc_1] Título: 584 Esc | Inventário: 584 Esc", on)
        no_inventory = self._service(self._settings(CHAT_UNTITLED_CARD_BY_INVENTORY=True))._format_docs_compact(docs=[{"title": ""}])
        self.assertIn("Título: sem título", no_inventory)


class TestListAsPlanned(_Base):
    def _service_on(self, **extra):
        return self._service(self._settings(
            CHAT_ANSWER_ALL_CARDS=True, CHAT_ANSWER_LIST_PLAN=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True, **extra))

    def test_a_complete_list_stays_and_what_comes_after_goes(self) -> None:
        text = ("Encontrei três esculturas.\n1. Fé – Modelo em barro.\n2. **Caridade** – Grupo com duas crianças.\n"
                "3. 584 Esc – Santa Ana e a Virgem.\n\nTodas estão na visita virtual.")
        self.assertEqual(
            self._service_on()._mention_every_card(text, CARDS, language="pt", list_planned=True),
            "Encontrei três esculturas.\n1. Fé – Modelo em barro.\n2. **Caridade** – Grupo com duas crianças.\n3. 584 Esc – Santa Ana e a Virgem.",
        )

    def test_a_card_left_out_or_said_elsewhere_gets_its_line(self) -> None:
        text = ("Encontrei esculturas:\n1. Fé – Modelo em barro.\n2. Caridade – Grupo com Santa Ana.\n\n"
                "A peça Santa Ana ensinando a Virgem a ler (Inv. 584 Esc) também está no museu.")
        self.assertEqual(
            self._service_on()._mention_every_card(text, CARDS, language="pt", list_planned=True),
            "Encontrei esculturas:\n1. Fé – Modelo em barro.\n2. Caridade – Grupo com Santa Ana.\n"
            "3. 584 Esc – Grupo escultórico de Santa Ana ensinando a Virgem a ler assente sobre base quadrangular.",
        )

    def test_lines_follow_the_cards_order_and_numbers(self) -> None:
        text = "Eis:\n- Caridade: grupo.\n- Fé: modelo.\n- Fénix: outra coisa."
        self.assertEqual(
            self._service_on()._mention_every_card(text, CARDS, language="pt", list_planned=True),
            "Eis:\n1. Fé: modelo.\n2. Caridade: grupo.\n"
            "3. 584 Esc – Grupo escultórico de Santa Ana ensinando a Virgem a ler assente sobre base quadrangular.",
        )

    def test_prose_becomes_the_list(self) -> None:
        text = "No museu há obras notáveis. A Fé e a Caridade são modelos em barro, e há ainda um grupo de Santa Ana."
        self.assertEqual(
            self._service_on()._mention_every_card(text, CARDS, language="pt", list_planned=True),
            "No museu há obras notáveis.\n1. Fé – Modelo em barro para estátua.\n"
            "2. Caridade – Grupo escultórico com uma figura feminina e duas crianças, encomendado por D. Maria I.\n"
            "3. 584 Esc – Grupo escultórico de Santa Ana ensinando a Virgem a ler assente sobre base quadrangular.",
        )

    def test_one_line_per_piece_without_markers_keeps_her_sentences(self) -> None:
        text = ("O museu tem três esculturas.\nA Fé, do século XVIII, é um modelo em barro.\n"
                "O grupo é notável.\n**Caridade** mostra duas crianças.\nTodas estão na visita.")
        self.assertEqual(
            self._service_on()._mention_every_card(text, CARDS, language="pt", list_planned=True),
            "O museu tem três esculturas.\n1. A Fé, do século XVIII, é um modelo em barro.\n2. **Caridade** mostra duas crianças.\n"
            "3. 584 Esc – Grupo escultórico de Santa Ana ensinando a Virgem a ler assente sobre base quadrangular.",
        )

    def test_a_line_about_another_piece_with_the_same_title_is_not_taken(self) -> None:
        cards = [_card("b1", "1638 Mov", "Biombo (par)", "Biombo de seis folhas."), _card("b2", "1639 Mov", "Biombo (par)", "Outro biombo.")]
        text = "Dois biombos.\nO Biombo (par) de 1639 Mov tem pó de prata."
        self.assertEqual(self._service_on()._mention_every_card(text, cards, language="pt", list_planned=True),
                         "Dois biombos.\n1. Biombo (par) (1638 Mov) – Biombo de seis folhas.\n2. O Biombo (par) de 1639 Mov tem pó de prata.")

    def test_english_fallback_line_has_no_portuguese_sentence(self) -> None:
        text = "I found sculptures:\n1. Fé – A clay model."
        self.assertEqual(
            self._service_on()._mention_every_card(text, CARDS, language="en", list_planned=True),
            "I found sculptures:\n1. Fé – A clay model.\n2. Caridade (63 Esc)\n3. 584 Esc",
        )

    def test_same_title_cards_are_told_apart_by_inventory(self) -> None:
        cards = [_card("j1", "1 Pint", "São Jerónimo", "Pintura a óleo."), _card("j2", "2 Pint", "São Jerónimo", "Gravura.")]
        service = self._service_on()
        self.assertEqual(service._answer_list_names(cards), ["São Jerónimo (1 Pint)", "São Jerónimo (2 Pint)"])
        text = "Dois:\n1. São Jerónimo (2 Pint) – a gravura.\n2. São Jerónimo – o óleo."
        self.assertEqual(service._mention_every_card(text, cards, language="pt", list_planned=True),
                         "Dois:\n1. São Jerónimo – o óleo.\n2. São Jerónimo (2 Pint) – a gravura.")

    def test_without_the_plan_in_the_prompt_nothing_changes(self) -> None:
        text = "A mais antiga é a Fé. A Caridade e a 584 Esc são posteriores.\n\nQuer saber mais?"
        service = self._service_on()
        self.assertEqual(service._mention_every_card(text, CARDS, language="pt"), text)
        off = self._service(self._settings(CHAT_ANSWER_ALL_CARDS=True))
        listed = "Eis:\n1. Fé – modelo.\n2. Caridade – grupo.\n3. 584 Esc – grupo.\n\nFecho."
        self.assertEqual(off._mention_every_card(listed, CARDS, language="pt", list_planned=True), listed)


class TestMissingCardsInList(_Base):
    TEXT = "Encontrei esculturas:\n\n1. **Fé** – Modelo em barro.\n\nSão estudos para estátuas."

    def test_missing_cards_become_items_of_the_list(self) -> None:
        service = self._service(self._settings(
            CHAT_ANSWER_ALL_CARDS=True, CHAT_MISSING_CARDS_IN_LIST=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True))
        self.assertEqual(
            service._mention_every_card(self.TEXT, CARDS, language="pt"),
            "Encontrei esculturas:\n\n1. **Fé** – Modelo em barro.\n"
            "2. **Caridade** – Grupo escultórico com uma figura feminina e duas crianças, encomendado por D. Maria I.\n"
            "3. **584 Esc** – Grupo escultórico de Santa Ana ensinando a Virgem a ler assente sobre base quadrangular.\n"
            "\nSão estudos para estátuas.",
        )

    def test_bullets_and_separator_follow_the_last_item(self) -> None:
        service = self._service(self._settings(CHAT_ANSWER_ALL_CARDS=True, CHAT_MISSING_CARDS_IN_LIST=True))
        text = "Eis:\n- Fé: modelo em barro.\n- Caridade: grupo."
        self.assertEqual(service._mention_every_card(text, CARDS[:2] + [_card("d", "9 Esc", "Diana")], language="pt"),
                         "Eis:\n- Fé: modelo em barro.\n- Caridade: grupo.\n- Diana (9 Esc)")

    def test_without_a_list_the_last_line_stays(self) -> None:
        service = self._service(self._settings(
            CHAT_ANSWER_ALL_CARDS=True, CHAT_MISSING_CARDS_IN_LIST=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True))
        self.assertEqual(service._mention_every_card("Destaco a Fé.", CARDS, language="pt"),
                         "Destaco a Fé.\n\nNos cartões estão também: **Caridade** (63 Esc), **584 Esc**.")
        legacy = self._service(self._settings(CHAT_ANSWER_ALL_CARDS=True))
        self.assertEqual(legacy._mention_every_card("Destaco a Fé.", CARDS, language="pt"),
                         "Destaco a Fé.\n\nNos cartões estão também: **Caridade** (63 Esc), a peça 584 Esc.")

    def test_first_sentence_is_one_line_with_a_full_stop(self) -> None:
        service = self._service(self._settings())
        self.assertEqual(service._first_sentence("Encomenda de D. Maria I para o convento.\nOutra frase."), "Encomenda de D. Maria I para o convento.")
        self.assertEqual(service._first_sentence("Grupo escultórico"), "Grupo escultórico.")
        self.assertEqual(service._first_sentence(None), "")
        long = service._first_sentence("palavra " * 60)
        self.assertTrue(long.endswith("…") and len(long) <= 161)
        comma = service._first_sentence("Numa paisagem rochosa em que se pode observar casario, " + "palavra " * 40)
        self.assertEqual(comma, "Numa paisagem rochosa em que se pode observar casario.")


if __name__ == "__main__":
    import unittest

    unittest.main()
