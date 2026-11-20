"""A resposta só fala da visita virtual quando o visitante a pergunta
(CHAT_TOUR_PRESENCE_ONLY_IF_ASKED) e o streaming só envia texto que fica na
resposta final (CHAT_STREAM_COMMITTED_LIST). docs/lista-da-resposta.md.

    python -m unittest tests.test_tour_presence_only_if_asked
"""
from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.schemas.chat import ChatAuthorCard, TourNavigationTarget
from tests.test_answer_list_plan import CARDS
from tests.test_author_before_answer import LIST, WROTE, _Flow, _Pre
from tests.test_tour_location import _Base

MACHADO = ChatAuthorCard(name="Machado de Castro, Joaquim", artifact_count=7, in_tour_count=3)


class TestTourTalkLeavesTheItems(_Base):
    def _service_with(self, **flags):
        base = dict(CHAT_ANSWER_ALL_CARDS=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True, CHAT_ANSWER_LIST_PLAN=True,
                    CHAT_ANSWER_BOLD_NAMES=True, CHAT_TOUR_PRESENCE_ONLY_IF_ASKED=True)
        base.update(flags)
        return self._service(self._settings(**base))

    def test_off_by_default(self) -> None:
        settings = Settings(_env_file=None)
        self.assertFalse(settings.CHAT_TOUR_PRESENCE_ONLY_IF_ASKED)
        self.assertFalse(settings.CHAT_STREAM_COMMITTED_LIST)

    def test_what_is_said_about_the_tour_leaves_and_the_description_stays(self) -> None:
        service = self._service_with()
        for sentence, expected in [
            ("Não está presente na visita virtual.", ""),
            ("Presente na visita virtual.", ""),
            ("Sim, encontra-se na visita virtual.", ""),
            ("Na visita virtual: sim", ""),
            ("Modelo em barro para estátua, não está presente na visita virtual.", "Modelo em barro para estátua."),
            ("Figura feminina modelada em vulto pleno, presente na visita virtual.", "Figura feminina modelada em vulto pleno."),
            ("Grupo escultórico de Santa Ana. Está nesta visita.", "Grupo escultórico de Santa Ana."),
            ("Modelo em barro mas não está na visita virtual.", "Modelo em barro."),
            ("It is not in this tour.", ""),
            ("Modelo em barro para estátua.", "Modelo em barro para estátua."),
            ("A Virgem na Visitação, em barro.", "A Virgem na Visitação, em barro."),
        ]:
            with self.subTest(sentence=sentence):
                self.assertEqual(service._without_tour_talk(sentence), expected)

    def test_items_keep_the_name_and_lose_the_tour(self) -> None:
        service = self._service_with()
        wrote = ("Encontrei 3 peças de Machado de Castro nesta visita virtual:\n\n"
                 "1. **Fé** – Não está presente na visita virtual.\n"
                 "2. Caridade – Grupo escultórico com duas crianças, presente na visita virtual.\n"
                 "3. 584 Esc (584 Esc): Na visita virtual: sim\n\n"
                 "Apenas a Caridade está acessível nesta visita virtual.")
        self.assertEqual(service._strip_tour_talk(wrote, CARDS).split("\n"), [
            "Encontrei 3 peças de Machado de Castro:", "",
            "1. **Fé**",
            "2. Caridade – Grupo escultórico com duas crianças.",
            "3. 584 Esc (584 Esc)",
            "", "Apenas a Caridade está acessível nesta visita virtual.",   # fora da lista: a garantia da lista corta o fecho
        ])

    def test_text_that_does_not_talk_about_the_tour_is_untouched(self) -> None:
        service = self._service_with()
        for text in ("", "Encontrei duas:\n1. Fé – modelo em barro.\n2. Caridade – grupo.", "As peças estão na visita virtual desde 2024."):
            self.assertEqual(service._strip_tour_talk(text, CARDS), text)

    def test_a_line_left_with_only_the_name_gets_the_sentence_of_the_record(self) -> None:
        service = self._service_with()
        wrote = service._strip_tour_talk(
            "Aqui estão:\n1. Fé – Na visita virtual: sim\n2. **Caridade** – Não está na visita virtual.\n3. 584 Esc – Grupo de Santa Ana.", CARDS)
        text = service._bold_names_in_list(service._mention_every_card(wrote, CARDS, language="pt", list_planned=True), CARDS)
        self.assertEqual(text.split("\n"), [
            "Aqui estão:",
            "1. **Fé** – Modelo em barro para estátua.",
            "2. **Caridade** – Grupo escultórico com uma figura feminina e duas crianças, encomendado por D. Maria I.",
            "3. **584 Esc** – Grupo de Santa Ana.",
        ])
        # sem a flag, a linha só com o nome fica como a AMÁLIA a deixou
        off = self._service_with(CHAT_TOUR_PRESENCE_ONLY_IF_ASKED=False)
        self.assertIn("1. **Fé**\n", off._bold_names_in_list(off._mention_every_card("Aqui estão:\n1. Fé\n2. Caridade – x.\n3. 584 Esc – y.", CARDS, language="pt", list_planned=True), CARDS))

    def test_asking_about_the_tour(self) -> None:
        service = self._service_with()
        for message in ("Que obras de Machado de Castro estão nesta visita?", "está na visita virtual?", "which ones are in this tour?"):
            self.assertTrue(service._asks_about_tour(message), message)
        for message in ("Encontra as obras de Machado de Castro neste museu?", "retratos de Sequeira", ""):
            self.assertFalse(service._asks_about_tour(message), message)
        self.assertTrue(service._asks_about_tour("retratos", retrieval_request={"tour_scope": {"in_tour": True}}))

    def test_the_authors_opening_without_the_tour_count(self) -> None:
        service = self._service_with(CHAT_AUTHOR_ANSWER_COUNTS=True)
        self.assertEqual(service._author_counts_opening(author=MACHADO, total=7, shown=7, language="pt", tour_presence=False),
                         "O museu tem 7 peças de Joaquim Machado de Castro. São estas:")
        self.assertEqual(service._author_counts_opening(author=MACHADO, total=7, shown=7, language="pt"),
                         "O museu tem 7 peças de Joaquim Machado de Castro, 3 delas nesta visita. São estas:")

    def test_prompt_cards_without_the_tour_line(self) -> None:
        service = self._service(self._settings(CHAT_COMPACT_ANSWER_CONTEXT=True, CHAT_IN_TOUR_FROM_NAVIGATION=True))
        targets = [TourNavigationTarget(overlay_id="ov", panorama_key="p", inventory_id="60 Esc")]
        common = dict(artifact_results=CARDS, image_matches=[], page=1, page_size=8, total=3, prefer_artifact_results=True,
                      include_image_matches_section=False, navigation_targets=targets)
        self.assertIn("Na visita virtual: sim", service._build_visible_results_retrieval_context(**common))
        self.assertNotIn("Na visita virtual", service._build_visible_results_retrieval_context(**common, tour_presence=False))


class TestCommittedListStream(_Base):
    """O streaming só envia texto que vai estar na resposta final."""

    def _run(self, text: str, size: int, *, opening: str | None = None, **flags):
        base = dict(CHAT_ANSWER_ALL_CARDS=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True, CHAT_ANSWER_LIST_PLAN=True,
                    CHAT_ANSWER_BOLD_NAMES=True, CHAT_TOUR_PRESENCE_ONLY_IF_ASKED=True)
        base.update(flags)
        service = self._service(self._settings(**base))
        sent: list[str] = []

        def finish_text(raw: str, report: dict | None = None) -> str:
            out = service._mention_every_card(service._strip_tour_talk(raw, CARDS), CARDS, language="pt", list_planned=True, report=report)
            out = service._bold_names_in_list(out, CARDS)
            return f"{opening}\n\n{out[out.index('1. '):]}" if opening else out

        def factory():
            async def emit(delta: str) -> None:
                sent.append(delta)
            return emit

        async def run() -> str:
            wrapped, finish = service._committed_list_stream(factory, finish_text, opening=opening)
            emit = wrapped()
            seen: list[str] = []
            for start in range(0, len(text), size):
                await emit(text[start:start + size])
                seen.append("".join(sent))
            final = finish_text(text)
            for partial in seen:
                # tudo o que foi enviado, em qualquer momento, é o princípio do texto final
                assert final.startswith(partial), (partial, final)
            before_finish = "".join(sent)
            await finish(final)
            return before_finish, final

        before_finish, final = asyncio.run(run())
        return before_finish, "".join(sent), final

    FOLLOWED = ("Aqui estão as peças:\n\n1. Fé – Modelo em barro.\n2. Caridade – Grupo escultórico.\n3. 584 Esc – Grupo de Santa Ana.\n\n"
                "Todas as peças estão presentes nesta visita virtual.")
    TEXTS = {
        "lista seguida e fecho": FOLLOWED,
        "itens sobre a visita": "Aqui estão:\n1. Fé – Não está presente na visita virtual.\n2. Caridade – Na visita virtual: não\n3. 584 Esc – Presente na visita virtual.",
        "falta um cartão": "Aqui estão:\n1. Fé – Modelo em barro.\n3. 584 Esc – Grupo de Santa Ana.",
        "fora de ordem": "Aqui estão:\n1. Caridade – Grupo.\n2. Fé – Modelo.\n3. 584 Esc – Santa Ana.",
        "prosa": "Encontrei três obras. A Fé é um modelo em barro e a Caridade um grupo escultórico; há ainda a 584 Esc.",
        "sem marcadores": "Aqui estão as peças:\nFé – Modelo em barro.\nCaridade – Grupo escultórico.\n584 Esc – Grupo.",
    }

    def test_everything_sent_is_the_start_of_the_final_text(self) -> None:
        for name, text in self.TEXTS.items():
            for opening in (None, "O museu tem 3 peças de Joaquim Machado de Castro. São estas:"):
                for size in (1, 4, 9, 50, 10_000):
                    with self.subTest(text=name, opening=bool(opening), size=size):
                        _before, streamed, final = self._run(text, size, opening=opening)
                        self.assertEqual(streamed, final)

    def test_lines_arrive_as_they_will_stay_and_the_closing_never_shows(self) -> None:
        before_finish, streamed, final = self._run(self.FOLLOWED, 6)
        self.assertEqual(final, "Aqui estão as peças:\n1. **Fé** – Modelo em barro.\n2. **Caridade** – Grupo escultórico.\n3. **584 Esc** – Grupo de Santa Ana.")
        # as três linhas chegaram enquanto ela escrevia; o fecho nunca foi enviado
        self.assertEqual(before_finish, final)
        self.assertNotIn("Todas as peças", streamed)

    def test_prose_shows_nothing_until_the_final_list(self) -> None:
        before_finish, streamed, final = self._run(self.TEXTS["prosa"], 5)
        self.assertEqual(before_finish, "")
        self.assertEqual(streamed, final)
        self.assertIn("1. **Fé** – Modelo em barro para estátua.", final)

    def test_the_authors_opening_is_sent_with_the_first_token(self) -> None:
        opening = "O museu tem 3 peças de Joaquim Machado de Castro. São estas:"
        before_finish, _streamed, _final = self._run(self.TEXTS["prosa"], 5, opening=opening)
        self.assertEqual(before_finish, f"{opening}\n\n")

    def test_a_card_she_skips_holds_the_lines_after_it(self) -> None:
        before_finish, streamed, final = self._run(self.TEXTS["falta um cartão"], 7)
        self.assertEqual(before_finish, "Aqui estão:\n1. **Fé** – Modelo em barro.\n")
        self.assertEqual(streamed, final)


class TestInTheSearch(_Flow):
    """A pesquisa de ponta a ponta, com o streaming ligado."""

    ALL = [f"doc_{n}" for n in range(1, 9)]
    FLAGS = dict(CHAT_ANSWER_ALL_CARDS=True, CHAT_ANSWER_LIST_PLAN=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True, CHAT_ANSWER_BOLD_NAMES=True,
                 CHAT_AUTHOR_ONLY_FILL_PAGE=True, CHAT_TOUR_PRESENCE_ONLY_IF_ASKED=True, CHAT_STREAM_COMMITTED_LIST=True,
                 CHAT_IN_TOUR_FROM_NAVIGATION=True)
    WROTE = ("Encontrei 8 peças nesta visita virtual:\n"
             + "\n".join(f"{n}. Peça l{n} – descrição {n}, presente na visita virtual." for n in range(1, 8))
             + "\n8. Peça l8 – Na visita virtual: não\n\nApenas algumas estão acessíveis nesta visita virtual.")

    def test_title_and_description_only_and_the_stream_is_the_final_text(self) -> None:
        service, llm = self._flow(_Pre(cards=self.ALL), self.WROTE, **self.FLAGS)
        response, streamed = self._ask(service)
        # nenhum cartão do prompt leva a linha da visita (a regra que fala dela continua nas instruções)
        self.assertNotIn("    Na visita virtual:", llm.prompts[0])
        self.assertEqual(len(llm.prompts), 1)
        lines = response.reply.split("\n")
        self.assertEqual(lines[0], "O museu tem 19 peças de Gregório Lopes. Os cartões mostram as primeiras 8 (as restantes estão em «Ver mais resultados»):")
        self.assertEqual(lines[2:9], [f"{n}. **Peça l{n}** – descrição {n}." for n in range(1, 8)])
        self.assertEqual(lines[9], "8. **Peça l8** (inv l8)")   # a ficha de teste não tem descrição: fica o nome e o n.º
        self.assertEqual(len(lines), 10)
        self.assertNotIn("visita", response.reply)
        self.assertEqual(streamed, response.reply)

    def test_when_the_visitor_asks_about_the_tour_it_is_said(self) -> None:
        service, llm = self._flow(_Pre(cards=self.ALL), self.WROTE, **self.FLAGS)
        response, streamed = self._ask(service, "Que obras de Gregório Lopes estão nesta visita?")
        self.assertIn("    Na visita virtual:", llm.prompts[0])
        self.assertIn("presente na visita virtual", response.reply)
        self.assertEqual(streamed, response.reply)

    def test_without_the_list_in_the_prompt_the_stream_is_as_before(self) -> None:
        service, llm = self._flow(_Pre(cards=self.ALL), WROTE, **{**self.FLAGS, "CHAT_ANSWER_LIST_PLAN": False, "CHAT_ANSWER_ALL_CARDS": False})
        response, streamed = self._ask(service)
        self.assertTrue(response.reply.endswith(LIST))
        self.assertEqual(streamed, response.reply)
