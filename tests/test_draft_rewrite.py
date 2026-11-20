"""O pré-processamento escreve o rascunho e a AMÁLIA só o reescreve (CHAT_DRAFT_THEN_REWRITE).
docs/rascunho-e-reescrita.md.

    python -m unittest tests.test_draft_rewrite
"""
from __future__ import annotations

import asyncio
import json
import re

from app.core.config import Settings
from app.prompts.chat_prompts import build_draft_opening_prompt, build_draft_sentence_prompt, build_rewrite_prompt
from app.schemas.chat import ResponseFormatObject
from app.services.draft_rewrite import AnswerDraft, DraftRewriteMerger, merge_rewrite
from app.services.llm_service import LLMResponse, LLMServiceError
from tests.test_answer_list_plan import CARDS, _card
from tests.test_author_before_answer import _Flow, _Pre
from tests.test_tour_location import _Base

DRAFT = AnswerDraft(
    opening="Estas são as esculturas encontradas.",
    names=["Fé", "Caridade", "584 Esc"],
    lines=["1. **Fé** – Modelo em barro.", "2. **Caridade** – Grupo com duas crianças.", "3. **584 Esc** – Santa Ana a ensinar a Virgem."],
)
REWRITTEN = ("Aqui tem as esculturas que encontrei.\n\n"
             "1. **Fé** – Um modelo em barro.  \n2. **Caridade** – Um grupo com duas crianças.\n3. **584 Esc** – Santa Ana ensina a Virgem a ler.\n\n"
             "Espero que goste da visita!")
FINAL = ("Aqui tem as esculturas que encontrei.\n\n"
         "1. **Fé** – Um modelo em barro.\n2. **Caridade** – Um grupo com duas crianças.\n3. **584 Esc** – Santa Ana ensina a Virgem a ler.")


class TestMerger(_Base):
    def test_off_by_default(self) -> None:
        self.assertFalse(Settings(_env_file=None).CHAT_DRAFT_THEN_REWRITE)

    def test_her_lines_stay_and_what_she_adds_does_not(self) -> None:
        self.assertEqual(merge_rewrite(DRAFT, REWRITTEN), (FINAL, 3))

    def test_a_line_that_is_not_the_drafts_gives_way_to_the_draft(self) -> None:
        for name, wrote, expected, kept in [
            ("nada", "", DRAFT.text, 0),
            ("outro nome", "Olá.\n1. **Fé** – a.\n2. **Esperança** – b.\n3. **584 Esc** – c.",
             "Olá.\n\n1. **Fé** – a.\n2. **Caridade** – Grupo com duas crianças.\n3. **584 Esc** – c.", 2),
            ("sem negrito", "Olá.\n1. Fé – a.\n2. **Caridade** – b.\n3. **584 Esc** – c.",
             "Olá.\n\n1. **Fé** – Modelo em barro.\n2. **Caridade** – b.\n3. **584 Esc** – c.", 2),
            ("sem frase", "Olá.\n1. **Fé**\n2. **Caridade** – b.\n3. **584 Esc** –",
             "Olá.\n\n1. **Fé** – Modelo em barro.\n2. **Caridade** – b.\n3. **584 Esc** – Santa Ana a ensinar a Virgem.", 1),
            ("salta uma linha", "Olá.\n1. **Fé** – a.\n3. **584 Esc** – c.",
             "Olá.\n\n1. **Fé** – a.\n2. **Caridade** – Grupo com duas crianças.\n3. **584 Esc** – c.", 2),
            ("fora de ordem", "1. **Fé** – a.\n3. **584 Esc** – c.\n2. **Caridade** – b.",
             f"{DRAFT.opening}\n\n1. **Fé** – a.\n2. **Caridade** – Grupo com duas crianças.\n3. **584 Esc** – c.", 2),
            ("linhas a mais", "Olá.\n1. **Fé** – a.\n2. **Caridade** – b.\n3. **584 Esc** – c.\n4. **Outra** – d.\nE mais isto.",
             "Olá.\n\n1. **Fé** – a.\n2. **Caridade** – b.\n3. **584 Esc** – c.", 3),
            ("prosa", "Encontrei três esculturas muito bonitas, todas em barro.",
             "Encontrei três esculturas muito bonitas, todas em barro.\n\n" + "\n".join(DRAFT.lines), 0),
        ]:
            with self.subTest(case=name):
                self.assertEqual(merge_rewrite(DRAFT, wrote), (expected, kept))

    def test_an_opening_that_already_talks_about_the_pieces_gives_way_to_the_drafts(self) -> None:
        # caso real: a AMÁLIA abriu com um parágrafo sobre as três primeiras peças e depois a lista
        wrote = "Bem-vindos. Contemplamos aqui **Fé**, modelo em barro, e **Caridade**, grupo com crianças.\n\n1. **Fé** – a.\n2. **Caridade** – b.\n3. **584 Esc** – c."
        self.assertEqual(merge_rewrite(DRAFT, wrote)[0], f"{DRAFT.opening}\n\n1. **Fé** – a.\n2. **Caridade** – b.\n3. **584 Esc** – c.")
        long_opening = "Bem-vindos! " + "Esta é uma coleção notável de escultura em barro. " * 8
        self.assertTrue(merge_rewrite(DRAFT, long_opening + "\n1. **Fé** – a.")[0].startswith(DRAFT.opening))

    def test_a_fixed_opening_is_never_replaced(self) -> None:
        fixed = AnswerDraft(opening="O museu tem 3 peças de X. São estas:", names=DRAFT.names, lines=DRAFT.lines, fixed_opening=True)
        text, kept = merge_rewrite(fixed, REWRITTEN)
        self.assertTrue(text.startswith("O museu tem 3 peças de X. São estas:\n\n1. **Fé** – Um modelo em barro."))
        self.assertEqual(kept, 3)

    def test_the_text_only_grows(self) -> None:
        merger = DraftRewriteMerger(DRAFT)
        seen = [merger.text()]
        for line in REWRITTEN.split("\n"):
            merger.feed(line)
            seen.append(merger.text())
        merger.finish()
        final = merger.text()
        self.assertEqual(final, FINAL)
        for earlier, later in zip(seen, seen[1:] + [final]):
            self.assertTrue(later.startswith(earlier), (earlier, later))


class TestRewriteStream(_Base):
    TEXTS = [REWRITTEN, "", "1. **Fé** – a.\n3. **584 Esc** – c.", "Só prosa, sem lista nenhuma.", "Olá.\n1. Fé – a.\n2. **Caridade** – b.\n3. **584 Esc** – c."]

    def _run(self, text: str, size: int, draft: AnswerDraft = DRAFT) -> tuple[list[str], str]:
        service = self._service(self._settings())
        sent: list[str] = []

        def factory():
            async def emit(delta: str) -> None:
                sent.append(delta)
            return emit

        async def run() -> str:
            wrapped, finish = service._rewrite_stream(factory, draft, lambda line: line)
            emit = wrapped()
            for start in range(0, len(text), size):
                await emit(text[start:start + size])
            final, _kept = merge_rewrite(draft, text)
            partial = "".join(sent)
            assert final.startswith(partial), (partial, final)
            await finish(final)
            return final

        final = asyncio.run(run())
        return sent, final

    def test_what_is_sent_is_the_final_text(self) -> None:
        fixed = AnswerDraft(opening="O museu tem 3 peças de X. São estas:", names=DRAFT.names, lines=DRAFT.lines, fixed_opening=True)
        for draft in (DRAFT, fixed):
            for text in self.TEXTS:
                for size in (1, 3, 8, 40, 10_000):
                    with self.subTest(text=text[:20], size=size, fixed=draft.fixed_opening):
                        sent, final = self._run(text, size, draft)
                        self.assertEqual("".join(sent), final)

    def test_lines_arrive_one_by_one(self) -> None:
        sent, _final = self._run(REWRITTEN, 5)
        self.assertEqual(sent[:4], ["Aqui tem as esculturas que encontrei.\n\n", "1. **Fé** – Um modelo em barro.\n",
                                    "2. **Caridade** – Um grupo com duas crianças.\n", "3. **584 Esc** – Santa Ana ensina a Virgem a ler."])
        self.assertNotIn("Espero", "".join(sent))


class _Thinker:
    """O modelo de pré-processamento: uma frase por ficha e a abertura; guarda os pedidos."""

    def __init__(self, *, fail: set[str] = frozenset(), opening: str = "Encontrei estas esculturas.") -> None:
        self.fail, self.opening, self.prompts = set(fail), opening, []

    async def generate(self, *, message, response_format=ResponseFormatObject(type="json_object"), **_kwargs):
        self.prompts.append(message)
        if "FICHAS:\n" in message:
            reply: object = {"abertura": self.opening}
        else:
            title = re.search(r"\[doc_1\] Título: (.*?) \|", message)[1]
            if title in self.fail:
                raise RuntimeError("modelo em baixo")
            reply = {"frase": f"frase da peça {title}"}
        return LLMResponse(text=json.dumps(reply), model="pre", response_format=response_format, parsed_json=reply, finish_reason="stop")


class TestDraft(_Base):
    def _draft(self, thinker: _Thinker, cards=CARDS, **kwargs) -> AnswerDraft:
        service = self._service(self._settings(CHAT_COMPACT_ANSWER_CONTEXT=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True, CHAT_IN_TOUR_FROM_NAVIGATION=True))
        service.preprocess_llm_service = thinker
        return asyncio.run(service._draft_answer(user_message="esculturas de barro", cards=cards, language="pt", **kwargs))

    def test_one_request_per_piece_with_only_its_record(self) -> None:
        thinker = _Thinker()
        draft = self._draft(thinker)
        self.assertEqual(draft.text, "Encontrei estas esculturas.\n\n1. **Fé** – frase da peça Fé.\n2. **Caridade** – frase da peça Caridade.\n"
                                     "3. **584 Esc** – frase da peça 584 Esc.")
        self.assertEqual(draft.names, ["Fé", "Caridade", "584 Esc"])
        per_piece = [prompt for prompt in thinker.prompts if "FICHA:\n" in prompt]
        self.assertEqual(len(per_piece), 3)
        for prompt in per_piece:
            # a ficha de uma peça só, com o pedido, e sem a linha da visita
            self.assertEqual(prompt.count("Título: "), 1)
            self.assertIn("«esculturas de barro»", prompt)
            self.assertNotIn("Na visita virtual", prompt)
        self.assertEqual(len(thinker.prompts), 4)   # mais a abertura, com as três fichas
        self.assertEqual(draft.info["sentences_from_model"], 3)

    def test_a_sentence_that_fails_is_the_first_of_the_record(self) -> None:
        draft = self._draft(_Thinker(fail={"Caridade"}))
        self.assertEqual(draft.lines[1], "2. **Caridade** – Grupo escultórico com uma figura feminina e duas crianças, encomendado por D. Maria I.")
        self.assertEqual(draft.info["sentences_from_model"], 2)

    def test_the_authors_opening_is_not_asked_for(self) -> None:
        thinker = _Thinker()
        draft = self._draft(thinker, fixed_opening="O museu tem 3 peças de X. São estas:")
        self.assertEqual((draft.opening, draft.fixed_opening, len(thinker.prompts)), ("O museu tem 3 peças de X. São estas:", True, 3))

    def test_repeated_titles_carry_the_inventory_number(self) -> None:
        cards = [_card("e", "1 Pint", "São Jerónimo"), _card("f", "2 Pint", "São Jerónimo"), CARDS[0]]
        draft = self._draft(_Thinker(), cards)
        self.assertEqual(draft.lines[:2], ["1. **São Jerónimo** (1 Pint) – frase da peça São Jerónimo.", "2. **São Jerónimo** (2 Pint) – frase da peça São Jerónimo."])
        text, kept = merge_rewrite(draft, "Olá.\n1. **São Jerónimo** (1 Pint) – a.\n2. **São Jerónimo** (2 Pint): b.\n3. **Fé** – c.")
        self.assertEqual((text.split("\n")[2:], kept), (["1. **São Jerónimo** (1 Pint) – a.", "2. **São Jerónimo** (2 Pint): b.", "3. **Fé** – c."], 3))

    def test_prompts(self) -> None:
        sentence = build_draft_sentence_prompt(user_message="retratos", card="[doc_1] Título: Fé | Inventário: 60 Esc")
        self.assertIn("uma frase com no máximo 15 palavras", sentence)
        self.assertNotIn("O visitante pediu", build_draft_sentence_prompt(user_message=None, card="x"))
        self.assertIn('{"abertura": "..."}', build_draft_opening_prompt(user_message="retratos", cards="x"))
        rewrite = build_rewrite_prompt(museum_name="Museu Nacional de Arte Antiga", draft=DRAFT.text)
        self.assertTrue(rewrite.endswith(f"TEXTO:\n{DRAFT.text}"))
        self.assertIn("Mantém exatamente as mesmas linhas", rewrite)
        self.assertIn("Rewrite the text below in English", build_rewrite_prompt(museum_name="MNAA", draft="x", language="en"))


class _FlowThinker(_Pre):
    """Como _Pre, e também escreve as frases e a abertura do rascunho."""

    async def generate(self, *, message, response_format=ResponseFormatObject(type="json_object"), **kwargs):
        if "FICHAS:\n" in message or "FICHA:\n" in message:
            self.prompts.append(message)
            if "FICHAS:\n" in message:
                reply: object = {"abertura": "Encontrei estas obras do pintor."}
            else:
                reply = {"frase": "frase de " + re.search(r"\[doc_1\] Título: (.*?) \|", message)[1]}
            return LLMResponse(text=json.dumps(reply), model="pre", response_format=response_format, parsed_json=reply, finish_reason="stop")
        return await super().generate(message=message, response_format=response_format, **kwargs)


class TestInTheSearch(_Flow):
    ALL = [f"doc_{n}" for n in range(1, 9)]
    FLAGS = dict(CHAT_ANSWER_ALL_CARDS=True, CHAT_ANSWER_LIST_PLAN=True, CHAT_UNTITLED_CARD_BY_INVENTORY=True, CHAT_ANSWER_BOLD_NAMES=True,
                 CHAT_AUTHOR_ONLY_FILL_PAGE=True, CHAT_TOUR_PRESENCE_ONLY_IF_ASKED=True, CHAT_STREAM_COMMITTED_LIST=True,
                 CHAT_DRAFT_THEN_REWRITE=True)
    DRAFT_LINES = [f"{n}. **Peça l{n}** – frase de Peça l{n}." for n in range(1, 9)]
    AUTHOR_OPENING = "O museu tem 19 peças de Gregório Lopes. Os cartões mostram as primeiras 8 (as restantes estão em «Ver mais resultados»):"

    def test_the_final_model_gets_the_draft_and_only_rewrites(self) -> None:
        wrote = "Com todo o gosto!\n\n" + "\n".join(f"{n}. **Peça l{n}** – reescrita {n}." for n in range(1, 9)) + "\n\nBoa visita!"
        service, llm = self._flow(_FlowThinker(cards=self.ALL), wrote, **self.FLAGS)
        response, streamed = self._ask(service)
        # o prompt da AMÁLIA é o rascunho: sem cartões, sem regras
        self.assertEqual(len(llm.prompts), 1)
        self.assertTrue(llm.prompts[0].startswith("És a Amália, guia de uma visita virtual ao Museu Nacional de Arte Antiga.\nReescreve o texto abaixo"))
        self.assertTrue(llm.prompts[0].endswith("TEXTO:\n" + self.AUTHOR_OPENING + "\n\n" + "\n".join(self.DRAFT_LINES)))
        self.assertNotIn("[doc_", llm.prompts[0])
        # a abertura do autor fica; as linhas são as dela; o fecho não entra
        self.assertEqual(response.reply, self.AUTHOR_OPENING + "\n\n" + "\n".join(f"{n}. **Peça l{n}** – reescrita {n}." for n in range(1, 9)))
        self.assertEqual(streamed, response.reply)
        self.assertEqual(service.session_store.get(response.conversation_id).history[-1].text, response.reply)

    def test_a_search_that_is_not_by_author_opens_with_the_models_sentence(self) -> None:
        wrote = "Aqui ficam as obras.\n" + "\n".join(f"{n}. **Peça {c}** – reescrita." for n, c in enumerate(["l1", "o1", "l2", "o2", "l3", "o3", "l4", "o4"], 1))
        pre = _FlowThinker(cards=self.ALL, only="B")
        service, llm = self._flow(pre, wrote, **self.FLAGS)
        response, streamed = self._ask(service, "Mostra-me obras de Gregório Lopes e de outros pintores.")
        self.assertIn("TEXTO:\nEncontrei estas obras do pintor.\n\n1. **Peça l1** – frase de Peça l1.\n2. **Peça o1**", llm.prompts[0])
        self.assertTrue(response.reply.startswith("Aqui ficam as obras.\n\n1. **Peça l1** – reescrita.\n2. **Peça o1** – reescrita."))
        self.assertEqual(streamed, response.reply)
        # uma frase por cartão e a abertura, além da escolha dos cartões e das duas perguntas do autor
        self.assertEqual(sum("FICHA:\n" in p for p in pre.prompts), 8)
        self.assertEqual(sum("FICHAS:\n" in p for p in pre.prompts), 1)

    def test_what_she_gets_wrong_stays_as_drafted(self) -> None:
        wrote = "1. **Peça l1** – reescrita 1.\n2. Peça l2 sem negrito.\n3. **Outra peça** – inventada."
        service, llm = self._flow(_FlowThinker(cards=self.ALL), wrote, **self.FLAGS)
        response, streamed = self._ask(service)
        self.assertEqual(response.reply.split("\n")[2:], ["1. **Peça l1** – reescrita 1.", *self.DRAFT_LINES[1:]])
        self.assertEqual(streamed, response.reply)

    def test_when_the_final_model_is_down_the_draft_is_the_answer(self) -> None:
        service, llm = self._flow(_FlowThinker(cards=self.ALL), "x", **self.FLAGS)

        async def down(**_kwargs):
            raise LLMServiceError("AMÁLIA em baixo")

        llm.generate_stream = down
        llm.generate = down
        response, streamed = self._ask(service)
        self.assertEqual(response.reply, self.AUTHOR_OPENING + "\n\n" + "\n".join(self.DRAFT_LINES))
        self.assertEqual(streamed, response.reply)

    def test_flag_off_the_answer_is_written_as_before(self) -> None:
        service, llm = self._flow(_FlowThinker(cards=self.ALL), "Encontrei:\n1. Peça l1 – a.", **{**self.FLAGS, "CHAT_DRAFT_THEN_REWRITE": False})
        self._ask(service)
        self.assertIn("CARTÕES CONFIRMADOS:", llm.prompts[0])

    def test_one_card_keeps_the_usual_prompt(self) -> None:
        service, llm = self._flow(_FlowThinker(cards=["doc_1"]), "A Peça l1 é uma pintura.", **self.FLAGS)
        response, _streamed = self._ask(service)
        self.assertEqual(len(response.artifact_results), 8)   # a página do autor enche-se: há lista
        service, llm = self._flow(_FlowThinker(cards=["doc_1"], name="B"), "A Peça l1 é uma pintura.", **self.FLAGS)
        response, _streamed = self._ask(service, "Mostra-me a peça da rua Gregório Lopes.")
        self.assertEqual(len(response.artifact_results), 1)
        self.assertIn("CARTÕES CONFIRMADOS:", llm.prompts[0])
